import copy
import torch
import torch.nn as nn

# --------------- External components ---------------
from utils.misc import multiclass_nms

# --------------- Model components ---------------
try:
    from .backbone import ResNet
    from .fpn      import BasicFPN
    from .head     import FcosRTHead
except:
    from  backbone import ResNet
    from  fpn      import BasicFPN
    from  head     import FcosRTHead


# --------------------- Real-time FCOS ---------------------
class FcosRT(nn.Module):
    def __init__(self, 
                 cfg,
                 conf_thresh  :float = 0.05,
                 nms_thresh   :float = 0.6,
                 topk_results :int   = 1000,
                 ca_nms       :bool  = False):
        super(FcosRT, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.ca_nms = ca_nms
        self.nms_thresh   = nms_thresh
        self.conf_thresh  = conf_thresh
        self.num_classes  = cfg.num_classes
        self.topk_results = topk_results

        # ---------------------- Network Parameters ----------------------
        ## Backbone
        self.backbone = ResNet(cfg.backbone, cfg.bk_norm, cfg.res5_dilation, cfg.freeze_at, cfg.use_pretrained)
        self.pyramid_feats = self.backbone.feat_dims

        ## Feature pyramid network
        self.fpn = BasicFPN(self.pyramid_feats, cfg.fpn_dim, cfg.fpn_p6_feat, cfg.fpn_p7_feat, cfg.fpn_p6_from_c5)
        
        ## Heads
        self.head = FcosRTHead(self.fpn.out_dim, cfg.cls_head_dim, cfg.reg_head_dim, cfg.num_cls_heads,
                               cfg.num_reg_heads, cfg.head_act, cfg.head_norm, cfg.num_classes, cfg.out_stride)

    def post_process(self, cls_preds, box_preds):
        """
        Input:
            cls_preds: List(Tensor) [[B, H x W, C], ...]
            box_preds: List(Tensor) [[B, H x W, 4], ...]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for cls_pred_i, box_pred_i in zip(cls_preds, box_preds):
            cls_pred_i = cls_pred_i[0]
            box_pred_i = box_pred_i[0]
            
            # (H x W x C,)
            scores_i = cls_pred_i.sigmoid().flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_results, box_pred_i.size(0))

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores_i.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            topk_idxs = topk_idxs[keep_idxs]

            # final scores
            scores = topk_scores[keep_idxs]
            # final labels
            labels = topk_idxs % self.num_classes
            # final bboxes
            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            bboxes = box_pred_i[anchor_idxs]

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # to cpu & numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # NMS
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, self.ca_nms)

        return bboxes, scores, labels

    def forward(self, src, src_mask=None):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(src)

        # ---------------- Neck ----------------
        pyramid_feats = self.fpn(pyramid_feats)

        # ---------------- Heads ----------------
        outputs = self.head(pyramid_feats, src_mask)

        if not self.training:
            # ---------------- PostProcess ----------------
            cls_pred = outputs["pred_cls"]
            box_pred = outputs["pred_box"]
            bboxes, scores, labels = self.post_process(cls_pred, box_pred)
            # normalize bbox
            bboxes[..., 0::2] /= src.shape[-1]
            bboxes[..., 1::2] /= src.shape[-2]
            bboxes = bboxes.clip(0., 1.)

            outputs = {
                'scores': scores,
                'labels': labels,
                'bboxes': bboxes
            }

        return outputs 

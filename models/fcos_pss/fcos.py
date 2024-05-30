import copy
import torch
import torch.nn as nn

# --------------- External components ---------------
from utils.misc import multiclass_nms

# --------------- Model components ---------------
try:
    from .backbone import ResNet
    from .fpn      import BasicFPN
    from .head     import FcosPSSHead
except:
    from  backbone import ResNet
    from  fpn      import BasicFPN
    from  head     import FcosPSSHead


# --------------------- Real-time FCOS ---------------------
class FcosPSS(nn.Module):
    def __init__(self, 
                 cfg,
                 conf_thresh  :float = 0.05,
                 topk_results :int   = 1000,
                 ca_nms       :bool  = False):
        super(FcosPSS, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.ca_nms = ca_nms
        self.conf_thresh  = conf_thresh
        self.num_classes  = cfg.num_classes
        self.topk_results = topk_results

        # ---------------------- Network Parameters ----------------------
        ## Backbone
        self.backbone = ResNet(cfg.backbone, cfg.bk_norm, cfg.res5_dilation, cfg.freeze_at, cfg.use_pretrained)

        ## Feature pyramid network
        self.backbone_fpn = BasicFPN(self.backbone.feat_dims, cfg.fpn_dim, cfg.fpn_p6_feat, cfg.fpn_p7_feat, cfg.fpn_p6_from_c5)
        
        ## Heads
        self.detection_head = FcosPSSHead(self.backbone_fpn.out_dim, cfg.cls_head_dim, cfg.reg_head_dim, cfg.num_cls_heads,
                                          cfg.num_reg_heads, cfg.head_act, cfg.head_norm, cfg.num_classes, cfg.out_stride)

    def post_process(self, cls_preds, box_preds, pss_preds):
        """
        Input:
            cls_preds: List(Tensor) [[B, H x W, C], ...]
            box_preds: List(Tensor) [[B, H x W, 4], ...]
            pss_preds: List(Tensor) [[B, H x W, 1], ...]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for cls_pred_i, box_pred_i, pss_preds_i in zip(cls_preds, box_preds, pss_preds):
            cls_pred_i = cls_pred_i[0]
            box_pred_i = box_pred_i[0]
            
            # Reshape: [H, W, C] -> [HWC,]
            scores_i = (cls_pred_i.sigmoid() * pss_preds_i.sigmoid()).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_results, box_pred_i.size(0))
            predicted_prob, topk_idxs = scores_i.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # Filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            topk_idxs = topk_idxs[keep_idxs]

            # Final scores & labels & bboxes
            scores = topk_scores[keep_idxs]
            labels = topk_idxs % self.num_classes
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

        return bboxes, scores, labels

    def inference_o2o(self, src):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(src)

        # ---------------- Neck ----------------
        pyramid_feats = self.backbone_fpn(pyramid_feats)

        # ---------------- Heads ----------------
        outputs = self.detection_head(pyramid_feats)

        # ---------------- PostProcess ----------------
        cls_pred = outputs["pred_cls"]
        box_pred = outputs["pred_box"]
        pss_pred = outputs["pred_pss"]
        bboxes, scores, labels = self.post_process(cls_pred, box_pred, pss_pred)
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

    def forward(self, src, src_mask=None):
        if not self.training:
            return self.inference_o2o(src)
        else:
            # ---------------- Backbone ----------------
            pyramid_feats = self.backbone(src)

            # ---------------- Neck ----------------
            pyramid_feats = self.backbone_fpn(pyramid_feats)

            # ---------------- Heads ----------------
            outputs = self.detection_head(pyramid_feats, src_mask)

            return outputs 

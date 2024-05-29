# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from .criterion import FcosRTCriterion, FcosE2ECriterion
from .fcos import FcosRT, FcosE2E, FcosE2Ev2


def build_model(args, cfg, is_val=False):
    # ------------ build object detector ------------
    ## RT-FCOS    
    if   'fcos_rt' in args.model:
        model = FcosRT(cfg          = cfg,
                       conf_thresh  = cfg.train_conf_thresh if is_val else cfg.test_conf_thresh,
                       nms_thresh   = cfg.train_nms_thresh  if is_val else cfg.test_nms_thresh,
                       topk_results = cfg.train_topk        if is_val else cfg.test_topk,
                       )
        criterion = FcosRTCriterion(cfg) if is_val else None
    elif 'fcos_e2e' in args.model:
        model = FcosE2Ev2(cfg          = cfg,
                        conf_thresh  = cfg.train_conf_thresh if is_val else cfg.test_conf_thresh,
                        topk_results = cfg.train_topk        if is_val else cfg.test_topk,
                        )
        criterion = FcosE2ECriterion(cfg) if is_val else None
    else:
        raise NotImplementedError("Unknown detector: {}".args.model)

    if is_val:
        # ------------ Keep training from the given weight ------------
        if args.resume is not None and args.resume.lower() != "none":
            print('Load model from the checkpoint: ', args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            model.load_state_dict(checkpoint_state_dict)

        return model, criterion

    else:      
        return model
    
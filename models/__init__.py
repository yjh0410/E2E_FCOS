# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from .criterion import SetCriterion
from .fcos_rt  import FcosRT
from .fcos_e2e import FcosE2E


def build_model(args, cfg, is_val=False):
    # ------------ build object detector ------------
    ## RT-FCOS    
    if   'fcos_rt' in args.model:
        model = FcosRT(cfg          = cfg,
                       conf_thresh  = cfg.train_conf_thresh if is_val else cfg.test_conf_thresh,
                       nms_thresh   = cfg.train_nms_thresh  if is_val else cfg.test_nms_thresh,
                       topk_results = cfg.train_topk        if is_val else cfg.test_topk,
                       )
    elif 'fcos_e2e' in args.model:
        model = FcosE2E(cfg          = cfg,
                        conf_thresh  = cfg.train_conf_thresh if is_val else cfg.test_conf_thresh,
                        topk_results = cfg.train_topk        if is_val else cfg.test_topk,
                       )
    else:
        raise NotImplementedError("Unknown detector: {}".args.model)

    # -------------- Build Criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = SetCriterion(cfg)

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
    
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from .fcos import FcosPSS
from .criterion import SetCriterion


def build_fcos_pss(args, cfg, is_val=False):
    # ------------ build object detector ------------
    ## PSS-FCOS    
    model = FcosPSS(cfg          = cfg,
                    conf_thresh  = cfg.train_conf_thresh if is_val else cfg.test_conf_thresh,
                    topk_results = cfg.train_topk        if is_val else cfg.test_topk,
                    )
    criterion = SetCriterion(cfg) if is_val else None

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
    
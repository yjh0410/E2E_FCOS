# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from .fcos.build     import build_fcos_rt
from .fcos_e2e.build import build_fcos_e2e


def build_model(args, cfg, is_val=False):
    if   'fcos_rt' in args.model:
        model, criterion = build_fcos_rt(args, cfg, is_val)

    elif 'fcos_e2e' in args.model:
        model, criterion = build_fcos_e2e(args, cfg, is_val)
        
    else:
        raise NotImplementedError("Unknown detector: {}".format(args.model))

    if criterion is None:
        return model
    else:
        return model, criterion
    
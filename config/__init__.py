# ----------------------- Model Config -----------------------
from .fcos_config      import build_fcos_config

def build_config(args):
    # FCOS
    if   "fcos" in args.model:
        cfg = build_fcos_config(args)
    else:
        raise NotImplementedError('Unknown Model: {}'.format(args.model))

    # Print model config
    cfg.print_config()

    return cfg

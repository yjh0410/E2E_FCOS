import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet

try:
    from .modules import FrozenBatchNorm2d
except:
    from  modules import FrozenBatchNorm2d


# IN1K-Cls pretrained weights
model_urls = {
    'resnet18':  resnet.ResNet18_Weights,
    'resnet34':  resnet.ResNet34_Weights,
    'resnet50':  resnet.ResNet50_Weights,
    'resnet101': resnet.ResNet101_Weights,
}


# -------------------- ResNet series --------------------
class ResNet(nn.Module):
    def __init__(self, model_name, bk_norm="bn", res5_dilation=False, freeze_at=0, use_pretrained=False):
        super().__init__()
        # Pretrained
        if use_pretrained:
            pretrained_weights = model_urls[model_name].IMAGENET1K_V1
        else:
            pretrained_weights = None

        # Norm layer
        print("- Norm layer of backbone: {}".format(bk_norm))
        if   bk_norm == 'bn':
            norm_layer = nn.BatchNorm2d
        elif bk_norm == 'frozed_bn':
            norm_layer = FrozenBatchNorm2d
        else:
            raise NotImplementedError("Unknown norm type: {}".format(bk_norm))

        # Backbone
        backbone = getattr(torchvision.models, model_name)(
            replace_stride_with_dilation=[False, False, res5_dilation],
            norm_layer=norm_layer, weights=pretrained_weights)
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.feat_dims = [128, 256, 512] if model_name in ('resnet18', 'resnet34') else [512, 1024, 2048]
 
        # Freeze
        print("- Freeze at {}".format(freeze_at))
        if freeze_at >= 0:
            for name, parameter in backbone.named_parameters():
                if   freeze_at == 0: # Only freeze stem layer
                    if 'layer1' not in name and 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                        parameter.requires_grad_(False)
                elif freeze_at == 1: # Freeze stem layer + layer1
                    if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                        parameter.requires_grad_(False)
                elif freeze_at == 2: # Freeze stem layer + layer1 + layer2
                    if 'layer3' not in name and 'layer4' not in name:
                        parameter.requires_grad_(False)
                elif freeze_at == 3: # Freeze stem layer + layer1 + layer2 + layer3
                    if 'layer4' not in name:
                        parameter.requires_grad_(False)
                else: # Freeze all resnet's layers
                    parameter.requires_grad_(False)

    def forward(self, x):
        xs = self.body(x)
        pyramid_feats = []
        for name, fmp in xs.items():
            pyramid_feats.append(fmp)

        return pyramid_feats


if __name__ == '__main__':

    class FcosBaseConfig(object):
        def __init__(self):
            self.backbone = "resnet18"
            self.bk_norm = "frozed_bn"
            self.res5_dilation = False
            self.use_pretrained = True
            self.freeze_at = 0

    cfg = FcosBaseConfig()
    model = ResNet(cfg.backbone, cfg.bk_norm, cfg.res5_dilation, cfg.freeze_at, cfg.use_pretrained)

    x = torch.randn(2, 3, 320, 320)
    output = model(x)
    for y in output:
        print(y.size())

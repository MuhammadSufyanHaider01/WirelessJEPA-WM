# shufflenet_tv_in_timm.py
# Register torchvision ShuffleNetV2 into timm's model registry.

import torch
import torch.nn as nn
import torchvision.models as tvm
from timm.models.registry import register_model
from timm.layers import SelectAdaptivePool2d


class TVShuffleNetV2(nn.Module):
    """
    Torchvision ShuffleNetV2 wrapped to look like a timm backbone.
    - supports num_classes=0 (features only)
    - supports global_pool: 'avg' (default), 'max', 'avgmax', or '' (no pool)
    Exposes:
      - .num_features
      - .feature_info  (timm-compatible feature metadata)
      - .get_downsample_ratio()
      - .forward_features(x) -> (N, C, H, W)
    """
    def __init__(self, variant='shufflenet_v2_x0_5', pretrained=False,
                 num_classes=1000, global_pool='avg'):
        super().__init__()

        # Build base torchvision model
        weights_kw = {}
        if pretrained:
            # Handle both modern and legacy torchvision weight APIs
            try:
                enum_map = {
                    'shufflenet_v2_x0_5': tvm.ShuffleNet_V2_X0_5_Weights,
                    'shufflenet_v2_x1_0': tvm.ShuffleNet_V2_X1_0_Weights,
                    'shufflenet_v2_x1_5': tvm.ShuffleNet_V2_X1_5_Weights,
                    'shufflenet_v2_x2_0': tvm.ShuffleNet_V2_X2_0_Weights,
                }
                weights_kw['weights'] = enum_map[variant].DEFAULT
            except Exception:
                weights_kw['pretrained'] = True

        tv_builder = getattr(tvm, variant)
        self.base = tv_builder(**weights_kw)

        # Adjust the input layer for (C=2, H=256, W=256)
        self.base.conv1 = nn.Conv2d(
            2, 24, kernel_size=3, stride=2, padding=1, bias=False
        )


        # Timм-style head settings
        self.num_features = getattr(self.base.fc, 'in_features', 1024)
        self.pool_type = global_pool  # 'avg' | 'max' | 'avgmax' | '' (none)

        # Classifier head: if num_classes==0, we behave like timm(num_classes=0)
        if num_classes and num_classes > 0:
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool or 'avg')
            self.classifier = nn.Linear(self.num_features, num_classes)
        else:
            self.global_pool = None
            self.classifier = nn.Identity()

        # Make a timm-compatible feature_info (channels & reduction per stage)
        # Torchvision ShuffleNetV2 reductions: 8 (stage2), 16 (stage3), 32 (stage4), 32 (conv5)
        # Channels come from the builder (x0_5 -> [24, 48, 96, 192, 1024], etc.)
        # We can infer them from the actual modules:
        #   stage2 out_chs, stage3 out_chs, stage4 out_chs, conv5 out_chs
        # For torchvision, these are stable per variant — we hardcode by variant.
        ch_map = {
            'shufflenet_v2_x0_5': [48, 96, 192, 1024],
            'shufflenet_v2_x1_0': [116, 232, 464, 1024],
            'shufflenet_v2_x1_5': [176, 352, 704, 1024],
            'shufflenet_v2_x2_0': [244, 488, 976, 2048],
        }
        c2, c3, c4, c5 = ch_map[variant]
        self.feature_info = [
            dict(num_chs=c2, reduction=8,  module='stage2'),  # after stage2
            dict(num_chs=c3, reduction=16, module='stage3'),  # after stage3
            dict(num_chs=c4, reduction=32, module='stage4'),  # after stage4
            dict(num_chs=c5, reduction=32, module='conv5'),   # after conv5
        ]

    # --- timm-like API helpers ---
    def get_downsample_ratio(self) -> int:
        return 32

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # mirror torchvision forward up to conv5 (pre-pool, pre-fc)
        x = self.base.conv1(x)
        x = self.base.maxpool(x)
        x = self.base.stage2(x)
        x = self.base.stage3(x)
        x = self.base.stage4(x)
        x = self.base.conv5(x)     # (N, C, H, W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        if self.global_pool is not None and self.pool_type != '':
            x = self.global_pool(x).flatten(1)  # (N, C)
        if isinstance(self.classifier, nn.Identity):
            return x
        return self.classifier(x)


# ---- Register factory funcs so timm.create_model(...) works ----

@register_model
def shufflenet_v2_x0_5_torchvision(pretrained=False, num_classes=1000, global_pool='avg', **kwargs):
    return TVShuffleNetV2('shufflenet_v2_x0_5', pretrained=pretrained,
                          num_classes=num_classes, global_pool=global_pool)

@register_model
def shufflenet_v2_x1_0_torchvision(pretrained=False, num_classes=1000, global_pool='avg', **kwargs):
    return TVShuffleNetV2('shufflenet_v2_x1_0', pretrained=pretrained,
                          num_classes=num_classes, global_pool=global_pool)

@register_model
def shufflenet_v2_x1_5_torchvision(pretrained=False, num_classes=1000, global_pool='avg', **kwargs):
    return TVShuffleNetV2('shufflenet_v2_x1_5', pretrained=pretrained,
                          num_classes=num_classes, global_pool=global_pool)

@register_model
def shufflenet_v2_x2_0_torchvision(pretrained=False, num_classes=1000, global_pool='avg', **kwargs):
    return TVShuffleNetV2('shufflenet_v2_x2_0', pretrained=pretrained,
                          num_classes=num_classes, global_pool=global_pool)

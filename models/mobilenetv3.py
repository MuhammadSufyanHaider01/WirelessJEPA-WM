# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in under
# https://github.com/keyu-tian/SparK/blob/main/LICENSE ur.
from typing import List

import torch
from timm.models.mobilenetv3 import MobileNetV3

# hack: inject the `get_downsample_ratio` function into `timm.models.mobilenetv3.MobileNetV3`
def get_downsample_ratio(self: MobileNetV3) -> int:
	# MobileNetV3 (like most modern convnets) downsamples by 32x by default
	return 32

# hack: inject the `get_feature_map_channels` function into `timm.models.mobilenetv3.MobileNetV3`
def get_feature_map_channels(self: MobileNetV3) -> List[int]:
	# self.feature_info is maintained by timm
	return [info['num_chs'] for info in self.feature_info[1:]]

# hack: override the forward function of `timm.models.mobilenetv3.MobileNetV3`
def forward(self, x, hierarchical=False):
	if hierarchical:
		# Use feature_info to get intermediate features
		feats = []
		for i, block in enumerate(self.blocks):
			x = block(x)
			# feature_info[1:] gives the indices of the feature maps we want
			for idx, info in enumerate(self.feature_info[1:]):
				if info['module'] == f'blocks.{i}':
					feats.append(x)
		# If not all features found, fallback to using feature_info hooks
		if len(feats) < len(self.feature_info[1:]):
			# fallback: use feature_info hooks (slower, but robust)
			feats = [f['hook'].output for f in self.feature_info[1:]]
		return feats
	else:
		x = self.forward_features(x)
		x = self.forward_head(x)
		return x

MobileNetV3.get_downsample_ratio = get_downsample_ratio
MobileNetV3.get_feature_map_channels = get_feature_map_channels
MobileNetV3.forward = forward


@torch.no_grad()
def convnet_test():
	from timm.models import create_model
	cnn = create_model('mobilenetv3_small_100')
	print('get_downsample_ratio:', cnn.get_downsample_ratio())
	print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
	downsample_ratio = cnn.get_downsample_ratio()
	feature_map_channels = cnn.get_feature_map_channels()

	# check the forward function
	B, C, H, W = 4, 3, 224, 224
	inp = torch.rand(B, C, H, W)
	feats = cnn(inp, hierarchical=True)
	assert isinstance(feats, list)
	assert len(feats) == len(feature_map_channels)
	print([tuple(t.shape) for t in feats])

	# check the downsample ratio
	feats = cnn(inp, hierarchical=True)
	assert feats[-1].shape[-2] == H // downsample_ratio
	assert feats[-1].shape[-1] == W // downsample_ratio

	# check the channel number
	for feat, ch in zip(feats, feature_map_channels):
		assert feat.ndim == 4
		assert feat.shape[1] == ch


if __name__ == '__main__':
	convnet_test()

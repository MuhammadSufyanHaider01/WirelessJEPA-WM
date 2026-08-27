# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import timm

from lightly.models.utils import deactivate_requires_grad
from lightly.transforms.ijepa_transform import IJEPATransform

# import random


# from lightly.models.utils import get_weight_decay_parameters
# from lightly.utils.lars import LARS
# from lightly.utils.scheduler import CosineWarmupScheduler

from timm.models.layers import trunc_normal_
from timm.layers import LayerNorm2d

from pretrain.trainer_common import LightlyModelMomentum, main_pretrain

import models.sparse_encoder as sparse_encoder

from pretrain.online_classification_benchmark import OnlineLinearClassificationBenckmark
from pretrain.iqfm_masks import WirelessMaskGenerator, masked_l2_loss, resize_mask

# import models.X does more than just importing the `models` module!
# It also replaces some convnext models in the `timm` model registry with a ConvNext implementation that supports sparsity.
import models.convnext
from data.hdf5_iqfmfolder import IQTransformations, ExclusiveComposeTransforms

class IJEPA_CNN(LightlyModelMomentum):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if hasattr(self.backbone, 'sparse'):
            self.backbone.sparse = True
            self.backbone_momentum.sparse = False
        self.backbone_sparse = sparse_encoder.dense_model_to_sparse(self.backbone)

        # The sparse backbone doesn't work for online eval (TODO_ fix this)
        # For now use the momentum backbone for online eval
        self.backbone_for_online_eval = self.backbone_momentum

        self.mask_token = nn.Parameter(torch.zeros(1, self.backbone.num_features, 1, 1))
        trunc_normal_(self.mask_token, mean=0, std=.02, a=-.02, b=.02)

        if self.cfg.backbone.name.lower().startswith('resnet') or self.cfg.backbone.name.lower().startswith('wide_resnet'):
            norm_cls = nn.BatchNorm2d
        elif self.cfg.backbone.name.lower().startswith('shufflenet') or self.cfg.backbone.name.lower().startswith('mobilenet'):
            norm_cls = nn.BatchNorm2d
        elif self.cfg.backbone.name.lower().startswith('convnext'):
            norm_cls = LayerNorm2d

        if self.cfg.use_projection_head:
            proj_layers = []
            for i in range(1):
                proj_layers.append(nn.Conv2d(self.backbone.num_features, 
                                            self.backbone.num_features, 
                                            kernel_size=1,
                                            padding='same'))
                proj_layers.append(norm_cls(self.backbone.num_features))
                proj_layers.append(nn.ReLU(inplace=True))
            self.projection_head = nn.Sequential(*proj_layers)

            self.projection_head_momentum = copy.deepcopy(self.projection_head)
            deactivate_requires_grad(self.projection_head_momentum)

            self.projection_head = sparse_encoder.dense_model_to_sparse(self.projection_head)
           
        pred_layers = []
        for i in range(self.cfg.predictor.n_layers):
            if self.cfg.predictor.get("dw_sep_conv", False):
                # Depthwise separable convolution
                pred_layers.append(nn.Conv2d(self.backbone.num_features, 
                                             self.backbone.num_features, 
                                             kernel_size=self.cfg.predictor.kernel_size,
                                             padding='same',
                                             groups=self.backbone.num_features))
                pred_layers.append(nn.Conv2d(self.backbone.num_features, 
                                             self.backbone.num_features, 
                                             kernel_size=1,
                                             padding='same'))
            else:
                pred_layers.append(nn.Conv2d(self.backbone.num_features, 
                                            self.backbone.num_features, 
                                            kernel_size=self.cfg.predictor.kernel_size,
                                            padding='same'))
            pred_layers.append(norm_cls(self.backbone.num_features))
            pred_layers.append(nn.ReLU(inplace=True))
        self.predictor = nn.Sequential(*pred_layers)

        self.criterion = masked_l2_loss

    def setup_transform(self):
        # self.transform = IJEPATransform(self.input_size) # RandomResizedCrop, RandomHorizontalFlip, ToTensor
        # ch_drop = ChannelDropping(drop_prob=0)
        # ch_mask = ChannelMasking(mask_prob=0.9, mask_length=160)
        # self.transform = ExclusiveComposeTransforms([
        #     IQTransformations(
        #         noise_std=0,
        #         time_shift_max=60,
        #         amplitude_scale=(1, 1),
        #         phase_shift_max=0,
        #         apply_prob=0.9,
        # )])
        self.transform = ExclusiveComposeTransforms([])

    def setup(self, stage: str) -> None:
        super().setup(stage)
        self.setup_masking()

    def setup_masking(self):
        """Initialize paper-faithful IQFM masks independently of dataset setup."""
        if self.cfg.backbone.name.lower().startswith('resnet') or self.cfg.backbone.name.lower().startswith('wide_resnet'):
            downsample_ratio = (32, 32)
        else:
            ratio = self.backbone.get_downsample_ratio()
            downsample_ratio = (ratio, ratio)

        self.latent_shape = (
            self.input_size // downsample_ratio[0],
            self.input_size // downsample_ratio[1],
        )
        self.mask_generator = WirelessMaskGenerator(
            input_size=(self.input_size, self.input_size),
            strategy=self.cfg.mask.strategy,
            patch_size=self.cfg.mask.patch_size,
            mask_ratio=self.cfg.get("mask_ratio", None),
            mask_ratio_choices=self.cfg.mask.get("mask_ratio_choices", None),
            multi_block_kwargs=self.cfg.mask.get("mutli_block_kwargs", None),
        )
        self.fmap_h = self.mask_generator.height
        self.fmap_w = self.mask_generator.width

    def get_views_to_log_from_batch(self, batch):
        inp_bchw = batch[0]
        context_mask_b1ff, target_mask_b1ff = self.mask(inp_bchw.shape[0], inp_bchw.device)  # (B, 1, f, f)

        context_mask_b1hw = resize_mask(context_mask_b1ff, inp_bchw.shape[-2:])
        target_mask_b1hw = resize_mask(target_mask_b1ff, inp_bchw.shape[-2:])

        context_bchw = inp_bchw * context_mask_b1hw
        target_bchw = inp_bchw * target_mask_b1hw
        return [inp_bchw, context_bchw, target_bchw]
    
    def contrastive_acc_eval(self, dataset, file_paths=None):
        sparse_encoder._cur_active = torch.ones_like(sparse_encoder._cur_active)
        return contrastive_acc_eval(self.backbone_momentum, dataset, input_size=self.input_size)
    
    def eval_feature_descriptors(self, dataset):
        sparse_encoder._cur_active = torch.ones_like(sparse_encoder._cur_active)
        return eval_feature_descriptors(
            self.backbone_momentum,
            dataset,
            cfg_name=self.cfg.name,
            current_epoch=self.current_epoch,
        )

    # def on_validation_epoch_end(self) -> None:
    #     mask_shape = sparse_encoder._cur_active.shape
    #     sparse_encoder._cur_active = torch.ones((1, 1, mask_shape[2], mask_shape[3]),
    #                                              device=sparse_encoder._cur_active.device)
    #     super().on_validation_epoch_end()
    
    def mask(self, B: int, device, generator=None):
        return self.mask_generator(B, device=device, generator=generator)
               

    def forward(self, x):
        inp_bchw = x
        # step1. Mask
        context_mask_b1ff, target_mask_b1ff  = self.mask(inp_bchw.shape[0], inp_bchw.device)  # (B, 1, f, f)

        # Adapt each paper mask geometry to the encoder latent grid.
        context_mask_b1ff = resize_mask(context_mask_b1ff, self.latent_shape)
        target_mask_b1ff = resize_mask(target_mask_b1ff, self.latent_shape)

        sparse_encoder._cur_active = context_mask_b1ff    # (B, 1, f, f)

        active_b1hw = resize_mask(context_mask_b1ff, inp_bchw.shape[-2:])

        masked_bchw = inp_bchw * active_b1hw
        
        # step2. Encode
        features_bcff = self.backbone_sparse(masked_bchw)

        # step 3. Project
        if self.projection_head is not None:
            features_bcff = self.projection_head(features_bcff)
        
        if features_bcff.shape[-2:] != context_mask_b1ff.shape[-2:]:
            raise RuntimeError(
                f"Backbone produced {features_bcff.shape[-2:]}, expected latent mask {context_mask_b1ff.shape[-2:]}"
            )

        # step 4. Fill-in mask tokens
        mask_tokens = self.mask_token.expand_as(features_bcff) # expands singleton dimensions to match the shape of features_bcff
        # where context_mask_b1ff is True, use features_bcff, where it's False (i.e. where it masked out a patch) use mask_tokens
        features_m_bcff = torch.where(context_mask_b1ff.expand_as(features_bcff), features_bcff, mask_tokens.to(features_bcff.dtype))   # fill in empty (non-active) positions with [mask] tokens

        z = self.predictor(features_m_bcff)

        return z, context_mask_b1ff, target_mask_b1ff

    def forward_momentum(self, x):
        z = self.backbone_momentum(x)
        if self.projection_head_momentum is not None:
            z = self.projection_head_momentum(z)
        return z.detach()

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        x = batch[0]
        p, _, target_mask_b1ff = self.forward(x)
        h = self.forward_momentum(x)
        loss = self.criterion(p, h, target_mask_b1ff)
        self.log(f"{metric_label}/ijepa_loss", loss, on_epoch=True)
        return loss
    
    # def configure_optimizers(self):
    #     # Don't use weight decay for batch norm, bias parameters, and classification
    #     # head to improve performance.
    #     params, params_no_weight_decay = get_weight_decay_parameters(
    #         [
    #             self.backbone_sparse,
    #             self.predictor,
    #         ] + 
    #         ([self.projection_head] if self.projection_head is not None else [])
    #     )
    #     param_groups = [
    #             {"name": "model", "params": params},
    #             {
    #                 "name": "model_no_weight_decay",
    #                 "params": params_no_weight_decay,
    #                 "weight_decay": 0.0,
    #             },
    #     ]
    #     # optimizer = torch.optim.AdamW(
    #     #     param_groups,
    #     #     lr=self.lr,
    #     #     weight_decay=self.cfg.optimizer.weight_decay,
    #     # )
    #     optimizer = LARS(
    #         param_groups,
    #         lr=self.cfg.optimizer.lr * self.cfg.optimizer.batch_size * self.trainer.world_size / 256,
    #         momentum=0.9,
    #         weight_decay=self.cfg.optimizer.weight_decay,
    #     )
    #     scheduler = {
    #         "scheduler": CosineWarmupScheduler(
    #             optimizer=optimizer,
    #             warmup_epochs=int(
    #                 self.trainer.estimated_stepping_batches
    #                 / self.trainer.max_epochs
    #                 * 10
    #             ),
    #             max_epochs=int(self.trainer.estimated_stepping_batches),
    #         ),
    #         "interval": "step",
    #     }
    #     return [optimizer], [scheduler]


@hydra.main(version_base="1.2", config_path="configs/", config_name="ijepacnn.yaml")
def pretrain_byol(cfg: DictConfig):
    main_pretrain(cfg, IJEPA_CNN)

def seed_all(seed: int):
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed_all(42)
    pretrain_byol()

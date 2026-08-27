"""Paper-faithful mask generation and objective helpers for WirelessJEPA."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pretrain.ijepa_mask import MultiBlockMask


PAPER_PATCH_SIZES = {
    "random": (64, 32),
    "antenna": (64, 256),
    "time": (256, 32),
    "multi-block": (64, 32),
}

STRATEGY_ALIASES = {
    "column": "time",
    "row": "antenna",
}


def _pair(value):
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"Expected a pair, got {value}")
    return (int(value[0]), int(value[1]))


def resize_mask(mask, size):
    """Resize a boolean BCHW mask with nearest-neighbor interpolation."""
    return F.interpolate(mask.float(), size=_pair(size), mode="nearest").to(torch.bool)


def masked_l2_loss(prediction, target, target_mask):
    """Squared L2 feature regression averaged over masked spatial indices."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} vs {target.shape}"
        )
    if target_mask.shape[0] != prediction.shape[0] or target_mask.shape[-2:] != prediction.shape[-2:]:
        raise ValueError(
            f"Target mask {target_mask.shape} is incompatible with features {prediction.shape}"
        )

    per_location = (prediction - target).pow(2).sum(dim=1, keepdim=True)
    target_mask = target_mask.to(device=per_location.device, dtype=per_location.dtype)
    return (per_location * target_mask).sum() / target_mask.sum().clamp_min(1.0)


class WirelessMaskGenerator:
    """Generate random, antenna, time, or multi-block masks on an IQ grid."""

    def __init__(
        self,
        input_size,
        strategy,
        patch_size,
        mask_ratio=None,
        mask_ratio_choices=None,
        multi_block_kwargs=None,
    ):
        self.input_size = _pair(input_size)
        self.strategy = STRATEGY_ALIASES.get(str(strategy), str(strategy))
        self.patch_size = _pair(patch_size)

        if self.strategy not in PAPER_PATCH_SIZES:
            raise ValueError(
                f"Unknown IQFM mask strategy {strategy!r}; expected one of {tuple(PAPER_PATCH_SIZES)}"
            )
        if self.patch_size != PAPER_PATCH_SIZES[self.strategy]:
            raise ValueError(
                f"{self.strategy} requires paper patch size {PAPER_PATCH_SIZES[self.strategy]}, "
                f"got {self.patch_size}"
            )

        height, width = self.input_size
        patch_height, patch_width = self.patch_size
        if height % patch_height or width % patch_width:
            raise ValueError(
                f"Input size {self.input_size} is not divisible by patch size {self.patch_size}"
            )
        self.height = height // patch_height
        self.width = width // patch_width

        self.mask_ratio = None if mask_ratio is None else float(mask_ratio)
        self.mask_ratio_choices = tuple(float(v) for v in (mask_ratio_choices or ()))

        self.multi_block_mask = None
        if self.strategy == "multi-block":
            kwargs = dict(multi_block_kwargs or {})
            self.multi_block_mask = MultiBlockMask(
                input_size=self.input_size,
                patch_size=self.patch_size,
                **kwargs,
            )
        elif not self.mask_ratio_choices and self.mask_ratio is None:
            raise ValueError(f"{self.strategy} masking requires mask_ratio or mask_ratio_choices")

    def _select_mask_ratio(self, generator=None):
        if not self.mask_ratio_choices:
            ratio = self.mask_ratio
        else:
            index = torch.randint(len(self.mask_ratio_choices), (1,), generator=generator).item()
            ratio = self.mask_ratio_choices[index]
        if ratio is None or not 0.0 < ratio < 1.0:
            raise ValueError(f"Mask ratio must be between 0 and 1, got {ratio}")
        return ratio

    def _random_mask(self, batch_size, device, generator=None):
        total = self.height * self.width
        ratio = self._select_mask_ratio(generator)
        len_keep = round(total * (1.0 - ratio))
        if not 0 < len_keep < total:
            raise ValueError(
                f"Mask ratio {ratio} leaves {len_keep}/{total} context patches; both sets must be non-empty"
            )

        indices = torch.rand(batch_size, total, generator=generator).argsort(dim=1)
        indices = indices[:, :len_keep].to(device)
        context = torch.zeros(batch_size, total, dtype=torch.bool, device=device)
        context.scatter_(dim=1, index=indices, value=True)
        context = context.view(batch_size, 1, self.height, self.width)
        return context, context.logical_not()

    def __call__(self, batch_size, device=None, generator=None):
        device = torch.device("cpu") if device is None else torch.device(device)
        if self.strategy == "multi-block":
            context, target = self.multi_block_mask(batch_size)
            return (
                context.unsqueeze(1).to(device=device, dtype=torch.bool),
                target.unsqueeze(1).to(device=device, dtype=torch.bool),
            )
        return self._random_mask(batch_size, device=device, generator=generator)

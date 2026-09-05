"""Reusable JEPA, VAE, side-state, and action-conditioned dynamics modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from data.hdf5_iqfmfolder import upsample_antenna_axis
from pretrain.iqfm_masks import WirelessMaskGenerator, PAPER_PATCH_SIZES, resize_mask

try:
    import timm
    import models.shufflenet_tv_in_timm  # registers the torchvision-compatible model
    import models.sparse_encoder as sparse_encoder
except Exception:  # pragma: no cover - optional for light unit tests
    timm = None
    sparse_encoder = None


def upsample_iq_batch(x: torch.Tensor) -> torch.Tensor:
    """Map [B,2,4,256] to WirelessJEPA's [B,2,256,256] grid."""
    if x.ndim != 4 or tuple(x.shape[1:]) != (2, 4, 256):
        raise ValueError(f"Expected [B,2,4,256], got {tuple(x.shape)}")
    return x.repeat_interleave(64, dim=2)


class RFEncoder(nn.Module):
    """Dense or sparse-inference wrapper around the ShuffleNet RF backbone."""
    def __init__(
        self,
        latent_dim: int = 128,
        backbone_name: str = "shufflenet_v2_x0_5_torchvision",
        sparse: bool = False,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for RFEncoder")
        backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="")
        self.sparse = bool(sparse)
        if self.sparse:
            if sparse_encoder is None:
                raise ImportError("sparse encoder utilities are required for sparse RFEncoder")
            backbone = sparse_encoder.dense_model_to_sparse(backbone)
        self.backbone = backbone
        self.projection = nn.Conv2d(self.backbone.num_features, latent_dim, kernel_size=1)
        self.latent_dim = int(latent_dim)
        # ShuffleNet feature maps are downsampled by 32. Sparse masks must
        # live at the lowest feature-map resolution so every sparse stage can
        # expand them by an integer factor.
        self.sparse_mask_stride = 32

    def encode_map(
        self,
        x: torch.Tensor,
        already_upsampled: bool = False,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        image = x if already_upsampled else upsample_iq_batch(x)
        if self.sparse:
            if active_mask is None:
                active_mask = torch.ones(
                    image.shape[0], 1,
                    max(1, image.shape[-2] // self.sparse_mask_stride),
                    max(1, image.shape[-1] // self.sparse_mask_stride),
                    dtype=torch.bool, device=image.device,
                )
            elif tuple(active_mask.shape[-2:]) == tuple(image.shape[-2:]):
                # Convert an optional pixel-space mask to the sparse base grid.
                kernel = self.sparse_mask_stride
                active_mask = F.max_pool2d(
                    active_mask.to(dtype=torch.float32),
                    kernel_size=kernel, stride=kernel,
                ).to(dtype=torch.bool)
            sparse_encoder._cur_active = active_mask.to(
                device=image.device, dtype=torch.bool
            )
        fmap = self.backbone(image)
        return self.projection(fmap)

    def forward(self, x: torch.Tensor, already_upsampled: bool = False) -> torch.Tensor:
        return self.encode_map(x, already_upsampled).mean(dim=(-2, -1))


class MaskedSpatialPredictor(nn.Module):
    """WirelessJEPA-style spatial predictor with depthwise 3x3 blocks."""
    def __init__(self, channels: int, layers: int = 3):
        super().__init__()
        blocks = []
        for _ in range(int(layers)):
            blocks.extend([
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=False),
                # GroupNorm has identical behavior in train/eval mode; this
                # avoids a predictor-statistics mismatch during frozen
                # representation validation and downstream inference.
                nn.GroupNorm(1, channels),
                nn.ReLU(inplace=True),
            ])
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _masked_feature_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str = "normalized",
):
    """Compute target-only JEPA loss and return collapse diagnostics."""
    if mode == "normalized":
        # Per-token normalization prevents a near-zero predictor/teacher pair
        # from winning raw MSE merely by shrinking all RF features.
        prediction_for_loss = F.normalize(prediction, dim=1, eps=1e-6)
        target_for_loss = F.normalize(target.detach(), dim=1, eps=1e-6)
    elif mode == "centered":
        prediction_for_loss = prediction - prediction.mean(dim=1, keepdim=True)
        target_for_loss = target.detach() - target.detach().mean(dim=1, keepdim=True)
        prediction_for_loss = F.normalize(prediction_for_loss, dim=1, eps=1e-6)
        target_for_loss = F.normalize(target_for_loss, dim=1, eps=1e-6)
    elif mode == "raw":
        prediction_for_loss, target_for_loss = prediction, target.detach()
    else:
        raise ValueError(f"Unknown JEPA feature loss mode {mode!r}")
    mask = target_mask.to(prediction_for_loss.dtype)
    per_location = (prediction_for_loss - target_for_loss).pow(2).sum(dim=1, keepdim=True)
    loss = (per_location * mask).sum() / mask.sum().clamp_min(1.0)
    with torch.no_grad():
        target_std = target.detach().flatten(2).std(dim=-1, unbiased=False).mean()
        prediction_std = prediction.detach().flatten(2).std(dim=-1, unbiased=False).mean()
        target_norm = target.detach().flatten(2).norm(dim=1).mean()
        prediction_norm = prediction.detach().flatten(2).norm(dim=1).mean()
        cosine = F.cosine_similarity(prediction.detach(), target.detach(), dim=1, eps=1e-6).mean()
    diagnostics = {
        "target_std": target_std,
        "prediction_std": prediction_std,
        "target_norm": target_norm,
        "prediction_norm": prediction_norm,
        "cosine": cosine,
    }
    return loss, diagnostics

def _mask_for_strategy(
    strategy: str,
    batch_size: int,
    device: torch.device,
    mask_ratio: float = 0.5,
    antenna_mask_ratio_choices=None,
):
    patch_size = PAPER_PATCH_SIZES[strategy]
    kwargs = dict(input_size=(256, 256), strategy=strategy, patch_size=patch_size)
    if strategy == "multi-block":
        kwargs.update(mask_ratio=mask_ratio, multi_block_kwargs={
            "aspect_ratio": (0.75, 1.5), "enc_mask_scale": (1.0, 1.0),
            "min_keep": 3, "num_enc_blocks": 1, "num_pred_blocks": 4,
            "pred_mask_scale": (0.15, 0.2),
        })
    elif strategy == "antenna":
        # A four-antenna patch grid has only four rows.  Randomly choosing
        # .25/.5/.75 changes the task from masking one row to masking three
        # rows and makes validation noisy.  Ratio sweeps remain available
        # when explicitly registered by the caller.
        choices = antenna_mask_ratio_choices or (mask_ratio,)
        kwargs.update(mask_ratio_choices=tuple(float(value) for value in choices))
    else:
        kwargs.update(mask_ratio=mask_ratio)
    generator = WirelessMaskGenerator(**kwargs)
    return generator(batch_size, device=device)


class JEPAWorldModelEncoder(nn.Module):
    """Masked latent predictor with EMA teacher and full-input RF inference."""
    def __init__(
        self,
        latent_dim: int = 128,
        strategy: str = "multi-block",
        mask_ratio: float = 0.5,
        antenna_mask_ratio_choices=None,
        predictor_variant: str = "original",
        sparse_context: bool = True,
        feature_loss: str = "normalized",
    ):
        super().__init__()
        self.predictor_variant = str(predictor_variant)
        self.sparse_context = bool(sparse_context)
        self.feature_loss = str(feature_loss)
        self.context = RFEncoder(latent_dim=latent_dim, sparse=self.sparse_context)
        self.teacher = RFEncoder(latent_dim=latent_dim, sparse=False)
        if self.predictor_variant == "legacy":
            self.predictor = nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim, 1), nn.GELU(),
                nn.Conv2d(latent_dim, latent_dim, 1),
            )
        elif self.predictor_variant == "original":
            self.predictor = MaskedSpatialPredictor(latent_dim, layers=3)
            self.mask_token = nn.Parameter(torch.zeros(1, latent_dim, 1, 1))
            nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown predictor variant {predictor_variant!r}")
        self.strategy = strategy
        self.mask_ratio = float(mask_ratio)
        self.antenna_mask_ratio_choices = tuple(
            float(value) for value in (antenna_mask_ratio_choices or (mask_ratio,))
        )
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.load_state_dict(self.context.state_dict())

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for target, source in zip(self.teacher.parameters(), self.context.parameters()):
            target.data.mul_(momentum).add_(source.data, alpha=1.0 - momentum)

    def forward(self, x: torch.Tensor, strategy: Optional[str] = None):
        strategy = strategy or self.strategy
        context_mask, target_mask = _mask_for_strategy(
            strategy, x.shape[0], x.device, self.mask_ratio,
            antenna_mask_ratio_choices=self.antenna_mask_ratio_choices,
        )
        x_up = upsample_iq_batch(x)
        with torch.no_grad():
            target_map = self.teacher.encode_map(x_up, already_upsampled=True)
        latent_shape = target_map.shape[-2:]
        context_mask = resize_mask(context_mask, latent_shape)
        target_mask = resize_mask(target_mask, latent_shape)
        input_mask = resize_mask(context_mask, x_up.shape[-2:]).to(x_up.dtype)
        if self.sparse_context:
            context_map = self.context.encode_map(
                x_up * input_mask, already_upsampled=True, active_mask=context_mask,
            )
        else:
            context_map = self.context.encode_map(x_up * input_mask, already_upsampled=True)
        if self.predictor_variant == "original":
            # Explicit mask tokens tell the predictor which latent positions
            # are missing; masked zeros alone are ambiguous in I/Q signals.
            predictor_input = torch.where(
                context_mask.expand_as(context_map), context_map,
                self.mask_token.to(dtype=context_map.dtype),
            )
        else:
            predictor_input = context_map
        prediction_map = self.predictor(predictor_input)
        loss, diagnostics = _masked_feature_loss(
            prediction_map, target_map, target_mask, mode=self.feature_loss,
        )
        return {
            "loss": loss, "prediction": prediction_map, "target": target_map,
            "context_mask": context_mask, "target_mask": target_mask,
            **diagnostics,
        }

    @torch.no_grad()
    def encode_rf(self, x: torch.Tensor) -> torch.Tensor:
        return self.context(x)


class RFVAE(nn.Module):
    """Dimension-matched reconstruction baseline for the controlled study."""
    def __init__(self, latent_dim: int = 128, beta: float = 1e-3):
        super().__init__()
        self.encoder = RFEncoder(latent_dim=latent_dim)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)
        self.decoder_in = nn.Linear(latent_dim, latent_dim * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 2, 4, 2, 1),
        )
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x: torch.Tensor):
        x_up = upsample_iq_batch(x)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(self.decoder_in(z).view(-1, self.latent_dim, 8, 8))
        return reconstruction, mu, logvar

    def loss(self, x: torch.Tensor):
        x_up = upsample_iq_batch(x)
        reconstruction, mu, logvar = self(x)
        recon = F.mse_loss(reconstruction, x_up)
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()
        return recon + self.beta * kl, {"reconstruction": recon.detach(), "kl": kl.detach()}

    @torch.no_grad()
    def encode_rf(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu


class SideInfoEncoder(nn.Module):
    def __init__(self, input_dim: int = 8, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim))
        self.output_dim = output_dim

    def forward(self, side: torch.Tensor) -> torch.Tensor:
        return self.net(side)


class LatentStateEncoder(nn.Module):
    def __init__(self, rf_dim: int = 128, side_dim: int = 32):
        super().__init__()
        self.rf = rf_dim
        self.side = side_dim
        self.side_encoder = SideInfoEncoder(8, side_dim)

    @property
    def output_dim(self):
        return self.rf + self.side

    def forward(self, rf: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        return torch.cat([rf, self.side_encoder(side)], dim=-1)


@dataclass
class MDNOutput:
    log_pi: torch.Tensor
    mu: torch.Tensor
    log_sigma: torch.Tensor
    hidden: Tuple[torch.Tensor, torch.Tensor]
    kpi: torch.Tensor


class ActionConditionedMDNLSTM(nn.Module):
    """Action-conditioned stochastic latent dynamics with KPI heads."""
    def __init__(self, latent_dim: int = 160, hidden_dim: int = 256, mixtures: int = 5):
        super().__init__()
        self.latent_dim, self.hidden_dim, self.mixtures = latent_dim, hidden_dim, mixtures
        self.lstm = nn.LSTM(latent_dim + 2, hidden_dim, batch_first=True)
        self.log_pi = nn.Linear(hidden_dim, latent_dim * mixtures)
        self.mu = nn.Linear(hidden_dim, latent_dim * mixtures)
        self.log_sigma = nn.Linear(hidden_dim, latent_dim * mixtures)
        self.kpi_head = nn.Linear(hidden_dim, 3)  # secrecy, legitimate success, leakage logits

    def forward(self, z: torch.Tensor, action: torch.Tensor, hidden=None) -> MDNOutput:
        sequence = z.ndim == 3
        if not sequence:
            z, action = z.unsqueeze(1), action.unsqueeze(1)
        output, hidden = self.lstm(torch.cat([z, action], dim=-1), hidden)
        b, t, _ = output.shape
        reshape = lambda layer: layer(output).view(b, t, self.latent_dim, self.mixtures)
        result = MDNOutput(
            F.log_softmax(reshape(self.log_pi), dim=-1), reshape(self.mu),
            torch.clamp(reshape(self.log_sigma), -7.0, 5.0), hidden, self.kpi_head(output),
        )
        if not sequence:
            result = MDNOutput(result.log_pi[:, 0], result.mu[:, 0], result.log_sigma[:, 0], result.hidden, result.kpi[:, 0])
        return result

    def nll(self, output: MDNOutput, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 2:
            target = target.unsqueeze(1)
            log_pi, mu, log_sigma = output.log_pi.unsqueeze(1), output.mu.unsqueeze(1), output.log_sigma.unsqueeze(1)
        else:
            log_pi, mu, log_sigma = output.log_pi, output.mu, output.log_sigma
        normal = -0.5 * (((target.unsqueeze(-1) - mu) / log_sigma.exp()).pow(2) + 2 * log_sigma + torch.log(torch.tensor(2.0 * torch.pi, device=target.device)))
        return -torch.logsumexp(log_pi + normal, dim=-1).mean()


class PowerController(nn.Module):
    """Compact bounded two-power actor used by PPO and controller probes.

    The policy is a sigmoid-squashed diagonal Gaussian.  The trainable
    log-standard-deviation is clamped when constructing the distribution so
    that long real-environment runs cannot silently lose all exploration (or
    become numerically unstable near the action boundaries).
    """
    def __init__(
        self,
        input_dim: int = 416,
        hidden_dim: int = 128,
        initial_log_std: float = -0.7,
        min_log_std: float = -2.5,
        max_log_std: float = 1.0,
    ):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 2))
        self.log_std = nn.Parameter(torch.full((2,), float(initial_log_std)))
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

    def forward(self, features: torch.Tensor):
        return self.net(features)

    def distribution(self, features: torch.Tensor):
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        return torch.distributions.Normal(self.net(features), log_std.exp())

    def action(self, features: torch.Tensor, deterministic: bool = False):
        distribution = self.distribution(features)
        raw = distribution.mean if deterministic else distribution.rsample()
        bounded = torch.sigmoid(raw)
        log_prob = self.log_prob(features, bounded)
        return bounded, log_prob

    def log_prob(self, features: torch.Tensor, bounded: torch.Tensor):
        bounded = bounded.clamp(1e-5, 1.0 - 1e-5)
        raw = torch.logit(bounded)
        distribution = self.distribution(features)
        return distribution.log_prob(raw).sum(-1) - torch.log(bounded * (1.0 - bounded) + 1e-6).sum(-1)

    def entropy(self, features: torch.Tensor) -> torch.Tensor:
        """Base-distribution entropy used as a stable exploration bonus.

        The exact entropy of a sigmoid-transformed Gaussian has no convenient
        closed form.  The Normal entropy is a useful, monotonic proxy and is
        independent of sampled actions, which makes it suitable for PPO.
        """
        return self.distribution(features).entropy().sum(-1)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = ["RFEncoder", "JEPAWorldModelEncoder", "RFVAE", "SideInfoEncoder", "LatentStateEncoder", "ActionConditionedMDNLSTM", "PowerController", "parameter_count", "upsample_iq_batch"]

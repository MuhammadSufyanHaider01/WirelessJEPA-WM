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
except Exception:  # pragma: no cover - optional for light unit tests
    timm = None


def upsample_iq_batch(x: torch.Tensor) -> torch.Tensor:
    """Map [B,2,4,256] to WirelessJEPA's [B,2,256,256] grid."""
    if x.ndim != 4 or tuple(x.shape[1:]) != (2, 4, 256):
        raise ValueError(f"Expected [B,2,4,256], got {tuple(x.shape)}")
    return x.repeat_interleave(64, dim=2)


class RFEncoder(nn.Module):
    """Dense inference wrapper around the existing ShuffleNet RF backbone."""
    def __init__(self, latent_dim: int = 128, backbone_name: str = "shufflenet_v2_x0_5_torchvision"):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for RFEncoder")
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="")
        self.projection = nn.Conv2d(self.backbone.num_features, latent_dim, kernel_size=1)
        self.latent_dim = int(latent_dim)

    def encode_map(self, x: torch.Tensor, already_upsampled: bool = False) -> torch.Tensor:
        image = x if already_upsampled else upsample_iq_batch(x)
        fmap = self.backbone(image)
        return self.projection(fmap)

    def forward(self, x: torch.Tensor, already_upsampled: bool = False) -> torch.Tensor:
        return self.encode_map(x, already_upsampled).mean(dim=(-2, -1))


def _mask_for_strategy(strategy: str, batch_size: int, device: torch.device, mask_ratio: float = 0.5):
    patch_size = PAPER_PATCH_SIZES[strategy]
    kwargs = dict(input_size=(256, 256), strategy=strategy, patch_size=patch_size)
    if strategy == "multi-block":
        kwargs.update(mask_ratio=mask_ratio, multi_block_kwargs={
            "aspect_ratio": (0.75, 1.5), "enc_mask_scale": (1.0, 1.0),
            "min_keep": 3, "num_enc_blocks": 1, "num_pred_blocks": 4,
            "pred_mask_scale": (0.15, 0.2),
        })
    elif strategy == "antenna":
        kwargs.update(mask_ratio_choices=(0.25, 0.5, 0.75))
    else:
        kwargs.update(mask_ratio=mask_ratio)
    generator = WirelessMaskGenerator(**kwargs)
    return generator(batch_size, device=device)


class JEPAWorldModelEncoder(nn.Module):
    """Masked latent predictor with EMA teacher and full-input RF inference."""
    def __init__(self, latent_dim: int = 128, strategy: str = "multi-block", mask_ratio: float = 0.5):
        super().__init__()
        self.context = RFEncoder(latent_dim=latent_dim)
        self.teacher = RFEncoder(latent_dim=latent_dim)
        self.predictor = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 1), nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, 1),
        )
        self.strategy = strategy
        self.mask_ratio = float(mask_ratio)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.load_state_dict(self.context.state_dict())

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for target, source in zip(self.teacher.parameters(), self.context.parameters()):
            target.data.mul_(momentum).add_(source.data, alpha=1.0 - momentum)

    def forward(self, x: torch.Tensor, strategy: Optional[str] = None):
        strategy = strategy or self.strategy
        context_mask, target_mask = _mask_for_strategy(strategy, x.shape[0], x.device, self.mask_ratio)
        x_up = upsample_iq_batch(x)
        latent_shape = self.context.encode_map(x_up, already_upsampled=True).shape[-2:]
        context_mask = resize_mask(context_mask, latent_shape)
        target_mask = resize_mask(target_mask, latent_shape)
        # Match the target mask to the input grid; zeroing the context is the
        # dense equivalent of the sparse context encoder used by WirelessJEPA.
        input_mask = resize_mask(context_mask, x_up.shape[-2:]).to(x_up.dtype)
        context_map = self.context.encode_map(x_up * input_mask, already_upsampled=True)
        prediction_map = self.predictor(context_map)
        with torch.no_grad():
            target_map = self.teacher.encode_map(x_up, already_upsampled=True)
        mask = target_mask.to(prediction_map.dtype)
        loss = ((prediction_map - target_map).pow(2).sum(dim=1, keepdim=True) * mask).sum() / mask.sum().clamp_min(1.0)
        return {"loss": loss, "prediction": prediction_map, "target": target_map,
                "context_mask": context_mask, "target_mask": target_mask}

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

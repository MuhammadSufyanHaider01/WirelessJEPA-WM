"""Staged V1 training entry point for JEPA/VAE, MDN-LSTM, and PPO."""
from __future__ import annotations

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data.hap_uav import PilotWindowDataset, TrajectoryDataset
from environment.hap_uav import HapUavConfig, HapUavEnv
from models.jepa_wm import (
    ActionConditionedMDNLSTM,
    JEPAWorldModelEncoder,
    LatentStateEncoder,
    PowerController,
    RFVAE,
    parameter_count,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def save_state(path: Path, model: nn.Module, metrics: dict, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "metrics": metrics, **extra}
    torch.save(payload, path)


def make_tensorboard_writer(path: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(str(path))
    except Exception:
        return None


def append_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _split_pilot_dataset(dataset: PilotWindowDataset, val_fraction: float, seed: int):
    """Create deterministic, disjoint train/validation window subsets."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    count = len(dataset)
    if count < 2 or val_fraction == 0.0:
        return dataset, None
    val_count = max(1, int(round(count * val_fraction)))
    val_count = min(val_count, count - 1)
    indices = np.random.default_rng(seed).permutation(count)
    return Subset(dataset, indices[val_count:].tolist()), Subset(dataset, indices[:val_count].tolist())


@torch.no_grad()
def _jepa_validation_loss(model, loader, device, strategy: str):
    if loader is None:
        return float("nan")
    model.eval()
    values = []
    for batch in loader:
        result = model(batch.to(device, non_blocking=True), strategy=strategy)
        values.append(float(result["loss"].detach().cpu()))
    return float(np.mean(values)) if values else float("nan")


def train_jepa(args) -> Path:
    seed_everything(args.seed)
    device = device_from(args.device)
    dataset = PilotWindowDataset(args.data, normalize=args.normalize)
    train_dataset, val_dataset = _split_pilot_dataset(dataset, args.val_fraction, args.seed)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.workers)
    val_loader = (DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)
                  if val_dataset is not None else None)
    model = JEPAWorldModelEncoder(args.latent_dim, args.strategy, args.mask_ratio).to(device)
    optimizer = torch.optim.AdamW(
        list(model.context.parameters()) + list(model.predictor.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    history = []
    out = Path(args.output)
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    writer = make_tensorboard_writer(Path(args.output).with_suffix(".tensorboard"))
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            result = model(x, strategy=args.strategy)
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.update_teacher(min(1.0, args.ema_start + (1.0 - args.ema_start) * epoch / max(args.epochs - 1, 1)))
            losses.append(float(result["loss"].detach().cpu()))
        # Keep validation-mask draws comparable between epochs for reliable
        # early stopping instead of comparing unrelated random masks.
        torch.manual_seed(args.seed + 10_000 + epoch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + 10_000 + epoch)
        val_loss = _jepa_validation_loss(model, val_loader, device, args.strategy)
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_loss": val_loss, "strategy": args.strategy}
        history.append(row)
        if writer:
            writer.add_scalar("jepa/loss", row["loss"], epoch + 1)
            if np.isfinite(val_loss):
                writer.add_scalar("jepa/val_loss", val_loss, epoch + 1)
        print(json.dumps(row), flush=True)
        improved = np.isfinite(val_loss) and val_loss < best_val - args.min_delta
        if improved or (val_dataset is None and epoch == 0):
            best_val = val_loss
            best_epoch = epoch + 1
            stale_epochs = 0
            save_state(out, model, {"history": history}, latent_dim=args.latent_dim, strategy=args.strategy,
                       best_epoch=best_epoch, best_val_loss=best_val, train_count=len(train_dataset),
                       val_count=len(val_dataset) if val_dataset is not None else 0)
        else:
            stale_epochs += 1
        if args.patience > 0 and val_dataset is not None and stale_epochs >= args.patience:
            print(json.dumps({"early_stopping": True, "best_epoch": best_epoch,
                              "best_val_loss": best_val, "patience": args.patience}), flush=True)
            break
    if writer: writer.close()
    if not out.exists():
        save_state(out, model, {"history": history}, latent_dim=args.latent_dim, strategy=args.strategy,
                   best_epoch=best_epoch, best_val_loss=best_val, train_count=len(train_dataset),
                   val_count=len(val_dataset) if val_dataset is not None else 0)
    append_csv(out.with_suffix(".csv"), history)
    return out


def train_vae(args) -> Path:
    seed_everything(args.seed)
    device = device_from(args.device)
    dataset = PilotWindowDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.workers)
    model = RFVAE(args.latent_dim, args.beta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    writer = make_tensorboard_writer(Path(args.output).with_suffix(".tensorboard"))
    for epoch in range(args.epochs):
        model.train(); values = []
        for batch in loader:
            loss, parts = model.loss(batch.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            values.append((float(loss.detach().cpu()), float(parts["reconstruction"]), float(parts["kl"])))
        row = {"epoch": epoch + 1, "loss": float(np.mean([v[0] for v in values])), "reconstruction": float(np.mean([v[1] for v in values])), "kl": float(np.mean([v[2] for v in values]))}
        history.append(row)
        if writer:
            writer.add_scalar("vae/loss", row["loss"], epoch + 1); writer.add_scalar("vae/reconstruction", row["reconstruction"], epoch + 1); writer.add_scalar("vae/kl", row["kl"], epoch + 1)
        print(json.dumps(row), flush=True)
    if writer: writer.close()
    out = Path(args.output)
    save_state(out, model, {"history": history}, latent_dim=args.latent_dim, beta=args.beta)
    append_csv(out.with_suffix(".csv"), history)
    return out


def _load_representation(kind: str, path: str, device: torch.device):
    payload = torch.load(path, map_location=device)
    if kind == "jepa":
        model = JEPAWorldModelEncoder(payload.get("latent_dim", 128), payload.get("strategy", "multi-block"))
    elif kind == "vae":
        model = RFVAE(payload.get("latent_dim", 128), payload.get("beta", 1e-3))
    else:
        raise ValueError(f"unknown representation {kind}")
    model.load_state_dict(payload["model"]); model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def train_mdn(args) -> Path:
    seed_everything(args.seed)
    device = device_from(args.device)
    representation = _load_representation(args.representation, args.representation_checkpoint, device)
    dataset = TrajectoryDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    side_encoder = LatentStateEncoder(128, 32).to(device)
    dynamics = ActionConditionedMDNLSTM(160, args.hidden_dim, args.mixtures).to(device)
    optimizer = torch.optim.AdamW(list(dynamics.parameters()) + list(side_encoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    writer = make_tensorboard_writer(Path(args.output).with_suffix(".tensorboard"))
    for epoch in range(args.epochs):
        dynamics.train(); side_encoder.train(); values = []
        for batch in loader:
            iq = batch["iq"].to(device); b, t = iq.shape[:2]
            with torch.no_grad():
                rf = representation.encode_rf(iq.reshape(b * t, *iq.shape[2:])).reshape(b, t, 128)
            z = torch.cat([rf, side_encoder.side_encoder(batch["side"].to(device))], dim=-1)
            actions = batch["action"].to(device)
            out = dynamics(z[:, :-1], actions[:, :-1])
            target = z[:, 1:].detach()
            nll = dynamics.nll(out, target)
            kpi_loss = torch.zeros((), device=device)
            if "secrecy_rate" in batch:
                secrecy = torch.clamp(batch["secrecy_rate"][:, 1:].to(device) / 0.5e6, 0.0, 1.0)
                kpi_loss = kpi_loss + torch.nn.functional.mse_loss(torch.sigmoid(out.kpi[..., 0]), secrecy)
                kpi_loss = kpi_loss + torch.nn.functional.binary_cross_entropy_with_logits(out.kpi[..., 1], batch["success"][:, 1:].to(device))
                kpi_loss = kpi_loss + torch.nn.functional.binary_cross_entropy_with_logits(out.kpi[..., 2], batch["leakage"][:, 1:].to(device))
            loss = nll + args.kpi_weight * kpi_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(dynamics.parameters()) + list(side_encoder.parameters()), 5.0); optimizer.step()
            values.append((float(loss.detach().cpu()), float(nll.detach().cpu()), float(kpi_loss.detach().cpu())))
        row = {"epoch": epoch + 1, "loss": float(np.mean([v[0] for v in values])), "nll": float(np.mean([v[1] for v in values])), "kpi_loss": float(np.mean([v[2] for v in values]))}
        history.append(row)
        if writer:
            writer.add_scalar("mdn/loss", row["loss"], epoch + 1); writer.add_scalar("mdn/nll", row["nll"], epoch + 1); writer.add_scalar("mdn/kpi_loss", row["kpi_loss"], epoch + 1)
        print(json.dumps(row), flush=True)
    if writer: writer.close()
    out = Path(args.output)
    save_state(out, dynamics, {"history": history}, side_encoder=side_encoder.state_dict(), latent_dim=160, hidden_dim=args.hidden_dim, mixtures=args.mixtures, representation=args.representation)
    append_csv(out.with_suffix(".csv"), history)
    return out


class ValueNet(nn.Module):
    def __init__(self, input_dim=416):
        super().__init__(); self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.Tanh(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def _feature(obs, representation, side_encoder, dynamics, hidden, device, memory=True, side_only=False):
    iq = torch.from_numpy(obs["iq"]).to(device).unsqueeze(0)
    side = torch.from_numpy(obs["side"]).to(device).unsqueeze(0)
    with torch.no_grad():
        rf = representation.encode_rf(iq)
        if side_only: rf = torch.zeros_like(rf)
        z = torch.cat([rf, side_encoder.side_encoder(side)], dim=-1)
        h = hidden[0].transpose(0, 1).reshape(1, -1) if hidden is not None else torch.zeros(1, 256, device=device)
        features = torch.cat([z, h], dim=-1)
    return features


def train_ppo(args) -> Path:
    seed_everything(args.seed)
    device = device_from(args.device)
    representation = _load_representation(args.representation, args.representation_checkpoint, device)
    dynamics_payload = torch.load(args.dynamics_checkpoint, map_location=device)
    dynamics = ActionConditionedMDNLSTM(160, dynamics_payload.get("hidden_dim", 256), dynamics_payload.get("mixtures", 5)).to(device)
    dynamics.load_state_dict(dynamics_payload["model"]); dynamics.eval()
    side_encoder = LatentStateEncoder(128, 32).to(device); side_encoder.load_state_dict(dynamics_payload["side_encoder"]); side_encoder.eval()
    actor = PowerController(416, args.actor_hidden).to(device)
    critic = ValueNet(416).to(device)
    optimizer = torch.optim.AdamW(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)
    env_cfg = HapUavConfig(max_steps=args.episode_length)
    history = []
    writer = make_tensorboard_writer(Path(args.output).with_suffix(".tensorboard"))
    for episode in range(args.episodes):
        env = HapUavEnv(env_cfg); obs, _ = env.reset(seed=args.seed + episode)
        hidden = None; features = []; actions = []; old_logp = []; rewards = []; values = []; dones = []
        episode_metrics = {key: [] for key in ("age_h", "age_u", "secrecy_rate", "secrecy_gate", "mu_h", "mu_u", "pt_dbm", "pj_dbm")}
        for _ in range(args.episode_length):
            f = _feature(obs, representation, side_encoder, dynamics, hidden, device, args.memory, args.side_only)
            with torch.no_grad():
                a, lp = actor.action(f); value = critic(f)
            next_obs, reward, term, trunc, info = env.step(a.squeeze(0).cpu().numpy())
            for key in episode_metrics:
                if key in info:
                    episode_metrics[key].append(float(info[key]))
            with torch.no_grad():
                rf = representation.encode_rf(torch.from_numpy(obs["iq"]).to(device).unsqueeze(0))
                if args.side_only: rf = torch.zeros_like(rf)
                z = torch.cat([rf, side_encoder.side_encoder(torch.from_numpy(obs["side"]).to(device).unsqueeze(0))], dim=-1)
                if args.memory:
                    hidden = dynamics(z, a, hidden).hidden
                else:
                    hidden = None
            features.append(f.squeeze(0)); actions.append(a.squeeze(0)); old_logp.append(lp.squeeze(0)); rewards.append(reward); values.append(value.squeeze(0)); dones.append(term or trunc)
            obs = next_obs
            if term or trunc: break
        with torch.no_grad():
            returns = []; gae = torch.zeros((), device=device); next_value = torch.zeros((), device=device)
            for i in reversed(range(len(rewards))):
                delta = torch.tensor(rewards[i], device=device) + args.gamma * next_value * (1.0 - float(dones[i])) - values[i]
                gae = delta + args.gamma * args.gae_lambda * (1.0 - float(dones[i])) * gae
                returns.insert(0, gae + values[i]); next_value = values[i]
            returns = torch.stack(returns); old_values = torch.stack(values); old_logp_t = torch.stack(old_logp); features_t = torch.stack(features); actions_t = torch.stack(actions)
        advantages = (returns - old_values); advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(args.ppo_epochs):
            logp = actor.log_prob(features_t, actions_t); ratio = torch.exp(logp - old_logp_t); clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
            policy_loss = -(torch.minimum(ratio * advantages, clipped * advantages)).mean()
            value_loss = torch.nn.functional.mse_loss(critic(features_t), returns)
            loss = policy_loss + 0.5 * value_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0); optimizer.step()
        row = {"episode": episode + 1, "return": float(sum(rewards)), "mean_reward": float(np.mean(rewards)), "policy_loss": float(policy_loss.detach().cpu()), "value_loss": float(value_loss.detach().cpu())}
        for key, values_for_key in episode_metrics.items():
            if values_for_key:
                row[key] = float(np.mean(values_for_key))
        history.append(row)
        if writer:
            writer.add_scalar("ppo/return", row["return"], episode + 1); writer.add_scalar("ppo/policy_loss", row["policy_loss"], episode + 1); writer.add_scalar("ppo/value_loss", row["value_loss"], episode + 1)
        print(json.dumps(row), flush=True)
    if writer: writer.close()
    out = Path(args.output); save_state(out, actor, {"history": history}, critic=critic.state_dict(), representation=args.representation, memory=args.memory, side_only=args.side_only, actor_hidden=args.actor_hidden); append_csv(out.with_suffix(".csv"), history); return out


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    def common(p):
        p.add_argument("--data", required=True); p.add_argument("--output", required=True); p.add_argument("--device", default="auto"); p.add_argument("--seed", type=int, default=42); p.add_argument("--epochs", type=int, default=1); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--workers", type=int, default=0); p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--weight-decay", type=float, default=1e-4)
    p = sub.add_parser("jepa"); common(p); p.add_argument("--latent-dim", type=int, default=128); p.add_argument("--strategy", default="multi-block", choices=("random", "antenna", "time", "multi-block")); p.add_argument("--mask-ratio", type=float, default=.5); p.add_argument("--ema-start", type=float, default=.996); p.add_argument("--val-fraction", type=float, default=.1); p.add_argument("--patience", type=int, default=10, help="epochs without validation improvement before stopping; 0 disables"); p.add_argument("--min-delta", type=float, default=0.0); p.add_argument("--normalize", action="store_true", help="per-window max-magnitude normalization"); p.set_defaults(func=train_jepa)
    p = sub.add_parser("vae"); common(p); p.add_argument("--latent-dim", type=int, default=128); p.add_argument("--beta", type=float, default=1e-3); p.set_defaults(func=train_vae)
    p = sub.add_parser("mdn"); common(p); p.add_argument("--representation", choices=("jepa", "vae"), required=True); p.add_argument("--representation-checkpoint", required=True); p.add_argument("--hidden-dim", type=int, default=256); p.add_argument("--mixtures", type=int, default=5); p.add_argument("--kpi-weight", type=float, default=.1); p.set_defaults(func=train_mdn)
    p = sub.add_parser("ppo"); p.add_argument("--representation", choices=("jepa", "vae"), required=True); p.add_argument("--representation-checkpoint", required=True); p.add_argument("--dynamics-checkpoint", required=True); p.add_argument("--output", required=True); p.add_argument("--device", default="auto"); p.add_argument("--seed", type=int, default=42); p.add_argument("--episodes", type=int, default=20); p.add_argument("--episode-length", type=int, default=100); p.add_argument("--actor-hidden", type=int, default=128); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--gamma", type=float, default=.99); p.add_argument("--gae-lambda", type=float, default=.95); p.add_argument("--clip", type=float, default=.2); p.add_argument("--ppo-epochs", type=int, default=4); p.add_argument("--memory", action="store_true"); p.add_argument("--side-only", action="store_true"); p.set_defaults(func=train_ppo)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args(); parsed.func(parsed)

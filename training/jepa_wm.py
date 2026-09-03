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
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _feature(obs, representation, side_encoder, dynamics, hidden, device, memory=True, side_only=False):
    """Encode one causal observation once and return ``(features, z)``.

    ``z`` is reused for the frozen MDN state update after the action.  The old
    implementation encoded the same pilot twice per step, which became a
    major cost at the 5,000-step horizon.
    """
    iq = torch.from_numpy(obs["iq"]).to(device).unsqueeze(0)
    side = torch.from_numpy(obs["side"]).to(device).unsqueeze(0)
    with torch.no_grad():
        rf = representation.encode_rf(iq)
        if side_only:
            rf = torch.zeros_like(rf)
        z = torch.cat([rf, side_encoder.side_encoder(side)], dim=-1)
        hidden_dim = int(getattr(dynamics, "hidden_dim", 256))
        h = (
            hidden[0].transpose(0, 1).reshape(1, -1)
            if hidden is not None and memory
            else torch.zeros(1, hidden_dim, device=device)
        )
        features = torch.cat([z, h], dim=-1)
    return features, z


def _gae_returns(rewards, values, terminal_flags, bootstrap, gamma, gae_lambda, reward_scale):
    """Compute GAE targets, bootstrapping rollout truncations but not terminals."""
    scaled_rewards = rewards * float(reward_scale)
    returns = []
    advantages = []
    gae = torch.zeros((), device=rewards.device)
    next_value = bootstrap
    for index in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - float(terminal_flags[index])
        delta = scaled_rewards[index] + gamma * next_value * nonterminal - values[index]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[index])
        next_value = values[index]
    return torch.stack(returns), torch.stack(advantages)


def _ppo_update(actor, critic, actor_optimizer, critic_optimizer, features, actions,
                old_logp, returns, advantages, args):
    """Run shuffled minibatch PPO updates for one recurrent rollout chunk."""
    count = int(features.shape[0])
    normalized_advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    minibatch_size = max(1, min(int(args.minibatch_size), count))
    records = []
    for _ in range(args.ppo_epochs):
        permutation = torch.randperm(count, device=features.device)
        epoch_records = []
        for begin in range(0, count, minibatch_size):
            indices = permutation[begin:begin + minibatch_size]
            batch_features = features[indices]
            batch_actions = actions[indices]
            batch_old_logp = old_logp[indices]
            batch_returns = returns[indices]
            batch_advantages = normalized_advantages[indices]

            new_logp = actor.log_prob(batch_features, batch_actions)
            ratio = torch.exp((new_logp - batch_old_logp).clamp(-20.0, 20.0))
            clipped_ratio = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
            surrogate = torch.minimum(ratio * batch_advantages, clipped_ratio * batch_advantages)
            entropy = actor.entropy(batch_features).mean()
            policy_loss = -surrogate.mean()
            actor_loss = policy_loss - args.entropy_coef * entropy
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            actor_optimizer.step()

            value_prediction = critic(batch_features)
            value_loss = torch.nn.functional.mse_loss(value_prediction, batch_returns)
            critic_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            critic_optimizer.step()

            with torch.no_grad():
                # Measure KL after the actor step; using ``new_logp`` here
                # would compare the policy with itself and always report ~0.
                post_update_logp = actor.log_prob(batch_features, batch_actions)
                log_ratio = (post_update_logp - batch_old_logp).clamp(-20.0, 20.0)
                # A non-negative sample estimate of KL(old || new).
                approx_kl = (torch.exp(log_ratio) - 1.0 - log_ratio).mean()
                clip_fraction = (torch.abs(ratio - 1.0) > args.clip).float().mean()
            epoch_records.append({
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "approx_kl": float(approx_kl.cpu()),
                "clip_fraction": float(clip_fraction.cpu()),
            })
        records.extend(epoch_records)
        if args.target_kl > 0.0 and epoch_records:
            epoch_kl = float(np.mean([record["approx_kl"] for record in epoch_records]))
            if epoch_kl > args.target_kl:
                break
    if not records:
        return {key: 0.0 for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")}
    return {key: float(np.mean([record[key] for record in records])) for key in records[0]}


def train_ppo(args) -> Path:
    seed_everything(args.seed)
    if args.episode_length < 1:
        raise ValueError("episode_length must be positive")
    if args.rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    device = device_from(args.device)
    representation = _load_representation(args.representation, args.representation_checkpoint, device)
    dynamics_payload = torch.load(args.dynamics_checkpoint, map_location=device)
    dynamics = ActionConditionedMDNLSTM(
        160, dynamics_payload.get("hidden_dim", 256), dynamics_payload.get("mixtures", 5)
    ).to(device)
    dynamics.load_state_dict(dynamics_payload["model"])
    dynamics.eval()
    side_encoder = LatentStateEncoder(128, 32).to(device)
    side_encoder.load_state_dict(dynamics_payload["side_encoder"])
    side_encoder.eval()
    feature_dim = 128 + 32 + int(dynamics.hidden_dim)
    actor = PowerController(
        feature_dim,
        args.actor_hidden,
        initial_log_std=args.initial_log_std,
        min_log_std=args.min_log_std,
        max_log_std=args.max_log_std,
    ).to(device)
    critic = ValueNet(feature_dim).to(device)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)
    env_cfg = HapUavConfig(max_steps=args.episode_length)
    history = []
    out = Path(args.output)
    best_path = out.with_name(out.stem + "_best" + out.suffix)
    best_return = float("-inf")
    writer = make_tensorboard_writer(out.with_suffix(".tensorboard"))
    metric_keys = ("age_h", "age_u", "secrecy_rate", "secrecy_gate", "mu_h", "mu_u", "pt_dbm", "pj_dbm")

    for episode in range(args.episodes):
        env = HapUavEnv(env_cfg)
        obs, _ = env.reset(seed=args.seed + episode)
        hidden = None
        episode_done = False
        episode_return = 0.0
        episode_steps = 0
        episode_metrics = {key: [] for key in metric_keys}
        update_records = []

        # The environment horizon is 5,000 by default, while PPO updates every
        # rollout_steps transitions.  Hidden state is carried across chunks.
        while not episode_done and episode_steps < args.episode_length:
            features = []
            actions = []
            old_logp = []
            values = []
            rewards = []
            terminal_flags = []
            chunk_size = min(args.rollout_steps, args.episode_length - episode_steps)
            for _ in range(chunk_size):
                current_features, z = _feature(
                    obs, representation, side_encoder, dynamics, hidden, device,
                    args.memory, args.side_only,
                )
                with torch.no_grad():
                    action, logp = actor.action(current_features)
                    value = critic(current_features)
                next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
                for key in metric_keys:
                    if key in info:
                        episode_metrics[key].append(float(info[key]))
                if args.memory:
                    with torch.no_grad():
                        hidden = dynamics(z, action, hidden).hidden
                else:
                    hidden = None
                features.append(current_features.squeeze(0).detach())
                actions.append(action.squeeze(0).detach())
                old_logp.append(logp.squeeze(0).detach())
                values.append(value.squeeze(0).detach())
                rewards.append(float(reward))
                # A true termination has no value bootstrap.  A rollout or
                # external truncation still bootstraps the next state.
                terminal_flags.append(bool(terminated))
                episode_return += float(reward)
                episode_steps += 1
                obs = next_obs
                episode_done = bool(terminated or truncated)
                if episode_done:
                    break

            features_t = torch.stack(features)
            actions_t = torch.stack(actions)
            old_logp_t = torch.stack(old_logp)
            values_t = torch.stack(values)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            with torch.no_grad():
                if terminal_flags[-1]:
                    bootstrap = torch.zeros((), device=device)
                else:
                    next_features, _ = _feature(
                        obs, representation, side_encoder, dynamics, hidden, device,
                        args.memory, args.side_only,
                    )
                    bootstrap = critic(next_features).squeeze(0)
                returns_t, advantages_t = _gae_returns(
                    rewards_t, values_t, terminal_flags, bootstrap,
                    args.gamma, args.gae_lambda, args.reward_scale,
                )
            update_records.append(_ppo_update(
                actor, critic, actor_optimizer, critic_optimizer,
                features_t, actions_t, old_logp_t, returns_t, advantages_t, args,
            ))

        row = {
            "episode": episode + 1,
            "steps": episode_steps,
            "return": float(episode_return),
            "mean_reward": float(episode_return / max(episode_steps, 1)),
            "rollout_updates": len(update_records),
            "policy_loss": float(np.mean([record["policy_loss"] for record in update_records])),
            "value_loss": float(np.mean([record["value_loss"] for record in update_records])),
            "entropy": float(np.mean([record["entropy"] for record in update_records])),
            "approx_kl": float(np.mean([record["approx_kl"] for record in update_records])),
            "clip_fraction": float(np.mean([record["clip_fraction"] for record in update_records])),
            "log_std_pt": float(actor.log_std[0].clamp(args.min_log_std, args.max_log_std).detach().cpu()),
            "log_std_pj": float(actor.log_std[1].clamp(args.min_log_std, args.max_log_std).detach().cpu()),
        }
        for key, values_for_key in episode_metrics.items():
            if values_for_key:
                row[key] = float(np.mean(values_for_key))
        history.append(row)
        if writer:
            for key, value in row.items():
                if key not in {"episode"} and np.isfinite(value):
                    writer.add_scalar("ppo/" + key, value, episode + 1)
            writer.flush()
        if episode_return > best_return:
            best_return = episode_return
            save_state(
                best_path, actor, {"history": history, "best_return": best_return},
                critic=critic.state_dict(), representation=args.representation,
                memory=args.memory, side_only=args.side_only, actor_hidden=args.actor_hidden,
                feature_dim=feature_dim, episode_length=args.episode_length,
                rollout_steps=args.rollout_steps, min_log_std=args.min_log_std,
                max_log_std=args.max_log_std,
            )
        print(json.dumps(row), flush=True)

    if writer:
        writer.close()
    save_state(
        out, actor, {"history": history, "best_return": best_return},
        critic=critic.state_dict(), representation=args.representation,
        memory=args.memory, side_only=args.side_only, actor_hidden=args.actor_hidden,
        feature_dim=feature_dim, episode_length=args.episode_length,
        rollout_steps=args.rollout_steps, min_log_std=args.min_log_std,
        max_log_std=args.max_log_std,
    )
    append_csv(out.with_suffix(".csv"), history)
    return out

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    def common(p):
        p.add_argument("--data", required=True); p.add_argument("--output", required=True); p.add_argument("--device", default="auto"); p.add_argument("--seed", type=int, default=42); p.add_argument("--epochs", type=int, default=1); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--workers", type=int, default=0); p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--weight-decay", type=float, default=1e-4)
    p = sub.add_parser("jepa"); common(p); p.add_argument("--latent-dim", type=int, default=128); p.add_argument("--strategy", default="multi-block", choices=("random", "antenna", "time", "multi-block")); p.add_argument("--mask-ratio", type=float, default=.5); p.add_argument("--ema-start", type=float, default=.996); p.add_argument("--val-fraction", type=float, default=.1); p.add_argument("--patience", type=int, default=10, help="epochs without validation improvement before stopping; 0 disables"); p.add_argument("--min-delta", type=float, default=0.0); p.add_argument("--normalize", action="store_true", help="per-window max-magnitude normalization"); p.set_defaults(func=train_jepa)
    p = sub.add_parser("vae"); common(p); p.add_argument("--latent-dim", type=int, default=128); p.add_argument("--beta", type=float, default=1e-3); p.set_defaults(func=train_vae)
    p = sub.add_parser("mdn"); common(p); p.add_argument("--representation", choices=("jepa", "vae"), required=True); p.add_argument("--representation-checkpoint", required=True); p.add_argument("--hidden-dim", type=int, default=256); p.add_argument("--mixtures", type=int, default=5); p.add_argument("--kpi-weight", type=float, default=.1); p.set_defaults(func=train_mdn)
    p = sub.add_parser("ppo"); p.add_argument("--representation", choices=("jepa", "vae"), required=True); p.add_argument("--representation-checkpoint", required=True); p.add_argument("--dynamics-checkpoint", required=True); p.add_argument("--output", required=True); p.add_argument("--device", default="auto"); p.add_argument("--seed", type=int, default=42); p.add_argument("--episodes", type=int, default=500); p.add_argument("--episode-length", type=int, default=5000); p.add_argument("--rollout-steps", type=int, default=512, help="transitions per recurrent PPO update; does not shorten the episode"); p.add_argument("--minibatch-size", type=int, default=256); p.add_argument("--actor-hidden", type=int, default=128); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--critic-lr", type=float, default=1e-3); p.add_argument("--weight-decay", type=float, default=0.0); p.add_argument("--gamma", type=float, default=.99); p.add_argument("--gae-lambda", type=float, default=.95); p.add_argument("--clip", type=float, default=.2); p.add_argument("--ppo-epochs", type=int, default=4); p.add_argument("--entropy-coef", type=float, default=.01); p.add_argument("--target-kl", type=float, default=.03); p.add_argument("--max-grad-norm", type=float, default=.5); p.add_argument("--reward-scale", type=float, default=1.0); p.add_argument("--initial-log-std", type=float, default=.7); p.add_argument("--min-log-std", type=float, default=-2.5); p.add_argument("--max-log-std", type=float, default=1.0); p.add_argument("--memory", action="store_true"); p.add_argument("--side-only", action="store_true"); p.set_defaults(func=train_ppo)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args(); parsed.func(parsed)

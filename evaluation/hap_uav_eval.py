"""Evaluation, genie-oracle benchmarking, and confidence intervals for V1."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

from environment.hap_uav import HapUavConfig, HapUavEnv
from models.jepa_wm import ActionConditionedMDNLSTM, LatentStateEncoder, PowerController
from training.jepa_wm import _load_representation


def genie_action(env: HapUavEnv, step_db: int = 1) -> np.ndarray:
    """One-step full-state grid oracle; never passed into policy observations."""
    best, best_reward = np.zeros(2, dtype=np.float32), -float("inf")
    values = np.arange(env.config.min_power_dbm, env.config.max_power_dbm + 0.1, step_db)
    span = env.config.max_power_dbm - env.config.min_power_dbm
    for pt in values:
        for pj in values:
            candidate = np.array([(pt - env.config.min_power_dbm) / span, (pj - env.config.min_power_dbm) / span], dtype=np.float32)
            clone = copy.deepcopy(env)
            _, reward, _, _, _ = clone.step(candidate)
            if reward > best_reward:
                best_reward, best = reward, candidate
    return best


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1) if values.size > 1 else 0.0), "ci95": float(1.96 * values.std(ddof=1) / np.sqrt(values.size) if values.size > 1 else 0.0)}


def evaluate_policy(policy, episodes=10, episode_length=100, seed=42, scenario=None):
    rows = []
    for episode in range(episodes):
        env = HapUavEnv(HapUavConfig(max_steps=episode_length))
        obs, _ = env.reset(seed=seed + episode, scenario=scenario)
        totals = {"return": 0.0, "age_h": [], "age_u": [], "secrecy_rate": [], "secrecy_outage": [], "pt_dbm": [], "pj_dbm": [], "success": [], "leakage": []}
        state = policy.reset() if hasattr(policy, "reset") else None
        for _ in range(episode_length):
            action = policy(obs, env) if callable(policy) else np.zeros(2, dtype=np.float32)
            obs, reward, term, trunc, info = env.step(action)
            totals["return"] += reward
            totals["age_h"].append(info["age_h"]); totals["age_u"].append(info["age_u"]); totals["secrecy_rate"].append(info["secrecy_rate"]); totals["secrecy_outage"].append(float(not info["secrecy_gate"]))
            totals["pt_dbm"].append(info["pt_dbm"]); totals["pj_dbm"].append(info["pj_dbm"]); totals["success"].append(float(info["mu_h"])); totals["leakage"].append(float(info["mu_u"]))
            if term or trunc: break
        row = {"episode": episode, "return": totals["return"]}
        for key in totals:
            if key != "return": row[key] = float(np.mean(totals[key]))
        rows.append(row)
    summary = {key: _stats([row[key] for row in rows]) for key in rows[0] if key != "episode"}
    return rows, summary



class LearnedPolicy:
    """Deterministic checkpoint policy using only partial observations."""
    def __init__(self, representation, dynamics, side_encoder, actor, device, memory=True, side_only=False):
        self.representation = representation
        self.dynamics = dynamics
        self.side_encoder = side_encoder
        self.actor = actor
        self.device = device
        self.memory = memory
        self.side_only = side_only
        self.hidden = None

    def reset(self):
        self.hidden = None
        return self

    def __call__(self, obs, env):
        iq = torch.from_numpy(obs["iq"]).to(self.device).unsqueeze(0)
        side = torch.from_numpy(obs["side"]).to(self.device).unsqueeze(0)
        with torch.no_grad():
            rf = self.representation.encode_rf(iq)
            if self.side_only:
                rf = torch.zeros_like(rf)
            z = torch.cat([rf, self.side_encoder.side_encoder(side)], dim=-1)
            h = self.hidden[0].transpose(0, 1).reshape(1, -1) if self.hidden is not None else torch.zeros(1, 256, device=self.device)
            features = torch.cat([z, h], dim=-1)
            action, _ = self.actor.action(features, deterministic=True)
            if self.memory:
                self.hidden = self.dynamics(z, action, self.hidden).hidden
            else:
                self.hidden = None
        return action.squeeze(0).cpu().numpy().astype(np.float32)


def load_learned_policy(representation_kind, representation_checkpoint, dynamics_checkpoint, actor_checkpoint, device="cpu", memory=True, side_only=False):
    device = torch.device(device)
    representation = _load_representation(representation_kind, representation_checkpoint, device)
    dynamics_payload = torch.load(dynamics_checkpoint, map_location=device)
    dynamics = ActionConditionedMDNLSTM(160, dynamics_payload.get("hidden_dim", 256), dynamics_payload.get("mixtures", 5)).to(device)
    dynamics.load_state_dict(dynamics_payload["model"]); dynamics.eval()
    side_encoder = LatentStateEncoder(128, 32).to(device); side_encoder.load_state_dict(dynamics_payload["side_encoder"]); side_encoder.eval()
    actor_payload = torch.load(actor_checkpoint, map_location=device)
    actor = PowerController(416, actor_payload.get("actor_hidden", 128)).to(device)
    actor.load_state_dict(actor_payload["model"]); actor.eval()
    return LearnedPolicy(representation, dynamics, side_encoder, actor, device, memory, side_only)

def _simple_policy(name):
    if name == "random": return lambda obs, env: env.rng.random(2).astype(np.float32)
    if name == "fixed": return lambda obs, env: np.array([1.0, 0.0], dtype=np.float32)
    if name == "balanced": return lambda obs, env: np.array([.5, .5], dtype=np.float32)
    if name == "oracle": return lambda obs, env: genie_action(env)
    raise ValueError(name)


def evaluate_named(names, output, episodes=10, episode_length=100, seed=42):
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    all_rows, summaries = [], {}
    for name in names:
        rows, summary = evaluate_policy(_simple_policy(name), episodes, episode_length, seed)
        summaries[name] = summary
        for row in rows: all_rows.append({"policy": name, **row})
    with output.open("w", newline="") as handle:
        fields = sorted({key for row in all_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    output.with_suffix(".json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    return summaries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", default=["random", "fixed", "balanced", "oracle"], choices=("random", "fixed", "balanced", "oracle"))
    p.add_argument("--learned-representation", choices=("jepa", "vae"))
    p.add_argument("--representation-checkpoint")
    p.add_argument("--dynamics-checkpoint")
    p.add_argument("--actor-checkpoint")
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--side-only", action="store_true")
    p.add_argument("--episodes", type=int, default=10); p.add_argument("--episode-length", type=int, default=100); p.add_argument("--seed", type=int, default=42); p.add_argument("--device", default="cpu"); p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    all_rows, summaries = [], {}
    for name in args.policies:
        rows, summary = evaluate_policy(_simple_policy(name), args.episodes, args.episode_length, args.seed)
        summaries[name] = summary
        all_rows.extend({"policy": name, **row} for row in rows)
    if args.learned_representation:
        required = (args.representation_checkpoint, args.dynamics_checkpoint, args.actor_checkpoint)
        if any(value is None for value in required):
            p.error("learned evaluation requires representation, dynamics, and actor checkpoints")
        name = args.learned_representation + ("_side_only" if args.side_only else "") + ("_no_memory" if args.no_memory else "_memory")
        policy = load_learned_policy(args.learned_representation, *required, device=args.device, memory=not args.no_memory, side_only=args.side_only)
        rows, summary = evaluate_policy(policy, args.episodes, args.episode_length, args.seed)
        summaries[name] = summary; all_rows.extend({"policy": name, **row} for row in rows)
    with output.open("w", newline="") as handle:
        fields = sorted({key for row in all_rows for key in row}); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    output.with_suffix(".json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__": main()

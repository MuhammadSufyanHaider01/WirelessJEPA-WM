"""Synthetic HAP--UAV datasets for JEPA and latent-dynamics training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from environment.hap_uav import HapUavConfig, HapUavEnv


class PilotWindowDataset(Dataset):
    """Unlabelled raw pilot windows; antenna upsampling happens on demand."""
    def __init__(self, path: str | Path, normalize: bool = False):
        self.path = str(path)
        self._file = None
        self._pid = None
        with h5py.File(self.path, "r") as f:
            self.length = int(f["iq"].shape[0])
            self.shape = tuple(f["iq"].shape[1:])
        if self.shape != (2, 4, 256):
            raise ValueError(f"Expected raw I/Q shape (2,4,256), got {self.shape}")
        self.normalize = normalize

    def _open(self):
        import os
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            self.close()
            self._file = h5py.File(self.path, "r")
            self._pid = pid

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self._open()
        x = torch.from_numpy(np.asarray(self._file["iq"][index], dtype=np.float32))
        if self.normalize:
            x = x / (x.abs().amax() + 1e-6)
        return x

    def close(self):
        if self._file is not None:
            self._file.close()
        self._file = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TrajectoryDataset(Dataset):
    """Ordered partial-observation trajectories stored one episode per row."""
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._file = None
        self._pid = None
        with h5py.File(self.path, "r") as f:
            self.length = int(f["iq"].shape[0])
            self.episode_length = int(f["iq"].shape[1])
            self.keys = tuple(f.keys())

    def _open(self):
        import os
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            self.close()
            self._file = h5py.File(self.path, "r")
            self._pid = pid

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self._open()
        f = self._file
        result = {
            "iq": torch.from_numpy(np.asarray(f["iq"][index], dtype=np.float32)),
            "side": torch.from_numpy(np.asarray(f["side"][index], dtype=np.float32)),
            "action": torch.from_numpy(np.asarray(f["action"][index], dtype=np.float32)),
            "reward": torch.from_numpy(np.asarray(f["reward"][index], dtype=np.float32)),
            "done": torch.from_numpy(np.asarray(f["done"][index], dtype=np.bool_)),
        }
        for key in ("secrecy_rate", "success", "leakage"):
            if key in f:
                result[key] = torch.from_numpy(np.asarray(f[key][index], dtype=np.float32))
        if "scenario_id" in f:
            result["scenario_id"] = int(f["scenario_id"][index])
        return result

    def close(self):
        if self._file is not None:
            self._file.close()
        self._file = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _scenario_for(seed: int, config: HapUavConfig) -> dict:
    trajectories = ("static", "linear", "random_walk", "circular")
    return {
        "trajectory": trajectories[seed % len(trajectories)],
        "uav_initial_xy_m": (
            5_000.0 + float((seed % 11) - 5) * 100.0,
            float(((seed // 11) % 11) - 5) * 100.0,
        ),
    }


def generate_pilot_hdf5(path: str | Path, count: int = 10_000, config: Optional[HapUavConfig] = None, seed: int = 42) -> dict:
    """Generate unlabelled windows and write only compact raw I/Q tensors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = config or HapUavConfig(max_steps=1)
    shape = (int(count), 2, 4, cfg.pilot_samples)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("iq", shape=shape, dtype="f4", chunks=(1, 2, 4, cfg.pilot_samples), compression="lzf")
        seeds = np.zeros(count, dtype=np.int64)
        for index in range(count):
            sample_seed = int(seed + index)
            env = HapUavEnv(cfg)
            obs, _ = env.reset(seed=sample_seed, scenario=_scenario_for(sample_seed, cfg))
            ds[index] = obs["iq"]
            seeds[index] = sample_seed
        f.create_dataset("seed", data=seeds)
        f.attrs["raw_shape"] = shape[1:]
        f.attrs["split_policy"] = "one independent scenario seed per window"
        f.attrs["normalization"] = "per-window max magnitude"
    return {"path": str(path), "count": count, "shape": shape[1:]}


def generate_trajectory_hdf5(path: str | Path, episodes: int = 256, length: int = 100, config: Optional[HapUavConfig] = None, seed: int = 42) -> dict:
    """Generate ordered transitions with a broad random/grid action mixture."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = config or HapUavConfig(max_steps=length)
    iq = np.zeros((episodes, length, 2, 4, cfg.pilot_samples), dtype=np.float32)
    side = np.zeros((episodes, length, 8), dtype=np.float32)
    action = np.zeros((episodes, length, 2), dtype=np.float32)
    reward = np.zeros((episodes, length), dtype=np.float32)
    done = np.zeros((episodes, length), dtype=np.bool_)
    secrecy_rate = np.zeros((episodes, length), dtype=np.float32)
    success = np.zeros((episodes, length), dtype=np.float32)
    leakage = np.zeros((episodes, length), dtype=np.float32)
    scenario_ids = np.arange(episodes, dtype=np.int64)
    for episode in range(episodes):
        episode_seed = int(seed + episode)
        env = HapUavEnv(cfg)
        obs, _ = env.reset(seed=episode_seed, scenario=_scenario_for(episode_seed, cfg))
        rng = np.random.default_rng(episode_seed + 10_000)
        for t in range(length):
            iq[episode, t] = obs["iq"]
            side[episode, t] = obs["side"]
            # Random/grid mixture deliberately covers the full action square.
            if t % 4 == 0:
                a = np.array([rng.integers(0, 31), rng.integers(0, 31)], dtype=np.float32) / 30.0
            else:
                a = rng.random(2, dtype=np.float32)
            action[episode, t] = a
            obs, r, term, trunc, info = env.step(a)
            reward[episode, t] = r
            secrecy_rate[episode, t] = float(info["secrecy_rate"])
            success[episode, t] = float(info["mu_h"])
            leakage[episode, t] = float(info["mu_u"])
            done[episode, t] = term or trunc
    with h5py.File(path, "w") as f:
        f.create_dataset("iq", data=iq, compression="lzf")
        f.create_dataset("side", data=side, compression="lzf")
        f.create_dataset("action", data=action, compression="lzf")
        f.create_dataset("reward", data=reward, compression="lzf")
        f.create_dataset("done", data=done, compression="lzf")
        f.create_dataset("secrecy_rate", data=secrecy_rate, compression="lzf")
        f.create_dataset("success", data=success, compression="lzf")
        f.create_dataset("leakage", data=leakage, compression="lzf")
        f.create_dataset("scenario_id", data=scenario_ids)
        f.attrs["split_policy"] = "episodes are independent scenario seeds"
    return {"path": str(path), "episodes": episodes, "length": length}


def generate_expert_hdf5(path: str | Path, episodes: int = 256, length: int = 100, config: Optional[HapUavConfig] = None, seed: int = 42, grid_step_db: int = 1) -> dict:
    """Generate one-step full-state genie trajectories for benchmarking.

    The oracle sees simulator-only instantaneous channels. Its actions are an
    upper-bound benchmark and must not be used as labels for the partial
    observation policy in the primary experiment.
    """
    from evaluation.hap_uav_eval import genie_action
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    cfg = config or HapUavConfig(max_steps=length)
    iq = np.zeros((episodes, length, 2, 4, cfg.pilot_samples), dtype=np.float32)
    side = np.zeros((episodes, length, 8), dtype=np.float32)
    action = np.zeros((episodes, length, 2), dtype=np.float32)
    reward = np.zeros((episodes, length), dtype=np.float32)
    done = np.zeros((episodes, length), dtype=np.bool_)
    secrecy_rate = np.zeros((episodes, length), dtype=np.float32)
    success = np.zeros((episodes, length), dtype=np.float32)
    leakage = np.zeros((episodes, length), dtype=np.float32)
    scenario_ids = np.arange(episodes, dtype=np.int64)
    for episode in range(episodes):
        episode_seed = int(seed + episode)
        env = HapUavEnv(cfg); obs, _ = env.reset(seed=episode_seed, scenario=_scenario_for(episode_seed, cfg))
        for t in range(length):
            iq[episode, t] = obs["iq"]; side[episode, t] = obs["side"]
            a = genie_action(env, step_db=grid_step_db); action[episode, t] = a
            obs, r, term, trunc, info = env.step(a)
            reward[episode, t] = r; secrecy_rate[episode, t] = float(info["secrecy_rate"])
            success[episode, t] = float(info["mu_h"]); leakage[episode, t] = float(info["mu_u"])
            done[episode, t] = term or trunc
    with h5py.File(path, "w") as f:
        for key, value in (("iq", iq), ("side", side), ("action", action), ("reward", reward), ("done", done), ("secrecy_rate", secrecy_rate), ("success", success), ("leakage", leakage), ("scenario_id", scenario_ids)):
            f.create_dataset(key, data=value, compression="lzf" if value.ndim > 1 else None)
        f.attrs["split_policy"] = "episodes are independent scenario seeds"
        f.attrs["oracle"] = "full-state one-step grid upper bound; not a policy label"
        f.attrs["grid_step_db"] = int(grid_step_db)
    return {"path": str(path), "episodes": episodes, "length": length, "grid_step_db": grid_step_db}


def write_manifest(path: str | Path, **entries) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")


__all__ = ["PilotWindowDataset", "TrajectoryDataset", "generate_pilot_hdf5", "generate_trajectory_hdf5", "generate_expert_hdf5", "write_manifest"]

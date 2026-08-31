"""Generate V1 synthetic pilot and ordered trajectory datasets."""
from __future__ import annotations

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from pathlib import Path

from data.hap_uav import generate_pilot_hdf5, generate_trajectory_hdf5, write_manifest
from environment.hap_uav import HapUavConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data/hap_uav")
    p.add_argument("--pilot-windows", type=int, default=100_000)
    p.add_argument("--trajectories", type=int, default=2_000)
    p.add_argument("--trajectory-length", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.pilot_windows, args.trajectories, args.trajectory_length = 16, 4, 8
    root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    cfg = HapUavConfig(max_steps=args.trajectory_length)
    pilot = generate_pilot_hdf5(root / "pilot_train.h5", args.pilot_windows, cfg, args.seed)
    trajectories = generate_trajectory_hdf5(root / "trajectories_train.h5", args.trajectories, args.trajectory_length, cfg, args.seed + 100_000)
    write_manifest(root / "manifest.json", config=cfg.__dict__, pilot=pilot, trajectories=trajectories, split_policy="whole independent scenario seeds")
    print(root / "manifest.json")


if __name__ == "__main__":
    main()

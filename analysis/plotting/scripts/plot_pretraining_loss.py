#!/usr/bin/env python3
"""Plot epoch-level JEPA pretraining loss for all masking strategies.

The completed runs use Lightning CSV and TensorBoard loggers. TensorBoard
files are preferred because they preserve the pre-wall-time-limit history for
resumed antenna and multi-block jobs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:  # pragma: no cover - depends on the project environment
    raise SystemExit(
        "TensorBoard is required. Run this script in the wireless-jepa environment."
    ) from exc


MASK_ORDER = ["random", "antenna", "time", "multi-block"]
MASK_LABELS = {
    "random": "Random",
    "antenna": "Antenna",
    "time": "Time",
    "multi-block": "Multi-block",
}
COLORS = ["#3B6EA8", "#D95F02", "#2E8B57", "#756BB1"]
LINE_STYLES = ["-", "--", "-.", ":"]


def mask_from_run_name(run_name: str) -> str | None:
    for mask in MASK_ORDER:
        if run_name.startswith(f"WirelessJEPA_{mask}_"):
            return mask
    return None


def load_event_loss(run_dir: Path) -> dict[int, float]:
    """Load and merge epoch losses from all TensorBoard event files."""
    event_dir = run_dir / "version_0" / "tensorboard" / "version_0"
    points: dict[int, float] = {}
    for event_path in sorted(event_dir.glob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", [])
        if "train_metrics/ijepa_loss_epoch" not in tags:
            continue
        losses = accumulator.Scalars("train_metrics/ijepa_loss_epoch")
        epochs = {
            item.step: item.value
            for item in accumulator.Scalars("epoch")
        } if "epoch" in tags else {}
        for item in losses:
            epoch = epochs.get(item.step)
            if epoch is None:
                continue
            # Later event files contain the resumed value for a repeated epoch.
            points[int(round(epoch))] = float(item.value)
    return dict(sorted(points.items()))


def load_csv_loss(run_dir: Path) -> dict[int, float]:
    """Fallback loader for runs where only CSV metrics are available."""
    csv_path = run_dir / "version_0" / "csv" / "version_0" / "metrics.csv"
    points: dict[int, float] = {}
    if not csv_path.exists():
        return points
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            epoch = row.get("epoch", "")
            value = row.get("train_metrics/ijepa_loss_epoch", "")
            if epoch and value:
                points[int(float(epoch))] = float(value)
    return dict(sorted(points.items()))


def load_losses(results_root: Path) -> dict[str, dict[int, float]]:
    losses: dict[str, dict[int, float]] = {}
    for run_dir in sorted(results_root.glob("WirelessJEPA_*/")):
        mask = mask_from_run_name(run_dir.name)
        if mask is None:
            continue
        points = load_event_loss(run_dir)
        if not points:
            points = load_csv_loss(run_dir)
        if points:
            losses[mask] = points

    missing = [mask for mask in MASK_ORDER if mask not in losses]
    if missing:
        raise RuntimeError(f"No epoch-level training loss found for: {missing}")
    return losses


def plot(losses: dict[str, dict[int, float]], output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=180)
    for index, mask in enumerate(MASK_ORDER):
        points = losses[mask]
        epochs = list(points)
        values = [points[epoch] for epoch in epochs]
        ax.plot(
            epochs,
            values,
            color=COLORS[index],
            linestyle=LINE_STYLES[index],
            linewidth=2.0,
            label=MASK_LABELS[mask],
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("JEPA training loss")
    ax.set_xlim(left=0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#D0D0D0", loc="best")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_root / "artifacts/pretrain_lightly/ijepacnn_iqfm",
        help="Directory containing WirelessJEPA_<mask> run directories.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=repo_root / "analysis/plotting/figures/pretraining_loss_by_mask",
        help="Output path without an extension; PNG and PDF are written.",
    )
    args = parser.parse_args()
    losses = load_losses(args.results_root)
    plot(losses, args.output_stem)
    for mask in MASK_ORDER:
        points = losses[mask]
        print(f"{MASK_LABELS[mask]}: epochs {min(points)}–{max(points)} ({len(points)} points)")
    print(f"Wrote {args.output_stem.with_suffix('.png')}")
    print(f"Wrote {args.output_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot frozen-encoder downstream accuracy by masking strategy.

The script reads the ``Benchmark results`` lines emitted by
``pretrain/eval_ijepacnn.py`` and writes grouped linear-probe accuracy bars
for every completed ``*_samples500`` run.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MASK_ORDER = ["random", "antenna", "time", "multi-block"]
MASK_LABELS = {
    "random": "Random",
    "antenna": "Antenna",
    "time": "Time",
    "multi-block": "Multi-block",
}
TASK_ORDER = ["aoa", "mod", "rf", "rml", "radar"]
TASK_LABELS = {
    "aoa": "AoA",
    "mod": "Modulation",
    "rf": "RF fingerprint",
    "rml": "RML modulation",
    "radar": "Radar",
}
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]


def load_linear_top1(results_root: Path) -> dict[tuple[str, str], float]:
    """Return final linear top-1 accuracy in percent for each mask/task pair."""
    values: dict[tuple[str, str], float] = {}
    value_pattern = re.compile(r"'([^']+)': tensor\(([-+0-9.eE]+)")
    result_pattern = re.compile(r"Benchmark results: (\{.*\})")

    for log_path in sorted(results_root.glob("*_samples500/eval.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = result_pattern.findall(text)
        if not matches:
            continue
        metrics = {key: float(value) for key, value in value_pattern.findall(matches[-1])}
        run_name = log_path.parent.name.removesuffix("_samples500")
        try:
            mask, task = run_name.rsplit("_", 1)
        except ValueError:
            continue
        if "lin_top1_final" in metrics:
            values[(mask, task)] = 100.0 * metrics["lin_top1_final"]

    expected = [(mask, task) for mask in MASK_ORDER for task in TASK_ORDER]
    missing = [pair for pair in expected if pair not in values]
    if missing:
        raise RuntimeError(f"Missing completed metrics: {missing}")
    return values


def plot(values: dict[tuple[str, str], float], output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(TASK_ORDER))
    width = 0.19

    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=180)
    for index, mask in enumerate(MASK_ORDER):
        accuracies = [values[(mask, task)] for task in TASK_ORDER]
        bars = ax.bar(
            x + (index - 1.5) * width,
            accuracies,
            width,
            label=MASK_LABELS[mask],
            color=COLORS[index],
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, accuracy in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                accuracy + 1.2,
                f"{accuracy:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_title("WirelessJEPA Masking Ablation — 500-shot Linear Probe", weight="bold", pad=14)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Downstream task")
    ax.set_xticks(x, [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.text(
        0,
        -0.19,
        "Linear top-1 validation accuracy; AoA uses 80/20 train/validation because the dataset has 100 samples/class.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_root / "artifacts/downstream_linear",
        help="Directory containing *_samples500/eval.log files.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=repo_root / "analysis/plotting/figures/accuracy_500shots_bar",
        help="Output path without an extension; PNG and PDF are written.",
    )
    args = parser.parse_args()
    values = load_linear_top1(args.results_root)
    plot(values, args.output_stem)
    print(f"Wrote {args.output_stem.with_suffix('.png')}")
    print(f"Wrote {args.output_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

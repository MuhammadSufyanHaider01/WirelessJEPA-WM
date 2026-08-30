#!/usr/bin/env python3
"""Plot frozen-encoder downstream accuracy by masking strategy.

The script reads the ``Benchmark results`` lines emitted by
``pretrain/eval_ijepacnn.py`` and writes grouped top-1 accuracy bars for
every completed ``*_samples500`` run. By default it writes one figure for
the linear probe and one for kNN.
"""

from __future__ import annotations

import argparse
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
# A high-contrast, colorblind-friendly palette; hatches provide a second
# visual encoding so the mask identity remains clear in print/grayscale.
COLORS = ["#3B6EA8", "#D95F02", "#2E8B57", "#756BB1"]
HATCHES = ["///", "xx", "...", "++"]


def load_top1_metrics(results_root: Path) -> dict[str, dict[tuple[str, str], float]]:
    """Return final linear and kNN top-1 accuracy in percent.

    The evaluator stores probabilities in the final ``Benchmark results``
    dictionary. Keeping both metrics in one parser ensures the two plots are
    generated from exactly the same completed runs.
    """
    values: dict[str, dict[tuple[str, str], float]] = {
        "linear": {},
        "knn": {},
    }
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
            values["linear"][(mask, task)] = 100.0 * metrics["lin_top1_final"]
        if "knn_top1" in metrics:
            values["knn"][(mask, task)] = 100.0 * metrics["knn_top1"]

    expected = [(mask, task) for mask in MASK_ORDER for task in TASK_ORDER]
    for metric, metric_values in values.items():
        missing = [pair for pair in expected if pair not in metric_values]
        if missing:
            raise RuntimeError(f"Missing completed {metric} metrics: {missing}")
    return values


def plot(
    values: dict[tuple[str, str], float],
    output_stem: Path,
    *,
    title: str,
    note: str,
) -> None:
    """Write a grouped-bar figure in PNG and PDF formats."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(TASK_ORDER))
    width = 0.19

    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=180)
    for index, mask in enumerate(MASK_ORDER):
        accuracies = [values[(mask, task)] for task in TASK_ORDER]
        ax.bar(
            x + (index - 1.5) * width,
            accuracies,
            width,
            label=MASK_LABELS[mask],
            color=COLORS[index],
            edgecolor="#303030",
            linewidth=0.8,
            hatch=HATCHES[index],
            alpha=0.92,
        )

    ax.set_title(title, weight="bold", pad=18)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Downstream task")
    ax.set_xticks(x, [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )
    ax.text(
        0,
        -0.19,
        note,
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)



def plot_comparison(
    metrics: dict[str, dict[tuple[str, str], float]],
    output_stem: Path,
    *,
    note: str,
) -> None:
    """Write one two-panel figure comparing linear probing and kNN."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(TASK_ORDER))
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.8), sharey=True, dpi=180)

    for panel_index, (ax, metric, panel_title) in enumerate(
        zip(
            axes,
            ("linear", "knn"),
            ("(a) Linear probing", "(b) kNN"),
        )
    ):
        for index, mask in enumerate(MASK_ORDER):
            accuracies = [metrics[metric][(mask, task)] for task in TASK_ORDER]
            ax.bar(
                x + (index - 1.5) * width,
                accuracies,
                width,
                label=MASK_LABELS[mask] if panel_index == 0 else "_nolegend_",
                color=COLORS[index],
                edgecolor="#303030",
                linewidth=0.8,
                hatch=HATCHES[index],
                alpha=0.92,
            )
        ax.set_title(panel_title, weight="bold", pad=12)
        ax.set_xlabel("Downstream task")
        ax.set_xticks(x, [TASK_LABELS[task] for task in TASK_ORDER])
        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Top-1 accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        "WirelessJEPA Downstream Evaluation — Linear Probe vs kNN",
        weight="bold",
        y=1.02,
    )
    fig.text(0.5, 0.015, note, ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.055, 1, 0.90))
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
        help="Linear-probe output path without an extension; PNG and PDF are written.",
    )
    parser.add_argument(
        "--knn-output-stem",
        type=Path,
        default=repo_root / "analysis/plotting/figures/accuracy_500shots_knn_bar",
        help="kNN output path without an extension; PNG and PDF are written.",
    )
    parser.add_argument(
        "--metric",
        choices=("linear", "knn", "both"),
        default="both",
        help="Which metric to plot (default: both).",
    )
    parser.add_argument(
        "--comparison-output-stem",
        type=Path,
        default=repo_root / "analysis/plotting/figures/accuracy_500shots_linear_knn",
        help="Combined two-panel output path without an extension; written when --metric=both.",
    )
    args = parser.parse_args()
    metrics = load_top1_metrics(args.results_root)
    note = (
        "Top-1 validation accuracy; AoA uses 80/20 train/validation because "
        "the dataset has 100 samples/class."
    )

    if args.metric in ("linear", "both"):
        plot(
            metrics["linear"],
            args.output_stem,
            title="WirelessJEPA Masking Ablation — 500-shot Linear Probe",
            note=f"Linear {note[0].lower() + note[1:]}",
        )
        print(f"Wrote {args.output_stem.with_suffix('.png')}")
        print(f"Wrote {args.output_stem.with_suffix('.pdf')}")

    if args.metric in ("knn", "both"):
        plot(
            metrics["knn"],
            args.knn_output_stem,
            title="WirelessJEPA Masking Ablation — 500-shot kNN",
            note=f"kNN {note[0].lower() + note[1:]}",
        )
        print(f"Wrote {args.knn_output_stem.with_suffix('.png')}")
        print(f"Wrote {args.knn_output_stem.with_suffix('.pdf')}")

    if args.metric == "both":
        plot_comparison(metrics, args.comparison_output_stem, note=note)
        print(f"Wrote {args.comparison_output_stem.with_suffix('.png')}")
        print(f"Wrote {args.comparison_output_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

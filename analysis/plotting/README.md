# Plotting

This directory keeps analysis scripts separate from generated figures.

## Recreate the downstream accuracy figures

From the repository root, activate the project environment and run:

```bash
source "$HOME/software/init-conda"
conda activate wireless-jepa
python analysis/plotting/scripts/plot_downstream_accuracy.py
```

The script reads the completed downstream logs under
`artifacts/downstream_linear/` and writes the two retained figures:

- `figures/accuracy_500shots_bar.png` and `.pdf` — linear-probe top-1 accuracy
- `figures/accuracy_500shots_linear_knn.png` and `.pdf` — combined linear-probe/kNN comparison

The combined figure places both probing methods side by side with the same
masking and task order. Each masking strategy is encoded with both a distinct
color and hatch pattern so it remains readable in grayscale or print. The
current repository contains five tasks (`aoa`, `mod`, `rf`, `rml`, and `radar`);
it is a comparable local result, not a complete reproduction of the paper's
five OOD Figure 3 tasks.

## Pretraining loss by masking strategy

Run:

```bash
python analysis/plotting/scripts/plot_pretraining_loss.py
```

This writes `figures/pretraining_loss_by_mask.png` and `.pdf`, with one
line/legend entry per masking strategy. It plots the epoch-level
`train_metrics/ijepa_loss_epoch` values and merges the TensorBoard event files
from resumed jobs so every mask covers epochs 0–99.

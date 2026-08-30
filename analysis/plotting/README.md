# Plotting

This directory keeps analysis scripts separate from generated figures.

## Recreate the downstream accuracy figure

From the repository root, activate the project environment and run:

```bash
source "$HOME/software/init-conda"
conda activate wireless-jepa
python analysis/plotting/scripts/plot_downstream_accuracy.py
```

The script reads the completed downstream logs under
`artifacts/downstream_linear/` and, by default, writes both metrics:

- `figures/accuracy_500shots_bar.png` and `.pdf` — final linear-probe top-1 accuracy
- `figures/accuracy_500shots_knn_bar.png` and `.pdf` — kNN top-1 accuracy
- `figures/accuracy_500shots_linear_knn.png` and `.pdf` — combined two-panel comparison

The combined figure places linear probing and kNN side by side with the same
masking and task order. Use `--metric linear` or `--metric knn` to generate
only one standalone figure. Each masking strategy is encoded with both a
distinct color and hatch pattern so it remains readable in grayscale or print.
The current repository contains five
tasks (`aoa`, `mod`, `rf`, `rml`, and `radar`); it is a comparable local
result, not a complete reproduction of the paper's five OOD Figure 3 tasks.

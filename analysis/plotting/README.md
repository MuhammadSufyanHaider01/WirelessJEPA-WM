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
`artifacts/downstream_linear/` and writes:

- `figures/accuracy_500shots_bar.png`
- `figures/accuracy_500shots_bar.pdf`

The chart shows final linear top-1 validation accuracy for each available
masking strategy and downstream task. The current repository contains five
tasks (`aoa`, `mod`, `rf`, `rml`, and `radar`); it is a comparable local
result, not a complete reproduction of the paper's five OOD Figure 3 tasks.

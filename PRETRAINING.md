# WirelessJEPA pretraining

The reproduction uses the IQFM corpus with 777,878 training samples and a
337,500-sample held-out split. Each sample has shape `(4, 2, 256)` before the
paper's nearest-neighbor antenna upsampling to `(2, 256, 256)`.

## 1. Prepare the environment

```bash
source ~/software/init-conda
conda env create -f environment-wirelessjepa.yml
conda activate wireless-jepa
```

## 2. Connect the local datasets

The HDF5 files remain under the ignored `datasets/` directory. From the
repository root:

```bash
ln -s ../datasets/train_256_100_256_22.h5 data/iqfm-train.h5
ln -s ../datasets/test_256_100_256_22.h5 data/iqfm-val.h5
python3 data/sample_h5.py \
    --src data/iqfm-val.h5 \
    --dst data/iqfm-val-100.h5 \
    --per-class 100 \
    --seed 42
```

`iqfm-val-100.h5` must contain 22,500 samples: 100 for each of 225 AoA
classes. All three paths are ignored by Git.

## 3. Run preflight tests

```bash
python3 -m unittest -v tests/test_wirelessjepa_pretraining.py
python3 pretrain/smoke_iqfm_pretraining.py --device cpu --batch-size 1
```

For a CUDA smoke test on ARC:

```bash
sbatch wirelessjepa-smoke.slurm
```

Do not submit full pretraining until the smoke job completes successfully for
all four masks.

## 4. Submit all four full runs together

The array launcher starts four independent models concurrently (one model per
masking strategy), keeping the paper comparisons valid:

```bash
sbatch wirelessjepa-pretrain-all.slurm
```

Each task writes Slurm stdout to
`artifacts/slurm-wirelessjepa-all-<job>_<task>.out`. For every run,
checkpoints and local metrics are stored under
`artifacts/pretrain_lightly/ijepacnn_iqfm/<run-name>/version_<n>/`:

- TensorBoard event files: `tensorboard/version_0/`
- CSV metrics (training loss, validation loss, learning rate, weight decay,
  and online linear-evaluation metrics): `csv/version_0/`
- `last.ckpt` and the best checkpoint selected by `val_metrics/lin_top1`

View metrics while jobs are running with `tensorboard --logdir artifacts` or
inspect `metrics.csv` in each run's CSV directory.

If GPUs are unavailable in the default partition, override it at submission
time, for example: `sbatch --partition=gpu-h100 wirelessjepa-pretrain-all.slurm`.

For a single full run, use:

```bash
sbatch cnn-jepa.slurm ijepacnn_iqfm_random
sbatch cnn-jepa.slurm ijepacnn_iqfm_antenna
sbatch cnn-jepa.slurm ijepacnn_iqfm_time
sbatch cnn-jepa.slurm ijepacnn_iqfm_multiblock
```

The four canonical configurations reproduce the paper's mask geometry:

| Config | Mask | Patch size |
|---|---|---:|
| `ijepacnn_iqfm_random` | Random baseline | `64 x 32` |
| `ijepacnn_iqfm_antenna` | Complete antenna bands | `64 x 256` |
| `ijepacnn_iqfm_time` | Complete time bands | `256 x 32` |
| `ijepacnn_iqfm_multiblock` | Contiguous spatio-temporal blocks | `64 x 32` |

W&B is disabled by default. Authenticate with `wandb login` and append
`wandb=true` to the `sbatch` command only when remote logging is wanted. Never
store API keys in a tracked shell script.

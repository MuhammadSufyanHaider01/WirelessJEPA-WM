#!/bin/bash
# Full-data multi-block JEPA representation training on the idle RTX 6000.
# The trainer evaluates a deterministic 10% held-out pilot split each epoch,
# keeps the best checkpoint, and stops after PATIENCE stale epochs.
#SBATCH --job-name=jepa-wm
#SBATCH --partition=gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=artifacts/slurm-%x-%j.out

set -eo pipefail
source "${HOME}/software/init-conda"
conda activate wireless-jepa
set -u

cd "${SLURM_SUBMIT_DIR:?Submit this job from the WirelessJEPA repository root}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

DATA="${JEPA_WM_DATA:-${PWD}/artifacts/jepa_wm_v1/datasets/pilot_train.h5}"
OUTPUT="${JEPA_WM_OUTPUT:-${PWD}/artifacts/jepa_wm_v1/checkpoints/jepa_multiblock_rtx6000.pt}"
EPOCHS="${JEPA_WM_EPOCHS:-100}"
PATIENCE="${JEPA_WM_PATIENCE:-10}"
BATCH_SIZE="${JEPA_WM_BATCH_SIZE:-64}"
WORKERS="${JEPA_WM_WORKERS:-4}"
mkdir -p "$(dirname "${OUTPUT}")"

echo "Starting JEPA on ${HOSTNAME}; data=${DATA}; output=${OUTPUT}; epochs=${EPOCHS}; patience=${PATIENCE}" >&2
srun python3 -u training/jepa_wm.py jepa \
  --data "${DATA}" \
  --output "${OUTPUT}" \
  --device cuda \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --strategy multi-block \
  --mask-ratio 0.5 \
  --val-fraction 0.10 \
  --patience "${PATIENCE}" \
  --min-delta 0.0 \
  --normalize

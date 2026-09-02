#!/bin/bash
# Full JEPA-conditioned MDN-LSTM and side-MLP training.
# JEPA is frozen; only the dynamics model and 8->64->32 side encoder update.
#SBATCH --job-name=mdn-wm
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=artifacts/slurm-%x-%j.out

set -eo pipefail
source "${HOME}/software/init-conda"
conda activate wireless-jepa
set -u

cd "${SLURM_SUBMIT_DIR:?Submit this job from the WirelessJEPA repository root}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

DATA="${JEPA_WM_TRAJECTORIES:-${PWD}/artifacts/jepa_wm_v1/datasets/trajectories_train.h5}"
REPRESENTATION="${JEPA_WM_CHECKPOINT:-${PWD}/artifacts/jepa_wm_v1/checkpoints/jepa_multiblock_gpu.pt}"
OUTPUT="${JEPA_WM_MDN_OUTPUT:-${PWD}/artifacts/jepa_wm_v1/checkpoints/mdn_jepa_a100.pt}"
EPOCHS="${JEPA_WM_MDN_EPOCHS:-50}"
BATCH_SIZE="${JEPA_WM_MDN_BATCH_SIZE:-4}"
WORKERS="${JEPA_WM_MDN_WORKERS:-4}"
mkdir -p "$(dirname "${OUTPUT}")"

# Fail before loading the large trajectory file if this node is unsupported.
python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the Slurm allocation")
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
supported = torch.cuda.get_arch_list()
print(f"CUDA device: {torch.cuda.get_device_name(0)}; capability={arch}; supported_arches={supported}", flush=True)
if arch not in supported:
    raise SystemExit(f"PyTorch build does not support {arch}; choose A100, H100, or V100")
PY

echo "Starting MDN/side-MLP on ${HOSTNAME}; data=${DATA}; representation=${REPRESENTATION}; output=${OUTPUT}; epochs=${EPOCHS}" >&2
srun python3 -u training/jepa_wm.py mdn \
  --data "${DATA}" \
  --representation jepa \
  --representation-checkpoint "${REPRESENTATION}" \
  --output "${OUTPUT}" \
  --device cuda \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --hidden-dim 256 \
  --mixtures 5 \
  --kpi-weight 0.1

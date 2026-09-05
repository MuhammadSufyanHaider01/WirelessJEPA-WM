#!/bin/bash
# Full-data JEPA representation training on a CUDA-compatible GPU.
# The trainer evaluates a deterministic 10% held-out pilot split each epoch,
# keeps the best checkpoint, and stops after PATIENCE stale epochs.
#SBATCH --job-name=jepa-wm
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=artifacts/slurm-%x-%A_%a.out

set -eo pipefail
source "${HOME}/software/init-conda"
conda activate wireless-jepa
set -u

cd "${SLURM_SUBMIT_DIR:?Submit this job from the WirelessJEPA repository root}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

DATA="${JEPA_WM_DATA:-${PWD}/artifacts/jepa_wm_v1/datasets/pilot_train.h5}"
ARRAY_STRATEGY=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  case "${SLURM_ARRAY_TASK_ID}" in
    0) ARRAY_STRATEGY="random" ;;
    1) ARRAY_STRATEGY="antenna" ;;
    2) ARRAY_STRATEGY="time" ;;
    *) echo "Unsupported JEPA array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
  esac
fi
STRATEGY="${JEPA_WM_STRATEGY:-${ARRAY_STRATEGY:-multi-block}}"
case "${STRATEGY}" in
  random|antenna|time|multi-block) ;;
  *) echo "Unsupported JEPA strategy ${STRATEGY}" >&2; exit 2 ;;
esac
DEFAULT_STEM="${STRATEGY//-/_}"
OUTPUT="${JEPA_WM_OUTPUT:-${PWD}/artifacts/jepa_wm_v1/checkpoints/jepa_${DEFAULT_STEM}_gpu_v4.pt}"
EPOCHS="${JEPA_WM_EPOCHS:-100}"
PATIENCE="${JEPA_WM_PATIENCE:-30}"
LR="${JEPA_WM_LR:-3e-5}"
VAL_REPEATS="${JEPA_WM_VAL_REPEATS:-8}"
WARMUP_EPOCHS="${JEPA_WM_WARMUP_EPOCHS:-3}"
FINAL_LR_RATIO="${JEPA_WM_FINAL_LR_RATIO:-0.1}"
GRAD_CLIP="${JEPA_WM_GRAD_CLIP:-1.0}"
MIN_DELTA="${JEPA_WM_MIN_DELTA:-0.0001}"
EMA_START="${JEPA_WM_EMA_START:-0.9999}"
EMA_END="${JEPA_WM_EMA_END:-0.9999}"
ANTENNA_RATIOS="${JEPA_WM_ANTENNA_RATIOS:-0.25}"
if [[ -n "${JEPA_WM_MASK_RATIO:-}" ]]; then
  MASK_RATIO="${JEPA_WM_MASK_RATIO}"
elif [[ "${STRATEGY}" == "multi-block" ]]; then
  MASK_RATIO="0.5"
else
  MASK_RATIO="0.25"
fi
MASK_RATIO_START="${JEPA_WM_MASK_RATIO_START:-}"
MASK_CURRICULUM_EPOCHS="${JEPA_WM_MASK_CURRICULUM_EPOCHS:-0}"
BATCH_SIZE="${JEPA_WM_BATCH_SIZE:-64}"
WORKERS="${JEPA_WM_WORKERS:-4}"
mkdir -p "$(dirname "${OUTPUT}")"
CURRICULUM_ARGS=(--mask-curriculum-epochs "${MASK_CURRICULUM_EPOCHS}")
if [[ -n "${MASK_RATIO_START}" ]]; then
  CURRICULUM_ARGS+=(--mask-ratio-start "${MASK_RATIO_START}")
fi

# Fail early if the node GPU is not supported by this PyTorch build.
python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the Slurm allocation")
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
supported = torch.cuda.get_arch_list()
print(f"CUDA device: {torch.cuda.get_device_name(0)}; capability={arch}; supported_arches={supported}", flush=True)
if arch not in supported:
    raise SystemExit(f"PyTorch build does not support {arch}; choose a CUDA-compatible GPU (A100/H100/V100)")
PY

echo "Starting JEPA on ${HOSTNAME}; strategy=${STRATEGY}; data=${DATA}; output=${OUTPUT}; epochs=${EPOCHS}; patience=${PATIENCE}; lr=${LR}" >&2
srun python3 -u training/jepa_wm.py jepa \
  --data "${DATA}" \
  --output "${OUTPUT}" \
  --device cuda \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --strategy "${STRATEGY}" \
  --mask-ratio "${MASK_RATIO}" \
  "${CURRICULUM_ARGS[@]}" \
  --antenna-ratios ${ANTENNA_RATIOS} \
  --lr "${LR}" \
  --ema-start "${EMA_START}" \
  --ema-end "${EMA_END}" \
  --final-lr-ratio "${FINAL_LR_RATIO}" \
  --grad-clip "${GRAD_CLIP}" \
  --val-fraction 0.10 \
  --val-repeats "${VAL_REPEATS}" \
  --warmup-epochs "${WARMUP_EPOCHS}" \
  --patience "${PATIENCE}" \
  --min-delta "${MIN_DELTA}" \
  --predictor-variant original \
  --feature-loss normalized \
  --min-feature-std 1e-4 \
  --normalize

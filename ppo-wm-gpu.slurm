#!/bin/bash
# Full real-environment PPO controller for the frozen JEPA+MDN stack.
# JEPA, the side MLP, and MDN-LSTM are frozen; actor/critic update only.
#SBATCH --job-name=ppo-wm
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

REPRESENTATION="${JEPA_WM_CHECKPOINT:-${PWD}/artifacts/jepa_wm_v1/checkpoints/jepa_multiblock_gpu.pt}"
DYNAMICS="${JEPA_WM_MDN_CHECKPOINT:-${PWD}/artifacts/jepa_wm_v1/checkpoints/mdn_jepa_a100.pt}"
OUTPUT="${JEPA_WM_PPO_OUTPUT:-${PWD}/artifacts/jepa_wm_v1/checkpoints/ppo_jepa_5000_v2.pt}"
EPISODES="${JEPA_WM_PPO_EPISODES:-500}"
EPISODE_LENGTH="${JEPA_WM_PPO_EPISODE_LENGTH:-5000}"
ROLLOUT_STEPS="${JEPA_WM_PPO_ROLLOUT_STEPS:-512}"
MINIBATCH_SIZE="${JEPA_WM_PPO_MINIBATCH_SIZE:-256}"
PPO_EPOCHS="${JEPA_WM_PPO_EPOCHS:-4}"
PPO_LR="${JEPA_WM_PPO_LR:-1e-4}"
CRITIC_LR="${JEPA_WM_PPO_CRITIC_LR:-5e-4}"
ENTROPY_START="${JEPA_WM_PPO_ENTROPY_START:-0.01}"
ENTROPY_END="${JEPA_WM_PPO_ENTROPY_END:-0.0}"
ENTROPY_ANNEAL="${JEPA_WM_PPO_ENTROPY_ANNEAL:-0.30}"
TARGET_KL="${JEPA_WM_PPO_TARGET_KL:-0.015}"
VALUE_CLIP="${JEPA_WM_PPO_VALUE_CLIP:-0.2}"
EVAL_INTERVAL="${JEPA_WM_PPO_EVAL_INTERVAL:-25}"
EVAL_EPISODES="${JEPA_WM_PPO_EVAL_EPISODES:-3}"
INITIAL_LOG_STD="${JEPA_WM_PPO_INITIAL_LOG_STD:-0.7}"
MAX_LOG_STD="${JEPA_WM_PPO_MAX_LOG_STD:-0.7}"
mkdir -p "$(dirname "${OUTPUT}")"

# Fail before a long rollout if this node is unsupported by PyTorch.
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

echo "Starting PPO on ${HOSTNAME}; representation=${REPRESENTATION}; dynamics=${DYNAMICS}; output=${OUTPUT}; episodes=${EPISODES}; horizon=${EPISODE_LENGTH}; rollout=${ROLLOUT_STEPS}" >&2
srun python3 -u training/jepa_wm.py ppo \
  --representation jepa \
  --representation-checkpoint "${REPRESENTATION}" \
  --dynamics-checkpoint "${DYNAMICS}" \
  --output "${OUTPUT}" \
  --device cuda \
  --episodes "${EPISODES}" \
  --episode-length "${EPISODE_LENGTH}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --minibatch-size "${MINIBATCH_SIZE}" \
  --ppo-epochs "${PPO_EPOCHS}" \
  --lr "${PPO_LR}" \
  --critic-lr "${CRITIC_LR}" \
  --entropy-coef-start "${ENTROPY_START}" \
  --entropy-coef-end "${ENTROPY_END}" \
  --entropy-anneal-fraction "${ENTROPY_ANNEAL}" \
  --target-kl "${TARGET_KL}" \
  --value-clip "${VALUE_CLIP}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --eval-episodes "${EVAL_EPISODES}" \
  --initial-log-std "${INITIAL_LOG_STD}" \
  --max-log-std "${MAX_LOG_STD}" \
  --memory

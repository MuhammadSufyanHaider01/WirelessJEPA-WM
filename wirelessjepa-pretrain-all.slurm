#!/bin/bash
# Submit all four paper masking strategies as independent, concurrent runs.
# Optional Hydra overrides may follow the script name, for example:
#   sbatch wirelessjepa-pretrain-all.slurm trainer.max_epochs=100

# The current ARC snapshot has an idle 4xA100 node in bigmem. Override this
# partition at submission time when a different GPU queue is preferred.
#SBATCH --job-name=wirelessjepa-all
#SBATCH --partition=bigmem
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/slurm-%x-%A_%a.out

set -eo pipefail

source "${HOME}/software/init-conda"
conda activate wireless-jepa
set -u

cd "${SLURM_SUBMIT_DIR:?Submit this job from the WirelessJEPA repository root}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

configs=(
    ijepacnn_iqfm_random
    ijepacnn_iqfm_antenna
    ijepacnn_iqfm_time
    ijepacnn_iqfm_multiblock
)

task_id="${SLURM_ARRAY_TASK_ID:?Slurm array task id is required}"
if (( task_id < 0 || task_id >= ${#configs[@]} )); then
    echo "Unsupported array task id: ${task_id}" >&2
    exit 2
fi
config_name="${configs[$task_id]}"
echo "Starting ${config_name} on ${HOSTNAME}" >&2

srun python3 -u pretrain/train_ijepacnn.py --config-name "${config_name}" "$@"

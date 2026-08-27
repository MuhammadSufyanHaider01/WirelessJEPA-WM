#!/bin/bash
# Submit one paper mask at a time, for example:
#   sbatch cnn-jepa.slurm ijepacnn_iqfm_random
# Hydra overrides may follow the config name, for example: wandb=true

#SBATCH --job-name=wirelessjepa
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/slurm-%x-%j.out

set -eo pipefail

source "${HOME}/software/init-conda"
conda activate wireless-jepa
set -u

cd "${SLURM_SUBMIT_DIR:?Submit this job from the WirelessJEPA repository root}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

config_name="${1:-ijepacnn_iqfm_random}"
if [ "$#" -gt 0 ]; then
    shift
fi

case "${config_name}" in
    ijepacnn_iqfm_random|ijepacnn_iqfm_antenna|ijepacnn_iqfm_time|ijepacnn_iqfm_multiblock)
        ;;
    *)
        echo "Unsupported config: ${config_name}" >&2
        exit 2
        ;;
esac

srun python3 -u pretrain/train_ijepacnn.py --config-name "${config_name}" "$@"

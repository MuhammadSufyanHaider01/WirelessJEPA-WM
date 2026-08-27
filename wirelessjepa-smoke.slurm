#!/bin/bash

#SBATCH --job-name=wirelessjepa-smoke
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/slurm-%x-%j.out

set -eo pipefail

source "${HOME}/software/init-conda"
conda activate wireless-jepa
set -u

cd "${SLURM_SUBMIT_DIR:?Submit this job from the WirelessJEPA repository root}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

srun python3 -u pretrain/smoke_iqfm_pretraining.py \
    --device cuda \
    --batch-size 1

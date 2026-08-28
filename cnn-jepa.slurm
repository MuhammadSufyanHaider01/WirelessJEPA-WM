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
#SBATCH --signal=B:USR1@600
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

hydra_overrides=("$@")
continuation_submitted=0

schedule_continuation() {
    if [ "${continuation_submitted}" -eq 1 ]; then
        return 0
    fi
    continuation_submitted=1

    local launcher="${WIRELESSJEPA_LAUNCHER:-${SLURM_SUBMIT_DIR}/cnn-jepa.slurm}"
    local sbatch_args=(--dependency="afterany:${SLURM_JOB_ID}")
    if [ -n "${SLURM_JOB_PARTITION:-}" ]; then
        sbatch_args+=(--partition="${SLURM_JOB_PARTITION}")
    fi
    if [ -n "${SLURM_JOB_QOS:-}" ]; then
        sbatch_args+=(--qos="${SLURM_JOB_QOS}")
    fi
    if [ -n "${SLURM_JOB_ACCOUNT:-}" ]; then
        sbatch_args+=(--account="${SLURM_JOB_ACCOUNT}")
    fi

    local next_job
    if next_job="$(sbatch --parsable "${sbatch_args[@]}" \
        --export="ALL,WIRELESSJEPA_AUTO_RESUME=1" \
        "${launcher}" "${config_name}" "${hydra_overrides[@]}")"; then
        echo "Submitted automatic continuation ${next_job} for ${config_name} after ${SLURM_JOB_ID}" >&2
    else
        continuation_submitted=0
        echo "Failed to submit automatic continuation for ${config_name}" >&2
    fi
}

trap schedule_continuation USR1
trap 'schedule_continuation; exit 0' TERM

srun python3 -u pretrain/train_ijepacnn.py --config-name "${config_name}" "${hydra_overrides[@]}"

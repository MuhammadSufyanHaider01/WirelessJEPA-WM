#!/bin/bash
# Run the available downstream-task x masking-strategy matrix in one job.
# The evaluator is a frozen-encoder linear/k-NN benchmark, matching the
# paper's primary low-shot protocol. Set WIRELESSJEPA_DOWNSTREAM_SAMPLES to
# change the per-class training shot count (default: 500).

#SBATCH --job-name=wirelessjepa-downstream
#SBATCH --partition=gpu-v100
#SBATCH --qos=gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=artifacts/slurm-%x-%j.out

set -euo pipefail

source "${HOME}/software/init-conda"
conda activate wireless-jepa

cd "${SLURM_SUBMIT_DIR:?Submit this job from the WirelessJEPA repository root}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

sample_count="${WIRELESSJEPA_DOWNSTREAM_SAMPLES:-500}"
results_root="${WIRELESSJEPA_DOWNSTREAM_RESULTS:-artifacts/downstream_linear}"
mkdir -p "${results_root}"

continuation_submitted=0
schedule_continuation() {
    if [ "${continuation_submitted}" -eq 1 ]; then
        return 0
    fi
    continuation_submitted=1

    local launcher="${WIRELESSJEPA_DOWNSTREAM_LAUNCHER:-${SLURM_SUBMIT_DIR}/wirelessjepa-downstream-all.slurm}"
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
        --export="ALL,WIRELESSJEPA_DOWNSTREAM_SAMPLES=${sample_count},WIRELESSJEPA_DOWNSTREAM_RESULTS=${results_root}" \
        "${launcher}")"; then
        echo "Submitted downstream continuation ${next_job} after ${SLURM_JOB_ID}" >&2
    else
        continuation_submitted=0
        echo "Failed to submit downstream continuation" >&2
    fi
}

trap schedule_continuation USR1
trap 'schedule_continuation; exit 0' TERM

declare -a checkpoints=(
    "random|artifacts/pretrain_lightly/ijepacnn_iqfm/WirelessJEPA_random_shufflenet_v2_x0_5_torchvision_predL3K3_lr0.005_wd0.005_bs256/version_0/last.ckpt"
    "antenna|artifacts/pretrain_lightly/ijepacnn_iqfm/WirelessJEPA_antenna_shufflenet_v2_x0_5_torchvision_predL3K3_lr0.01_wd0.01_bs256/version_0/last-v1.ckpt"
    "time|artifacts/pretrain_lightly/ijepacnn_iqfm/WirelessJEPA_time_shufflenet_v2_x0_5_torchvision_predL3K3_lr0.01_wd0.01_bs256/version_0/last.ckpt"
    "multi-block|artifacts/pretrain_lightly/ijepacnn_iqfm/WirelessJEPA_multi-block_shufflenet_v2_x0_5_torchvision_predL3K3_lr0.005_wd0.005_bs256/version_0/last-v1.ckpt"
)

# These are the downstream HDF5 files currently present in this checkout.
# The legacy evaluator also names deepbeam, jamming, tpi, inter, and craw,
# but their HDF5 files are not present and are intentionally not submitted.
declare -a tasks=(
    "aoa|data/iqfm-val-100.h5"
    "mod|data/iqfm-val-100.h5"
    "rf|datasets/RF_FingerPrinting.h5"
    "rml|datasets/rml2016_wifi_4ch_minSNR0.h5"
    "radar|datasets/RADAR_RadCom_prepared.h5"
)

failed=0
for checkpoint_spec in "${checkpoints[@]}"; do
    IFS='|' read -r mask_name checkpoint <<< "${checkpoint_spec}"
    if [ ! -f "${checkpoint}" ]; then
        echo "Missing checkpoint for ${mask_name}: ${checkpoint}" >&2
        failed=1
        continue
    fi

    for task_spec in "${tasks[@]}"; do
        IFS='|' read -r task_name data_path <<< "${task_spec}"
        if [ ! -f "${data_path}" ]; then
            echo "Skipping ${mask_name}/${task_name}; missing dataset: ${data_path}" >&2
            failed=1
            continue
        fi

        run_name="${mask_name}_${task_name}_samples${sample_count}"
        run_dir="${results_root}/${run_name}"
        done_marker="${run_dir}/complete"
        if [ -f "${done_marker}" ]; then
            echo "Skipping completed ${run_name}"
            continue
        fi
        mkdir -p "${run_dir}"
        echo "Running ${run_name}"
        if srun --exclusive python3 -u pretrain/eval_ijepacnn.py \
            --task "${task_name}" \
            --data-path "${data_path}" \
            --sample "${sample_count}" \
            --checkpoint "${checkpoint}" \
            > "${run_dir}/eval.log" 2>&1; then
            printf 'completed\n' > "${done_marker}"
        else
            echo "Downstream evaluation failed for ${run_name}; see ${run_dir}/eval.log" >&2
            failed=1
        fi
    done
done

if [ "${failed}" -ne 0 ]; then
    echo "One or more downstream combinations failed." >&2
    exit 1
fi
echo "All available downstream combinations completed."

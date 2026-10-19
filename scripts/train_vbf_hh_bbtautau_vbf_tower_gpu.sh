#!/bin/bash
# Joint HH+VBF 5-feature training where the VBF branch cannot change the shared encoder.
#
# MODE=frozen (default): load the HH-only checkpoint, freeze its weights (shared embedding,
#   shared encoder, HH branch) and train only the VBF branch. HH performance is exactly the
#   HH-only model. Requires STATE_DICT to point at the HH-only checkpoint.
# MODE=detach: train everything from scratch in one run. The VBF branch reads the shared
#   encoder output but its gradient is blocked, so only HH trains the shared layers.
#
# Example:
#   STATE_DICT=/local/norman/UHHAnalysis/SPANet/outputs/260917_spanet_hh_only_5features_unique/version_0/checkpoints/<best>.ckpt \
#       sbatch scripts/train_vbf_hh_bbtautau_vbf_tower_gpu.sh
#   MODE=detach sbatch scripts/train_vbf_hh_bbtautau_vbf_tower_gpu.sh
#
# EPOCHS and LIMIT_DATASET can be overridden for a smoke test.
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem-per-gpu 64G
#SBATCH --cpus-per-task 8
#SBATCH --time 72:00:00
#SBATCH -o /local/norman/UHHAnalysis/SPANet/slurm/slurm-%x-%j-%N.out

set -euo pipefail

cd /home/norman/SPANet
mkdir -p /local/norman/UHHAnalysis/SPANet/slurm /local/norman/UHHAnalysis/SPANet/outputs

MODE=${MODE:-frozen}
TRAINING_FILE=/local/norman/UHHAnalysis/SPANet/hdf5/22_23_v14_vbf_signal_train.h5
VALIDATION_FILE=/local/norman/UHHAnalysis/SPANet/hdf5/22_23_v14_vbf_signal_val.h5
EVENT_FILE=event_files/vbf_hh_bbtautau_5features.yaml
OUTPUT_DIR=/local/norman/UHHAnalysis/SPANet/outputs
STATE_DICT=${STATE_DICT:-}
EPOCHS=${EPOCHS:-}
LIMIT_DATASET=${LIMIT_DATASET:-}
BATCH_SIZE=${BATCH_SIZE:-512}
DIAGNOSTICS_BATCHES=${DIAGNOSTICS_BATCHES:-20}
VALIDATE_EVERY=${VALIDATE_EVERY:-1}
LOG_EVERY=${LOG_EVERY:-50}

EXTRA_ARGS=()
case "${MODE}" in
    frozen)
        OPTIONS_FILE=options_files/vbf_hh_bbtautau/vbf_tower_frozen.json
        RUN_NAME=${RUN_NAME:-$(date +%y%m%d)_spanet_vbf_tower_frozen_5features}
        if [[ -z "${STATE_DICT}" || ! -f "${STATE_DICT}" ]]; then
            echo "MODE=frozen needs STATE_DICT pointing at the HH-only checkpoint, got '${STATE_DICT}'" >&2
            exit 1
        fi
        EXTRA_ARGS+=("-sf" "${STATE_DICT}" "-fsf")
        ;;
    detach)
        OPTIONS_FILE=options_files/vbf_hh_bbtautau/vbf_tower_detach.json
        RUN_NAME=${RUN_NAME:-$(date +%y%m%d)_spanet_vbf_tower_detach_5features}
        ;;
    *)
        echo "Unknown MODE '${MODE}', use 'frozen' or 'detach'" >&2
        exit 1
        ;;
esac

if [[ -n "${EPOCHS}" ]]; then
    EXTRA_ARGS+=("-e" "${EPOCHS}")
fi
if [[ -n "${LIMIT_DATASET}" ]]; then
    EXTRA_ARGS+=("-p" "${LIMIT_DATASET}")
fi

echo "Host: $(hostname)"
echo "Mode: ${MODE}"
echo "Training file: ${TRAINING_FILE}"
echo "Validation file: ${VALIDATION_FILE}"
echo "Options file: ${OPTIONS_FILE}"
echo "Event file: ${EVENT_FILE}"
echo "State dict: ${STATE_DICT:-none}"
echo "Run name: ${RUN_NAME}"
echo "Epochs override: ${EPOCHS:-from options file}"
echo "Dataset limit: ${LIMIT_DATASET:-full} percent"
echo "Batch size: ${BATCH_SIZE}"
nvidia-smi -L

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

./run.sh python -m spanet.train \
    -ef "${EVENT_FILE}" \
    -tf "${TRAINING_FILE}" \
    -vf "${VALIDATION_FILE}" \
    -of "${OPTIONS_FILE}" \
    -l "${OUTPUT_DIR}" \
    -n "${RUN_NAME}" \
    -b "${BATCH_SIZE}" \
    -g 1 \
    --diagnostics-batches "${DIAGNOSTICS_BATCHES}" \
    --validate-every-n-epochs "${VALIDATE_EVERY}" \
    --log-every-n-steps "${LOG_EVERY}" \
    "${EXTRA_ARGS[@]}"

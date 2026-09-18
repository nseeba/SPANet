#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem-per-gpu 64G
#SBATCH --cpus-per-task 8
#SBATCH --time 48:00:00
#SBATCH -o /local/norman/UHHAnalysis/SPANet/slurm/slurm-%x-%j-%N.out

set -euo pipefail

SPANET_DIR=${SPANET_DIR:-/home/norman/SPANet}
TRAINING_FILE=${TRAINING_FILE:-/local/norman/UHHAnalysis/SPANet/hdf5/22pre_v14_vbf_signal_all.h5}
VALIDATION_FILE=${VALIDATION_FILE:-}
OPTIONS_FILE=${OPTIONS_FILE:-options_files/vbf_hh_bbtautau/train.json}
EVENT_FILE=${EVENT_FILE:-event_files/vbf_hh_bbtautau.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-/local/norman/UHHAnalysis/SPANet/outputs}
RUN_NAME=${RUN_NAME:-vbf_hh_bbtautau_train}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_GPU=${NUM_GPU:-1}
DIAGNOSTICS_BATCHES=${DIAGNOSTICS_BATCHES:-10}
FP16=${FP16:-0}
VALIDATE_EVERY_N_EPOCHS=${VALIDATE_EVERY_N_EPOCHS:-1}
LOG_EVERY_N_STEPS=${LOG_EVERY_N_STEPS:-50}
DEVICE_STATS=${DEVICE_STATS:-0}
CHECKPOINT=${CHECKPOINT:-}

mkdir -p /local/norman/UHHAnalysis/SPANet/slurm "${OUTPUT_DIR}"

cd "${SPANET_DIR}"

echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Training file: ${TRAINING_FILE}"
echo "Validation file: ${VALIDATION_FILE:-internal split}"
echo "Options file: ${OPTIONS_FILE}"
echo "Event file: ${EVENT_FILE}"
echo "Run name: ${RUN_NAME}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "GPUs requested by SPANet: ${NUM_GPU}"
echo "Diagnostics batches: ${DIAGNOSTICS_BATCHES}"
echo "Mixed precision: ${FP16}"
echo "Validate every N epochs: ${VALIDATE_EVERY_N_EPOCHS}"
echo "Log every N steps: ${LOG_EVERY_N_STEPS}"
echo "Device stats: ${DEVICE_STATS}"
echo "Checkpoint: ${CHECKPOINT:-none}"

env | grep CUDA || true
nvidia-smi -L
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRA_ARGS=()
if [[ "${FP16}" == "1" ]]; then
    EXTRA_ARGS+=("-fp16")
fi
if [[ "${DEVICE_STATS}" == "1" ]]; then
    EXTRA_ARGS+=("--device-stats")
fi
if [[ -n "${VALIDATION_FILE}" ]]; then
    EXTRA_ARGS+=("-vf" "${VALIDATION_FILE}")
fi
if [[ -n "${CHECKPOINT}" ]]; then
    EXTRA_ARGS+=("-cf" "${CHECKPOINT}")
fi

./run.sh python -m spanet.train \
    -ef "${EVENT_FILE}" \
    -tf "${TRAINING_FILE}" \
    -of "${OPTIONS_FILE}" \
    -l "${OUTPUT_DIR}" \
    -n "${RUN_NAME}" \
    -e "${EPOCHS}" \
    -b "${BATCH_SIZE}" \
    -g "${NUM_GPU}" \
    --diagnostics-batches "${DIAGNOSTICS_BATCHES}" \
    --validate-every-n-epochs "${VALIDATE_EVERY_N_EPOCHS}" \
    --log-every-n-steps "${LOG_EVERY_N_STEPS}" \
    "${EXTRA_ARGS[@]}"

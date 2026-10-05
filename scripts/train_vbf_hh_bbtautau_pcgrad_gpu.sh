#!/bin/bash
# Joint HH+VBF 5-feature training with opt-in PCGrad.
# EPOCHS and LIMIT_DATASET can be overridden for a smoke test.
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem-per-gpu 64G
#SBATCH --cpus-per-task 8
#SBATCH --time 96:00:00
#SBATCH -o /local/norman/UHHAnalysis/SPANet/slurm/slurm-%x-%j-%N.out

set -euo pipefail

cd /home/norman/SPANet
mkdir -p /local/norman/UHHAnalysis/SPANet/slurm /local/norman/UHHAnalysis/SPANet/outputs

TRAINING_FILE=/local/norman/UHHAnalysis/SPANet/hdf5/22_23_v14_vbf_signal_train.h5
VALIDATION_FILE=/local/norman/UHHAnalysis/SPANet/hdf5/22_23_v14_vbf_signal_val.h5
OPTIONS_FILE=options_files/vbf_hh_bbtautau/pcgrad.json
EVENT_FILE=event_files/vbf_hh_bbtautau_5features.yaml
OUTPUT_DIR=/local/norman/UHHAnalysis/SPANet/outputs
RUN_NAME=${RUN_NAME:-260918_spanet_pcgrad_5features}
EPOCHS=${EPOCHS:-50}
LIMIT_DATASET=${LIMIT_DATASET:-}
CHECKPOINT=${CHECKPOINT:-}
BATCH_SIZE=${BATCH_SIZE:-512}
DIAGNOSTICS_BATCHES=${DIAGNOSTICS_BATCHES:-20}
VALIDATE_EVERY=${VALIDATE_EVERY:-1}
LOG_EVERY=${LOG_EVERY:-50}
FP16=${FP16:-0}

echo "Host: $(hostname)"
echo "Training file: ${TRAINING_FILE}"
echo "Validation file: ${VALIDATION_FILE}"
echo "Options file: ${OPTIONS_FILE}"
echo "Event file: ${EVENT_FILE}"
echo "Run name: ${RUN_NAME}"
echo "Epochs: ${EPOCHS}"
echo "Dataset limit: ${LIMIT_DATASET:-full} percent"
echo "Checkpoint: ${CHECKPOINT:-none}"
echo "Batch size: ${BATCH_SIZE}"
echo "Diagnostics batches: ${DIAGNOSTICS_BATCHES}"
echo "Validate every N epochs: ${VALIDATE_EVERY}"
echo "FP16: ${FP16}"
nvidia-smi -L

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRA_ARGS=()
if [[ -n "${LIMIT_DATASET}" ]]; then
    EXTRA_ARGS+=("-p" "${LIMIT_DATASET}")
fi
if [[ -n "${CHECKPOINT}" ]]; then
    EXTRA_ARGS+=("-cf" "${CHECKPOINT}")
fi
if [[ "${FP16}" == "1" ]]; then
    EXTRA_ARGS+=("-fp16")
fi

./run.sh python -m spanet.train \
    -ef "${EVENT_FILE}" \
    -tf "${TRAINING_FILE}" \
    -vf "${VALIDATION_FILE}" \
    -of "${OPTIONS_FILE}" \
    -l "${OUTPUT_DIR}" \
    -n "${RUN_NAME}" \
    -e "${EPOCHS}" \
    -b "${BATCH_SIZE}" \
    -g 1 \
    --diagnostics-batches "${DIAGNOSTICS_BATCHES}" \
    --validate-every-n-epochs "${VALIDATE_EVERY}" \
    --log-every-n-steps "${LOG_EVERY}" \
    "${EXTRA_ARGS[@]}"

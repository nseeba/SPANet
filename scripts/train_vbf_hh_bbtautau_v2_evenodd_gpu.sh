#!/bin/bash
# v2 training: VBF + ggF HH, original 5 jet inputs, one model per event-number parity.
# The even model trains on even event numbers and is tested on odd ones, and vice versa.
#
#   PARITY=even sbatch scripts/train_vbf_hh_bbtautau_v2_evenodd_gpu.sh
#   PARITY=odd  sbatch scripts/train_vbf_hh_bbtautau_v2_evenodd_gpu.sh
#
# EPOCHS and LIMIT_DATASET can be overridden for a smoke test, SEED for repeated trainings,
# TIME_LIMIT (DD:HH:MM:SS) stops training cleanly after that time, e.g. for learning-rate tests.
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem-per-gpu 64G
#SBATCH --cpus-per-task 8
#SBATCH --time 96:00:00
#SBATCH -o /local/norman/UHHAnalysis/SPANet/slurm/slurm-%x-%j-%N.out

set -euo pipefail

PARITY=${PARITY:?Set PARITY to even or odd}
if [[ "${PARITY}" != "even" && "${PARITY}" != "odd" ]]; then
    echo "PARITY must be even or odd, got ${PARITY}" >&2
    exit 1
fi

DATA_DIR=${DATA_DIR:-/local/norman/UHHAnalysis/SPANet/hdf5/v2_evenodd}
PREFIX=${PREFIX:-vbf_ggf_}
TRAINING_FILE=${DATA_DIR}/${PREFIX}${PARITY}_train.h5
VALIDATION_FILE=${DATA_DIR}/${PREFIX}${PARITY}_val.h5
TEST_FILE=${DATA_DIR}/${PREFIX}${PARITY}_test.h5
OPTIONS_FILE=${OPTIONS_FILE:-options_files/vbf_hh_bbtautau/v2_evenodd.json}
EVENT_FILE=${EVENT_FILE:-event_files/vbf_hh_bbtautau_5features.yaml}
OUTPUT_DIR=/local/norman/UHHAnalysis/SPANet/outputs
RUN_NAME=${RUN_NAME:-$(date +%y%m%d)_spanet_v2_vbf_ggf_5features_${PARITY}}
EPOCHS=${EPOCHS:-}
LIMIT_DATASET=${LIMIT_DATASET:-}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-512}
TIME_LIMIT=${TIME_LIMIT:-}

for f in "${TRAINING_FILE}" "${VALIDATION_FILE}"; do
    if [[ ! -f "${f}" ]]; then
        echo "Missing ${f}; run scripts/build_vbf_hh_bbtautau_v2_inputs.sh first" >&2
        exit 1
    fi
done

mkdir -p /local/norman/UHHAnalysis/SPANet/slurm "${OUTPUT_DIR}"
cd /home/norman/SPANet

EXTRA_ARGS=()
if [[ -n "${EPOCHS}" ]]; then
    EXTRA_ARGS+=("-e" "${EPOCHS}")
fi
if [[ -n "${LIMIT_DATASET}" ]]; then
    EXTRA_ARGS+=("-p" "${LIMIT_DATASET}")
fi
if [[ "${SEED}" != "0" ]]; then
    EXTRA_ARGS+=("-r" "${SEED}")
fi
if [[ -n "${TIME_LIMIT}" ]]; then
    EXTRA_ARGS+=("-t" "${TIME_LIMIT}")
fi

echo "Host: $(hostname)"
echo "Parity: ${PARITY}"
echo "Training file: ${TRAINING_FILE}"
echo "Validation file: ${VALIDATION_FILE}"
echo "Test file (other parity, for evaluation): ${TEST_FILE}"
echo "Options file: ${OPTIONS_FILE}"
echo "Event file: ${EVENT_FILE}"
echo "Run name: ${RUN_NAME}"
echo "Epochs override: ${EPOCHS:-from options file}"
echo "Dataset limit: ${LIMIT_DATASET:-full} percent"
echo "Seed: ${SEED}"
echo "Time limit: ${TIME_LIMIT:-none}"
echo "Git commit: $(git rev-parse --short HEAD) ($(git status --short | wc -l) uncommitted changes)"
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
    --diagnostics-batches 20 \
    --validate-every-n-epochs 1 \
    --log-every-n-steps 50 \
    "${EXTRA_ARGS[@]}"

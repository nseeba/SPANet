#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem-per-gpu 32G
#SBATCH --cpus-per-task 4
#SBATCH --time 04:00:00
#SBATCH -o /local/norman/UHHAnalysis/SPANet/slurm/slurm-%x-%j-%N.out

set -euo pipefail

SPANET_DIR=${SPANET_DIR:-/home/norman/SPANet}
RUN_DIR=${RUN_DIR:?Set RUN_DIR to the SPANet output version directory}
OUTPUT_FILE=${OUTPUT_FILE:?Set OUTPUT_FILE to the prediction HDF5 path}
BATCH_SIZE=${BATCH_SIZE:-512}
SPLIT=${SPLIT:-validation}
TEST_FILE=${TEST_FILE:-}
CHECKPOINT=${CHECKPOINT:-}
EVENT_FILE=${EVENT_FILE:-}
MAX_BATCHES=${MAX_BATCHES:-}
FP16=${FP16:-1}

mkdir -p /local/norman/UHHAnalysis/SPANet/slurm "$(dirname "${OUTPUT_FILE}")"

cd "${SPANET_DIR}"

echo "Host: $(hostname)"
echo "Run directory: ${RUN_DIR}"
echo "Output file: ${OUTPUT_FILE}"
echo "Split: ${SPLIT}"
echo "Test file: ${TEST_FILE:-from run options}"
echo "Checkpoint: ${CHECKPOINT:-best from run directory}"
echo "Event file: ${EVENT_FILE:-from run options}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max batches: ${MAX_BATCHES:-all}"
echo "FP16: ${FP16}"

ARGS=(
    utils/evaluate_hh_bbtautau_model.py
    "${RUN_DIR}"
    -o "${OUTPUT_FILE}"
    --split "${SPLIT}"
    --batch-size "${BATCH_SIZE}"
    --gpu
)

if [[ "${FP16}" == "1" ]]; then
    ARGS+=(--fp16)
fi

if [[ -n "${TEST_FILE}" ]]; then
    ARGS+=(--test-file "${TEST_FILE}")
fi

if [[ -n "${CHECKPOINT}" ]]; then
    ARGS+=(--checkpoint "${CHECKPOINT}")
fi

if [[ -n "${EVENT_FILE}" ]]; then
    ARGS+=(--event-file "${EVENT_FILE}")
fi

if [[ -n "${MAX_BATCHES}" ]]; then
    ARGS+=(--max-batches "${MAX_BATCHES}")
fi

./run.sh python "${ARGS[@]}"

#!/bin/bash
# HH-only 5-feature ablation. Keep this launcher separate from the joint HH+VBF run.
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem-per-gpu 64G
#SBATCH --cpus-per-task 8
#SBATCH --time 48:00:00
#SBATCH -o /local/norman/UHHAnalysis/SPANet/slurm/slurm-%x-%j-%N.out

set -euo pipefail

cd /home/norman/SPANet
mkdir -p /local/norman/UHHAnalysis/SPANet/slurm /local/norman/UHHAnalysis/SPANet/outputs

TRAINING_FILE=/local/norman/UHHAnalysis/SPANet/hdf5/22_23_v14_vbf_signal_train.h5
VALIDATION_FILE=/local/norman/UHHAnalysis/SPANet/hdf5/22_23_v14_vbf_signal_val.h5
OPTIONS_FILE=options_files/vbf_hh_bbtautau/hh_only_5features.json
EVENT_FILE=event_files/vbf_hh_bbtautau_hhb_only_5features.yaml
OUTPUT_DIR=/local/norman/UHHAnalysis/SPANet/outputs
RUN_NAME=260917_spanet_hh_only_5features_unique

echo "Host: $(hostname)"
echo "Training file: ${TRAINING_FILE}"
echo "Validation file: ${VALIDATION_FILE}"
echo "Options file: ${OPTIONS_FILE}"
echo "Event file: ${EVENT_FILE}"
echo "Run name: ${RUN_NAME}"
nvidia-smi -L

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

./run.sh python -m spanet.train \
    -ef "${EVENT_FILE}" \
    -tf "${TRAINING_FILE}" \
    -vf "${VALIDATION_FILE}" \
    -of "${OPTIONS_FILE}" \
    -l "${OUTPUT_DIR}" \
    -n "${RUN_NAME}" \
    -e 50 \
    -b 512 \
    -g 1 \
    --diagnostics-batches 20 \
    --validate-every-n-epochs 5 \
    --log-every-n-steps 50

#!/bin/bash
# Build the even/odd training files from the per-sample HDF5 files (VBF + ggF).
# Streams one input file at a time, so it needs little memory.
#
#   sbatch scripts/build_vbf_hh_bbtautau_v2_inputs.sh
#   KEEP_FRACTION="ggf=0.5" sbatch scripts/build_vbf_hh_bbtautau_v2_inputs.sh   # subsample ggF if it dominates
#SBATCH -p main
#SBATCH --cpus-per-task 2
#SBATCH --mem 24G
#SBATCH --time 24:00:00
#SBATCH -o /local/norman/UHHAnalysis/SPANet/slurm/slurm-%x-%j.out

set -euo pipefail

PARTS_DIR=${PARTS_DIR:-/local/norman/UHHAnalysis/SPANet/hdf5/v2_evenodd/parts}
OUTPUT_DIR=${OUTPUT_DIR:-/local/norman/UHHAnalysis/SPANet/hdf5/v2_evenodd}
PREFIX=${PREFIX:-vbf_ggf_}
VAL_FRACTION=${VAL_FRACTION:-0.1}
KEEP_FRACTION=${KEEP_FRACTION:-}

echo "Host: $(hostname)"
echo "Parts: ${PARTS_DIR}"
echo "Output: ${OUTPUT_DIR}/${PREFIX}{even,odd}_{train,val,test}.h5"
echo "Validation fraction: ${VAL_FRACTION}"
echo "Keep fraction: ${KEEP_FRACTION:-all events}"
find "${PARTS_DIR}" -name "*.h5" | sort

EXTRA_ARGS=()
if [[ -n "${KEEP_FRACTION}" ]]; then
    EXTRA_ARGS+=(--keep-fraction ${KEEP_FRACTION})
fi

cd /home/norman/SPANet
./run.sh python utils/build_evenodd_hdf5.py "${PARTS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --prefix "${PREFIX}" \
    --val-fraction "${VAL_FRACTION}" \
    "${EXTRA_ARGS[@]}"

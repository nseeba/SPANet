#!/bin/bash
# Convert ColumnFlow reduced-event parquets to SPANet HDF5, one job per (era, dataset).
# Uses the current converter: event/run/luminosityBlock stored in METADATA (for the even/odd split),
# VBF truth only for VBF samples, tagger placeholders mapped to -1, all files padded to MAX_JETS.
#
# Usage (one array job per era x dataset):
#   PROCESS_TYPE=vbf sbatch --array=0-39 scripts/convert_vbf_hh_bbtautau_v2.sh
#   PROCESS_TYPE=ggf DATASETS="hh_ggf_hbb_htt_kl1_kt1_powheg ..." sbatch --array=0-15 scripts/convert_vbf_hh_bbtautau_v2.sh
#
# Expected input layout (as for the existing prod24 VBF samples):
#   ${INPUT_BASE}/<era>/<dataset>/**/events_*.parquet
#SBATCH -p main
#SBATCH --cpus-per-task 2
#SBATCH --mem 48G
#SBATCH --time 12:00:00
#SBATCH -o /local/norman/UHHAnalysis/SPANet/slurm/slurm-%x-%A_%a.out

set -euo pipefail

PROCESS_TYPE=${PROCESS_TYPE:?Set PROCESS_TYPE to vbf, ggf or background}
INPUT_BASE=${INPUT_BASE:-/local/norman/UHHAnalysis/SPANet}
OUTPUT_BASE=${OUTPUT_BASE:-/local/norman/UHHAnalysis/SPANet/hdf5/v2_evenodd/parts}
MAX_JETS=${MAX_JETS:-16}
ERAS=(${ERAS:-22pre_v14 22post_v14 23pre_v14 23post_v14})

if [[ -n "${DATASETS:-}" ]]; then
    DATASET_LIST=(${DATASETS})
elif [[ "${PROCESS_TYPE}" == "vbf" ]]; then
    DATASET_LIST=(
        hh_vbf_hbb_htt_kv1_k2v1_kl1_madgraph
        hh_vbf_hbb_htt_kv1_k2v0_kl1_madgraph
        hh_vbf_hbb_htt_kv1p74_k2v1p37_kl14p4_madgraph
        hh_vbf_hbb_htt_kv2p12_k2v3p87_klm5p96_madgraph
        hh_vbf_hbb_htt_kvm0p012_k2v0p03_kl10p2_madgraph
        hh_vbf_hbb_htt_kvm0p758_k2v1p44_klm19p3_madgraph
        hh_vbf_hbb_htt_kvm0p962_k2v0p959_klm1p43_madgraph
        hh_vbf_hbb_htt_kvm1p21_k2v1p94_klm0p94_madgraph
        hh_vbf_hbb_htt_kvm1p6_k2v2p72_klm1p36_madgraph
        hh_vbf_hbb_htt_kvm1p83_k2v3p57_klm3p39_madgraph
    )
else
    echo "Set DATASETS for PROCESS_TYPE=${PROCESS_TYPE}" >&2
    exit 1
fi

TASK=${SLURM_ARRAY_TASK_ID:?Submit as an array job, e.g. --array=0-39}
N_DATASETS=${#DATASET_LIST[@]}
N_TASKS=$(( ${#ERAS[@]} * N_DATASETS ))
if (( TASK >= N_TASKS )); then
    echo "Task ${TASK} out of range (${N_TASKS} era x dataset combinations)" >&2
    exit 1
fi
ERA=${ERAS[$(( TASK / N_DATASETS ))]}
DATASET=${DATASET_LIST[$(( TASK % N_DATASETS ))]}
INPUT_DIR=${INPUT_BASE}/${ERA}/${DATASET}
OUTPUT_FILE=${OUTPUT_BASE}/${ERA}/${DATASET}.h5

echo "Host: $(hostname)"
echo "Task: ${TASK}/${N_TASKS}"
echo "Era: ${ERA}"
echo "Dataset: ${DATASET}"
echo "Process type: ${PROCESS_TYPE}"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_FILE}"
echo "Max jets: ${MAX_JETS}"

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "Input directory ${INPUT_DIR} does not exist" >&2
    exit 1
fi
mkdir -p "$(dirname "${OUTPUT_FILE}")"

cd /home/norman/SPANet
./run.sh python utils/convert_hh_bbtautau_parquet.py "${INPUT_DIR}" \
    -o "${OUTPUT_FILE}" \
    --process-type "${PROCESS_TYPE}" \
    --max-jets "${MAX_JETS}" \
    --extra-features hhbtag vbfjtag \
    --compression lzf

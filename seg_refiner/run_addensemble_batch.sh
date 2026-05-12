#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

SEG="seg_refiner/checkpoints/best.pt"
RIPSEG="seg_refiner/checkpoints/ripdetseg_binary_best.pt"

PB_VID="data/DJI_0011.MOV"
PB_MASK="script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png"
PB_GT="data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png"

HV_VID="data/DJI_0009.MOV"
HV_MASK="script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png"
HV_GT="script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png"

run() {
    local SCENE="$1" VID="$2" MASK="$3" GT="$4" ALGO="$5"
    local OUT="seg_refiner/results/addensemble_${SCENE}/${ALGO}.csv"
    mkdir -p "$(dirname "$OUT")"
    echo "=== ${SCENE} ${ALGO} addensemble ==="
    python seg_refiner/run_pipeline_addensemble.py \
        --video "$VID" --mask "$MASK" --gt "$GT" --algo "$ALGO" \
        --seg-ckpt "$SEG" --ripseg-ckpt "$RIPSEG" \
        --max-frames 600 --skip 2 --start-time 0 \
        --out "$OUT" 2>&1
}

# Most promising algos for each scene
run palmbeach "$PB_VID" "$PB_MASK" "$PB_GT" LK
run haulover  "$HV_VID" "$HV_MASK" "$HV_GT" LK
run haulover  "$HV_VID" "$HV_MASK" "$HV_GT" HOR_LK
run palmbeach "$PB_VID" "$PB_MASK" "$PB_GT" HOR_HS
echo ADDENSEMBLE_DONE

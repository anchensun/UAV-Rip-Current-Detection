#!/usr/bin/env bash
# Stage E inference: run 4 algos × {Palm Beach, Haulover} with the
# RipDetSeg-trained binary U-Net as the per-frame rip prior.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

CKPT="seg_refiner/checkpoints/ripdetseg_binary_best.pt"

PB_VID="data/DJI_0011.MOV"
PB_MASK="script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png"
PB_GT="data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png"

HV_VID="data/DJI_0009.MOV"
HV_MASK="script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png"
HV_GT="script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png"

run_clip() {
    local SCENE="$1" VID="$2" MASK="$3" GT="$4"
    local OUT="seg_refiner/results/stage_e_${SCENE}"
    mkdir -p "$OUT"
    for ALGO in HOR_HS HOR_LK HS LK; do
        echo "=== ${SCENE} ${ALGO} ==="
        python seg_refiner/run_pipeline_with_ripseg.py \
            --video "$VID" --mask "$MASK" --gt "$GT" \
            --algo "$ALGO" --max-frames 600 --skip 2 --start-time 0 \
            --ripseg-ckpt "$CKPT" \
            --out "${OUT}/${ALGO}.csv" 2>&1
    done
}
run_clip palmbeach "$PB_VID" "$PB_MASK" "$PB_GT"
run_clip haulover  "$HV_VID" "$HV_MASK" "$HV_GT"
echo STAGE_E_INFER_DONE

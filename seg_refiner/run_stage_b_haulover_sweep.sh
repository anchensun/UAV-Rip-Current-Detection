#!/usr/bin/env bash
# Stage B: Haulover 4-algo × 3-gate seg-gate sweep.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

VID="data/DJI_0009.MOV"
MASK="script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png"
GT="script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png"

for ALGO in HOR_HS HOR_LK HS LK; do
    for GATE in sea sea_x_nonfoam argmax; do
        OUT="seg_refiner/results/stage_b_haulover/${ALGO}_${GATE}.csv"
        mkdir -p "$(dirname "$OUT")"
        echo "=== ${ALGO} × ${GATE} ==="
        python seg_refiner/run_pipeline_with_seg.py \
            --video "$VID" --mask "$MASK" --gt "$GT" \
            --algo "$ALGO" --max-frames 600 --skip 2 --start-time 0 --gate-mode "$GATE" \
            --out "$OUT" 2>&1
    done
done
echo STAGE_B_DONE

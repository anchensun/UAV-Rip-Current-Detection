#!/usr/bin/env bash
# Re-run our seg-gated pipeline saving full heatmap arrays for both scenes,
# then apply Gaussian smoothing + ensemble. We need 2 algos (HOR_HS, LK) on
# each scene; LK and HOR_LK are similar so 2 is enough.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

OUT="seg_refiner/results/saved_heatmaps"
mkdir -p "$OUT"

PB_VID="data/DJI_0011.MOV"
PB_MASK="script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png"

HV_VID="data/DJI_0009.MOV"
HV_MASK="script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png"

for ALGO in HOR_HS HOR_LK LK; do
    echo "=== PB ${ALGO} ==="
    python seg_refiner/enhance_save_pipeline_heatmaps.py \
        --video "$PB_VID" --mask "$PB_MASK" --algo "$ALGO" \
        --max-frames 600 --skip 2 --start-time 0 --gate-mode sea \
        --out "${OUT}/PB_${ALGO}.npz" 2>&1
    echo "=== HV ${ALGO} ==="
    python seg_refiner/enhance_save_pipeline_heatmaps.py \
        --video "$HV_VID" --mask "$HV_MASK" --algo "$ALGO" \
        --max-frames 600 --skip 2 --start-time 0 --gate-mode sea \
        --out "${OUT}/HV_${ALGO}.npz" 2>&1
done
echo SAVE_HEATMAPS_DONE

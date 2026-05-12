#!/usr/bin/env bash
# Palm Beach (DJI_0011) — all 4 OF algos with the 6-class U-Net sea-gate.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

OUT="seg_refiner/results/palmbeach_sea"
mkdir -p "$OUT"

for ALGO in HOR_LK HS LK; do
    echo "=== PalmBeach sea-gate ${ALGO} ==="
    python seg_refiner/run_pipeline_with_seg.py \
        --video data/DJI_0011.MOV \
        --mask script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png \
        --gt   "data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png" \
        --algo "$ALGO" --max-frames 600 --skip 2 --start-time 0 --gate-mode sea \
        --out  "${OUT}/${ALGO}.csv" 2>&1
done
echo PB_ALL_DONE

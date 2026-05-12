#!/usr/bin/env bash
# Stage D: param sweep on best Palm Beach config (LK seg-gate).
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

VID="data/DJI_0011.MOV"
MASK="script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png"
GT="data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png"
OUT="seg_refiner/results/stage_d_pbsweep"
mkdir -p "$OUT"

for FLOW in 0.02 0.10; do
    for TOP in 0.25 0.40; do
        TAG="flow${FLOW}_top${TOP}"
        echo "=== LK seg-gate ${TAG} ==="
        python seg_refiner/run_pipeline_with_seg.py \
            --video "$VID" --mask "$MASK" --gt "$GT" \
            --algo LK --max-frames 600 --skip 2 --start-time 0 --gate-mode sea \
            --min-flow-mag "$FLOW" --top-fraction "$TOP" \
            --out "${OUT}/${TAG}.csv" 2>&1
    done
done
echo STAGE_D_DONE

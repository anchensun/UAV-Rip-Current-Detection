#!/usr/bin/env bash
# Apply the U-Net sea-gate to the 16 SBL-annotated DJI clips (HOR_HS only).
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

DATA="data/extracted/Rip Current Video/New Drone Videos"
MASKS="script/Rip_Current-main/New_Rip_Mask"
GT="data/extracted/Rip Current Video/RipOutlinesSBL/SBL_GT"
OUT="seg_refiner/results/sbl16_seagate"
mkdir -p "$OUT"

for clip in DJI_0006_1 DJI_0006_2 DJI_0006_3 DJI_0006_4 DJI_0006_5 DJI_0006_6 DJI_0006_7 DJI_0006_8 \
            DJI_0007_1 DJI_0007_2 DJI_0007_3 DJI_0007_4 DJI_0007_5 DJI_0007_6 DJI_0007_7 DJI_0007_8; do
    echo "=== ${clip} sea-gate HOR_HS ==="
    python seg_refiner/run_pipeline_with_seg.py \
        --video "${DATA}/${clip}.mp4" \
        --mask  "${MASKS}/${clip}_mask.png" \
        --gt    "${GT}/${clip}.png" \
        --algo HOR_HS --max-frames 600 --skip 2 --start-time 0 --gate-mode sea \
        --out  "${OUT}/${clip}.csv" 2>&1
done
echo SBL16_SEAGATE_DONE

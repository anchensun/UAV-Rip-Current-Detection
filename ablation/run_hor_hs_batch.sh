#!/usr/bin/env bash
# Run HOR_HS angular-tolerance ablation on 6 DJI clips, sequentially.
# Each clip: ~3 min on 12 cores. Total: ~18 min.

set -euo pipefail

cd "$(dirname "$0")/.."

source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

CLIPS=(DJI_0006_1 DJI_0006_3 DJI_0006_8 DJI_0007_3 DJI_0007_5 DJI_0007_8)
ALGO="${ALGO:-HOR_HS}"
OUTDIR="ablation/results/${ALGO,,}"
mkdir -p "$OUTDIR"

for c in "${CLIPS[@]}"; do
  echo "=== ${ALGO} on $c ==="
  python ablation/run_hor_hs_pipeline.py \
    --video "data/extracted/Rip Current Video/New Drone Videos/${c}.mp4" \
    --mask  "script/Rip_Current-main/New_Rip_Mask/${c}_mask.png" \
    --gt    "data/extracted/Rip Current Video/RipOutlinesSBL/SBL_GT/${c}.png" \
    --algo "$ALGO" --max-frames 600 --skip 2 --workers 12 \
    --out "${OUTDIR}/${c}.csv" 2>&1
done

echo "BATCH DONE: $ALGO"

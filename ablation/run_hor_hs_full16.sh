#!/usr/bin/env bash
# Extend HOR_HS / HOR_LK ablation to the remaining 10 DJI clips
# (the original batch handled 6; this picks up the rest to give a 16-clip total).

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

CLIPS=(DJI_0006_2 DJI_0006_4 DJI_0006_5 DJI_0006_6 DJI_0006_7 DJI_0007_1 DJI_0007_2 DJI_0007_4 DJI_0007_6 DJI_0007_7)
ALGO="${ALGO:-HOR_HS}"
WORKERS="${WORKERS:-8}"
OUTDIR="ablation/results/${ALGO,,}"
mkdir -p "$OUTDIR"

for c in "${CLIPS[@]}"; do
  if [ -f "${OUTDIR}/${c}.csv" ]; then
    echo "=== ${ALGO} on $c (already done, skipping) ==="
    continue
  fi
  echo "=== ${ALGO} on $c ==="
  python ablation/run_hor_hs_pipeline.py \
    --video "data/extracted/Rip Current Video/New Drone Videos/${c}.mp4" \
    --mask  "script/Rip_Current-main/New_Rip_Mask/${c}_mask.png" \
    --gt    "data/extracted/Rip Current Video/RipOutlinesSBL/SBL_GT/${c}.png" \
    --algo "$ALGO" --max-frames 600 --skip 2 --workers "$WORKERS" \
    --out "${OUTDIR}/${c}.csv" 2>&1
done
echo "FULL16 BATCH DONE: $ALGO"

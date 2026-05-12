#!/usr/bin/env bash
# Run the patched pipeline on DJI_0009.MOV (the actual manuscript Haulover
# raw video, per coauthor confirmation 2026-05-06).
# Reference-frame matching: the last frame of finalplot matches DJI_0009 t=20s
# (MSE=20.65 vs MSE=1770 for DJI_0008), so the GT analysis window is t=0-20s.

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

VIDEO="data/DJI_0009.MOV"
MASK="script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png"
GT="script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png"
START="${WINDOW:-0}"
SUFFIX="${SUFFIX:-_t${START}s}"
OUTDIR="ablation/results/haulover_dji0009${SUFFIX}"
mkdir -p "$OUTDIR"

# DJI_0009 is 60 fps; skip=2 -> 30 fps effective; 600 frame-pairs = 20 s.
for ALGO in HOR_HS HOR_LK HS LK; do
  echo "=== ${ALGO} on DJI_0009 t=${START}-$((START + 20))s ==="
  python ablation/run_hor_hs_pipeline.py \
    --video "$VIDEO" --mask "$MASK" --gt "$GT" \
    --algo "$ALGO" --max-frames 600 --skip 2 --workers 12 \
    --start-time "$START" \
    --out "${OUTDIR}/${ALGO}.csv" 2>&1
done
echo "DJI_0009 BATCH DONE: window t=${START}-$((START + 20))s"

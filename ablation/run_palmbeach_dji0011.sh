#!/usr/bin/env bash
# Run the patched pipeline on DJI_0011.MOV (the actual manuscript Palm Beach
# raw video, per coauthor confirmation) at the GT-matching 20s window.
# The reference frame (last frame of finalplot) matches t=20s, so the GT
# window is t=0-20s.
#
# Variant runs:
#   - WINDOW=0   : t = 0 .. 20s (default, matches reference last-frame)
#   - WINDOW=20  : t = 20 .. 40s (in case reference is first-frame)

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

VIDEO="data/DJI_0011.MOV"
MASK="script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png"
GT="data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png"
START="${WINDOW:-0}"
SUFFIX="${SUFFIX:-_t${START}s}"
OUTDIR="ablation/results/palmbeach_dji0011${SUFFIX}"
mkdir -p "$OUTDIR"

# DJI_0011 is 60 fps; skip=2 -> 30 fps effective; 600 frame-pairs = 20 s.
for ALGO in HOR_HS HOR_LK HS LK; do
  echo "=== ${ALGO} on DJI_0011 t=${START}-$((START + 20))s ==="
  python ablation/run_hor_hs_pipeline.py \
    --video "$VIDEO" --mask "$MASK" --gt "$GT" \
    --algo "$ALGO" --max-frames 600 --skip 2 --workers 12 \
    --start-time "$START" \
    --out "${OUTDIR}/${ALGO}.csv" 2>&1
done
echo "DJI_0011 BATCH DONE: window t=${START}-$((START + 20))s"

#!/usr/bin/env bash
# Run the patched HOR HS / HOR LK / HS / LK pipeline on Palm Beach
# (DJI_0047-002.MP4) at 1920x1080, accumulating per-pixel heatmaps for
# multiple angular thresholds. Compares clean float32 storage against the
# previous int8-buggy heatmaps for the manuscript's headline scene.

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

VIDEO="data/DJI_0047-002.MP4"
MASK="script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png"
GT="data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png"
OUTDIR="ablation/results/palmbeach"
mkdir -p "$OUTDIR"

# 30 fps source, 20 second window = 600 frame pairs at skip=1 (matches the
# manuscript's "T=20s" temporal-fusion window). Note: 0007/0006 clips were
# 60 fps with skip=2; this clip is already 30 fps so skip=1.
for ALGO in HOR_HS HOR_LK HS LK; do
  echo "=== ${ALGO} on PalmBeach (DJI_0047-002) ==="
  python ablation/run_hor_hs_pipeline.py \
    --video "$VIDEO" --mask "$MASK" --gt "$GT" \
    --algo "$ALGO" --max-frames 600 --skip 1 --workers 12 \
    --out "${OUTDIR}/${ALGO}.csv" 2>&1
done
echo "PALMBEACH BATCH DONE"

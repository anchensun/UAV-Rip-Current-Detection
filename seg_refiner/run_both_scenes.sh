#!/usr/bin/env bash
# Run the seg-gated pipeline on Palm Beach (DJI_0011) and Haulover (DJI_0009).
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

GATE="${GATE:-sea}"
MAX_FRAMES="${MAX_FRAMES:-600}"

run_scene() {
    local NAME="$1" VIDEO="$2" MASK="$3" GT="$4" START="$5"
    local OUT="seg_refiner/results/${NAME}_${GATE}"
    mkdir -p "$OUT"
    for ALGO in HOR_HS HOR_LK HS LK; do
        echo "=== ${NAME} ${GATE} ${ALGO} t=${START}-$((START+20))s ==="
        python seg_refiner/run_pipeline_with_seg.py \
            --video "$VIDEO" --mask "$MASK" --gt "$GT" \
            --algo "$ALGO" --max-frames "$MAX_FRAMES" --skip 2 \
            --start-time "$START" --gate-mode "$GATE" \
            --out "${OUT}/${ALGO}.csv" 2>&1
    done
}

run_scene "palmbeach"        \
    "data/DJI_0011.MOV"      \
    "script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png" \
    "data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png" \
    0

run_scene "haulover_dji0009" \
    "data/DJI_0009.MOV"      \
    "script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png" \
    "script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png" \
    0

echo "DONE both scenes (gate=$GATE)"

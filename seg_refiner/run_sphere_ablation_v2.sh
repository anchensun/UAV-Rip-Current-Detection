#!/usr/bin/env bash
# Ablation v2: correct ψ convention. ψ=90° = nadir. Sweep around nadir.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

run() {
    local SCENE="$1" VIDEO="$2" MASK="$3" GT="$4" PSI="$5"
    local OUT="seg_refiner/results/${SCENE}_sphere_v2_psi${PSI}"
    mkdir -p "$OUT"
    echo "=== ${SCENE} sphere ψ=${PSI}° HOR_HS ==="
    python seg_refiner/run_pipeline_with_seg_sphere.py \
        --video "$VIDEO" --mask "$MASK" --gt "$GT" \
        --algo HOR_HS --max-frames 600 --skip 2 --start-time 0 \
        --gate-mode sea --projection sphere \
        --camera-altitude 60 --camera-tilt-psi "$PSI" \
        --out "${OUT}/HOR_HS.csv" 2>&1
}

PB_VID="data/DJI_0011.MOV"
PB_MASK="script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png"
PB_GT="data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png"

HV_VID="data/DJI_0009.MOV"
HV_MASK="script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png"
HV_GT="script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png"

for PSI in 70 80 85 90; do
    run palmbeach        "$PB_VID" "$PB_MASK" "$PB_GT" "$PSI"
    run haulover_dji0009 "$HV_VID" "$HV_MASK" "$HV_GT" "$PSI"
done
echo SPHERE_V2_DONE

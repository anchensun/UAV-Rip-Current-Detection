#!/usr/bin/env bash
# Broad sphere ablation: all 4 OF algos × {60, 70, 80, 90, 100}° camera tilt
# on Haulover (DJI_0009.MOV). Save full heatmap arrays for each combo so we
# can compute per-T AUC and find the best sphere configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

VID="data/DJI_0009.MOV"
MASK="script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png"
GT="script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png"
OUT="seg_refiner/results/sphere_broad_haulover"
mkdir -p "$OUT"

# psi=90 = nadir baseline (identity-equivalent to no-projection)
# psi<90 = forward tilt; expected best is ~70 for Haulover (~20 deg tilt)
# psi>90 = backward tilt (unusual)
for PSI in 60 70 80 90 100; do
    for ALGO in HOR_HS HOR_LK HS LK; do
        TAG="${ALGO}_psi${PSI}"
        echo "=== ${TAG} ==="
        python seg_refiner/sphere_plus_smooth.py \
            --video "$VID" --mask "$MASK" --gt "$GT" \
            --algo "$ALGO" --max-frames 600 --skip 2 --start-time 0 \
            --gate-mode sea --projection sphere \
            --camera-altitude 60 --camera-tilt-psi "$PSI" \
            --out-npz "${OUT}/${TAG}.npz" 2>&1 | tail -20
    done
done
echo SPHERE_BROAD_DONE

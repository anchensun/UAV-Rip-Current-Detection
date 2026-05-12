#!/usr/bin/env bash
# Run sphere-projected seg-gated pipeline at scene-specific best ψ + save heatmap.
# Then compare smoothed-with-sphere vs smoothed-without-sphere.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

OUT="seg_refiner/results/sphere_plus_smooth"
mkdir -p "$OUT"

# Palm Beach: best ψ=90 (nadir).
echo === Palm Beach HOR_HS sphere ψ=90 + smoothing ===
python seg_refiner/sphere_plus_smooth.py \
    --video data/DJI_0011.MOV \
    --mask script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png \
    --gt   "data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png" \
    --algo HOR_HS --max-frames 600 --skip 2 --start-time 0 \
    --gate-mode sea --projection sphere \
    --camera-altitude 60 --camera-tilt-psi 90 \
    --out-npz "$OUT/PB_HORHS_sphere_psi90.npz" 2>&1

echo === Haulover HOR_HS sphere ψ=70 + smoothing ===
python seg_refiner/sphere_plus_smooth.py \
    --video data/DJI_0009.MOV \
    --mask script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png \
    --gt   script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png \
    --algo HOR_HS --max-frames 600 --skip 2 --start-time 0 \
    --gate-mode sea --projection sphere \
    --camera-altitude 60 --camera-tilt-psi 70 \
    --out-npz "$OUT/HV_HORHS_sphere_psi70.npz" 2>&1

echo === Haulover LK sphere ψ=70 + smoothing ===
python seg_refiner/sphere_plus_smooth.py \
    --video data/DJI_0009.MOV \
    --mask script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png \
    --gt   script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png \
    --algo LK --max-frames 600 --skip 2 --start-time 0 \
    --gate-mode sea --projection sphere \
    --camera-altitude 60 --camera-tilt-psi 70 \
    --out-npz "$OUT/HV_LK_sphere_psi70.npz" 2>&1

echo SPHERE_SMOOTH_DONE

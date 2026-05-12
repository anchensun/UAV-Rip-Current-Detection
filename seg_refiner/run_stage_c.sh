#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/anchensun/miniconda3/etc/profile.d/conda.sh
conda activate base

PB_VID="data/DJI_0011.MOV"
PB_MASK="script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png"
PB_GT="data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png"

HV_VID="data/DJI_0009.MOV"
HV_MASK="script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png"
HV_GT="script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png"

echo === Palm Beach 4-algo ensemble
python seg_refiner/run_pipeline_multialgo_ensemble.py \
    --video "$PB_VID" --mask "$PB_MASK" --gt "$PB_GT" \
    --out-dir seg_refiner/results/stage_c_palmbeach \
    --max-frames 600 --skip 2 --start-time 0 --gate-mode sea 2>&1

echo
echo === Haulover 4-algo ensemble
python seg_refiner/run_pipeline_multialgo_ensemble.py \
    --video "$HV_VID" --mask "$HV_MASK" --gt "$HV_GT" \
    --out-dir seg_refiner/results/stage_c_haulover \
    --max-frames 600 --skip 2 --start-time 0 --gate-mode sea 2>&1

echo STAGE_C_DONE

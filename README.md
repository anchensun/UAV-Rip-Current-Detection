# UAV Rip Current Detection

Code, trained model, and result CSVs for an end-to-end UAV-video rip-current detection pipeline. The pipeline combines CNN-based scene parsing, sphere-aware projection from image to ground coordinates with a tilt-aware Jacobian correction, dense optical-flow estimation with five algorithm variants, a per-pixel temporal accumulator gated by offshore-direction agreement, and a post-processing block of spatial Gaussian smoothing plus normalized cross-algorithm ensembling.

## Repository contents

```
script/Rip_Current-main/        Patched 2020 reference pipeline
  ├── OFlib.py                  Lucas-Kanade, Horn-Schunck, HOR variants
  ├── Typhoonmode.py            Typhoon optical flow (numpy 2.x compatible)
  ├── Compute_Heatmap.py        Per-pixel offshore-agreement accumulator
  ├── Compute_AUC.py            Precision-recall AUC against GT
  ├── GenDemo.py, *.py          Visualization + pipeline glue
  └── New_Rip_Mask/*.png        Per-scene sea masks

seg_refiner/                    Post-processing pipeline
  ├── train_seg.py              CNN sea-mask refiner (U-Net, ResNet-18)
  ├── train_ripdetseg_binary.py YOLOv8m-seg fine-tune on RipDetSeg
  ├── sphere_projection.py      Sphere → ground Jacobian
  ├── sphere_plus_smooth.py     Sphere ablation runner
  ├── run_pipeline_with_seg.py        Seg-gated pipeline
  ├── run_pipeline_with_seg_sphere.py +sphere projection
  ├── run_pipeline_with_yolo.py YOLOv8 baseline runner
  ├── run_pipeline_with_ripseg.py     RipDetSeg U-Net baseline
  ├── enhance_canonical_smoothing.py  σ-smoothing on canonical heatmaps
  ├── enhance_canonical_ensemble.py   Cross-algorithm normalized ensemble
  ├── smooth_and_evaluate.py    σ sweep + AUC
  ├── match_markup_frame.py     Annotator markup → reference-frame matcher
  ├── eval_frame_bounded.py     Frame-aware AUC evaluation
  ├── window_sweep_all_clips.py Window-length × start-position sweep
  ├── analyze_dji0006_1.py      Camera-motion deep-dive
  ├── checkpoints/best.pt       Trained CNN sea-mask refiner
  └── results/*.csv             AUC tables, motion drift, window sweeps

ablation/results/               Original ablation tables
  ├── _consolidated.csv         Master comparison table
  ├── DJI_*.csv                 Per-clip per-T AUC sweeps
  ├── sphere_sensitivity*.csv   Planar vs sphere @ different altitudes
  └── _head_to_head.csv         Framework vs YOLOv8m-seg on 6 clips
```

## Trained model

`seg_refiner/checkpoints/best.pt` — U-Net (ResNet-18 encoder, ImageNet-initialized, 14.3 M parameters) trained on 215 + 22 frames at 1920 × 1080 with six-class scene-parsing labels (void / sky / sea / land / foam / structure). Validation IoU: 0.728 (sea), 0.736 (land), 0.523 (foam).

Loaded by every seg-gated pipeline runner via `torch.load(.../seg_refiner/checkpoints/best.pt)`.

## Data

Raw UAV videos are **not included** in this repository. The DJI Phantom 3 Advanced captures (`DJI_0009.MOV` Haulover, `DJI_0011.MOV` Palm Beach, `DJI_0006_*.mp4` and `DJI_0007_*.mp4` South Beach 1 & 2) were provided by researchers at the Department of Earth & Environment, Florida International University, with permission for use in this study but without public redistribution rights. To obtain the videos, contact `asun@miami.edu`.

Per-scene sea masks (drawable from any frame) are included under `script/Rip_Current-main/New_Rip_Mask/`.

## Headline results

| Scene | Configuration | PR-AUC |
|---|---|---|
| Palm Beach | HOR HS + LK seg-gated ensemble, T=150°, σ=25 | **0.7067** |
| Haulover | Canonical HS+LK σ=25 ensemble + sphere-projected LK (w=0.4) | **0.7040** |
| South Beach 1, segment 1 | Late-10s window, HS+LK ensemble, T=30°, σ=25 | **0.6276** |

CSV reproducers for each table:

| Result | Reproducer | Output CSV |
|---|---|---|
| Per-algorithm AUC @ PB / HV | `script/Rip_Current-main/Compute_Heatmap.py` then `Compute_AUC.py` | `ablation/results/_consolidated.csv` |
| PB AUC under CNN seg-gate | `seg_refiner/run_pipeline_with_seg.py` | `seg_refiner/results/seg_gated_palmbeach.csv` |
| PB pipeline build-up | `seg_refiner/enhance_canonical_smoothing.py` | `seg_refiner/results/palmbeach_buildup.csv` |
| HV pipeline build-up | `seg_refiner/sphere_plus_smooth.py` | `seg_refiner/results/haulover_buildup.csv` |
| Sphere ψ ablation | `seg_refiner/sphere_plus_smooth.py` (ψ sweep) | `seg_refiner/results/sphere_haulover.csv` |
| Cross-scene window sweep | `seg_refiner/window_sweep_all_clips.py` | `seg_refiner/results/window_sweep_all_clips.csv` |
| Markup frame matches | `seg_refiner/match_markup_frame.py` | `seg_refiner/results/markup_frame_match.csv` |
| Motion drift (DJI_0006_1) | `seg_refiner/analyze_dji0006_1.py` | `seg_refiner/results/dji0006_1_motion.csv` |
| YOLOv8 head-to-head | `seg_refiner/run_pipeline_with_yolo.py` | `ablation/results/_head_to_head.csv` |

## Environment

Two Python environments are exercised:

- **`rip_current`** (legacy): the patched 2020 numpy/scipy/OpenCV pipeline. Compatible with NumPy ≥ 2.0 after the in-tree patches.
- **`base` / project default**: PyTorch 2.x + `segmentation_models_pytorch` 0.5.x for the CNN sea-mask refiner and YOLOv8 baseline.

Activation helper: `tools/activate_env.sh` (sets `LD_PRELOAD` for an Intel JIT stub on some Linux configs).

Minimal dependency stack (Python 3.10+):

```
numpy>=2.0  scipy  opencv-python  pillow
torch>=2.5  torchvision  segmentation_models_pytorch
ultralytics                         # YOLOv8 baseline
pyimgcompactor  matplotlib
```

## Contact

Anchen Sun — `asun@miami.edu` — University of Miami.

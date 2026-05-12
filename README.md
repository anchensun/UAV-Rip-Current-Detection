# UAV Rip Current Detection

End-to-end pipeline for detecting rip currents from raw UAV (Unmanned Aerial Vehicle) video, combining CNN-based scene parsing, sphere-aware projection from image to ground coordinates with a tilt-aware Jacobian correction, dense optical-flow estimation with five algorithm variants, a per-pixel temporal accumulator gated by offshore-direction agreement, and a post-processing block of spatial Gaussian smoothing plus normalized cross-algorithm ensembling.

This repository accompanies the manuscript:

> Anchen Sun et al. *Rip Current Detection in Nearshore Areas through UAV Video Analysis with Almost Local-Isometric Embedding Techniques on Sphere*. PLOS ONE (manuscript PONE-D-25-20427R1, R2 revision).

**Headline results** (PR-AUC against expert consensus annotations):

- **Palm Beach: 0.7067** — full pipeline (HOR HS + LK seg-gated ensemble at T=150°, σ=25)
- **Haulover: 0.7040** — canonical HS+LK σ=25 ensemble mixed (w=0.4) with sphere-projected LK
- **South Beach 1, segment 1: 0.6276** — cross-scene generalization under short-window protocol (10 s ending at reference frame, HS+LK ensemble, T=30°, σ=25)

## Repository layout

```
plos_latex_template.tex                 # Main manuscript (R2)
plos_latex_template_r1_clean.tex        # R1 baseline (read-only archival)
plos_latex_template_r1_diff_base.tex    # latexdiff input (placeholders aligned to R2)
plos_latex_template_R2_clean.pdf        # Compiled clean manuscript
plos_latex_template_R2_tracked.pdf      # latexdiff R1→R2 tracked-changes
Cover_Letter.{md,pdf}                   # Editor cover letter
Response_to_Reviewers.{md,pdf}          # Point-by-point response
letter_style.tex                        # Shared pandoc/xelatex letter style
fix_tracked_swaps.py                    # Post-processor for latexdiff output

main.bib, plos2015.bst                  # Bibliography + PLOS bibstyle

figures/                                # Manuscript figures (PNG/JPG/PDF)

script/Rip_Current-main/                # Patched 2020 reference pipeline
  ├── OFlib.py                          # Lucas-Kanade, Horn-Schunck, HOR variants
  ├── Typhoonmode.py                    # Typhoon optical flow (patched for numpy 2.x)
  ├── New_Rip_Mask/*_mask.png           # Per-scene sea masks
  └── *.py                              # Original pipeline scripts (audited)

seg_refiner/                            # R2 post-processing pipeline
  ├── train_seg.py                      # CNN sea-mask refiner (U-Net, ResNet-18)
  ├── train_ripdetseg_binary.py         # YOLOv8m-seg fine-tune on RipDetSeg
  ├── sphere_projection.py              # Sphere → ground Jacobian
  ├── sphere_plus_smooth.py             # Sphere ablation runner
  ├── run_pipeline_with_seg.py          # Full seg-gated pipeline
  ├── run_pipeline_with_seg_sphere.py   # +sphere
  ├── enhance_canonical_smoothing.py    # Smoothing on canonical heatmaps
  ├── enhance_canonical_ensemble.py     # Cross-algorithm ensemble
  ├── smooth_and_evaluate.py            # σ sweep evaluation
  ├── match_markup_frame.py             # SBL Markup → reference frame matcher
  ├── eval_frame_bounded.py             # Frame-aware AUC eval
  ├── window_sweep_all_clips.py         # Window-length × start-position sweep
  ├── analyze_dji0006_1.py              # DJI_0006_1 deep-dive (motion + windows)
  ├── checkpoints/best.pt               # Trained CNN sea-mask refiner (val IoU 0.728 sea, 0.736 land, 0.523 foam)
  └── results/                          # CSVs of all R2 ablations and sweeps

ablation/results/                       # Original ablation tables
  ├── _consolidated.csv                 # Master comparison table
  ├── DJI_*.csv                         # Per-clip per-T AUC sweeps
  ├── sphere_sensitivity*.csv           # Planar vs sphere @ different altitudes
  └── _head_to_head.csv                 # Framework vs YOLOv8m-seg on 6 clips
```

## Reproducing the headline results

Each result table in the manuscript ships with a runner script that reproduces it from the raw video files (provided separately by Florida International University; see Acknowledgments in the manuscript).

| Manuscript table / claim | Reproducer |
|---|---|
| `tbl2` per-algorithm AUC at PB & HV | `script/Rip_Current-main/Compute_Heatmap.py` then `Compute_AUC.py` |
| `tbl_seagate` PB AUC under seg gating | `seg_refiner/run_pipeline_with_seg.py` |
| `tbl_pb_final` PB pipeline build-up | `seg_refiner/enhance_canonical_smoothing.py` |
| `tbl_hv_final` HV pipeline build-up | `seg_refiner/sphere_plus_smooth.py` |
| `tbl_sphere` sphere ablation | `seg_refiner/sphere_plus_smooth.py` with ψ sweep |
| `tbl_short_window` cross-scene generalization | `seg_refiner/window_sweep_all_clips.py` |
| YOLOv8 head-to-head | `seg_refiner/run_pipeline_with_yolo.py` |

CSV outputs from each runner are in `seg_refiner/results/` and `ablation/results/`.

## What is *not* in this repository

- **Raw UAV videos** (the `data/` directory): the `DJI_0009.MOV` (Haulover), `DJI_0011.MOV` (Palm Beach), and the 16 `DJI_0006_*.mp4` / `DJI_0007_*.mp4` (South Beach) source files were provided by researchers from Florida International University with permission for use in this study but without rights for public redistribution. To obtain the videos, contact the corresponding author (`asun@miami.edu`).
- **Large intermediate `.npz` heatmap arrays** and **YOLOv8m-seg + RipDetSeg checkpoints** (~1 GB total): omitted to keep the repository under GitHub's recommended size. The CSV summary tables that derive from them are included; the .npz arrays themselves will be deposited on Zenodo on manuscript acceptance.

## Environment

The pipeline is exercised in two Python environments:

- **`rip_current` (legacy)**: original 2020 numpy/scipy/cv2 pipeline, patched for numpy ≥ 2.0 compatibility.
- **`base` / project-default**: PyTorch 2.x + `segmentation_models_pytorch` 0.5.x for the CNN sea-mask refiner and YOLOv8 baseline.

Activation helper: `tools/activate_env.sh` (sets `LD_PRELOAD` for an Intel JIT stub on some Linux configs).

## Citation

```bibtex
@article{sun2026ripcurrent,
  title  = {Rip Current Detection in Nearshore Areas through UAV Video Analysis with Almost Local-Isometric Embedding Techniques on Sphere},
  author = {Sun, Anchen and others},
  journal = {PLOS ONE},
  year   = {2026},
  note   = {Manuscript PONE-D-25-20427R1, R2 revision}
}
```

## License

Source code: MIT. Manuscript text and figures: © the authors, all rights reserved pending publication.

## Contact

Anchen Sun — `asun@miami.edu` — University of Miami.

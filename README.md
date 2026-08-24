# UAV Rip Current Detection

Code, trained model, and result CSVs for an end-to-end UAV-video rip-current detection pipeline. The pipeline combines CNN-based scene parsing, sphere-aware projection from image to ground coordinates with a tilt-aware Jacobian correction, dense optical-flow estimation with five algorithm variants, a per-pixel temporal accumulator gated by offshore-direction agreement, and a post-processing block of spatial Gaussian smoothing plus normalized cross-algorithm ensembling.

## Repository contents

```
script/Rip_Current-main/        Patched 2020 reference pipeline
  ├── OFlib.py                  Lucas-Kanade, Horn-Schunck, HOR variants
  ├── Typhoonmode.py            Typhoon optical flow (numpy 2.x compatible)
  ├── Gen_OPT_HeatMap.py        Per-pixel offshore-agreement accumulator
  ├── PerformanceTest.py        Precision-recall AUC against GT
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
  ├── enhance_canonical_haulover.py   σ-smoothing on canonical heatmaps
  ├── enhance_canonical_v2/v3/v4.py   Cross-algorithm normalized ensemble
  ├── smooth_and_evaluate.py    σ sweep + AUC
  ├── match_markup_frame.py     Annotator markup → reference-frame matcher
  ├── eval_frame_bounded.py     Frame-aware AUC evaluation
  ├── window_sweep_all_clips.py Window-length × start-position sweep
  ├── analyze_dji0006_1.py      Camera-motion deep-dive
  ├── checkpoints/best.pt       Trained CNN sea-mask refiner
  ├── results/README.md         >> directory map: which run backs which paper number
  └── results/*.csv             AUC tables, motion drift, window sweeps

ablation/results/               Angular-tolerance and sphere ablations
  ├── README.md                 >> directory map; READ THIS before using any
  │                                Palm Beach result (palmbeach/ is superseded)
  ├── _consolidated.csv         Master comparison table
  ├── DJI_*.csv                 Per-clip per-T AUC sweeps
  ├── sphere_sensitivity*.csv   Planar vs sphere @ different altitudes
  └── _head_to_head.csv         Framework vs YOLOv8m-seg on 6 clips
```

## Trained model

`seg_refiner/checkpoints/best.pt` — U-Net (ResNet-18 encoder, ImageNet-initialized, 14.3 M parameters) trained on **215 distinct** frames at 1920 × 1080 with six-class scene-parsing labels (void / sky / sea / land / foam / structure).

> **Two disclosures about this dataset**, both stated in the paper. (1) The released 22-frame "validation" split consists of byte-identical duplicates of training frames, so a genuine held-out split was not preserved when the dataset was packaged. The IoU historically reported as validation — 0.728 (sea), 0.736 (land), 0.523 (foam) — is therefore an **in-sample** measurement. Generalization is instead quantified by test-time transfer to the two evaluation scenes, no frame of which is in the training set: sea IoU 0.484 (Palm Beach), 0.575 (Haulover). Note that frame-level disjointness is not scene-level disjointness -- the Ocean Reef Park training capture lies about 131 m from the Palm Beach evaluation site and shares its setting, as the paper states. (2) The per-frame extraction manifest from the 2020 dataset assembly was not preserved, so per-capture provenance cannot be reconstructed from records. What *is* established, by dense pixel-matching over both evaluation videos, is that **no training frame is an extraction from either evaluation video** (minimum MSE 674; see `detect_test_leakage.py` below).

Loaded by every seg-gated pipeline runner via `torch.load(.../seg_refiner/checkpoints/best.pt)`.

**Two things you will notice if you load the checkpoint, explained here so they are not
mistaken for inconsistencies with the paper.** (1) The stored `num_classes` is **8**, not 6.
Labels are 3-bit BGR-encoded, so the tensor has eight channels, of which codes 3 and 7 are
never used (see the comment at `seg_refiner/train_seg.py:158`); the task is six-class, as
the paper says. (2) The checkpoint's stored `iou` field, and the per-epoch IoUs in
`seg_refiner/_train_log.txt`, are **patch-level** metrics used only to select the best epoch.
They do **not** match the paper's numbers, and the gap looks alarming for the foam class in
particular: every training epoch logs `foam=0.000`, while the paper reports foam IoU 0.523.
This is a protocol difference, not an inconsistency, and it is checkable — evaluating
`best.pt` on the same 22-frame split at **full frame** (native 1920x1080, single forward
pass, argmax) reproduces the paper exactly:

| class | full-frame IoU | full-frame F1 | paper |
|---|---|---|---|
| sea | 0.7283 | 0.8428 | 0.728 / 0.843 |
| land | 0.7364 | 0.8482 | 0.736 |
| foam | 0.5230 | 0.6868 | 0.523 |

Foam occupies only 0.23% of validation pixels, so on 512x512 random crops it is essentially
never the argmax; given the whole frame the model recovers it. Both figures are in-sample,
for the reason given above; the generalization numbers are the test-time transfer IoUs
(0.484 Palm Beach, 0.575 Haulover).

## Data

Raw UAV videos are **not included** in this repository. The DJI Phantom 3 Advanced captures (`DJI_0009.MOV` Haulover, `DJI_0011.MOV` Palm Beach, `DJI_0006_*.mp4` and `DJI_0007_*.mp4` South Beach 1 & 2) were collected by the Department of Earth & Environment, Florida International University (the affiliation of co-author S. B. Leatherman, who provided them for this study) and are currently held without public-redistribution rights. They will be deposited alongside the derived data in the archival Zenodo repository once the institutional release completes; until then, access requests are handled through the Department of Earth & Environment, Florida International University.

The training-only captures -- **Ocean Reef Park** (the capture the paper identifies as a principal source of the CNN scene-parsing frames, ~131 m from the Palm Beach evaluation site) and the South Beach captures -- are likewise not included, on the same terms. The 215 scene-parsing training frames themselves are image data and ship with the archival deposit, not this repository.

Per-scene sea masks (drawable from any frame) are included under `script/Rip_Current-main/New_Rip_Mask/`.

## Headline results

The paper reports **two operating points**, and leads with the untuned one.

| Scene | Operating point | Configuration | PR-AUC |
|---|---|---|---|
| Palm Beach | **neutral default** (headline) | HOR HS + LK seg-gated ensemble, **T=90°**, σ=25 | **0.636** |
| Haulover | **neutral default** (headline) | Canonical HS+LK σ=25 ensemble + sea-mask (natively T=90°) | **0.660** |
| Palm Beach | per-scene-selected (upper bound) | same recipe, T=150° | 0.7066 |
| Haulover | per-scene-selected (upper bound) | + sphere-projected LK term at T=15°, w=0.4, ψ=60° | 0.7040 |
| South Beach 1, segment 1 | per-scene-selected | Late-10s window, HS+LK ensemble, T=30°, σ=25 | 0.6276 |

`T = 90°` is not a tuned value: it is the boundary of the seaward half-plane (a velocity has a positive
offshore component iff it lies within ±90° of the offshore direction). The per-scene-selected rows are
obtained by selecting against the same expert annotation used for evaluation, and are reported in the paper
as an **optimistic upper bound, not an expected-performance estimate**.

CSV reproducers. Each row names the script that actually writes the listed output; where an
artifact was produced by a flag or assembled from a script's console output rather than written
directly, that is stated instead of implied.

| Result | Reproducer | Output |
|---|---|---|
| Per-algorithm AUC vs angular tolerance | `ablation/ablate_angle.py` | `ablation/results/hor_hs/`, `ablation/results/hor_lk/`, and the per-scene directories |
| Cross-run consolidation, incl. framework vs YOLOv8m-seg | `ablation/consolidate_results.py` | `ablation/results/_consolidated.csv`, `_head_to_head.csv`, `_digest.json` |
| Palm Beach AUC under the CNN sea-gate | `seg_refiner/run_pipeline_with_seg.py` | `seg_refiner/results/palmbeach_sea/` |
| Haulover sphere-projection sweep | `seg_refiner/sphere_plus_smooth.py` | `seg_refiner/results/haulover_dji0009_sphere_v2_psi*/` |
| Cross-scene window sweep | `seg_refiner/window_sweep_all_clips.py` | `seg_refiner/results/window_sweep_all_clips.csv` |
| Markup frame matches | `seg_refiner/match_markup_frame.py` | `seg_refiner/results/markup_frame_match.csv` |
| Motion drift (DJI_0006_1) | `seg_refiner/analyze_dji0006_1.py` | `seg_refiner/results/dji0006_1_motion.csv` |
| Frame-bounded extra-clip evaluation | `seg_refiner/eval_extra_clips.py` | `seg_refiner/results/extra_clips_eval.csv` |
| Planar-vs-sphere sensitivity and altitude bound | `ablation/sphere_projection_sensitivity.py` (writes to the path given by `--out`) | `ablation/results/sphere_sensitivity*.csv`, `sphere_altitude_bound.csv`, `sphere_altitude/` |
| Smoothing sensitivity of the headline recipe | `seg_refiner/smooth_and_evaluate.py` — prints the sweep to stdout; `sigma_sweep_headline.csv` was transcribed from that output rather than written by the script | `seg_refiner/results/sigma_sweep_headline.csv` |
| Canonical + smoothing + ensemble build-up | `seg_refiner/enhance_canonical_haulover.py`, `enhance_canonical_v2.py`, `enhance_canonical_v3.py`, `enhance_canonical_v4.py` | stdout tables reproduced in the paper |

The archived 2020 Haulover heatmap arrays that serve as the canonical base for the Haulover
result are fixed intermediate artifacts; they are distributed through the archival data deposit
rather than this repository, whose size limits exclude them.

## What can and cannot be checked from this repository

The raw UAV videos and the scene-parsing training frames are not redistributable here, so
not every number in the paper is recomputable from a bare clone. This section says plainly
which is which, so that a reader does not have to discover it by trying.

**Checkable directly from files in this repository** (no video needed — the CSVs are the
computed outputs):

- The whole training-frame leakage investigation: minimum pixel-MSE 674, median 1726 over
  the 215 distinct frames, none below 200 (`test_leakage_mse.csv`, filter `split=='train'`).
- The leakage-controlled and matched-random retraining results: 0.384→0.458 and 0.532→0.602
  leakage-free, 0.640 / 0.686 full-data, 0.491 / 0.579 random-exclusion, and the Haulover
  0.242→0.131 / 0.273→0.278 (`loso_no_pb_sea/`, `loso_no_hv_sea/`, `random_matched_pb/`,
  `palmbeach_sea/`, `palmbeach_dji0011_t0_20/`).
- The Palm Beach headline curve, both operating points: 0.6356 at the neutral `T = 90°` and
  0.7066 at `T = 150°` (`pb_headline_ensemble_perT.csv`).
- The degenerate control, 0.6183 (`degenerate_control.csv`).
- The multi-scene angular-tolerance analysis and both evaluation-scene curves
  (`ablation/results/`, plotted by `ablation/plot_multiscene_T.py`).
- The accumulation-length study (`accumulation_window.csv`) and the planar-vs-sphere
  altitude bound (`sphere_altitude_bound.csv`).

**Not recomputable from this repository alone**, because the inputs are excluded by size or
by the data restriction:

- The Haulover per-algorithm column and the Haulover build-up table, including the
  neutral-default 0.6599. These derive from the archived 2020 heatmap arrays, whose upstream
  dense optical-flow data was not retained; the arrays ship with the archival deposit.
- The sphere-projection tilt sweep table. Its inputs are large `.npz` heatmap stacks
  excluded by size; the per-run directories under `seg_refiner/results/*_sphere_v2_psi*/`
  are the released intermediate outputs.
- The segmenter IoU/F1 table, which needs the scene-parsing frames themselves.
- Anything requiring the raw captures, including re-running the optical-flow pipeline
  end-to-end.

Where a number is not reproducible here, the released intermediate artifact it was computed
from is named above, and the full inputs are in the archival deposit.


## Environment

Two Python environments are exercised:

- **`rip_current`** (legacy): the patched 2020 numpy/scipy/OpenCV pipeline. Compatible with NumPy ≥ 2.0 after the in-tree patches.
- **`base` / project default**: PyTorch 2.x + `segmentation_models_pytorch` 0.5.x for the CNN sea-mask refiner and YOLOv8 baseline.

On some Linux configurations the legacy environment needs `LD_PRELOAD` set for an Intel JIT stub before the 2020 pipeline will import; this is environment-specific and no helper script is shipped.

Minimal dependency stack (Python 3.10+):

```
numpy>=2.0  scipy  opencv-python  pillow  matplotlib
PyWavelets                          # required by OFlib.py / Typhoonmode.py
sympy  pyoptflow                    # used by the 2020 reference pipeline
torch>=2.5  torchvision  segmentation_models_pytorch
ultralytics                         # YOLOv8 baseline only
```

## Contact

Anchen Sun — `asun@miami.edu` — University of Miami.

## Third-round revision additions

Scripts and outputs added for the third round of revision.

| Analysis | Script | Output |
|---|---|---|
| **Training-frame leakage check** — dense pixel-MSE search of both evaluation videos against every training frame (Haulover across its full duration, every second frame of 24,576; Palm Beach across its entire decodable span) | `seg_refiner/detect_test_leakage.py` | `seg_refiner/results/test_leakage_mse.csv`, `results/leakage_audit_lowest.png`, `results/leakage_audit_highest.png` |
| **Leakage-controlled retraining** — segmenter retrained with every training frame the two-way matcher assigns to the held-out scene removed (76 for Palm Beach, 139 for Haulover) | `seg_refiner/train_seg.py --exclude-csv ... --exclude-scene ...` | `checkpoints/loso_no_pb.pt`, `checkpoints/loso_no_hv.pt`, `results/loso_no_pb_sea/`, `results/loso_no_hv_sea/`, training logs `_train_loso_no_*.txt` |
| **Matched random-exclusion control** — same *number* of frames removed at random, to separate training-set size from scene familiarity | `seg_refiner/train_seg.py` with `results/random_exclusion_matched.csv` | `checkpoints/random_matched.pt`, `results/random_matched_pb/` |
| Attempted per-source provenance attribution. **Outcome: negative** -- the dense matcher attributes *none* of the 215 frames to any searchable source video (all `UNMATCHED`), consistent with the extraction manifest not having been preserved | `seg_refiner/attribute_train_frames.py`, `attribute_train_frames_sources.py` | `results/train_frame_scenes.csv`, `results/train_frame_sources.csv` |
| **Segmenter test-time transfer IoU** on the two unseen evaluation scenes | `seg_refiner/eval_seg_iou_per_scene.py` | reported in the paper's Table 3 |
| **Multi-scene angular-tolerance consolidation** (Fig 10) | `ablation/plot_multiscene_T.py` | `figures/AUC_vs_T_multiscene.pdf` |
| **Degenerate control** — identical pipeline with the direction test removed entirely | `seg_refiner/degenerate_control.py` | `results/degenerate_control.csv` |
| **Full T-curve** of the exact headline ensemble recipe | `seg_refiner/pb_headline_ensemble_perT.py` | `results/pb_headline_ensemble_perT.csv` |
| **Accumulation-length study** (Fig 11) — per-frame vs accumulated AUC | `seg_refiner/accumulation_window_study.py` | `results/accumulation_window.csv` |
| **Planar-vs-sphere error over altitude** (Table 9) | `ablation/sphere_projection_sensitivity.py` | `ablation/results/sphere_altitude_bound.csv`, `ablation/results/sphere_altitude/` |
| — Smoothing sensitivity of the headline recipe | `seg_refiner/smooth_and_evaluate.py` | `results/sigma_sweep_headline.csv` |

An archival Zenodo deposit carrying this repository plus all redistributable derived data will be minted on
acceptance and cited in the paper.

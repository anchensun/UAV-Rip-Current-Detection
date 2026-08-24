# `seg_refiner/results/` — directory map

Per-scene subdirectories follow the pattern `<scene>_<variant>/`, each holding one
`T_deg,AUC` CSV plus a `.meta.json` per optical-flow algorithm recording the source video,
the segmenter checkpoint and the settings used. Palm Beach is always `DJI_0011.MOV` and
Haulover is always `DJI_0009.MOV`; the `.meta.json` is the authoritative record.

## Third-round (R3) additions — the leakage investigation

| Path | What it is |
|---|---|
| `test_leakage_mse.csv` | Dense pixel-MSE of every scene-parsing frame against both evaluation videos. `split` distinguishes the 215 distinct training frames from the 22 duplicated validation copies — **use `split=='train'` for the 215-frame statistics.** Minimum MSE 674, median 1726 over the 215; nothing below 200. |
| `leakage_audit_lowest.png`, `leakage_audit_highest.png` | Visual audit of the closest and furthest matched pairs |
| `train_frame_scenes.csv` | Deep-embedding nearest-neighbour attribution of each frame (237 rows: 215 train + 22 duplicates) |

> **The nearest-neighbour assignment is degenerate, and this is the point.** Of the 139
> training frames whose nearer evaluation video is Haulover, **134 match the same single
> frame**, `DJI_0009.MOV` at t = 401.9 s — an out-of-focus ground-level shot taken while the
> drone was on the sand. Those tiles are the blurred sand you see captioned `Hv@402s` in the
> montages. This is direct evidence that forcing every training frame onto its nearer of two
> evaluation videos measures nothing about provenance: a large majority of the assignments
> collapse onto one degenerate frame at a minimum MSE of ≥ 674. It also means the 139-frame
> Haulover exclusion behind `loso_no_hv.pt` should be read as *removing the frames that
> matcher flags*, which is what the paper says, and **not** as a scene-similarity control.
> The matched random-exclusion control is the one that isolates training-set size.

| `train_frame_sources.csv` | Dense pixel-MSE against the *training-side* source videos; every frame is `UNMATCHED`, i.e. no extraction match was found against any searchable source |
| `loso_no_pb_sea/`, `loso_no_hv_sea/` | Pipeline re-run with the leakage-controlled segmenters (76 / 139 most-similar frames removed) |
| `random_exclusion_matched.csv`, `random_matched_pb/` | **The matched control used in the paper**: 215 rows, 76 marked `RANDOM` — the same number the similarity-based exclusion removes |
| `random_exclusion_86.csv` | **Superseded, and retained only for completeness — not used for any reported number.** An earlier draw made before the duplicated-validation-split defect was found, so it is over all 237 released files rather than the 215 distinct frames. Its filename is also misleading: it marks 93 frames, not 86. It is kept rather than deleted so the record of what was run is complete; the paper's random control is `random_exclusion_matched.csv` above. |
| `degenerate_control.csv` | Direction test removed entirely (Palm Beach ensemble 0.6183) |
| `accumulation_window.csv` | PR-AUC vs accumulation length, both scenes (Fig 11) |
| `pb_headline_ensemble_perT.csv` | Full T-curve of the exact Palm Beach headline recipe: 0.6356 at the neutral `T=90°`, 0.7066 at `T=150°` |
| `sigma_sweep_headline.csv` | Gaussian-smoothing sensitivity of the headline recipe |

## A note on the audit montages and the video-redistribution restriction

`leakage_audit_*.png` and `attribution_audit_*.png` necessarily display imagery from the
UAV captures, which the paper states are held without public-redistribution rights. These
montages are not a redistribution of that material: each tile is a heavily downsampled
thumbnail (321x180 from a 1920x1080 source, ~17% linear scale), the aerial viewpoint is 50-70 m so no
individual is identifiable, and they exist solely so that a reader can verify by eye the
central claim of the leakage check -- that the closest-matching training and evaluation
frames are plainly different scenes. The manuscript commits to releasing exactly this
visual audit. The full-resolution frames and the source videos are not included here.

## Two sphere-ablation conventions — use the `_v2_` directories

The `*_seasphere_sphere_psi{0,5,15,30}/` directories were produced under an earlier tilt
parameterisation. `run_sphere_ablation_v2.sh` describes the later convention as the correct
one (`ψ = 90°` is true nadir, smaller ψ = greater forward tilt), and the paper's Table 8 and
the ψ = 60° ensemble term both use it. **The authoritative sphere results are the
`*_sphere_v2_psi{70,80,85,90}/` directories**; the four `_seasphere_sphere_psi*` directories
are retained for completeness and back no reported number.

## Earlier stages

`stage_b_*`, `stage_d_*`, `stage_e_*`, `stage_g_*` and `addensemble_*` are intermediate
runs from the development sequence, retained for auditability. `addensemble_palmbeach/`
holds the seg-gated per-algorithm peaks quoted in the paper's Palm Beach sea-gate table
(HOR HS 0.6402, LK 0.686).

Large intermediate heatmap arrays (`saved_heatmaps/`, `final_haulover/`, `reproduce_*`,
`*.npz`) are excluded from this repository by size and are distributed with the archival
data deposit instead.

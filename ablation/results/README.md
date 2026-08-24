# `ablation/results/` — what is in here, and which run the paper uses

Each per-scene subdirectory holds one CSV per optical-flow algorithm, with columns
`T_deg,AUC`: the precision-recall AUC of the patched 2020 pipeline at each angular
tolerance `T`, with **no CNN gating**. Alongside each CSV, a `.meta.json` records the
exact source video, mask, ground-truth annotation, algorithm and frame count that
produced it. **When in doubt about which capture a result came from, read the
`.meta.json` — it is the authoritative record.**

## Directory map

| Directory | Source video | Scene | Used in the paper for |
|---|---|---|---|
| `palmbeach_dji0011_t0_20/` | `DJI_0011.MOV` | Palm Beach (evaluation) | **Authoritative Palm Beach run.** Fig 9 (single-scene T sensitivity) and the Palm Beach curve of Fig 10 (18-scene sweep) |
| `palmbeach_dji0011_t10_30/`, `_t20_40/`, `_t40_60/` | `DJI_0011.MOV` | Palm Beach | Later 20-s windows, used in the short-window / camera-drift analysis |
| `haulover_dji0009_t0s/` | `DJI_0009.MOV` | Haulover (evaluation) | Haulover curve of Fig 10 |
| `hor_hs/`, `hor_lk/` | `DJI_0006_*.mp4`, `DJI_0007_*.mp4` | 16 South Beach sub-clips | The 16 grey per-scene curves of Fig 10; the short-window cross-scene evaluation |
| `sphere_altitude/` + `sphere_altitude_bound.csv` | synthetic | — | Table 9, planar-vs-sphere reconstruction error over altitude |
| `sphere_sensitivity*.csv` | synthetic | — | Earlier planar-vs-sphere sensitivity study |
| `_consolidated.csv`, `_head_to_head.csv`, `_digest.json` | — | — | Cross-run summary tables, incl. the framework-vs-YOLOv8m-seg comparison |

## ⚠ `palmbeach/` is superseded — do not use it

`ablation/results/palmbeach/` was produced by `run_palmbeach_batch.sh` on
**`DJI_0047-002.MP4`**, which is *not* the Palm Beach evaluation capture. The manuscript's
Palm Beach evaluation scene is **`DJI_0011.MOV`**, confirmed by the co-author who
collected the data; the corresponding run is `palmbeach_dji0011_t0_20/`, produced by the
later `run_palmbeach_dji0011.sh`.

The two runs give materially different curves — `palmbeach/` is flat then falls, while the
real `DJI_0011` scene rises to a peak at `T = 120°` — so mixing them up changes conclusions.

The superseded directory is retained rather than deleted so that the record of what was run
is complete and auditable. It contributes to **no** number, figure or table in the
manuscript. Every other Palm Beach result in this repository, in both `ablation/results/`
and `seg_refiner/results/`, is computed on `DJI_0011.MOV`.

## `DJI_0007_8_inv.csv` — a deliberate inverted-direction sanity control

`ablation/ablate_angle.py` supports flipping the estimated offshore direction by 180°
(it prints `(inverted offshore direction)` when it does). `DJI_0007_8_inv.csv` is that
control run on sub-clip DJI_0007_8; `DJI_0007_8.csv` is the same clip with the correct
direction.

**Read the two together, because the inverted run scores nominally higher at 7 of the 9
tolerances** (e.g. 0.0473 vs 0.0428 at `T = 90°`). That is not evidence that the offshore
direction is backwards. Both runs sit at PR-AUC ≈ 0.04–0.05, which is the chance level for
this sub-clip — annotated rip pixels cover only 1.6–9.6% of frame pixels on the South Beach
sub-clips, and a random scorer achieves the prevalence. The two curves are separated by less
than 0.005 AUC throughout, i.e. by noise at chance level.

The honest conclusion, which the paper states, is that the South Beach sub-clips carry too
little rip signal to discriminate between direction hypotheses at all. The evidence that the
direction test does something rests on the two expert-annotated evaluation scenes and on the
degenerate control (`seg_refiner/degenerate_control.py`), where the direction test is removed
entirely and the Palm Beach ensemble falls from 0.636 to 0.618 — a small but measured
contribution, reported as such.

This control is shipped rather than withheld precisely because it constrains what the
sub-clip evidence can support.

"""Palm Beach full-pipeline (2-algorithm smoothed ensemble) PR-AUC across the
angular-tolerance sweep, from the saved seg-gated heatmaps.

This is the exact headline recipe of Table `tbl_pb_final` (max-normalized mean
of Gaussian-smoothed sigma=25 seg-gated HOR Horn-Schunck + Lucas-Kanade
heatmaps, masked to the sea region): it reproduces the per-scene-selected upper
bound 0.7066 at T=150 deg and reports the full T-curve, in particular the
physically-neutral default T=90 deg (0.6356), so the default-versus-tuned
distinction is explicit.

NOTE: the input .npz heatmap stacks are large intermediate artifacts excluded
from this repository by size (see .gitignore); they ship with the archival data
deposit. The script cannot be re-run from a bare clone -- its committed output,
results/pb_headline_ensemble_perT.csv, is the reviewable artifact.

Inputs:  seg_refiner/results/saved_heatmaps/PB_{HOR_HS,LK}.npz  (9, H, W)
Output:  seg_refiner/results/pb_headline_ensemble_perT.csv
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "seg_refiner"))
from run_pipeline_with_seg import load_gt, compute_offshore, compute_auc_pr  # noqa: E402

SIGMA = 25
N_BINS = 2048   # matches smooth_and_evaluate.py (the script that produced 0.7067)


def main():
    mask = cv2.imread(str(ROOT / "script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png"),
                      cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    gt = load_gt(ROOT / "data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png", (H, W))
    _, _, sea_valid = compute_offshore(mask, 1.0 / 3.0)
    sv = sea_valid.astype(np.float32)

    hs = np.load(ROOT / "seg_refiner/results/saved_heatmaps/PB_HOR_HS.npz")
    lk = np.load(ROOT / "seg_refiner/results/saved_heatmaps/PB_LK.npz")
    Ts = hs["thresholds"]

    rows = []
    for j, T in enumerate(Ts):
        a = cv2.GaussianBlur(hs["heat"][j], (0, 0), SIGMA); a /= max(a.max(), 1e-6)
        b = cv2.GaussianBlur(lk["heat"][j], (0, 0), SIGMA); b /= max(b.max(), 1e-6)
        ens = ((a + b) / 2) * sv
        auc = compute_auc_pr(ens, gt, n_bins=N_BINS)
        rows.append({"T_deg": int(T), "AUC": round(auc, 4)})
        print(f"T={int(T):4d}  AUC={auc:.4f}")

    out = ROOT / "seg_refiner/results/pb_headline_ensemble_perT.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["T_deg", "AUC"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

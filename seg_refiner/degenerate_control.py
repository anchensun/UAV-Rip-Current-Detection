"""Degenerate control: the headline pipeline with the direction test REMOVED.

The Palm Beach headline AUC rises monotonically with the angular tolerance T all
the way to T = 150 deg, the largest value swept. That trend invites an obvious
question: what happens in the limit where the direction test admits everything?
This script answers it directly, and the answer qualifies how much of the score
the direction test actually carries.

With the direction test removed, the accumulator reduces to

    heat[i,j] = sum_t  gate_t[i,j] * sea_valid[i,j] * 1[|flow_t[i,j]| >= min_mag]

i.e. every frame in which a pixel is sea and shows any motion contributes,
regardless of direction. This is exactly the T -> 180 deg limit (the flow/offshore
angle Theta lies in [0,180], so 1[Theta < 180] is true almost everywhere). We then
apply the identical post-processing as the headline recipe (Gaussian sigma = 25,
per-algorithm max normalization, two-algorithm mean, sea mask) and evaluate.

If this control approaches the headline AUC, the direction-consistent
accumulation is contributing little on that scene and the result is carried by
segmentation gating plus spatial smoothing -- a conclusion the manuscript must
state. If it falls well below, the direction test is doing real work.

Output: seg_refiner/results/degenerate_control.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np
import torch

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "seg_refiner"))
sys.path.insert(0, str(ROOT / "script" / "Rip_Current-main"))

from run_pipeline_with_seg import (  # noqa: E402
    compute_offshore, load_gt, compute_auc_pr, load_seg_model, seg_gate_one, of_one,
)

SIGMA = 25
N_BINS = 2048
MIN_FLOW = 0.05
SKIP = 2
MAX_FRAMES = 600


def accumulate(video, mask_path, ckpt, algos, start_time=0.0):
    """Return {algo: heat_no_direction_test} plus geometry, in one video pass."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    _, _, sea_valid = compute_offshore(mask, 1.0 / 3.0)
    svf = sea_valid.astype(np.float32)

    cap = cv2.VideoCapture(str(video))
    if start_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resize = (vh, vw) != (H, W)
    grays, bgrs = [], []
    idx = 0
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        if idx % SKIP == 0:
            if resize:
                fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
            grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
            bgrs.append(fr)
        idx += 1
        if len(grays) > MAX_FRAMES + 1:
            break
    cap.release()
    n_pairs = min(len(grays) - 1, MAX_FRAMES)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_seg_model(Path(ckpt), device)

    heat = {a: np.zeros((H, W), np.float32) for a in algos}
    t0 = time.time()
    for i in range(n_pairs):
        gate = seg_gate_one(model, bgrs[i + 1], device, "sea")
        for algo in algos:
            res = of_one(grays[i], grays[i + 1], algo)
            if res is None:
                continue
            u, v = res
            if u.shape != (H, W):
                u = cv2.resize(u, (W, H)); v = cv2.resize(v, (W, H))
            fm = np.sqrt(u * u + v * v)
            valid = svf * (fm >= MIN_FLOW).astype(np.float32)
            # NO DIRECTION TEST: indicator is identically 1 wherever there is motion
            heat[algo] += gate * valid
        if (i + 1) % 150 == 0:
            print(f"    {i+1}/{n_pairs} pairs ({(i+1)/(time.time()-t0):.1f}/s)", flush=True)
    return heat, svf, (H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="PalmBeach", choices=["PalmBeach", "Haulover"])
    ap.add_argument("--ckpt", default=str(ROOT / "seg_refiner/checkpoints/best.pt"))
    args = ap.parse_args()

    cfg = {
        "PalmBeach": dict(
            video=ROOT / "data/DJI_0011.MOV",
            mask=ROOT / "script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png",
            gt=ROOT / "data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png",
        ),
        "Haulover": dict(
            video=ROOT / "data/DJI_0009.MOV",
            mask=ROOT / "script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png",
            gt=ROOT / "script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png",
        ),
    }[args.scene]

    algos = ["HOR_HS", "LK"]
    print(f"=== degenerate control (no direction test): {args.scene} ===", flush=True)
    heat, svf, (H, W) = accumulate(cfg["video"], cfg["mask"], args.ckpt, algos)
    gt = load_gt(cfg["gt"], (H, W))

    rows = []
    sm = {}
    for a in algos:
        h = cv2.GaussianBlur(heat[a], (0, 0), SIGMA)
        h /= max(h.max(), 1e-6)
        sm[a] = h
        auc = compute_auc_pr(h * svf, gt, N_BINS)
        rows.append({"scene": args.scene, "config": f"no_direction_test_{a}", "AUC": round(auc, 4)})
        print(f"  {a} (no direction test): AUC = {auc:.4f}", flush=True)

    ens = ((sm["HOR_HS"] + sm["LK"]) / 2) * svf
    auc_e = compute_auc_pr(ens, gt, N_BINS)
    rows.append({"scene": args.scene, "config": "no_direction_test_ensemble", "AUC": round(auc_e, 4)})
    print(f"  HOR_HS+LK ensemble (no direction test): AUC = {auc_e:.4f}", flush=True)

    out = ROOT / "seg_refiner/results/degenerate_control.csv"
    write_header = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "config", "AUC"])
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

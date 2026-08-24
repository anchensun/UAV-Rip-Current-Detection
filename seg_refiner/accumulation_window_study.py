"""Per-frame vs temporally-accumulated AUC.

Optical flow measures apparent tracer motion, not water motion, so the claim that
temporal accumulation filters out wave noise needs direct evidence rather than
assertion. This script provides it: it accumulates the offshore-direction
agreement indicator over a growing number of frame pairs and reports PR-AUC as a
function of accumulation length. A single frame's direction map is dominated by
wave/foam motion (low AUC); as accumulation length grows, the zero-mean
oscillatory wave component averages out while the persistent rip drift survives,
so AUC rises and saturates -- accumulation acts as a matched filter for steady
drift (SNR gain ~ sqrt(N) for zero-mean noise).

No CNN gate is used (canonical configuration) so the effect isolates temporal
accumulation alone. Fixed T = 90 deg (the offshore half-plane).

Outputs:
  seg_refiner/results/accumulation_window.csv   (scene, algo, n_pairs, sec, AUC)
  figures/AUC_vs_accumulation_window.pdf (+ .png)
"""
from __future__ import annotations
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script" / "Rip_Current-main"
sys.path.insert(0, str(SCRIPT_DIR))
from OFlib import higher_order, lucas_kanade  # noqa: E402

# reuse the geometry / AUC helpers from the seg pipeline
sys.path.insert(0, str(ROOT / "seg_refiner"))
from run_pipeline_with_seg import compute_offshore, load_gt, compute_auc_pr  # noqa: E402

T_FIXED = 90.0
SKIP = 2
MAX_FRAMES = 600
MIN_FLOW = 0.05

SCENES = {
    "Palm Beach": dict(
        video="data/DJI_0011.MOV",
        mask="script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png",
        gt="data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png",
    ),
    "Haulover": dict(
        video="data/DJI_0009.MOV",
        mask="script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png",
        gt="script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png",
    ),
}
ALGOS = ("HOR_HS", "LK")


def of_one(g0, g1, algo):
    if algo == "HOR_HS":
        u, v = higher_order(g0, g1, mode="HS")
    elif algo == "LK":
        u, v = lucas_kanade(g0, g1)
    else:
        raise ValueError(algo)
    if u.shape == (10, 10):
        return None
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(u, -1000, 1000).astype(np.float32), np.clip(v, -1000, 1000).astype(np.float32)


def load_grays(video, H, W, fps_holder):
    cap = cv2.VideoCapture(str(ROOT / video))
    fps_holder["fps"] = cap.get(cv2.CAP_PROP_FPS) or 60.0
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resize = (vh, vw) != (H, W)
    grays = []
    idx = 0
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        if idx % SKIP == 0:
            if resize:
                fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
            grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        idx += 1
        if len(grays) > MAX_FRAMES + 1:
            break
    cap.release()
    return grays


def main():
    rows = []
    for scene, cfg in SCENES.items():
        mask = cv2.imread(str(ROOT / cfg["mask"]), cv2.IMREAD_GRAYSCALE)
        H, W = mask.shape
        gt = load_gt(ROOT / cfg["gt"], (H, W))
        off_x, off_y, sea_valid = compute_offshore(mask, 1.0 / 3.0)
        sea_valid_f = sea_valid.astype(np.float32)
        fh = {}
        grays = load_grays(cfg["video"], H, W, fh)
        eff_fps = fh["fps"] / SKIP
        n_pairs = min(len(grays) - 1, MAX_FRAMES)
        # checkpoints (in pairs): 1 frame, 1s, 3s, 5s, 10s, full
        targets = sorted(set([1] + [int(round(s * eff_fps)) for s in (1, 3, 5, 10)] + [n_pairs]))
        targets = [t for t in targets if 1 <= t <= n_pairs]
        print(f"{scene}: {n_pairs} pairs @ {eff_fps:.1f} eff fps; checkpoints(pairs)={targets}")
        for algo in ALGOS:
            heat = np.zeros((H, W), dtype=np.float32)
            snap = {}
            for i in range(n_pairs):
                res = of_one(grays[i], grays[i + 1], algo)
                if res is not None:
                    u, v = res
                    if u.shape != (H, W):
                        u = cv2.resize(u, (W, H)); v = cv2.resize(v, (W, H))
                    fm = np.sqrt(u * u + v * v)
                    cos = np.clip((u * off_x + v * off_y) / np.maximum(fm, 1e-6), -1, 1)
                    ang = np.degrees(np.arccos(cos))
                    valid = sea_valid_f * (fm >= MIN_FLOW).astype(np.float32)
                    heat += (ang < T_FIXED).astype(np.float32) * valid
                if (i + 1) in targets:
                    snap[i + 1] = compute_auc_pr(heat * sea_valid_f, gt)
            for k in targets:
                rows.append({"scene": scene, "algo": algo, "n_pairs": k,
                             "sec": round(k / eff_fps, 2), "AUC": round(snap[k], 4)})
                print(f"  {algo:7s} k={k:4d} ({k/eff_fps:5.1f}s)  AUC={snap[k]:.4f}")

    out = ROOT / "seg_refiner" / "results" / "accumulation_window.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "algo", "n_pairs", "sec", "AUC"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False)
    for ax, scene in zip(axes, SCENES):
        for algo, col in (("HOR_HS", "#c0392b"), ("LK", "#2471a3")):
            pts = [(r["sec"], r["AUC"]) for r in rows if r["scene"] == scene and r["algo"] == algo]
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "o-", color=col, lw=2, label=algo)
        ax.set_xscale("log")
        ax.set_xlabel("accumulation length (s, log scale)")
        ax.set_ylabel("PR-AUC")
        ax.set_title(scene)
        ax.grid(alpha=0.25, which="both")
        ax.legend()
    fig.suptitle(r"Temporal accumulation suppresses wave noise: AUC rises with accumulation length ($T=90^\circ$, no CNN)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"AUC_vs_accumulation_window.{ext}", dpi=150, bbox_inches="tight")
    print("wrote figures/AUC_vs_accumulation_window.pdf")


if __name__ == "__main__":
    main()

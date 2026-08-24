"""Per-source-capture provenance counts for the 215 distinct training frames.

Gap (final audit): the paper never states how many training frames come from
each training-only capture, and the leakage check only searched the two
EVALUATION videos. This script runs the same dense pixel-MSE matcher against
the TRAINING-side source material: the South Beach 1/2 sub-clips and the two
unlabeled captures (DJI_0047, DJI_0008). A frame that is a genuine extraction
from one of these sources scores near-zero MSE, giving (a) per-capture counts
and (b) a direct check of whether South Beach source videos supplied training
frames (relevant to the sub-clip circularity disclosure).

Output: seg_refiner/results/train_frame_sources.csv + summary to stdout.
"""
from __future__ import annotations
import csv
import os
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
TR = ROOT / "data" / "extracted" / "Rip Current Video" / "TrainDataset_RGB" / "train" / "img" / "0"
NDV = ROOT / "data" / "extracted" / "Rip Current Video" / "New Drone Videos"
DH, DW = 54, 96
STRIDE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SOURCES = {
    "SouthBeach1": sorted(NDV.glob("DJI_0006_*.mp4")),
    "SouthBeach2": sorted(NDV.glob("DJI_0007_*.mp4")),
    "DJI_0047": [ROOT / "data/DJI_0047-002.MP4"],
    "DJI_0008": [ROOT / "data/DJI_0008.MOV"],
}


def gray_vec(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (DW, DH), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


def sample(path):
    cap = cv2.VideoCapture(str(path))
    vecs, idx, fail = [], 0, 0
    while True:
        ret, fr = cap.read()
        if not ret:
            fail += 1
            if fail > 60:
                break
            idx += 1
            continue
        fail = 0
        if idx % STRIDE == 0:
            vecs.append(gray_vec(fr))
        idx += 1
    cap.release()
    return vecs


def min_mse(Q, P):
    Qt = torch.from_numpy(Q).to(DEVICE); Pt = torch.from_numpy(P).to(DEVICE)
    q2 = (Qt * Qt).sum(1, keepdim=True); p2 = (Pt * Pt).sum(1, keepdim=True).T
    best = torch.full((Qt.shape[0],), 1e30, device=DEVICE)
    for j0 in range(0, Pt.shape[0], 4096):
        Pc = Pt[j0:j0 + 4096]
        d = (q2 + p2[:, j0:j0 + 4096] - 2.0 * (Qt @ Pc.T)) / Q.shape[1]
        best = torch.minimum(best, d.min(1).values)
    return best.cpu().numpy()


def main():
    frames = sorted(TR.glob("*.png"), key=lambda x: int(x.stem.replace("img", "")))
    Q = np.stack([gray_vec(cv2.imread(str(p))) for p in frames])
    print(f"{len(frames)} distinct training frames loaded", flush=True)

    per_source = {}
    for name, vids in SOURCES.items():
        pool = []
        for v in vids:
            if Path(v).exists():
                pool.extend(sample(v))
        if not pool:
            print(f"  {name}: NO POOL"); continue
        P = np.asarray(pool, np.float32)
        per_source[name] = min_mse(Q, P)
        print(f"  {name}: {len(pool)} pool frames -> min-MSE computed", flush=True)

    rows = []
    THRESH = 200  # same extraction criterion as the evaluation-video check
    counts = {k: 0 for k in per_source}; unmatched = 0
    for i, p in enumerate(frames):
        best_src = min(per_source, key=lambda s: per_source[s][i])
        row = {"frame": p.name, "best_source": best_src,
               "min_mse": round(float(per_source[best_src][i]), 1)}
        for s in per_source:
            row[f"mse_{s}"] = round(float(per_source[s][i]), 1)
        if per_source[best_src][i] < THRESH:
            counts[best_src] += 1
        else:
            row["best_source"] = "UNMATCHED"; unmatched += 1
        rows.append(row)

    out = ROOT / "seg_refiner/results/train_frame_sources.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n=== per-capture extraction counts (min pixel-MSE < 200) ===")
    for s, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {s:12s}: {c}")
    print(f"  UNMATCHED   : {unmatched}  (no source below MSE {THRESH})")
    mm = np.array([r["min_mse"] for r in rows])
    print(f"  min-MSE distribution: min={mm.min():.0f} med={np.median(mm):.0f} max={mm.max():.0f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

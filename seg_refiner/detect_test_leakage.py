"""Definitive test-scene leakage detector via dense exact-frame pixel-MSE.

The question this settles is whether the CNN scene-parsing training frames
were extracted from the two rip-current *evaluation* videos (Palm Beach =
DJI_0011, Haulover = DJI_0009). A training frame that is a genuine extraction
from one of those videos will have (near) zero pixel-MSE against some frame of
that video; a frame from a different capture will not. This is the "image-MSE
matching" the manuscript cites -- but done densely and against the correct
targets, so it actually detects extractions instead of forcing every frame into
a test bucket (as global-feature NN or 2-video MSE do).

Method:
  - Densely sample both test videos (every DENSE_STRIDE-th frame), downscale to
    96x54 grayscale.
  - For each of the 237 training frames, compute the minimum raw pixel-MSE to
    each test video (GPU, via ||q||^2+||p||^2-2 q.p). Near-zero -> extraction.
  - Report the min-MSE distribution (bimodal: extractions near 0, non-extractions
    high) and a leakage flag per frame at MSE_THRESH.

Outputs:
  seg_refiner/results/test_leakage_mse.csv
  seg_refiner/results/leakage_audit_lowest.png   (24 lowest-MSE frames + match)
  seg_refiner/results/leakage_audit_highest.png   (12 highest-MSE frames + match)
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
TR = ROOT / "data" / "extracted" / "Rip Current Video" / "TrainDataset_RGB"
RES = ROOT / "seg_refiner" / "results"
OUT_CSV = RES / "test_leakage_mse.csv"

TEST_VIDEOS = {"PalmBeach": ROOT / "data/DJI_0011.MOV",
               "Haulover":  ROOT / "data/DJI_0009.MOV"}
DH, DW = 54, 96
DENSE_STRIDE = 2          # sample every 2nd frame of each test video
MAX_SCAN = 60000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gray_vec(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (DW, DH), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


def sample_dense(path: Path):
    cap = cv2.VideoCapture(str(path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scan = min(n_total, MAX_SCAN)
    vecs, thumbs, tsecs = [], [], []
    idx = 0
    consec_fail = 0
    while idx < scan:
        ret, fr = cap.read()
        if not ret or fr is None:
            consec_fail += 1
            if consec_fail > 60:
                break
            idx += 1
            continue
        consec_fail = 0
        if idx % DENSE_STRIDE == 0:
            vecs.append(gray_vec(fr))
            thumbs.append(cv2.resize(fr, (320, 180)))
            tsecs.append(idx / fps)
        idx += 1
    cap.release()
    print(f"  {path.name}: scanned {idx}/{n_total} -> {len(vecs)} dense samples")
    return np.asarray(vecs, np.float32), thumbs, tsecs


def load_train_frames():
    out = []
    for split in ("train", "val"):
        for p in sorted((TR / split / "img" / "0").glob("*.png"),
                        key=lambda x: int(x.stem.replace("img", ""))):
            out.append((p.name, split, p))
    return out


def min_mse_to_pool(Q, P):
    """Q (N,D), P (M,D) -> for each q, (min_mse, argmin). Uses GPU."""
    Qt = torch.from_numpy(Q).to(DEVICE)
    Pt = torch.from_numpy(P).to(DEVICE)
    q2 = (Qt * Qt).sum(1, keepdim=True)               # (N,1)
    p2 = (Pt * Pt).sum(1, keepdim=True).T              # (1,M)
    best_mse = torch.full((Qt.shape[0],), 1e30, device=DEVICE)
    best_idx = torch.zeros(Qt.shape[0], dtype=torch.long, device=DEVICE)
    D = Q.shape[1]
    for j0 in range(0, Pt.shape[0], 4096):             # chunk to bound memory
        Pc = Pt[j0:j0 + 4096]
        d = q2 + p2[:, j0:j0 + 4096] - 2.0 * (Qt @ Pc.T)   # (N,chunk) sum-sq dist
        d = d / D                                       # -> MSE
        cmin, carg = d.min(1)
        upd = cmin < best_mse
        best_mse = torch.where(upd, cmin, best_mse)
        best_idx = torch.where(upd, carg + j0, best_idx)
    return best_mse.cpu().numpy(), best_idx.cpu().numpy()


def main():
    RES.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}; dense stride={DENSE_STRIDE}")
    pools = {name: sample_dense(p) for name, p in TEST_VIDEOS.items()}

    frames = load_train_frames()
    Q = np.stack([gray_vec(cv2.imread(str(p))) for _, _, p in frames])
    print(f"loaded {len(frames)} training frames")

    per_scene = {}
    for name in TEST_VIDEOS:
        P = pools[name][0]
        mse, arg = min_mse_to_pool(Q, P)
        per_scene[name] = (mse, arg)

    rows = []
    for i, (name, split, path) in enumerate(frames):
        mse_pb, arg_pb = per_scene["PalmBeach"][0][i], per_scene["PalmBeach"][1][i]
        mse_hv, arg_hv = per_scene["Haulover"][0][i], per_scene["Haulover"][1][i]
        if mse_pb <= mse_hv:
            best_test, min_mse, arg = "PalmBeach", mse_pb, arg_pb
        else:
            best_test, min_mse, arg = "Haulover", mse_hv, arg_hv
        rows.append({
            "frame": name, "split": split,
            "min_mse_pb": round(float(mse_pb), 2),
            "min_mse_hv": round(float(mse_hv), 2),
            "min_mse": round(float(min_mse), 2),
            "best_test": best_test,
            "match_time": round(pools[best_test][2][arg], 1),
            "match_arg": int(arg),
        })

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    mm = np.array([r["min_mse"] for r in rows])
    print("\n=== min pixel-MSE to nearest test-video frame (0-255^2 scale) ===")
    for pct in (0, 5, 10, 25, 50, 75, 90, 100):
        print(f"  p{pct:>3}: {np.percentile(mm, pct):8.2f}")
    for thr in (10, 25, 50, 100, 200):
        print(f"  frames with min_mse < {thr:4d}: {(mm < thr).sum():3d}/{len(mm)}")
    print(f"  wrote {OUT_CSV}")

    # audit: 24 lowest-MSE (most likely extractions) + 12 highest (clearly not)
    order = np.argsort(mm)
    def montage(idxs, out):
        tiles = []
        for i in idxs:
            r = rows[i]; _, _, path = frames[i]
            left = cv2.resize(cv2.imread(str(path)), (320, 180))
            right = pools[r["best_test"]][1][r["match_arg"]].copy()
            cv2.putText(left, f"{r['frame']} mse={r['min_mse']:.0f}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(right, f"{r['best_test'][:2]}@{r['match_time']:.0f}s", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            tiles.append(np.hstack([left, np.full((180, 3, 3), 255, np.uint8), right]))
        per_row = 2
        rws = [np.hstack(tiles[k:k + per_row]) for k in range(0, len(tiles), per_row)]
        wmax = max(im.shape[1] for im in rws)
        rws = [np.pad(im, ((0, 0), (0, wmax - im.shape[1]), (0, 0)), constant_values=255) for im in rws]
        cv2.imwrite(str(out), np.vstack(rws))
        print(f"  wrote {out}")
    montage(order[:24], RES / "leakage_audit_lowest.png")
    montage(order[-12:], RES / "leakage_audit_highest.png")


if __name__ == "__main__":
    main()

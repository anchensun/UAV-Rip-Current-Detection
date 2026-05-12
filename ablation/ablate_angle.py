"""Self-contained angular-tolerance ablation for the rip-current pipeline.

For each threshold T in degrees, computes:
  heat_T[x, y] = # of frames where the angle between optical flow direction
                 and the offshore direction at pixel (x, y) is below T.

Uses cv2.calcOpticalFlowFarneback for speed (the original pipeline's HOR
HornSchunck/Typhoon implementations are 10-100x slower; the angular signal
should be similar across dense optical-flow methods, which is what matters
for picking the threshold T).

Run:
    python ablate_angle.py \
        --video "<...>/New Drone Videos/DJI_0006_1.mp4" \
        --mask  "<...>/script/Rip_Current-main/New_Rip_Mask/DJI_0006_1_mask.png" \
        --gt    "<...>/data/extracted/Rip Current Video/RipOutlinesSBL/SBL_GT/DJI_0006_1.png" \
        --out   ablation/results/DJI_0006_1.csv \
        --max-frames 600 --skip 2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


THRESHOLDS_DEFAULT = [15, 30, 45, 60, 75, 90, 105, 120, 150]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--mask", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--skip", type=int, default=2,
                   help="Process every Nth frame (subsample 60fps -> 30fps with skip=2)")
    p.add_argument("--thresholds", type=int, nargs="+",
                   default=THRESHOLDS_DEFAULT)
    p.add_argument("--top-fraction", type=float, default=1.0/3.0,
                   help="Skyline fraction to ignore when building coastline")
    p.add_argument("--min-flow-mag", type=float, default=0.1,
                   help="Skip pixels with optical-flow magnitude below this (px/frame)")
    p.add_argument("--invert-offshore", action="store_true",
                   help="Flip the sign of the offshore direction. Use if heat ends up "
                        "higher OUTSIDE the rip than inside (sign convention check).")
    return p.parse_args()


def load_gt(gt_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load SBL ground truth as a binary mask of the requested (H, W) shape.

    Treats any non-black pixel as a rip-current annotation.
    """
    arr = np.asarray(Image.open(gt_path))
    if arr.ndim == 3:
        gray = arr[..., :3].max(axis=-1)
    else:
        gray = arr
    binary = gray > 127
    if binary.shape != shape:
        rh, rw = shape
        pil = Image.fromarray(binary.astype(np.uint8) * 255).resize((rw, rh), Image.NEAREST)
        binary = np.asarray(pil) > 127
    return binary


def compute_offshore(mask: np.ndarray, top_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build coastline + offshore direction field from a binary sea mask.

    Excludes image-frame edges from the coastline (so the gradient field
    doesn't get pulled toward the left/right/bottom image borders where
    the sea mask happens to extend to the frame).
    """
    H, W = mask.shape
    sea = (mask == 255).astype(np.uint8) * 255
    boundary = cv2.morphologyEx(sea, cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8))
    # Zero out the skyline (top fraction).
    cutoff = int(H * top_fraction)
    boundary[:cutoff] = 0
    # Zero out frame-edge artifacts: left, right, bottom strips.
    edge = 5
    boundary[:, :edge] = 0
    boundary[:, -edge:] = 0
    boundary[-edge:, :] = 0
    if boundary.sum() == 0:
        boundary[-1, :] = 255
    coastline_zero = (boundary == 0).astype(np.uint8)
    dist = cv2.distanceTransform(coastline_zero, cv2.DIST_L2, 5).astype(np.float32)
    off_y, off_x = np.gradient(dist)
    off_mag = np.sqrt(off_x ** 2 + off_y ** 2)
    sea_valid = (mask == 255) & (off_mag > 0.05)
    off_x = off_x / np.maximum(off_mag, 1e-6)
    off_y = off_y / np.maximum(off_mag, 1e-6)
    return off_x, off_y, sea_valid


def compute_auc_pr(heat: np.ndarray, gt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall AUC for an integer heatmap vs. binary ground truth.

    Uses cumulative histograms for O((H*W) + max_heat) instead of O(max_heat * H * W).
    Returns (auc, thresholds, precisions, recalls).
    """
    pos = heat[gt]
    neg = heat[~gt]
    pmax = int(heat.max())
    if pmax == 0 or pos.size == 0:
        return 0.0, np.array([0]), np.array([0.0]), np.array([0.0])
    pos_hist = np.bincount(pos.astype(np.int64), minlength=pmax + 1)
    neg_hist = np.bincount(neg.astype(np.int64), minlength=pmax + 1)
    # Cumulative count of pos/neg with heat >= th, for th = 0..pmax.
    pos_cum_geq = np.cumsum(pos_hist[::-1])[::-1]
    neg_cum_geq = np.cumsum(neg_hist[::-1])[::-1]
    p = pos_cum_geq / np.maximum(pos_cum_geq + neg_cum_geq, 1)
    r = pos_cum_geq / max(int(pos.size), 1)
    thresholds = np.arange(pmax + 1)
    # PR curve typically: as th rises, recall falls. Integrate -dp/dr.
    order = np.argsort(r)
    # numpy 2.x renamed trapz to trapezoid
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(trap(p[order], r[order]))
    return auc, thresholds, p, r


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {args.mask}")
    H, W = mask.shape
    print(f"Mask: {H}x{W}, white frac={(mask==255).mean():.4f}")

    gt = load_gt(args.gt, (H, W))
    print(f"GT positive frac: {gt.mean():.4%}")

    off_x, off_y, sea_valid = compute_offshore(mask, args.top_fraction)
    if args.invert_offshore:
        off_x = -off_x
        off_y = -off_y
        print("(inverted offshore direction)")
    print(f"Sea-valid pixels: {sea_valid.sum()} ({sea_valid.mean():.4%})")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {n_total} frames @ {fps:.1f} fps")

    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Video has no frames")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    n_T = len(args.thresholds)
    heat = np.zeros((n_T, H, W), dtype=np.int32)
    flow_pairs = 0
    frame_idx = 1

    while flow_pairs < args.max_frames:
        # advance `args.skip` frames (so consecutive pairs are skip apart)
        cur = None
        for _ in range(max(args.skip, 1)):
            ret, cur = cap.read()
            if not ret:
                break
            frame_idx += 1
        if not ret or cur is None:
            break
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, cur_gray, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        flow_mag = np.sqrt(flow_x ** 2 + flow_y ** 2)
        # angle between (flow_x, flow_y) and (off_x, off_y) in degrees, [0, 180].
        cos = (flow_x * off_x + flow_y * off_y) / np.maximum(flow_mag, 1e-6)
        cos = np.clip(cos, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos))
        valid = sea_valid & (flow_mag >= args.min_flow_mag)
        for i, T in enumerate(args.thresholds):
            heat[i] += ((angle < T) & valid).astype(np.int32)
        prev_gray = cur_gray
        flow_pairs += 1
        if flow_pairs % 50 == 0:
            print(f"  processed {flow_pairs} flow pairs (frame ~{frame_idx})")
    cap.release()
    print(f"Total flow pairs: {flow_pairs}")

    rows = []
    for i, T in enumerate(args.thresholds):
        auc, _, _, _ = compute_auc_pr(heat[i], gt)
        rows.append({
            "T_deg": T,
            "AUC": round(auc, 4),
            "heat_max": int(heat[i].max()),
            "heat_mean_in_GT": float(heat[i][gt].mean()) if gt.any() else 0.0,
            "heat_mean_in_nonGT": float(heat[i][~gt].mean()),
        })
        print(f"T={T:3d}°  AUC={auc:.4f}  heat_max={heat[i].max():4d}  "
              f"GT_mean={rows[-1]['heat_mean_in_GT']:.2f}  "
              f"nonGT_mean={rows[-1]['heat_mean_in_nonGT']:.2f}")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

"""Run the original HOR HS / HS / LK / HOR LK optical flow pipeline on a clip
and accumulate per-pixel heatmap at multiple angular thresholds T.

Faithful to the manuscript's algorithm: imports OFlib.higher_order, lucas_kanade,
HornSchunck.HS from the original 2020 codebase. Difference from the original
GenDemo.py: stores everything in float32 (no int8 wrap), accumulates heatmaps
for multiple T values in one pass, and uses cv2.distanceTransform for the
offshore direction (mathematically equivalent to but ~10000x faster than the
original Find_Dis_Matrix).

Run:
    python ablation/run_hor_hs_pipeline.py \
        --video "data/extracted/Rip Current Video/New Drone Videos/DJI_0006_3.mp4" \
        --mask  "script/Rip_Current-main/New_Rip_Mask/DJI_0006_3_mask.png" \
        --gt    "data/extracted/Rip Current Video/RipOutlinesSBL/SBL_GT/DJI_0006_3.png" \
        --algo  HOR_HS \
        --out   ablation/results/hor_hs/DJI_0006_3.csv \
        --max-frames 600 --skip 2 --workers 12
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# Make the original 2020 source importable.
SCRIPT_DIR = Path(__file__).resolve().parents[1] / "script" / "Rip_Current-main"
sys.path.insert(0, str(SCRIPT_DIR))
from OFlib import higher_order, lucas_kanade  # noqa: E402
from HornSchunck import HS as HS_baseline  # noqa: E402


THRESHOLDS_DEFAULT = [15, 30, 45, 60, 75, 90, 105, 120, 150]
ALGO_CHOICES = ["HOR_HS", "HOR_LK", "HS", "LK"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--mask", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--algo", required=True, choices=ALGO_CHOICES)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--skip", type=int, default=2)
    p.add_argument("--thresholds", type=int, nargs="+", default=THRESHOLDS_DEFAULT)
    p.add_argument("--top-fraction", type=float, default=1.0/3.0)
    p.add_argument("--min-flow-mag", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--save-heatmaps", action="store_true",
                   help="Save the per-T accumulated heatmap arrays as .npz.")
    p.add_argument("--start-time", type=float, default=0.0,
                   help="Seconds into the video at which to start the 20s evaluation window. "
                        "GT annotations correspond to a specific stable interval, not the whole video.")
    return p.parse_args()


def load_gt(gt_path: Path, shape: tuple[int, int]) -> np.ndarray:
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
    """Compute offshore direction field as gradient of distance-to-coastline.

    Implementation matches the original 2020 OptOffShoreDirection.DetCoLine
    behavior: detect the boundary between sea (white in mask) and non-sea,
    zero out the upper `top_fraction` (skyline), and force the bottom row
    to be coastline so that pixels far above the actual coastline (e.g.
    open ocean toward the horizon) have well-defined offshore direction.
    """
    H, W = mask.shape
    sea = (mask == 255).astype(np.uint8) * 255
    boundary = cv2.morphologyEx(sea, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    # Zero out the skyline (top fraction).
    cutoff_top = int(H * top_fraction)
    boundary[:cutoff_top] = 0
    # Force the bottom row to be coastline. This matches the original
    # DetCoLine in OptOffShoreDirection.py: when the mask doesn't mark the
    # full beach line at the bottom of the frame, treating the bottom edge
    # as coast keeps the offshore direction pointing UP everywhere.
    boundary[-1, :] = 255
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


def of_runner(arg):
    """Top-level helper for multiprocessing — pickles cleanly."""
    g_prev, g_curr, algo = arg
    if algo == "HOR_HS":
        u, v = higher_order(g_prev, g_curr, mode="HS")
    elif algo == "HOR_LK":
        u, v = higher_order(g_prev, g_curr, mode="LK")
    elif algo == "HS":
        u, v = HS_baseline(g_prev, g_curr)
    elif algo == "LK":
        u, v = lucas_kanade(g_prev, g_curr)
    else:
        raise ValueError(algo)
    # Some methods return (10,10) on failure
    if u.shape == (10, 10):
        return None
    # Note: higher_order returns u, v at a coarser resolution than the input!
    # The original pipeline ignores this — heatmap iterates over the OF shape
    # but uses the offshore field at full resolution. We reproduce that by
    # resizing OF up to full frame size.
    # LK can produce NaN/inf or huge values via near-zero det. Clip to a
    # reasonable physical range (max plausible flow ~100 px/frame).
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    u = np.clip(u, -1000.0, 1000.0).astype(np.float32)
    v = np.clip(v, -1000.0, 1000.0).astype(np.float32)
    return u, v


def compute_auc_pr(heat: np.ndarray, gt: np.ndarray) -> float:
    pos = heat[gt]
    neg = heat[~gt]
    pmax = int(heat.max())
    if pmax == 0 or pos.size == 0:
        return 0.0
    pos_hist = np.bincount(pos.astype(np.int64), minlength=pmax + 1)
    neg_hist = np.bincount(neg.astype(np.int64), minlength=pmax + 1)
    pos_cum_geq = np.cumsum(pos_hist[::-1])[::-1]
    neg_cum_geq = np.cumsum(neg_hist[::-1])[::-1]
    p = pos_cum_geq / np.maximum(pos_cum_geq + neg_cum_geq, 1)
    r = pos_cum_geq / max(int(pos.size), 1)
    order = np.argsort(r)
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trap(p[order], r[order]))


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    gt = load_gt(args.gt, (H, W))
    off_x, off_y, sea_valid = compute_offshore(mask, args.top_fraction)
    print(f"Mask {H}x{W}, GT pos frac {gt.mean():.4%}, sea_valid frac {sea_valid.mean():.4%}")

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video {n_total} frames @ {fps:.1f} fps; algo={args.algo} workers={args.workers}")
    if args.start_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_time * 1000)
        print(f"Seeking to t={args.start_time:.1f}s")

    # Read frames into a list of grayscale arrays (subsampled by `skip`).
    # If the video resolution differs from the mask, downsample frames so OF
    # is computed at the mask's resolution (matches manuscript's 1080p assumption).
    vid_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    needs_resize = (vid_H, vid_W) != (H, W)
    if needs_resize:
        print(f"Video {vid_W}x{vid_H} differs from mask {W}x{H}; resizing frames to mask resolution.")
    grays = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % args.skip == 0:
            if needs_resize:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        idx += 1
        if len(grays) > args.max_frames + 1:
            break
    cap.release()
    n_pairs = min(len(grays) - 1, args.max_frames)
    pairs = [(grays[i], grays[i + 1], args.algo) for i in range(n_pairs)]
    print(f"Will process {n_pairs} frame pairs (loaded {len(grays)} grayscale frames)")

    # Compute optical flow in parallel.
    n_T = len(args.thresholds)
    heat = np.zeros((n_T, H, W), dtype=np.int32)
    flow_pairs_done = 0
    t_of = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for result in ex.map(of_runner, pairs, chunksize=4):
            if result is None:
                continue
            u, v = result
            # Resize OF up to full frame if higher_order returned coarser
            if u.shape != (H, W):
                u = cv2.resize(u, (W, H))
                v = cv2.resize(v, (W, H))
            flow_x = u
            flow_y = v
            flow_mag = np.sqrt(flow_x ** 2 + flow_y ** 2)
            cos = (flow_x * off_x + flow_y * off_y) / np.maximum(flow_mag, 1e-6)
            cos = np.clip(cos, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos))
            valid = sea_valid & (flow_mag >= args.min_flow_mag)
            for i, T in enumerate(args.thresholds):
                heat[i] += ((angle < T) & valid).astype(np.int32)
            flow_pairs_done += 1
            if flow_pairs_done % 50 == 0:
                rate = flow_pairs_done / (time.time() - t_of)
                eta = (n_pairs - flow_pairs_done) / max(rate, 1e-3)
                print(f"  {flow_pairs_done}/{n_pairs} pairs done, "
                      f"{rate:.1f} pair/s, ETA {eta:.0f}s")
    of_dt = time.time() - t_of
    print(f"OF done in {of_dt:.1f}s ({flow_pairs_done / of_dt:.2f} pair/s)")

    rows = []
    for i, T in enumerate(args.thresholds):
        auc = compute_auc_pr(heat[i], gt)
        gt_mean = float(heat[i][gt].mean()) if gt.any() else 0.0
        non_mean = float(heat[i][~gt].mean())
        rows.append({
            "T_deg": T, "algo": args.algo, "AUC": round(auc, 4),
            "heat_max": int(heat[i].max()),
            "heat_mean_in_GT": gt_mean, "heat_mean_outside_GT": non_mean,
        })
        print(f"T={T:3d}°  AUC={auc:.4f}  max={heat[i].max():4d}  "
              f"GT_mean={gt_mean:.2f}  nonGT_mean={non_mean:.2f}")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    if args.save_heatmaps:
        npz_path = args.out.with_suffix(".npz")
        np.savez_compressed(npz_path, heat=heat, thresholds=np.array(args.thresholds))
        print(f"Wrote heatmaps {npz_path}")

    meta = {
        "video": str(args.video),
        "mask": str(args.mask),
        "gt": str(args.gt),
        "algo": args.algo,
        "thresholds": args.thresholds,
        "n_pairs": flow_pairs_done,
        "wall_seconds": round(time.time() - t_start, 1),
        "of_seconds": round(of_dt, 1),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.out} ({time.time() - t_start:.1f}s total)")


if __name__ == "__main__":
    main()

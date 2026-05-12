"""Patched OF pipeline with YOLOv8m-seg-as-rip-prior ensemble.

For every kept frame we run a fine-tuned YOLOv8m-seg model (RipDetSeg,
1-class "rip_currents") and collapse its instance masks weighted by
confidence into a single per-pixel rip-prior map P(rip | frame). We then
use this as a multiplicative gate on the OF directional-match indicator
when accumulating the temporal heatmap.

Usage:
    python seg_refiner/run_pipeline_with_yolo.py \
        --video data/DJI_0009.MOV \
        --mask  script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png \
        --gt    script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png \
        --algo  HOR_HS --max-frames 600 --skip 2 --start-time 0 \
        --yolo-ckpt /home/anchensun/Anchen/Research/Rip_Current/CVPR_RipCurrent/runs/segment/outputs/yolov8/e1_yolov8m_seg/weights/best.pt \
        --out   seg_refiner/results/haulover_dji0009_yolo/HOR_HS.csv
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script" / "Rip_Current-main"
sys.path.insert(0, str(SCRIPT_DIR))
from OFlib import higher_order, lucas_kanade  # noqa: E402
from HornSchunck import HS as HS_baseline    # noqa: E402

ALGO_CHOICES = ["HOR_HS", "HOR_LK", "HS", "LK"]
THRESHOLDS = [15, 30, 45, 60, 75, 90, 105, 120, 150]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--mask", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--algo", required=True, choices=ALGO_CHOICES)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--yolo-ckpt", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--skip", type=int, default=2)
    p.add_argument("--start-time", type=float, default=0.0)
    p.add_argument("--top-fraction", type=float, default=1.0/3.0)
    p.add_argument("--min-flow-mag", type=float, default=0.05)
    p.add_argument("--yolo-conf-min", type=float, default=0.05,
                   help="Confidence floor for YOLOv8 detections to count toward the prior.")
    p.add_argument("--yolo-imgsz", type=int, default=1024,
                   help="YOLOv8 inference image size (square).")
    p.add_argument("--ensemble", choices=["mul", "addnorm"], default="mul",
                   help="mul = OF * yolo_prior; addnorm = (OF/maxOF) + (yolo/maxyolo).")
    return p.parse_args()


def load_gt(gt_path, shape):
    arr = np.asarray(Image.open(gt_path))
    if arr.ndim == 3: gray = arr[..., :3].max(axis=-1)
    else: gray = arr
    binary = gray > 127
    if binary.shape != shape:
        rh, rw = shape
        pil = Image.fromarray(binary.astype(np.uint8) * 255).resize((rw, rh), Image.NEAREST)
        binary = np.asarray(pil) > 127
    return binary


def compute_offshore(mask, top_fraction):
    H, W = mask.shape
    sea = (mask == 255).astype(np.uint8) * 255
    boundary = cv2.morphologyEx(sea, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    cutoff_top = int(H * top_fraction)
    boundary[:cutoff_top] = 0
    boundary[-1, :] = 255
    if boundary.sum() == 0:
        boundary[-1, :] = 255
    coastline_zero = (boundary == 0).astype(np.uint8)
    dist = cv2.distanceTransform(coastline_zero, cv2.DIST_L2, 5).astype(np.float32)
    off_y, off_x = np.gradient(dist)
    off_mag = np.sqrt(off_x**2 + off_y**2)
    sea_valid = (mask == 255) & (off_mag > 0.05)
    off_x = off_x / np.maximum(off_mag, 1e-6)
    off_y = off_y / np.maximum(off_mag, 1e-6)
    return off_x, off_y, sea_valid


def of_one(g_prev, g_curr, algo):
    if algo == "HOR_HS": u, v = higher_order(g_prev, g_curr, mode="HS")
    elif algo == "HOR_LK": u, v = higher_order(g_prev, g_curr, mode="LK")
    elif algo == "HS": u, v = HS_baseline(g_prev, g_curr)
    elif algo == "LK": u, v = lucas_kanade(g_prev, g_curr)
    else: raise ValueError(algo)
    if u.shape == (10, 10): return None
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(u, -1000, 1000).astype(np.float32), np.clip(v, -1000, 1000).astype(np.float32)


def compute_auc_pr(heat, gt, n_bins=1024):
    pos = heat[gt]; neg = heat[~gt]
    if pos.size == 0 or pos.max() <= 0: return 0.0
    hi = float(heat.max())
    if hi <= 0: return 0.0
    bins = np.linspace(0, hi, n_bins + 1)
    pos_h, _ = np.histogram(pos, bins=bins)
    neg_h, _ = np.histogram(neg, bins=bins)
    pos_cum = np.cumsum(pos_h[::-1])[::-1]
    neg_cum = np.cumsum(neg_h[::-1])[::-1]
    p = pos_cum / np.maximum(pos_cum + neg_cum, 1)
    r = pos_cum / max(int(pos.size), 1)
    order = np.argsort(r)
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trap(p[order], r[order]))


def yolo_prior_one(model, frame_bgr, imgsz, conf_min):
    """Return per-pixel rip-prior float32 (H, W) in [0,1] for a single frame.
    For each detected instance, splat its mask weighted by confidence; take the
    pixel-wise max across instances."""
    H, W = frame_bgr.shape[:2]
    res = model.predict(frame_bgr, imgsz=imgsz, conf=conf_min, verbose=False)
    out = np.zeros((H, W), dtype=np.float32)
    if not res: return out
    r = res[0]
    if r.masks is None or r.boxes is None or r.masks.data is None:
        return out
    masks = r.masks.data.detach().cpu().numpy()  # (N, h, w)
    confs = r.boxes.conf.detach().cpu().numpy() if r.boxes.conf is not None else np.ones(masks.shape[0])
    if masks.size == 0: return out
    for m, c in zip(masks, confs):
        if c < conf_min: continue
        if m.shape != (H, W):
            m_full = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            m_full = m.astype(np.float32)
        out = np.maximum(out, m_full * float(c))
    return out


def main():
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
    print(f"Video {n_total} frames @ {fps:.1f} fps; algo={args.algo}")
    if args.start_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_time * 1000)
    vid_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    needs_resize = (vid_H, vid_W) != (H, W)

    # Pass 1: sample grayscale frames
    grays = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % args.skip == 0:
            if needs_resize:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        idx += 1
        if len(grays) > args.max_frames + 1: break
    cap.release()
    n_pairs = min(len(grays) - 1, args.max_frames)
    print(f"Loaded {len(grays)} grayscale frames; will process {n_pairs} pairs")

    # Load YOLO
    from ultralytics import YOLO
    print(f"Loading YOLOv8 from {args.yolo_ckpt}")
    yolo = YOLO(str(args.yolo_ckpt))

    # Pass 2: stream frames + run yolo + accumulate
    cap = cv2.VideoCapture(str(args.video))
    if args.start_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_time * 1000)

    n_T = len(THRESHOLDS)
    heat = np.zeros((n_T, H, W), dtype=np.float32)
    of_heat_only = np.zeros((n_T, H, W), dtype=np.float32)  # for ensemble add-norm
    yolo_prior_acc = np.zeros((H, W), dtype=np.float32)
    yolo_prior_max = np.zeros((H, W), dtype=np.float32)
    sea_valid_f = sea_valid.astype(np.float32)

    print(f"Streaming OF + YOLO on {n_pairs} pairs (ensemble={args.ensemble})...")
    t_loop = time.time()
    n_processed = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % args.skip == 0:
            kept_idx = idx // args.skip
            if kept_idx >= 1:
                pair = kept_idx - 1
                if pair >= n_pairs: break
                if needs_resize:
                    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                rip_prior = yolo_prior_one(yolo, frame, args.yolo_imgsz, args.yolo_conf_min)
                yolo_prior_acc += rip_prior
                yolo_prior_max = np.maximum(yolo_prior_max, rip_prior)

                result = of_one(grays[pair], grays[pair+1], args.algo)
                if result is not None:
                    u, v = result
                    if u.shape != (H, W):
                        u = cv2.resize(u, (W, H)); v = cv2.resize(v, (W, H))
                    flow_mag = np.sqrt(u*u + v*v)
                    cos = (u * off_x + v * off_y) / np.maximum(flow_mag, 1e-6)
                    cos = np.clip(cos, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos))
                    valid = sea_valid_f * (flow_mag >= args.min_flow_mag).astype(np.float32)

                    if args.ensemble == "mul":
                        w = rip_prior * valid
                        for j, T in enumerate(THRESHOLDS):
                            heat[j] += ((angle < T).astype(np.float32) * w)
                    else:  # addnorm: track OF heat separately, combine at end
                        for j, T in enumerate(THRESHOLDS):
                            of_heat_only[j] += ((angle < T).astype(np.float32) * valid)
                    n_processed += 1
                if (pair + 1) % 50 == 0:
                    rate = (pair + 1) / (time.time() - t_loop)
                    eta = (n_pairs - pair - 1) / max(rate, 1e-3)
                    print(f"  {pair+1}/{n_pairs} pairs, {rate:.2f}/s, ETA {eta:.0f}s")
        idx += 1
    cap.release()
    print(f"Loop done in {time.time()-t_loop:.1f}s; processed {n_processed} pairs")

    if args.ensemble == "addnorm":
        # Ensemble: normalize each modality, sum
        yolo_norm = yolo_prior_acc / max(float(yolo_prior_acc.max()), 1e-6)
        for j in range(n_T):
            of_n = of_heat_only[j] / max(float(of_heat_only[j].max()), 1e-6)
            heat[j] = of_n + yolo_norm * sea_valid_f

    rows = []
    for i, T in enumerate(THRESHOLDS):
        auc = compute_auc_pr(heat[i], gt)
        gt_mean = float(heat[i][gt].mean()) if gt.any() else 0.0
        non_mean = float(heat[i][~gt].mean())
        rows.append({"T_deg": T, "algo": args.algo, "AUC": round(auc, 4),
                     "heat_max": float(heat[i].max()),
                     "heat_mean_in_GT": gt_mean, "heat_mean_outside_GT": non_mean})
        print(f"T={T:3d}°  AUC={auc:.4f}  max={heat[i].max():.2f}  "
              f"GT_mean={gt_mean:.2f}  nonGT_mean={non_mean:.2f}")

    # Also report YOLOv8-only AUC on the time-averaged prior for context
    yolo_avg = yolo_prior_acc / max(n_processed, 1)
    yolo_only_auc = compute_auc_pr(yolo_avg * sea_valid_f, gt)
    print(f"\nYOLO-only (time-averaged, masked to sea): AUC={yolo_only_auc:.4f}")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    meta = {"video": str(args.video), "mask": str(args.mask), "gt": str(args.gt),
            "algo": args.algo, "yolo_ckpt": str(args.yolo_ckpt),
            "ensemble": args.ensemble, "yolo_conf_min": args.yolo_conf_min,
            "yolo_imgsz": args.yolo_imgsz,
            "yolo_only_auc_avg": round(yolo_only_auc, 4),
            "n_pairs": n_processed, "wall_seconds": round(time.time() - t_start, 1)}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.out} ({time.time()-t_start:.1f}s total)")


if __name__ == "__main__":
    main()

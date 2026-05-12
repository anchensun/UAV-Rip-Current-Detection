"""Run E1's trained YOLOv8m-seg frame-by-frame on DJI clips and compute
per-clip rip-detection metrics against SBL_GT.

For each clip:
  - Sample frames at a fixed step (default every 30 frames at 60 fps -> 2 fps).
  - Run YOLOv8 segmentation inference on each sampled frame.
  - Build a per-pixel frame-level confidence map: max(confidence) over all
    predicted rip instances at that pixel; 0 if no instance covers it.
  - Aggregate across frames into a single rip-likelihood map per clip
    (mean confidence is the default; --agg max or --agg count also supported).
  - Compute AUC vs SBL_GT using precision-recall sweep, plus IoU at the
    threshold that maximizes F1.

Run:
    python ablation/run_yolo_on_clips.py \
        --weights .../e1_yolov8m_seg/weights/best.pt \
        --gt-dir  data/extracted/Rip\\ Current\\ Video/RipOutlinesSBL/SBL_GT \
        --video-dir data/extracted/Rip\\ Current\\ Video/New\\ Drone\\ Videos \
        --clips DJI_0006_1 DJI_0006_3 DJI_0006_8 DJI_0007_3 DJI_0007_5 DJI_0007_8 \
        --out ablation/results/yolov8/per_clip.csv \
        --frame-step 30 --agg mean
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--video-dir", type=Path, required=True)
    p.add_argument("--clips", type=str, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frame-step", type=int, default=30,
                   help="Sample 1 frame every N (60fps source / step = effective fps)")
    p.add_argument("--agg", choices=["mean", "max", "count"], default="mean")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5,
                   help="NMS IoU threshold inside YOLO (not the eval threshold)")
    p.add_argument("--device", default="0")
    p.add_argument("--save-maps", action="store_true",
                   help="Save the per-clip aggregated rip-likelihood map as .npy")
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


def per_frame_pixel_map(model, frame_bgr: np.ndarray, *, imgsz: int, conf: float,
                        iou: float, device: str) -> np.ndarray:
    """Run YOLOv8 on one frame; return (H, W) float32 confidence map.

    Each pixel = max confidence of any predicted rip instance covering it; 0 elsewhere.
    """
    H, W = frame_bgr.shape[:2]
    out = np.zeros((H, W), dtype=np.float32)
    res = model.predict(source=frame_bgr, imgsz=imgsz, conf=conf, iou=iou,
                        device=device, verbose=False, retina_masks=True)
    if not res:
        return out
    r = res[0]
    if r.masks is None or len(r.masks) == 0:
        return out
    masks = r.masks.data.cpu().numpy().astype(bool)  # [N, mH, mW]
    scores = r.boxes.conf.cpu().numpy().astype(np.float32)
    # retina_masks=True returns at orig image resolution
    for i in range(masks.shape[0]):
        m = masks[i]
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        np.maximum(out, m * scores[i], out=out)
    return out


def auc_pr(score: np.ndarray, gt: np.ndarray, n_thresh: int = 200) -> tuple[float, float, float]:
    """Compute PR-AUC and best-F1 IoU using a fixed-grid threshold sweep.

    Returns (auc, best_iou, best_thresh).
    """
    s = score.ravel()
    g = gt.ravel()
    pos_total = int(g.sum())
    if pos_total == 0:
        return 0.0, 0.0, 0.0
    smin, smax = float(s.min()), float(s.max())
    if smax <= smin:
        return 0.0, 0.0, 0.0
    thresholds = np.linspace(smax, smin, n_thresh)
    tp = fp = 0
    p_arr = []
    r_arr = []
    iou_arr = []
    sorted_idx = np.argsort(-s)
    s_sorted = s[sorted_idx]
    g_sorted = g[sorted_idx].astype(np.int64)
    cum_tp = np.cumsum(g_sorted)
    cum_fp = np.cumsum(1 - g_sorted)
    for t_idx, t in enumerate(thresholds):
        # find largest k where s_sorted[k] >= t
        k = int(np.searchsorted(-s_sorted, -t, side="right"))
        if k == 0:
            continue
        tp = cum_tp[k - 1]
        fp = cum_fp[k - 1]
        fn = pos_total - tp
        p = tp / max(tp + fp, 1)
        r = tp / pos_total
        iou = tp / max(tp + fp + fn, 1)
        p_arr.append(p); r_arr.append(r); iou_arr.append((iou, t))
    if not p_arr:
        return 0.0, 0.0, 0.0
    p_arr = np.asarray(p_arr); r_arr = np.asarray(r_arr)
    order = np.argsort(r_arr)
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(trap(p_arr[order], r_arr[order]))
    best_iou, best_t = max(iou_arr, key=lambda x: x[0])
    return auc, float(best_iou), float(best_t)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    rows = []
    for clip in args.clips:
        video_path = args.video_dir / f"{clip}.mp4"
        gt_path = args.gt_dir / f"{clip}.png"
        cap = cv2.VideoCapture(str(video_path))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        gt = load_gt(gt_path, (H, W))
        print(f"\n=== {clip}: {n_total} frames, {H}x{W}, GT {gt.mean():.4%}")

        agg_map = np.zeros((H, W), dtype=np.float32)
        n_processed = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % args.frame_step == 0:
                pixmap = per_frame_pixel_map(model, frame,
                                             imgsz=args.imgsz, conf=args.conf,
                                             iou=args.iou, device=args.device)
                if args.agg == "mean":
                    agg_map = (agg_map * n_processed + pixmap) / (n_processed + 1)
                elif args.agg == "max":
                    np.maximum(agg_map, pixmap, out=agg_map)
                elif args.agg == "count":
                    agg_map += (pixmap > args.conf).astype(np.float32)
                n_processed += 1
            frame_idx += 1
        cap.release()

        auc, best_iou, best_t = auc_pr(agg_map, gt)
        rows.append({
            "clip": clip,
            "n_frames_inferred": n_processed,
            "agg": args.agg,
            "AUC_PR": round(auc, 4),
            "best_IoU": round(best_iou, 4),
            "best_threshold": round(best_t, 4),
            "GT_pos_frac": round(float(gt.mean()), 6),
            "agg_min": round(float(agg_map.min()), 4),
            "agg_max": round(float(agg_map.max()), 4),
            "agg_mean_in_GT": round(float(agg_map[gt].mean()) if gt.any() else 0.0, 4),
            "agg_mean_outside_GT": round(float(agg_map[~gt].mean()), 4),
        })
        print(f"  AUC_PR={auc:.4f}  best_IoU={best_iou:.4f} @ t={best_t:.3f}  "
              f"agg_max={agg_map.max():.3f}  GT_mean={float(agg_map[gt].mean()) if gt.any() else 0:.3f}")
        if args.save_maps:
            np.save(args.out.parent / f"{clip}_agg_{args.agg}.npy", agg_map)

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    aucs = [r["AUC_PR"] for r in rows]
    print(f"\nWrote {args.out} ({len(rows)} clips)")
    print(f"AUC_PR: mean={np.mean(aucs):.4f} median={np.median(aucs):.4f} "
          f"min={min(aucs):.4f} max={max(aucs):.4f}")


if __name__ == "__main__":
    main()

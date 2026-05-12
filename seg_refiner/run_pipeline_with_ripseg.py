"""Patched OF pipeline gated by a BINARY rip-segmentation U-Net trained
on RipDetSeg (NOT the 6-class scene-parser used by run_pipeline_with_seg).

The U-Net outputs a per-pixel sigmoid rip probability in [0,1]. We use it
as a multiplicative gate on the OF directional-agreement indicator
(same accumulator formula as run_pipeline_with_seg.py).

Streaming inference: one frame at a time, peak RAM < 4 GB.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script" / "Rip_Current-main"
sys.path.insert(0, str(SCRIPT_DIR))
from OFlib import higher_order, lucas_kanade
from HornSchunck import HS as HS_baseline

ALGO_CHOICES = ["HOR_HS", "HOR_LK", "HS", "LK"]
THRESHOLDS = [15, 30, 45, 60, 75, 90, 105, 120, 150]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--mask", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--algo", required=True, choices=ALGO_CHOICES)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ripseg-ckpt", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--skip", type=int, default=2)
    p.add_argument("--start-time", type=float, default=0.0)
    p.add_argument("--top-fraction", type=float, default=1.0/3.0)
    p.add_argument("--min-flow-mag", type=float, default=0.05)
    p.add_argument("--prior-power", type=float, default=1.0,
                   help="Exponent applied to the rip prior P(rip)^power for gating")
    p.add_argument("--prior-floor", type=float, default=0.0,
                   help="Minimum gate value (helps avoid zeroing all signal)")
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
    boundary = cv2.morphologyEx(sea, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    cutoff_top = int(H * top_fraction); boundary[:cutoff_top] = 0
    boundary[-1, :] = 255
    if boundary.sum() == 0: boundary[-1, :] = 255
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
    elif algo == "HS":   u, v = HS_baseline(g_prev, g_curr)
    elif algo == "LK":   u, v = lucas_kanade(g_prev, g_curr)
    else: raise ValueError(algo)
    if u.shape == (10, 10): return None
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(u, -1000, 1000).astype(np.float32), np.clip(v, -1000, 1000).astype(np.float32)


def compute_auc_pr(heat, gt, n_bins=1024):
    pos = heat[gt]; neg = heat[~gt]
    hi = float(heat.max())
    if pos.size == 0 or hi <= 0: return 0.0
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


_NORM_MEAN = None; _NORM_STD = None
@torch.no_grad()
def ripseg_one(model, frame_bgr, device):
    global _NORM_MEAN, _NORM_STD
    if _NORM_MEAN is None:
        _NORM_MEAN = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
        _NORM_STD  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    H, W = frame_bgr.shape[:2]
    rgb = frame_bgr[..., ::-1].astype(np.float32)/255.0
    t = torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to(device, non_blocking=True)
    t = (t - _NORM_MEAN) / _NORM_STD
    pad_h = (32-H%32)%32; pad_w = (32-W%32)%32
    if pad_h or pad_w: t = torch.nn.functional.pad(t, (0,pad_w,0,pad_h))
    logits = model(t)
    if pad_h or pad_w: logits = logits[..., :H, :W]
    return torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)


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
    print(f"Video {n_total} frames @ {fps:.1f} fps; algo={args.algo}; ripseg={args.ripseg_ckpt}")
    if args.start_time > 0: cap.set(cv2.CAP_PROP_POS_MSEC, args.start_time * 1000)
    vid_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    needs_resize = (vid_H, vid_W) != (H, W)

    grays = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % args.skip == 0:
            if needs_resize: frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        idx += 1
        if len(grays) > args.max_frames + 1: break
    cap.release()
    n_pairs = min(len(grays) - 1, args.max_frames)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ripseg_ckpt, map_location=device, weights_only=False)
    model = smp.Unet(encoder_name=ckpt.get("encoder", "resnet34"),
                      encoder_weights=None, in_channels=3, classes=1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cap = cv2.VideoCapture(str(args.video))
    if args.start_time > 0: cap.set(cv2.CAP_PROP_POS_MSEC, args.start_time * 1000)
    n_T = len(THRESHOLDS)
    heat = np.zeros((n_T, H, W), dtype=np.float32)
    rip_acc = np.zeros((H, W), dtype=np.float32)
    sea_valid_f = sea_valid.astype(np.float32)
    n_proc = 0
    t_loop = time.time()
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % args.skip == 0:
            kept = idx // args.skip
            if kept >= 1:
                pair = kept - 1
                if pair >= n_pairs: break
                if needs_resize: frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                rip = ripseg_one(model, frame, device)
                rip_acc += rip
                gate = np.power(np.maximum(rip, args.prior_floor), args.prior_power)
                result = of_one(grays[pair], grays[pair+1], args.algo)
                if result is not None:
                    u, v = result
                    if u.shape != (H, W):
                        u = cv2.resize(u, (W, H)); v = cv2.resize(v, (W, H))
                    flow_mag = np.sqrt(u*u + v*v)
                    cos = (u * off_x + v * off_y) / np.maximum(flow_mag, 1e-6)
                    cos = np.clip(cos, -1, 1)
                    angle = np.degrees(np.arccos(cos))
                    valid = sea_valid_f * (flow_mag >= args.min_flow_mag).astype(np.float32)
                    w = gate * valid
                    for j, T in enumerate(THRESHOLDS):
                        heat[j] += ((angle < T).astype(np.float32) * w)
                    n_proc += 1
                if (pair + 1) % 100 == 0:
                    rate = (pair + 1) / (time.time() - t_loop)
                    eta = (n_pairs - pair - 1) / max(rate, 1e-3)
                    print(f"  {pair+1}/{n_pairs} pairs, {rate:.2f}/s, ETA {eta:.0f}s")
        idx += 1
    cap.release()

    rows = []
    for i, T in enumerate(THRESHOLDS):
        auc = compute_auc_pr(heat[i], gt)
        rows.append({"T_deg": T, "algo": args.algo, "AUC": round(auc, 4),
                     "heat_max": float(heat[i].max())})
        print(f"T={T:3d}°  AUC={auc:.4f}  max={heat[i].max():.2f}")

    rip_avg = rip_acc / max(n_proc, 1)
    rip_only_auc = compute_auc_pr(rip_avg * sea_valid_f, gt)
    print(f"\nRip-prior-only (time-averaged, masked): AUC={rip_only_auc:.4f}")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    meta = {"video": str(args.video), "algo": args.algo,
            "ripseg_ckpt": str(args.ripseg_ckpt),
            "prior_power": args.prior_power, "prior_floor": args.prior_floor,
            "rip_only_auc_avg": round(rip_only_auc, 4),
            "n_pairs": n_proc, "wall_seconds": round(time.time() - t_start, 1)}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.out} ({time.time()-t_start:.1f}s total)")


if __name__ == "__main__":
    main()

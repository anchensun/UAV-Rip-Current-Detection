"""Patched OF pipeline + U-Net sea-gate + sphere/planar projection.

Same as run_pipeline_with_seg.py but the angle metric is computed in
GROUND space (after applying the manuscript's sphere/planar projection
to both the optical-flow vector and the offshore-direction field). A
projection mode of "none" disables the projection and reproduces the
exact behavior of run_pipeline_with_seg.py.

Usage:
    python seg_refiner/run_pipeline_with_seg_sphere.py \
        --video data/DJI_0011.MOV \
        --mask  script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png \
        --gt    "data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png" \
        --algo  HOR_HS --max-frames 600 --skip 2 --start-time 0 \
        --gate-mode sea \
        --projection sphere --camera-altitude 60 --camera-tilt 5 \
        --out seg_refiner/results/palmbeach_seasphere/HOR_HS.csv
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
sys.path.insert(0, str(ROOT / "seg_refiner"))
from OFlib import higher_order, lucas_kanade        # noqa: E402
from HornSchunck import HS as HS_baseline           # noqa: E402
from sphere_projection import (                       # noqa: E402
    build_pixel_to_ground, build_local_jacobian, focal_pix_default
)

ALGO_CHOICES = ["HOR_HS", "HOR_LK", "HS", "LK"]
THRESHOLDS = [15, 30, 45, 60, 75, 90, 105, 120, 150]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--mask", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--algo", required=True, choices=ALGO_CHOICES)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ckpt", type=Path,
                   default=ROOT / "seg_refiner" / "checkpoints" / "best.pt")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--skip", type=int, default=2)
    p.add_argument("--start-time", type=float, default=0.0)
    p.add_argument("--top-fraction", type=float, default=1.0/3.0)
    p.add_argument("--min-flow-mag", type=float, default=0.05)
    p.add_argument("--gate-mode", choices=["sea", "sea_x_nonfoam", "argmax", "none"],
                   default="sea")
    p.add_argument("--projection", choices=["none", "sphere", "planar"], default="none")
    p.add_argument("--camera-altitude", type=float, default=60.0,
                   help="UAV altitude above sea level in meters.")
    p.add_argument("--camera-tilt-psi", type=float, default=5.0,
                   help="Camera boresight tilt from nadir, degrees.")
    p.add_argument("--camera-tilt-theta", type=float, default=0.0,
                   help="Camera in-plane rotation, degrees.")
    p.add_argument("--camera-focal-mm", type=float, default=4.0)
    p.add_argument("--camera-sensor-mm", type=float, default=6.16)
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
    elif algo == "HS":   u, v = HS_baseline(g_prev, g_curr)
    elif algo == "LK":   u, v = lucas_kanade(g_prev, g_curr)
    else: raise ValueError(algo)
    if u.shape == (10, 10): return None
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(u, -1000, 1000).astype(np.float32), np.clip(v, -1000, 1000).astype(np.float32)


def compute_auc_pr(heat, gt, n_bins=1024):
    pos = heat[gt]; neg = heat[~gt]
    if pos.size == 0: return 0.0
    hi = float(heat.max());
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


def load_seg_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = smp.Unet(encoder_name=ckpt.get("encoder", "resnet18"),
                      encoder_weights=None, in_channels=3, classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval(); return model


_NORM_MEAN = None; _NORM_STD = None
@torch.no_grad()
def seg_gate_one(model, frame_bgr, device, gate_mode):
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
    sm = torch.softmax(logits, dim=1)[0]
    if gate_mode == "sea": gate = sm[2]
    elif gate_mode == "sea_x_nonfoam": gate = sm[2] * (1.0 - sm[5])
    elif gate_mode == "argmax": gate = (sm.argmax(dim=0) == 2).float()
    elif gate_mode == "none": gate = torch.ones_like(sm[2])
    else: raise ValueError(gate_mode)
    return gate.cpu().numpy().astype(np.float32)


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    gt = load_gt(args.gt, (H, W))
    off_x_pix, off_y_pix, sea_valid = compute_offshore(mask, args.top_fraction)
    print(f"Mask {H}x{W}, GT pos frac {gt.mean():.4%}, sea_valid frac {sea_valid.mean():.4%}")

    # Build the pixel->ground projection Jacobian if requested.
    if args.projection != "none":
        focal_pix = focal_pix_default(args.camera_focal_mm, args.camera_sensor_mm, W)
        Xg, Yg, valid = build_pixel_to_ground(
            H, W, focal_pix, h=args.camera_altitude,
            theta_deg=args.camera_tilt_theta, psi_deg=args.camera_tilt_psi,
            mode=args.projection)
        J = build_local_jacobian(Xg, Yg)
        # Update sea_valid: drop pixels where the projection is undefined.
        sea_valid = sea_valid & valid
        print(f"Projection {args.projection}: focal_pix={focal_pix:.0f}, "
              f"h={args.camera_altitude}m, psi={args.camera_tilt_psi}°. "
              f"sea_valid after projection: {sea_valid.mean():.4%}")
        # Pre-project the offshore-direction field (it's a constant field)
        # off in pixel-space convention: off_x_pix corresponds to du, off_y_pix to dv-with-image-down convention
        # We need to flip sign on off_y to get screen-up convention before applying J
        off_screen_u = off_x_pix
        off_screen_v = -off_y_pix
        off_gx = J[..., 0, 0] * off_screen_u + J[..., 0, 1] * off_screen_v
        off_gy = J[..., 1, 0] * off_screen_u + J[..., 1, 1] * off_screen_v
        off_gmag = np.sqrt(off_gx*off_gx + off_gy*off_gy)
        off_gx_n = off_gx / np.maximum(off_gmag, 1e-9)
        off_gy_n = off_gy / np.maximum(off_gmag, 1e-9)
    else:
        J = None; off_gx_n = None; off_gy_n = None

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video {n_total} frames @ {fps:.1f} fps; algo={args.algo}; projection={args.projection}; gate={args.gate_mode}")
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
    print(f"Loaded {len(grays)} frames; will process {n_pairs} pairs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading seg model from {args.ckpt} on {device}")
    seg_model = load_seg_model(args.ckpt, device)

    cap = cv2.VideoCapture(str(args.video))
    if args.start_time > 0: cap.set(cv2.CAP_PROP_POS_MSEC, args.start_time * 1000)

    n_T = len(THRESHOLDS)
    heat = np.zeros((n_T, H, W), dtype=np.float32)
    sea_valid_f = sea_valid.astype(np.float32)
    t_of = time.time()
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
                gate = seg_gate_one(seg_model, frame, device, args.gate_mode)
                result = of_one(grays[pair], grays[pair+1], args.algo)
                if result is not None:
                    u, v = result
                    if u.shape != (H, W):
                        u = cv2.resize(u, (W, H)); v = cv2.resize(v, (W, H))
                    flow_mag_pix = np.sqrt(u*u + v*v)
                    if args.projection == "none":
                        cos = (u * off_x_pix + v * off_y_pix) / np.maximum(flow_mag_pix, 1e-6)
                        cos = np.clip(cos, -1.0, 1.0)
                        flow_mag_for_thresh = flow_mag_pix
                    else:
                        # Project flow to ground and use ground-space angle
                        fu = u; fv = -v  # flip to screen-up convention
                        g_fx = J[..., 0, 0] * fu + J[..., 0, 1] * fv
                        g_fy = J[..., 1, 0] * fu + J[..., 1, 1] * fv
                        g_fmag = np.sqrt(g_fx*g_fx + g_fy*g_fy)
                        cos = (g_fx * off_gx_n + g_fy * off_gy_n) / np.maximum(g_fmag, 1e-9)
                        cos = np.clip(cos, -1.0, 1.0)
                        flow_mag_for_thresh = flow_mag_pix  # threshold the source pixel-flow magnitude
                    angle = np.degrees(np.arccos(cos))
                    valid_pix = sea_valid_f * (flow_mag_for_thresh >= args.min_flow_mag).astype(np.float32)
                    w = gate * valid_pix
                    for j, T in enumerate(THRESHOLDS):
                        heat[j] += ((angle < T).astype(np.float32) * w)
                if (pair + 1) % 100 == 0:
                    rate = (pair + 1) / (time.time() - t_of)
                    eta = (n_pairs - pair - 1) / max(rate, 1e-3)
                    print(f"  {pair+1}/{n_pairs} pairs, {rate:.2f}/s, ETA {eta:.0f}s")
        idx += 1
    cap.release()
    print(f"Loop done in {time.time()-t_of:.1f}s")

    rows = []
    for i, T in enumerate(THRESHOLDS):
        auc = compute_auc_pr(heat[i], gt)
        rows.append({"T_deg": T, "algo": args.algo, "AUC": round(auc, 4),
                     "heat_max": float(heat[i].max())})
        print(f"T={T:3d}°  AUC={auc:.4f}  max={heat[i].max():.2f}")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    meta = {"video": str(args.video), "mask": str(args.mask), "gt": str(args.gt),
            "algo": args.algo, "ckpt": str(args.ckpt), "gate_mode": args.gate_mode,
            "projection": args.projection,
            "altitude": args.camera_altitude,
            "tilt_psi": args.camera_tilt_psi,
            "tilt_theta": args.camera_tilt_theta,
            "wall_seconds": round(time.time() - t_start, 1)}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.out} ({time.time()-t_start:.1f}s total)")


if __name__ == "__main__":
    main()

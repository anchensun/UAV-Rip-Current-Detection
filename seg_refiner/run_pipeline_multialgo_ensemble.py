"""Stage C: produce 4-algo ensemble heatmaps.

Runs all 4 OF algorithms on the same frames, applies the shared 6-class
U-Net sea-gate, accumulates per-algo heatmaps, then reports AUC for:
  * each algo individually (re-confirms numbers)
  * mean ensemble: H_mean[T] = mean over algos of (H_algo[T] / H_algo[T].max())
  * max ensemble:  H_max[T]  = elementwise max of normalized H_algo[T]
  * geomean ensemble: H_gm[T] = (prod over algos of H_algo[T]/max)^(1/4)

We re-use grays + segmenter inference across algos (compute the gate ONCE
per frame), so this is faster than running 4 separate scripts.
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

THRESHOLDS = [15, 30, 45, 60, 75, 90, 105, 120, 150]
ALGOS = ["HOR_HS", "HOR_LK", "HS", "LK"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--mask", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--ckpt", type=Path,
                   default=ROOT / "seg_refiner" / "checkpoints" / "best.pt")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--skip", type=int, default=2)
    p.add_argument("--start-time", type=float, default=0.0)
    p.add_argument("--top-fraction", type=float, default=1.0/3.0)
    p.add_argument("--min-flow-mag", type=float, default=0.05)
    p.add_argument("--gate-mode", default="sea")
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
    pos_cum = np.cumsum(pos_h[::-1])[::-1]; neg_cum = np.cumsum(neg_h[::-1])[::-1]
    p = pos_cum / np.maximum(pos_cum + neg_cum, 1)
    r = pos_cum / max(int(pos.size), 1)
    order = np.argsort(r)
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trap(p[order], r[order]))


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
    else: raise ValueError(gate_mode)
    return gate.cpu().numpy().astype(np.float32)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    gt = load_gt(args.gt, (H, W))
    off_x, off_y, sea_valid = compute_offshore(mask, args.top_fraction)
    print(f"Mask {H}x{W}; GT pos frac {gt.mean():.3%}; sea_valid frac {sea_valid.mean():.3%}")

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.start_time > 0: cap.set(cv2.CAP_PROP_POS_MSEC, args.start_time * 1000)
    vid_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vid_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    needs_resize = (vid_H, vid_W) != (H, W)

    grays = []; idx = 0
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
    print(f"Loaded {len(grays)} grayscale frames; {n_pairs} pairs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    seg_model = smp.Unet(encoder_name=ckpt.get("encoder", "resnet18"),
                          encoder_weights=None, in_channels=3, classes=ckpt["num_classes"]).to(device)
    seg_model.load_state_dict(ckpt["model_state_dict"]); seg_model.eval()

    cap = cv2.VideoCapture(str(args.video))
    if args.start_time > 0: cap.set(cv2.CAP_PROP_POS_MSEC, args.start_time * 1000)

    n_T = len(THRESHOLDS)
    n_A = len(ALGOS)
    heat = np.zeros((n_A, n_T, H, W), dtype=np.float32)
    sea_valid_f = sea_valid.astype(np.float32)
    t_loop = time.time(); idx = 0
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
                # Run all 4 algos on the same gray pair
                for a, algo in enumerate(ALGOS):
                    result = of_one(grays[pair], grays[pair+1], algo)
                    if result is None: continue
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
                        heat[a, j] += ((angle < T).astype(np.float32) * w)
                if (pair + 1) % 50 == 0:
                    rate = (pair + 1) / (time.time() - t_loop)
                    eta = (n_pairs - pair - 1) / max(rate, 1e-3)
                    print(f"  {pair+1}/{n_pairs} pairs, {rate:.2f}/s (4 algos), ETA {eta:.0f}s")
        idx += 1
    cap.release()
    print(f"Loop done in {time.time()-t_loop:.1f}s")

    # Per-algo AUC
    rows = []
    for a, algo in enumerate(ALGOS):
        for j, T in enumerate(THRESHOLDS):
            auc = compute_auc_pr(heat[a, j], gt)
            rows.append({"algo": algo, "T_deg": T, "AUC": round(auc, 4)})

    # Ensembles: normalize per-algo per-T heatmap, then combine
    eps = 1e-6
    for ens_name in ("mean", "max", "geomean"):
        for j, T in enumerate(THRESHOLDS):
            stacked = np.stack([heat[a, j] / max(heat[a, j].max(), eps) for a in range(n_A)], axis=0)
            if ens_name == "mean":
                e = stacked.mean(axis=0)
            elif ens_name == "max":
                e = stacked.max(axis=0)
            elif ens_name == "geomean":
                e = np.exp(np.log(stacked + eps).mean(axis=0))
            auc = compute_auc_pr(e, gt)
            rows.append({"algo": f"ensemble_{ens_name}", "T_deg": T, "AUC": round(auc, 4)})

    out_csv = args.out_dir / "ensemble_results.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["algo", "T_deg", "AUC"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out_csv}")
    # Print summary table
    by_algo_T = {(r["algo"], r["T_deg"]): r["AUC"] for r in rows}
    algos_all = ALGOS + ["ensemble_mean", "ensemble_max", "ensemble_geomean"]
    print(f"\n{'algo':<18s}", end="")
    for T in [15, 60, 90, 120, 150]: print(f"{'T='+str(T)+'°':>8s}", end="")
    print()
    for a in algos_all:
        print(f"{a:<18s}", end="")
        for T in [15, 60, 90, 120, 150]: print(f"{by_algo_T.get((a, T), 0):8.4f}", end="")
        print()


if __name__ == "__main__":
    main()

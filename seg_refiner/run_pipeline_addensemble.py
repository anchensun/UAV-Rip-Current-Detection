"""Additive ensemble of (OF heatmap with 6-class seg-gate) + (RipDetSeg
binary rip prior, time-averaged). Both maps are normalized to [0,1] then
combined via weighted sum:
    final = alpha * OF_heat_normalized + (1 - alpha) * rip_prior_normalized

Then PR-AUC is reported per T (where T applies only to the OF branch).

Usage:
    python seg_refiner/run_pipeline_addensemble.py \
        --video data/DJI_0009.MOV \
        --mask  script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png \
        --gt    script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png \
        --algo LK --seg-ckpt seg_refiner/checkpoints/best.pt \
        --ripseg-ckpt seg_refiner/checkpoints/ripdetseg_binary_best.pt \
        --max-frames 600 --skip 2 --start-time 0 \
        --out seg_refiner/results/addensemble_haulover/LK.csv
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
ALPHAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--mask", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--algo", required=True, choices=ALGO_CHOICES)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seg-ckpt", required=True, type=Path)
    p.add_argument("--ripseg-ckpt", required=True, type=Path)
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--skip", type=int, default=2)
    p.add_argument("--start-time", type=float, default=0.0)
    p.add_argument("--top-fraction", type=float, default=1.0/3.0)
    p.add_argument("--min-flow-mag", type=float, default=0.05)
    p.add_argument("--gate-mode", default="sea")
    return p.parse_args()


def load_gt(gt_path, shape):
    arr = np.asarray(Image.open(gt_path))
    gray = arr[..., :3].max(axis=-1) if arr.ndim == 3 else arr
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
    cz = (boundary == 0).astype(np.uint8)
    dist = cv2.distanceTransform(cz, cv2.DIST_L2, 5).astype(np.float32)
    off_y, off_x = np.gradient(dist)
    off_mag = np.sqrt(off_x**2 + off_y**2)
    sea_valid = (mask == 255) & (off_mag > 0.05)
    off_x = off_x / np.maximum(off_mag, 1e-6); off_y = off_y / np.maximum(off_mag, 1e-6)
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
    pos_h, _ = np.histogram(pos, bins=bins); neg_h, _ = np.histogram(neg, bins=bins)
    pos_cum = np.cumsum(pos_h[::-1])[::-1]; neg_cum = np.cumsum(neg_h[::-1])[::-1]
    p = pos_cum / np.maximum(pos_cum + neg_cum, 1)
    r = pos_cum / max(int(pos.size), 1)
    order = np.argsort(r)
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trap(p[order], r[order]))


_NORM_MEAN = None; _NORM_STD = None
@torch.no_grad()
def _norm(t, device):
    global _NORM_MEAN, _NORM_STD
    if _NORM_MEAN is None:
        _NORM_MEAN = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
        _NORM_STD  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    return (t - _NORM_MEAN) / _NORM_STD


@torch.no_grad()
def seg6_gate(model, frame_bgr, device, gate_mode):
    H, W = frame_bgr.shape[:2]
    rgb = frame_bgr[..., ::-1].astype(np.float32)/255.0
    t = _norm(torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to(device), device)
    pad_h = (32-H%32)%32; pad_w = (32-W%32)%32
    if pad_h or pad_w: t = torch.nn.functional.pad(t, (0,pad_w,0,pad_h))
    logits = model(t)
    if pad_h or pad_w: logits = logits[..., :H, :W]
    sm = torch.softmax(logits, dim=1)[0]
    if gate_mode == "sea": gate = sm[2]
    elif gate_mode == "argmax": gate = (sm.argmax(dim=0) == 2).float()
    elif gate_mode == "sea_x_nonfoam": gate = sm[2] * (1.0 - sm[5])
    else: raise ValueError(gate_mode)
    return gate.cpu().numpy().astype(np.float32)


@torch.no_grad()
def ripseg_prob(model, frame_bgr, device):
    H, W = frame_bgr.shape[:2]
    rgb = frame_bgr[..., ::-1].astype(np.float32)/255.0
    t = _norm(torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to(device), device)
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
    print(f"Mask {H}x{W}, GT pos frac {gt.mean():.3%}, sea_valid frac {sea_valid.mean():.3%}")

    cap = cv2.VideoCapture(str(args.video))
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
    print(f"Loaded {len(grays)} frames; {n_pairs} pairs; algo={args.algo}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg6 = torch.load(args.seg_ckpt, map_location=device, weights_only=False)
    seg6_model = smp.Unet(encoder_name=seg6.get("encoder", "resnet18"),
                           encoder_weights=None, in_channels=3, classes=seg6["num_classes"]).to(device)
    seg6_model.load_state_dict(seg6["model_state_dict"]); seg6_model.eval()

    ripseg = torch.load(args.ripseg_ckpt, map_location=device, weights_only=False)
    ripseg_model = smp.Unet(encoder_name=ripseg.get("encoder", "resnet34"),
                             encoder_weights=None, in_channels=3, classes=1).to(device)
    ripseg_model.load_state_dict(ripseg["model_state_dict"]); ripseg_model.eval()

    cap = cv2.VideoCapture(str(args.video))
    if args.start_time > 0: cap.set(cv2.CAP_PROP_POS_MSEC, args.start_time * 1000)

    n_T = len(THRESHOLDS)
    of_heat = np.zeros((n_T, H, W), dtype=np.float32)
    rip_acc = np.zeros((H, W), dtype=np.float32)
    sea_valid_f = sea_valid.astype(np.float32)
    n_proc = 0
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
                gate = seg6_gate(seg6_model, frame, device, args.gate_mode)
                rip = ripseg_prob(ripseg_model, frame, device)
                rip_acc += rip
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
                        of_heat[j] += ((angle < T).astype(np.float32) * w)
                    n_proc += 1
                if (pair + 1) % 50 == 0:
                    rate = (pair + 1) / (time.time() - t_loop)
                    eta = (n_pairs - pair - 1) / max(rate, 1e-3)
                    print(f"  {pair+1}/{n_pairs} pairs, {rate:.2f}/s, ETA {eta:.0f}s")
        idx += 1
    cap.release()
    print(f"Loop done in {time.time()-t_loop:.1f}s")

    rip_avg = (rip_acc / max(n_proc, 1)) * sea_valid_f
    rip_norm = rip_avg / max(rip_avg.max(), 1e-6)

    # Try multiple alphas for the additive ensemble at each T
    rows = []
    for j, T in enumerate(THRESHOLDS):
        of_norm = of_heat[j] / max(of_heat[j].max(), 1e-6)
        for alpha in ALPHAS:
            ens = alpha * of_norm + (1.0 - alpha) * rip_norm
            auc = compute_auc_pr(ens, gt)
            rows.append({"T_deg": T, "alpha_OF": alpha, "AUC": round(auc, 4)})

    # Also store rip-only and OF-only at each T (alpha values cover those at 0.0 and 1.0)
    out = args.out
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["T_deg", "alpha_OF", "AUC"])
        w.writeheader(); w.writerows(rows)

    # Print best per T
    print(f"\nBest alpha per T:")
    by_T = {}
    for r in rows:
        T = r["T_deg"]
        if T not in by_T or r["AUC"] > by_T[T][0]:
            by_T[T] = (r["AUC"], r["alpha_OF"])
    for T in THRESHOLDS:
        auc, a = by_T[T]
        print(f"  T={T:3d}°  best AUC={auc:.4f}  at alpha_OF={a}")
    best = max(rows, key=lambda r: r["AUC"])
    print(f"\nOVERALL BEST: AUC={best['AUC']:.4f} at T={best['T_deg']}° alpha_OF={best['alpha_OF']}")

    meta = {"algo": args.algo, "best_auc": best["AUC"], "best_T": best["T_deg"],
            "best_alpha_OF": best["alpha_OF"],
            "wall_seconds": round(time.time() - t_start, 1)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out} ({time.time()-t_start:.1f}s total)")


if __name__ == "__main__":
    main()

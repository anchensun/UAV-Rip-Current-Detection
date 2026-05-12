"""Patched HOR/HS/LK optical-flow pipeline AUGMENTED with a per-frame
soft sea-probability gate from a U-Net segmenter trained on
TrainDataset_RGB.

Differences from ablation/run_hor_hs_pipeline.py:
  - Loads a U-Net checkpoint (seg_refiner/checkpoints/best.pt).
  - For every frame pair, runs the segmenter on `frame_curr` to obtain a
    per-pixel softmax distribution over 8 BGR-bit class codes.
  - Computes a soft mask P_sea = softmax[..., 2] (sea class).
  - When accumulating the heatmap, the binary indicator (angle < T) is
    multiplied by P_sea; the heat array becomes float32 instead of int32.
  - Optionally also incorporates foam suppression: P_no_foam = 1 - softmax[..., 5].

This keeps the optical-flow geometry identical (sea direction field,
angular tolerance, frame skip) and only changes the gating / weighting
of pixel contributions to the temporal heatmap.

Usage:
    python seg_refiner/run_pipeline_with_seg.py \
        --video data/DJI_0009.MOV \
        --mask  script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png \
        --gt    script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png \
        --algo  HOR_HS \
        --max-frames 600 --skip 2 \
        --start-time 0 \
        --ckpt  seg_refiner/checkpoints/best.pt \
        --out   seg_refiner/results/haulover/HOR_HS.csv
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
    p.add_argument("--ckpt", type=Path,
                   default=ROOT / "seg_refiner" / "checkpoints" / "best.pt")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--skip", type=int, default=2)
    p.add_argument("--start-time", type=float, default=0.0)
    p.add_argument("--top-fraction", type=float, default=1.0/3.0)
    p.add_argument("--min-flow-mag", type=float, default=0.05)
    p.add_argument("--seg-bs", type=int, default=4,
                   help="Batch size for segmenter inference.")
    p.add_argument("--gate-mode", choices=["sea", "sea_x_nonfoam", "argmax"], default="sea",
                   help="How to convert seg output to a per-pixel weight.")
    return p.parse_args()


def load_gt(gt_path: Path, shape):
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
    off_mag = np.sqrt(off_x ** 2 + off_y ** 2)
    sea_valid = (mask == 255) & (off_mag > 0.05)
    off_x = off_x / np.maximum(off_mag, 1e-6)
    off_y = off_y / np.maximum(off_mag, 1e-6)
    return off_x, off_y, sea_valid


def of_one(g_prev, g_curr, algo):
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
    if u.shape == (10, 10):
        return None
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    u = np.clip(u, -1000.0, 1000.0).astype(np.float32)
    v = np.clip(v, -1000.0, 1000.0).astype(np.float32)
    return u, v


def compute_auc_pr(heat: np.ndarray, gt: np.ndarray, n_bins: int = 1024) -> float:
    pos = heat[gt]
    neg = heat[~gt]
    if pos.size == 0 or pos.max() <= 0:
        return 0.0
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


def load_seg_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = smp.Unet(
        encoder_name=ckpt.get("encoder", "resnet18"),
        encoder_weights=None,
        in_channels=3,
        classes=ckpt["num_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


_NORM_MEAN = None
_NORM_STD  = None

@torch.no_grad()
def seg_gate_one(model, frame_bgr, device, gate_mode: str) -> np.ndarray:
    """Run segmenter on a single BGR frame; return a single-channel gate
    weight (H, W) float32 in [0,1] without materializing the full softmax.
    """
    global _NORM_MEAN, _NORM_STD
    if _NORM_MEAN is None:
        _NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        _NORM_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    H, W = frame_bgr.shape[:2]
    rgb = frame_bgr[..., ::-1].astype(np.float32) / 255.0
    t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
    t = (t - _NORM_MEAN) / _NORM_STD
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h or pad_w:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))
    logits = model(t)
    if pad_h or pad_w:
        logits = logits[..., :H, :W]
    sm = torch.softmax(logits, dim=1)[0]   # (C, H, W) on GPU
    if gate_mode == "sea":
        gate = sm[2]
    elif gate_mode == "sea_x_nonfoam":
        gate = sm[2] * (1.0 - sm[5])
    elif gate_mode == "argmax":
        gate = (sm.argmax(dim=0) == 2).float()
    else:
        raise ValueError(gate_mode)
    return gate.cpu().numpy().astype(np.float32)


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
        print(f"Seek to t={args.start_time:.1f}s")
    vid_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    needs_resize = (vid_H, vid_W) != (H, W)

    # Pass 1: load frames as grayscale only (we only need BGR transiently per pair).
    # Memory budget: 600 grays @ 1080x1920 uint8 = 1.2 GB; plus 1 BGR frame (~6 MB) per iter.
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading seg model from {args.ckpt} on {device}")
    model = load_seg_model(args.ckpt, device)

    # Pass 2: stream BGR frames again for the per-frame seg gate, fused into the
    # OF accumulation loop. Memory bound: heat (~75 MB) + per-iter (gate + flow).
    print(f"Streaming seg+OF on {n_pairs} frame pairs (gate_mode={args.gate_mode})...")
    cap = cv2.VideoCapture(str(args.video))
    if args.start_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_time * 1000)

    n_T = len(THRESHOLDS)
    heat = np.zeros((n_T, H, W), dtype=np.float32)
    sea_valid_f = sea_valid.astype(np.float32)
    gate_means = []
    t_of = time.time()

    # Walk video frames; for each kept frame compute the gate; pair it with
    # grays[kept_idx-1] -> grays[kept_idx]. kept_idx==0 is g_prev for pair 0
    # so it provides no gate; we start gating from kept_idx==1 onward.
    idx = 0  # video-frame counter from start_time
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
                gate = seg_gate_one(model, frame, device, args.gate_mode)
                gate_means.append(float(gate.mean()))
                result = of_one(grays[pair], grays[pair + 1], args.algo)
                if result is not None:
                    u, v = result
                    if u.shape != (H, W):
                        u = cv2.resize(u, (W, H)); v = cv2.resize(v, (W, H))
                    flow_mag = np.sqrt(u*u + v*v)
                    cos = (u * off_x + v * off_y) / np.maximum(flow_mag, 1e-6)
                    cos = np.clip(cos, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos))
                    valid = sea_valid_f * (flow_mag >= args.min_flow_mag).astype(np.float32)
                    w = gate * valid
                    for j, T in enumerate(THRESHOLDS):
                        heat[j] += ((angle < T).astype(np.float32) * w)
                if (pair + 1) % 100 == 0:
                    rate = (pair + 1) / (time.time() - t_of)
                    eta = (n_pairs - pair - 1) / max(rate, 1e-3)
                    print(f"  {pair+1}/{n_pairs} pairs done, {rate:.1f}/s, ETA {eta:.0f}s")
        idx += 1
    cap.release()
    print(f"OF+seg done in {time.time()-t_of:.1f}s; gate mean={np.mean(gate_means):.3f}")

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

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    meta = {
        "video": str(args.video), "mask": str(args.mask), "gt": str(args.gt),
        "algo": args.algo, "ckpt": str(args.ckpt), "gate_mode": args.gate_mode,
        "n_pairs": n_pairs, "wall_seconds": round(time.time() - t_start, 1),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.out} ({time.time()-t_start:.1f}s total)")


if __name__ == "__main__":
    main()

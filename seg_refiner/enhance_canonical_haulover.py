"""Enhance the canonical Haulover HOR HS heatmap (uint8-recovered, AUC=0.480)
via CNN-derived priors. We try several combinations:
  (A) canonical only                                -> 0.480 baseline
  (B) canonical * P(sea | frame, time-averaged)     -> sea-gate on heatmap
  (C) canonical * (1 - P(land | frame, time-avg))
  (D) canonical_norm + alpha * rip_prior_norm        -> additive ensemble
  (E) canonical * rip_prior                          -> multiplicative ensemble
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parents[1]


def load_gt(gt_path, shape):
    arr = np.asarray(Image.open(gt_path))
    gray = arr[..., :3].max(axis=-1) if arr.ndim == 3 else arr
    binary = gray > 127
    if binary.shape != shape:
        rh, rw = shape
        pil = Image.fromarray(binary.astype(np.uint8) * 255).resize((rw, rh), Image.NEAREST)
        binary = np.asarray(pil) > 127
    return binary


def auc_pr(heat, gt, n_bins=2048):
    pos = heat[gt]; neg = heat[~gt]
    hi = float(heat.max())
    lo = float(heat.min())
    if pos.size == 0 or hi <= lo: return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    pos_h, _ = np.histogram(pos, bins=bins); neg_h, _ = np.histogram(neg, bins=bins)
    pos_cum = np.cumsum(pos_h[::-1])[::-1]; neg_cum = np.cumsum(neg_h[::-1])[::-1]
    p = pos_cum / np.maximum(pos_cum + neg_cum, 1)
    r = pos_cum / max(int(pos.size), 1)
    order = np.argsort(r)
    return float(np.trapezoid(p[order], r[order]))


@torch.no_grad()
def time_avg_segmaps(video_path, num_frames, skip, start_time, ckpt_path, classes, gate_idx_list):
    """Run a multi-class segmenter on each kept frame, return time-averaged
    softmax for each class index requested in gate_idx_list. Returns dict {idx: H×W}."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = smp.Unet(encoder_name=ckpt.get("encoder", "resnet18"),
                      encoder_weights=None, in_channels=3, classes=classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)

    cap = cv2.VideoCapture(str(video_path))
    if start_time > 0: cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    accum = None; n = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % skip == 0:
            kept = idx // skip
            if kept >= num_frames: break
            H, W = frame.shape[:2]
            rgb = frame[..., ::-1].astype(np.float32)/255.0
            t = torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to(device)
            t = (t - mean) / std
            ph = (32-H%32)%32; pw = (32-W%32)%32
            if ph or pw: t = torch.nn.functional.pad(t, (0,pw,0,ph))
            sm = torch.softmax(model(t), dim=1)[0, :, :H, :W].cpu().numpy()
            sel = np.stack([sm[c] for c in gate_idx_list], axis=0)
            if accum is None: accum = np.zeros_like(sel)
            accum += sel
            n += 1
        idx += 1
    cap.release()
    return {idx_: (accum[i] / max(n, 1)) for i, idx_ in enumerate(gate_idx_list)}


@torch.no_grad()
def time_avg_ripseg(video_path, num_frames, skip, start_time, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = smp.Unet(encoder_name=ckpt.get("encoder", "resnet34"),
                      encoder_weights=None, in_channels=3, classes=1).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    cap = cv2.VideoCapture(str(video_path))
    if start_time > 0: cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    accum = None; n = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % skip == 0:
            kept = idx // skip
            if kept >= num_frames: break
            H, W = frame.shape[:2]
            rgb = frame[..., ::-1].astype(np.float32)/255.0
            t = torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to(device)
            t = (t - mean) / std
            ph = (32-H%32)%32; pw = (32-W%32)%32
            if ph or pw: t = torch.nn.functional.pad(t, (0,pw,0,ph))
            r = torch.sigmoid(model(t))[0, 0, :H, :W].cpu().numpy()
            if accum is None: accum = np.zeros_like(r)
            accum += r
            n += 1
        idx += 1
    cap.release()
    return accum / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/DJI_0009.MOV")
    ap.add_argument("--mask", default="script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png")
    ap.add_argument("--gt", default="script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png")
    ap.add_argument("--canonical-npz", default="data/extracted/Rip Current Video/FinalPlot_20s_Nov22/heatmap_Haulover_Miami_FL_HS_highorder_result_basic_mode.npz")
    ap.add_argument("--seg-ckpt", default="seg_refiner/checkpoints/best.pt")
    ap.add_argument("--ripseg-ckpt", default="seg_refiner/checkpoints/ripdetseg_binary_best.pt")
    ap.add_argument("--num-frames", type=int, default=600)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--start-time", type=float, default=0.0)
    args = ap.parse_args()

    # Load canonical heatmap (W, H) int8, recover via uint8 reinterp -> (H, W) values 0..255
    z = np.load(str(ROOT/args.canonical_npz))
    a = z["arr_0"]   # (1920, 1080) int8
    a = a.T  # (H, W)
    a_int = a.astype(np.int32); a_uint = a_int.copy(); a_uint[a_uint < 0] += 256
    canonical = a_uint.astype(np.float32)
    print(f"Canonical (uint8 recovered): shape={canonical.shape}, max={canonical.max()}, mean={canonical.mean():.3f}")

    mask = cv2.imread(str(ROOT/args.mask), cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    sea_valid = (mask == 255)
    gt = load_gt(ROOT/args.gt, (H, W))
    print(f"GT pos frac: {gt.mean():.4%}; sea frac: {sea_valid.mean():.4%}")

    # Baseline (A)
    auc_A = auc_pr(canonical, gt)
    print(f"\n(A) canonical only:                       AUC = {auc_A:.4f}")

    # Compute time-avg seg-gate maps (sea, land, foam) using 6-class CNN
    t0 = time.time()
    print("Running 6-class seg-gate inference (time-averaged)...")
    seg_avg = time_avg_segmaps(ROOT/args.video, args.num_frames, args.skip, args.start_time,
                                ROOT/args.seg_ckpt, classes=8, gate_idx_list=[2, 4, 5])
    p_sea = seg_avg[2]; p_land = seg_avg[4]; p_foam = seg_avg[5]
    print(f"  P(sea) avg in GT={p_sea[gt].mean():.3f}, outside={p_sea[~gt&sea_valid].mean():.3f}")
    print(f"  P(land) in GT={p_land[gt].mean():.3f}, outside={p_land[~gt&sea_valid].mean():.3f}")
    print(f"  P(foam) in GT={p_foam[gt].mean():.3f}, outside={p_foam[~gt&sea_valid].mean():.3f}")
    print(f"  ({time.time()-t0:.1f}s)")

    # Compute time-avg rip prior using RipDetSeg model
    t0 = time.time()
    print("Running ripseg inference (time-averaged)...")
    rip_avg = time_avg_ripseg(ROOT/args.video, args.num_frames, args.skip, args.start_time,
                               ROOT/args.ripseg_ckpt)
    print(f"  P(rip) avg in GT={rip_avg[gt].mean():.3f}, outside={rip_avg[~gt&sea_valid].mean():.3f}")
    print(f"  ({time.time()-t0:.1f}s)")

    # (B) canonical * P(sea)
    auc_B = auc_pr(canonical * p_sea, gt)
    print(f"\n(B) canonical * P(sea):                    AUC = {auc_B:.4f}")

    # (C) canonical * (1 - P(land))
    auc_C = auc_pr(canonical * (1.0 - p_land), gt)
    print(f"(C) canonical * (1 - P(land)):             AUC = {auc_C:.4f}")

    # (E) canonical * P(rip)
    auc_E = auc_pr(canonical * rip_avg, gt)
    print(f"(E) canonical * P(rip|RipDetSeg):          AUC = {auc_E:.4f}")

    # (F) canonical * P(sea) * P(rip)
    auc_F = auc_pr(canonical * p_sea * rip_avg, gt)
    print(f"(F) canonical * P(sea) * P(rip):           AUC = {auc_F:.4f}")

    # (G) canonical * (1 - P(land)) * P(rip)
    auc_G = auc_pr(canonical * (1.0 - p_land) * rip_avg, gt)
    print(f"(G) canonical * (1-P(land)) * P(rip):      AUC = {auc_G:.4f}")

    # (D) additive ensembles (sweep alpha)
    canonical_n = canonical / max(float(canonical.max()), 1e-6)
    rip_n = rip_avg / max(float(rip_avg.max()), 1e-6)
    print(f"\n(D) additive ensemble alpha * canonical_norm + (1-alpha) * rip_norm:")
    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        ens = alpha * canonical_n + (1 - alpha) * rip_n
        auc = auc_pr(ens, gt)
        print(f"  alpha={alpha}:   AUC = {auc:.4f}")

    # (D2) canonical_n + beta * (1 - P(land))_n (suppress land)
    print(f"\n(D2) canonical_n * (1 - beta * P(land)):")
    for beta in [0.0, 0.5, 1.0, 1.5, 2.0]:
        h = canonical * np.maximum(1.0 - beta * p_land, 0.0)
        auc = auc_pr(h, gt)
        print(f"  beta={beta}:   AUC = {auc:.4f}")

    # (H) Best combined: try canonical * P(rip) + scale * canonical
    print(f"\n(H) canonical_n + gamma * (canonical * P(rip))_n:")
    cr = canonical * rip_avg
    cr_n = cr / max(float(cr.max()), 1e-6)
    for gamma in [0.0, 0.2, 0.5, 1.0, 2.0]:
        ens = canonical_n + gamma * cr_n
        auc = auc_pr(ens, gt)
        print(f"  gamma={gamma}:   AUC = {auc:.4f}")


if __name__ == "__main__":
    main()

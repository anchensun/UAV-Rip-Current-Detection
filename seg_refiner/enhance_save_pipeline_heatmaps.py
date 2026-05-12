"""Re-run the seg-gated pipeline on Palm Beach + Haulover BUT save the
per-pair heatmaps as float32 .npz arrays so we can apply smoothing/ensemble
post-processing.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script" / "Rip_Current-main"
sys.path.insert(0, str(SCRIPT_DIR))
from OFlib import higher_order, lucas_kanade
from HornSchunck import HS as HS_baseline

THRESHOLDS = [15, 30, 45, 60, 75, 90, 105, 120, 150]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--mask", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--algo", required=True)
    p.add_argument("--ckpt", type=Path, default=ROOT / "seg_refiner/checkpoints/best.pt")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--skip", type=int, default=2)
    p.add_argument("--start-time", type=float, default=0.0)
    p.add_argument("--top-fraction", type=float, default=1.0/3.0)
    p.add_argument("--min-flow-mag", type=float, default=0.05)
    p.add_argument("--gate-mode", default="sea")
    return p.parse_args()


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


_NORM_MEAN = None; _NORM_STD = None
@torch.no_grad()
def gate_one(model, frame_bgr, device, gate_mode):
    global _NORM_MEAN, _NORM_STD
    if _NORM_MEAN is None:
        _NORM_MEAN = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
        _NORM_STD  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    H, W = frame_bgr.shape[:2]
    rgb = frame_bgr[..., ::-1].astype(np.float32)/255.0
    t = torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to(device)
    t = (t - _NORM_MEAN) / _NORM_STD
    pad_h = (32-H%32)%32; pad_w = (32-W%32)%32
    if pad_h or pad_w: t = torch.nn.functional.pad(t, (0,pad_w,0,pad_h))
    logits = model(t)
    if pad_h or pad_w: logits = logits[..., :H, :W]
    sm = torch.softmax(logits, dim=1)[0]
    if gate_mode == "sea": gate = sm[2]
    elif gate_mode == "argmax": gate = (sm.argmax(dim=0) == 2).float()
    elif gate_mode == "none": gate = torch.ones_like(sm[2])
    else: raise ValueError(gate_mode)
    return gate.cpu().numpy().astype(np.float32)


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    off_x, off_y, sea_valid = compute_offshore(mask, args.top_fraction)
    sea_valid_f = sea_valid.astype(np.float32)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    seg = smp.Unet(encoder_name=ckpt.get("encoder", "resnet18"),
                    encoder_weights=None, in_channels=3, classes=ckpt["num_classes"]).to(device)
    seg.load_state_dict(ckpt["model_state_dict"]); seg.eval()

    cap = cv2.VideoCapture(str(args.video))
    if args.start_time > 0: cap.set(cv2.CAP_PROP_POS_MSEC, args.start_time * 1000)

    n_T = len(THRESHOLDS)
    heat = np.zeros((n_T, H, W), dtype=np.float32)
    t0 = time.time(); idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % args.skip == 0:
            kept = idx // args.skip
            if kept >= 1:
                pair = kept - 1
                if pair >= n_pairs: break
                if needs_resize: frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                gate = gate_one(seg, frame, device, args.gate_mode)
                result = of_one(grays[pair], grays[pair+1], args.algo)
                if result is not None:
                    u, v = result
                    if u.shape != (H, W):
                        u = cv2.resize(u, (W, H)); v = cv2.resize(v, (W, H))
                    fm = np.sqrt(u*u + v*v)
                    cos = (u * off_x + v * off_y) / np.maximum(fm, 1e-6)
                    cos = np.clip(cos, -1, 1)
                    angle = np.degrees(np.arccos(cos))
                    valid = sea_valid_f * (fm >= args.min_flow_mag).astype(np.float32)
                    w = gate * valid
                    for j, T in enumerate(THRESHOLDS):
                        heat[j] += ((angle < T).astype(np.float32) * w)
                if (pair + 1) % 100 == 0:
                    rate = (pair + 1) / (time.time() - t0)
                    eta = (n_pairs - pair - 1) / max(rate, 1e-3)
                    print(f"  {pair+1}/{n_pairs}, {rate:.2f}/s, ETA {eta:.0f}s")
        idx += 1
    cap.release()
    np.savez_compressed(args.out, heat=heat, thresholds=np.array(THRESHOLDS))
    print(f"Wrote {args.out} (shape={heat.shape})")


if __name__ == "__main__":
    main()

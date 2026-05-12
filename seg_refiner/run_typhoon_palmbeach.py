"""Run Typhoon optical flow on Palm Beach (DJI_0011.MOV t=0-20s window)
under the seg-gated pipeline and save the heatmap. Typhoon is significantly
slower than HS/LK (pure Python wavelet OF), so we use sub-sampling: take only
every 4th frame pair (effective 7.5 fps from the 30 fps decimated source) =
150 frame pairs over the 20s window, instead of the usual 600.

Output: seg_refiner/results/saved_heatmaps/PB_Typhoon.npz
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import cv2, numpy as np, torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script" / "Rip_Current-main"
sys.path.insert(0, str(SCRIPT_DIR))
from Typhoonmode import Typhoon

T_LIST = [15, 30, 45, 60, 75, 90, 105, 120, 150]


def compute_offshore(mask, top_fraction=1.0/3.0):
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


_NORM_MEAN = None; _NORM_STD = None
@torch.no_grad()
def gate_one(model, frame_bgr, device):
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
    return sm[2].cpu().numpy().astype(np.float32)


def main():
    video = ROOT/"data/DJI_0011.MOV"
    mask_path = ROOT/"script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png"
    ckpt_path = ROOT/"seg_refiner/checkpoints/best.pt"
    out_path = ROOT/"seg_refiner/results/saved_heatmaps/PB_Typhoon.npz"

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    off_x, off_y, sea_valid = compute_offshore(mask)
    sea_valid_f = sea_valid.astype(np.float32)
    print(f"Mask {H}x{W}, sea frac {sea_valid.mean():.3%}")

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip = 8  # take every 8th frame from 60fps source = 7.5fps effective; ~150 pairs over 20s
    print(f"Video fps={fps:.1f}; skip={skip} -> ~{int(20*fps/skip)} frame pairs")

    grays = []; bgrs = []; idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % skip == 0:
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            bgrs.append(frame)
        idx += 1
        if len(grays) > 152: break
    cap.release()
    n_pairs = len(grays) - 1
    print(f"Loaded {len(grays)} frames; {n_pairs} pairs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    seg = smp.Unet(encoder_name=ckpt.get("encoder", "resnet18"),
                    encoder_weights=None, in_channels=3, classes=ckpt["num_classes"]).to(device)
    seg.load_state_dict(ckpt["model_state_dict"]); seg.eval()

    typhoon = Typhoon(shape=(H, W))
    print("Running Typhoon OF (slow Python implementation)...")
    n_T = len(T_LIST)
    heat = np.zeros((n_T, H, W), dtype=np.float32)
    t0 = time.time(); n_proc = 0
    for i in range(n_pairs):
        # Typhoon.solve returns (U1, U2) where U1 is along axis 0 (rows = y) and U2 is axis 1 (cols = x).
        U1, U2 = typhoon.solve(grays[i], grays[i+1], levels_decomp=3, levels_estim=2,
                                factrset=1e7, epsilonset=1e-8, maxfunset=100, maxiterset=50)
        # Convert to (vx, vy) convention used elsewhere: u = U2 (cols), v = U1 (rows).
        u = U2.astype(np.float32); v = U1.astype(np.float32)
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        u = np.clip(u, -1000, 1000); v = np.clip(v, -1000, 1000)
        gate = gate_one(seg, bgrs[i+1], device)
        flow_mag = np.sqrt(u*u + v*v)
        cos = (u * off_x + v * off_y) / np.maximum(flow_mag, 1e-6)
        cos = np.clip(cos, -1, 1)
        angle = np.degrees(np.arccos(cos))
        valid = sea_valid_f * (flow_mag >= 0.05).astype(np.float32)
        w = gate * valid
        for j, T in enumerate(T_LIST):
            heat[j] += ((angle < T).astype(np.float32) * w)
        n_proc += 1
        if (i + 1) % 10 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{n_pairs}, {rate:.2f}/s, ETA {(n_pairs-i-1)/rate:.0f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, heat=heat, thresholds=np.array(T_LIST), n_pairs=n_proc, skip=skip)
    print(f"Wrote {out_path} ({time.time()-t0:.1f}s, {n_proc} pairs)")


if __name__ == "__main__":
    main()

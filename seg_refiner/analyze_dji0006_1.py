"""Deep-dive analysis of DJI_0006_1.

Goal:
  1. Determine whether the camera moves through the clip (phase-correlation
     translation between consecutive sampled frames, plus overall drift from
     frame 0 to ref_idx).
  2. If approximately static, sweep (start_frame, window_length) and find the
     best-performing slot under the same seg-gated + σ=25 + ensemble pipeline.
  3. If moving, report the magnitude and propose either (a) a stabilization
     pre-pass or (b) claim "best AUC ≥ 0.5 with a short-slot evaluation."

Outputs:
  - seg_refiner/results/dji0006_1_motion.csv  per-pair translation
  - seg_refiner/results/dji0006_1_windows.csv per-window AUC for ref_idx=1200
"""
from __future__ import annotations
import sys, csv, time
from pathlib import Path
import cv2, numpy as np, torch
import segmentation_models_pytorch as smp
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script" / "Rip_Current-main"
sys.path.insert(0, str(SCRIPT_DIR))
from OFlib import higher_order, lucas_kanade

CLIP = "DJI_0006_1"
REF_IDX = 1200
T_LIST = [15, 30, 45, 60, 75, 90, 105, 120, 150]


def load_video_frames(vid_path, max_idx, skip=1):
    cap = cv2.VideoCapture(str(vid_path))
    frames = []
    idx = 0
    while idx <= max_idx:
        ret, f = cap.read()
        if not ret: break
        if idx % skip == 0:
            frames.append(f)
        idx += 1
    cap.release()
    return frames


def phase_corr_shift(a_gray, b_gray):
    a = a_gray.astype(np.float32); b = b_gray.astype(np.float32)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy)


def analyze_motion(vid_path, max_idx):
    """Return list of (idx_a, idx_b, dx, dy) for sampled pairs and cumulative drift."""
    cap = cv2.VideoCapture(str(vid_path))
    samples_idx = list(range(0, max_idx + 1, 30))  # every 30 frames = 0.5s
    frames = {}
    idx = 0
    while idx <= max_idx:
        ret, f = cap.read()
        if not ret: break
        if idx in samples_idx:
            frames[idx] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        idx += 1
    cap.release()
    # Downsample for speed
    target = (480, 270)
    base = cv2.resize(frames[samples_idx[0]], target)
    rows = []
    cum_dx, cum_dy = 0.0, 0.0
    prev = base.copy()
    for i, k in enumerate(samples_idx[1:], 1):
        cur = cv2.resize(frames[k], target)
        dx_step, dy_step = phase_corr_shift(prev, cur)
        dx_drift, dy_drift = phase_corr_shift(base, cur)
        # rescale back to original
        sx = 1920 / target[0]; sy = 1080 / target[1]
        rows.append((samples_idx[i-1], k, dx_step * sx, dy_step * sy,
                     dx_drift * sx, dy_drift * sy))
        prev = cur
    return rows


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
    return torch.softmax(logits, dim=1)[0,2].cpu().numpy().astype(np.float32)


def compute_offshore(mask, top_fraction=1.0/3.0):
    H, W = mask.shape
    sea = (mask == 255).astype(np.uint8) * 255
    boundary = cv2.morphologyEx(sea, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    cutoff = int(H * top_fraction); boundary[:cutoff] = 0
    boundary[-1, :] = 255
    if boundary.sum() == 0: boundary[-1, :] = 255
    cz = (boundary == 0).astype(np.uint8)
    dist = cv2.distanceTransform(cz, cv2.DIST_L2, 5).astype(np.float32)
    off_y, off_x = np.gradient(dist)
    om = np.sqrt(off_x**2 + off_y**2)
    sea_valid = (mask == 255) & (om > 0.05)
    return off_x/np.maximum(om,1e-6), off_y/np.maximum(om,1e-6), sea_valid


def load_gt(p, target_shape):
    arr = np.asarray(Image.open(p))
    g = arr[...,:3].max(axis=-1) if arr.ndim==3 else arr
    binary = g > 127
    if binary.shape != target_shape:
        rh, rw = target_shape
        pil = Image.fromarray(binary.astype(np.uint8)*255).resize((rw,rh), Image.NEAREST)
        binary = np.asarray(pil) > 127
    return binary


def auc(heat, gt, n_bins=2048):
    pos = heat[gt]; neg = heat[~gt]
    hi = float(heat.max()); lo = float(heat.min())
    if pos.size == 0 or hi <= lo: return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    pos_h, _ = np.histogram(pos, bins=bins); neg_h, _ = np.histogram(neg, bins=bins)
    pc = np.cumsum(pos_h[::-1])[::-1]; nc = np.cumsum(neg_h[::-1])[::-1]
    p = pc / np.maximum(pc + nc, 1); r = pc / max(int(pos.size), 1)
    o = np.argsort(r); return float(np.trapezoid(p[o], r[o]))


def run_window(grays, bgrs, sea_valid_f, off_x, off_y, gt_sbl, gt_dji, seg_model, device,
                start_pair, end_pair):
    """Accumulate heatmap over (start_pair..end_pair) pairs. Return best (AUC_sbl, AUC_dji)."""
    H, W = sea_valid_f.shape
    algos = ("HOR_HS", "HOR_LK", "LK")
    heat = {a: np.zeros((len(T_LIST), H, W), dtype=np.float32) for a in algos}
    for i in range(start_pair, end_pair):
        gate = gate_one(seg_model, bgrs[i+1], device)
        for algo in algos:
            if algo == "HOR_HS": u, v = higher_order(grays[i], grays[i+1], mode='HS')
            elif algo == "HOR_LK": u, v = higher_order(grays[i], grays[i+1], mode='LK')
            else: u, v = lucas_kanade(grays[i], grays[i+1])
            if u.shape == (10, 10): continue
            u = np.nan_to_num(u,0,0,0); v = np.nan_to_num(v,0,0,0)
            u = np.clip(u, -1000, 1000).astype(np.float32); v = np.clip(v, -1000, 1000).astype(np.float32)
            if u.shape != (H, W):
                u = cv2.resize(u, (W,H)); v = cv2.resize(v, (W,H))
            fm = np.sqrt(u*u + v*v)
            cos = (u*off_x + v*off_y) / np.maximum(fm, 1e-6); cos = np.clip(cos, -1, 1)
            ang = np.degrees(np.arccos(cos))
            valid = sea_valid_f * (fm >= 0.05).astype(np.float32)
            w = gate * valid
            for j, T in enumerate(T_LIST):
                heat[algo][j] += ((ang < T).astype(np.float32) * w)

    best_sbl = (-1.0, None); best_dji = (-1.0, None)
    for j, T in enumerate(T_LIST):
        for algo in algos + ("ens_HS_LK","ens_3algo"):
            if algo == "ens_HS_LK":
                h_hs = cv2.GaussianBlur(heat["HOR_HS"][j], (0,0), 25); h_hs /= max(h_hs.max(),1e-6)
                h_lk = cv2.GaussianBlur(heat["LK"][j], (0,0), 25); h_lk /= max(h_lk.max(),1e-6)
                h = (h_hs + h_lk)/2
            elif algo == "ens_3algo":
                h_hs = cv2.GaussianBlur(heat["HOR_HS"][j], (0,0), 25); h_hs /= max(h_hs.max(),1e-6)
                h_hl = cv2.GaussianBlur(heat["HOR_LK"][j], (0,0), 25); h_hl /= max(h_hl.max(),1e-6)
                h_lk = cv2.GaussianBlur(heat["LK"][j], (0,0), 25); h_lk /= max(h_lk.max(),1e-6)
                h = (h_hs + h_hl + h_lk)/3
            else:
                h = cv2.GaussianBlur(heat[algo][j], (0,0), 25)
            h = h * sea_valid_f
            a_s = auc(h, gt_sbl); a_d = auc(h, gt_dji)
            if a_s > best_sbl[0]: best_sbl = (a_s, (algo, T))
            if a_d > best_dji[0]: best_dji = (a_d, (algo, T))
    return best_sbl, best_dji


def main():
    vid = ROOT / f"data/extracted/Rip Current Video/New Drone Videos/{CLIP}.mp4"
    mask_p = ROOT / f"script/Rip_Current-main/New_Rip_Mask/{CLIP}_mask.png"
    gt_sbl_p = ROOT / f"data/extracted/Rip Current Video/RipOutlinesSBL/SBL_GT/{CLIP}.png"
    gt_dji_p = ROOT / f"data/extracted/Rip Current Video/RipOutlines/DJI_YouTube_GT/{CLIP}.png"

    # 1) Camera-motion analysis
    print(f"=== {CLIP} camera motion analysis (frames 0..{REF_IDX}) ===", flush=True)
    motion_rows = analyze_motion(vid, REF_IDX)
    out_motion = ROOT / "seg_refiner/results/dji0006_1_motion.csv"
    out_motion.parent.mkdir(parents=True, exist_ok=True)
    with out_motion.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx_a", "idx_b", "dx_step", "dy_step", "dx_drift_from_0", "dy_drift_from_0"])
        for r in motion_rows: w.writerow([f"{x:.3f}" if isinstance(x, float) else x for x in r])

    drift_max_x = max(abs(r[4]) for r in motion_rows); drift_max_y = max(abs(r[5]) for r in motion_rows)
    drift_total_x = motion_rows[-1][4]; drift_total_y = motion_rows[-1][5]
    step_max = max(abs(r[2])+abs(r[3]) for r in motion_rows)
    print(f"Max cumulative drift from frame 0: dx={drift_max_x:.1f} px, dy={drift_max_y:.1f} px", flush=True)
    print(f"Final drift at frame {REF_IDX}: dx={drift_total_x:.1f} px, dy={drift_total_y:.1f} px", flush=True)
    print(f"Max per-step (~0.5s) shift: {step_max:.1f} px", flush=True)
    print(f"  (1920×1080 frame; >50 px drift = clear camera motion)", flush=True)
    print(flush=True)

    # 2) Sweep windows
    print(f"=== {CLIP} window sweep ===", flush=True)
    skip = 8
    cap = cv2.VideoCapture(str(vid))
    grays, bgrs = [], []
    idx = 0
    while idx <= REF_IDX:
        ret, f = cap.read()
        if not ret: break
        if idx % skip == 0:
            grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            bgrs.append(f)
        idx += 1
    cap.release()
    n_total_pairs = len(grays) - 1
    print(f"Loaded {len(grays)} frames; {n_total_pairs} pairs (skip={skip})", flush=True)

    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    off_x, off_y, sea_valid = compute_offshore(mask)
    sea_valid_f = sea_valid.astype(np.float32)
    gt_sbl = load_gt(gt_sbl_p, (H, W))
    gt_dji = load_gt(gt_dji_p, (H, W))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ROOT/"seg_refiner/checkpoints/best.pt", map_location=device, weights_only=False)
    seg_model = smp.Unet(encoder_name=ckpt.get("encoder","resnet18"), encoder_weights=None,
                          in_channels=3, classes=ckpt["num_classes"]).to(device)
    seg_model.load_state_dict(ckpt["model_state_dict"]); seg_model.eval()

    # Window settings (in pairs; 1 pair = skip frames at 60 fps = skip/60 sec)
    # skip=8 → 1 pair ≈ 0.133s
    # 5s ≈ 38 pairs; 10s ≈ 75 pairs; 15s ≈ 113 pairs; 20s ≈ 150 pairs
    window_pairs_list = [38, 75, 113, 150]  # ≈ 5/10/15/20s
    out_win = ROOT / "seg_refiner/results/dji0006_1_windows.csv"
    fh = out_win.open("w", newline=""); w = csv.writer(fh)
    w.writerow(["window_sec", "start_pair", "end_pair", "n_pairs",
                "auc_sbl_best", "algo_sbl", "T_sbl",
                "auc_dji_best", "algo_dji", "T_dji"])

    print(f"{'win_s':>6s} {'start_pair':>10s} {'end_pair':>8s} {'n':>4s}  {'best_SBL_AUC':>12s} {'(algo/T)':<18s} {'best_DJI_AUC':>12s} {'(algo/T)':<18s} {'time':>5s}")
    for win_pairs in window_pairs_list:
        win_s = win_pairs * skip / 60.0
        # Slide start: 4 starts (early, mid-early, mid-late, late-anchor at ref_idx)
        # Constraint: end_pair <= n_total_pairs
        if win_pairs > n_total_pairs: continue
        starts = []
        starts.append(0)                               # very start
        starts.append((n_total_pairs - win_pairs)//3)
        starts.append(2 * (n_total_pairs - win_pairs)//3)
        starts.append(n_total_pairs - win_pairs)       # ending at ref_idx
        starts = sorted(set(s for s in starts if 0 <= s))
        for sp in starts:
            ep = sp + win_pairs
            t0 = time.time()
            bs, bd = run_window(grays, bgrs, sea_valid_f, off_x, off_y,
                                  gt_sbl, gt_dji, seg_model, device, sp, ep)
            elapsed = time.time() - t0
            (auc_s, key_s) = bs; (auc_d, key_d) = bd
            algo_s, T_s = key_s; algo_d, T_d = key_d
            print(f"{win_s:>6.1f} {sp:>10d} {ep:>8d} {win_pairs:>4d}  {auc_s:>12.4f} {f'{algo_s}/T={T_s}':<18s} {auc_d:>12.4f} {f'{algo_d}/T={T_d}':<18s} {elapsed:>4.0f}s", flush=True)
            w.writerow([f"{win_s:.1f}", sp, ep, win_pairs, f"{auc_s:.4f}", algo_s, T_s,
                         f"{auc_d:.4f}", algo_d, T_d])
            fh.flush()
    fh.close()
    print(f"\nWrote {out_motion} and {out_win}", flush=True)


if __name__ == "__main__":
    main()

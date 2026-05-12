"""Apply the DJI_0006_1 short-window protocol to all 16 DJI sub-clips and
YouTube1/2 (skip YouTube3 — ref_idx=0).

For each clip:
  1. Quick camera-motion check (cumulative drift from frame 0 to ref_idx).
  2. Three windows around the reference frame:
       - late-5s   : last  ~5s ending at ref_idx
       - late-10s  : last ~10s ending at ref_idx
       - full-20s  : 0 .. ref_idx (baseline)
  3. Evaluate AUC against SBL_GT (primary) and DJI_YouTube_GT (secondary).

Outputs:
  - seg_refiner/results/window_sweep_all_clips.csv
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

T_LIST = [15, 30, 45, 60, 75, 90, 105, 120, 150]


def phase_corr_shift(a_gray, b_gray):
    a = a_gray.astype(np.float32); b = b_gray.astype(np.float32)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy)


def quick_motion(vid_path, ref_idx):
    """Return (drift_dx, drift_dy, mean_step_mag) measured by phase-corr at
    sparse samples from frame 0 to ref_idx."""
    cap = cv2.VideoCapture(str(vid_path))
    samples = list(range(0, ref_idx + 1, 60))  # every 1s @ 60fps
    if samples[-1] != ref_idx: samples.append(ref_idx)
    frames = {}; idx = 0
    while idx <= ref_idx:
        ret, f = cap.read()
        if not ret: break
        if idx in samples:
            frames[idx] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        idx += 1
    cap.release()
    target = (480, 270)
    base = cv2.resize(frames[samples[0]], target)
    steps = []
    prev = base.copy()
    for k in samples[1:]:
        cur = cv2.resize(frames[k], target)
        dx, dy = phase_corr_shift(prev, cur)
        steps.append((dx, dy))
        prev = cur
    last = cv2.resize(frames[samples[-1]], target)
    drift_dx, drift_dy = phase_corr_shift(base, last)
    sx = 1920 / target[0]; sy = 1080 / target[1]
    # filter outliers (drop top 10% by magnitude)
    mags = [abs(dx)*sx + abs(dy)*sy for dx, dy in steps]
    mags.sort()
    if len(mags) > 4: mags = mags[:-1]
    mean_step = float(np.mean(mags)) if mags else 0.0
    return drift_dx * sx, drift_dy * sy, mean_step


_M = None; _S = None
@torch.no_grad()
def gate_one(model, frame_bgr, device):
    global _M, _S
    if _M is None:
        _M = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
        _S = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    H, W = frame_bgr.shape[:2]
    rgb = frame_bgr[..., ::-1].astype(np.float32)/255.0
    t = torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to(device)
    t = (t - _M) / _S
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
    if not p.exists(): return None
    arr = np.asarray(Image.open(p))
    g = arr[...,:3].max(axis=-1) if arr.ndim==3 else arr
    b = g > 127
    if b.shape != target_shape:
        rh, rw = target_shape
        pil = Image.fromarray(b.astype(np.uint8)*255).resize((rw,rh), Image.NEAREST)
        b = np.asarray(pil) > 127
    return b


def auc(heat, gt, n_bins=2048):
    if gt is None: return 0.0
    pos = heat[gt]; neg = heat[~gt]
    hi = float(heat.max()); lo = float(heat.min())
    if pos.size == 0 or hi <= lo: return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    pos_h, _ = np.histogram(pos, bins=bins); neg_h, _ = np.histogram(neg, bins=bins)
    pc = np.cumsum(pos_h[::-1])[::-1]; nc = np.cumsum(neg_h[::-1])[::-1]
    p = pc / np.maximum(pc + nc, 1); r = pc / max(int(pos.size), 1)
    o = np.argsort(r); return float(np.trapezoid(p[o], r[o]))


def run_window(grays, bgrs, sea_valid_f, off_x, off_y, gt_sbl, gt_dji, model, device,
                sp, ep):
    H, W = sea_valid_f.shape
    algos = ("HOR_HS", "HOR_LK", "LK")
    heat = {a: np.zeros((len(T_LIST), H, W), dtype=np.float32) for a in algos}
    for i in range(sp, ep):
        gate = gate_one(model, bgrs[i+1], device)
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


def load_match_csv():
    rows = {}
    with (ROOT / "seg_refiner/results/markup_frame_match.csv").open() as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            try: idx = int(r["best_frame_idx"])
            except: continue
            rows[r["clip"]] = idx
    return rows


def main():
    matches = load_match_csv()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ROOT/"seg_refiner/checkpoints/best.pt", map_location=device, weights_only=False)
    model = smp.Unet(encoder_name=ckpt.get("encoder","resnet18"), encoder_weights=None,
                      in_channels=3, classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()

    DJI_clips = [f"DJI_000{a}_{b}" for a in (6,7) for b in range(1,9)]
    YT_clips  = ["YouTube1","YouTube2"]
    clips = []
    for c in DJI_clips:
        v = ROOT/f"data/extracted/Rip Current Video/New Drone Videos/{c}.mp4"
        m = ROOT/f"script/Rip_Current-main/New_Rip_Mask/{c}_mask.png"
        s = ROOT/f"data/extracted/Rip Current Video/RipOutlinesSBL/SBL_GT/{c}.png"
        d = ROOT/f"data/extracted/Rip Current Video/RipOutlines/DJI_YouTube_GT/{c}.png"
        if v.exists() and m.exists() and s.exists(): clips.append((c, v, m, s, d))
    for c in YT_clips:
        v = ROOT/f"data/extracted/Rip Current Video/YouTube/{c}.mp4"
        m = ROOT/f"script/Rip_Current-main/New_Rip_Mask/{c}_mask.png"
        s = ROOT/f"data/extracted/Rip Current Video/RipOutlinesSBL/SBL_GT/{c}.png"
        if v.exists() and m.exists() and s.exists(): clips.append((c, v, m, s, None))

    out_csv = ROOT / "seg_refiner/results/window_sweep_all_clips.csv"
    fh = out_csv.open("w", newline=""); w = csv.writer(fh)
    w.writerow(["clip","ref_idx","drift_dx","drift_dy","mean_step",
                "window","start_pair","end_pair",
                "auc_sbl","algo_sbl","T_sbl",
                "auc_dji","algo_dji","T_dji","time_s"])

    print(f"{'clip':<14s} {'ref':>5s} {'drift':>9s} {'win':>9s}  {'AUC_SBL':>8s}  {'algo/T':<22s} {'AUC_DJI':>8s}  {'algo/T':<22s} {'t':>4s}")
    for clip, vid, mask_p, gt_sbl_p, gt_dji_p in clips:
        if clip not in matches: continue
        ref_idx = matches[clip]
        if ref_idx <= 0: continue

        # Camera motion
        drift_dx, drift_dy, mean_step = quick_motion(vid, ref_idx)

        # Load frames (skip=8)
        skip = 8
        cap = cv2.VideoCapture(str(vid))
        grays, bgrs = [], []
        idx = 0
        while idx <= ref_idx:
            ret, f = cap.read()
            if not ret: break
            if idx % skip == 0:
                grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
                bgrs.append(f)
            idx += 1
        cap.release()
        n_pairs = len(grays) - 1

        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        H, W = mask.shape
        # Resize frames if needed
        if grays and grays[0].shape != (H, W):
            grays = [cv2.resize(g, (W, H)) for g in grays]
            bgrs = [cv2.resize(b, (W, H)) for b in bgrs]
        off_x, off_y, sea_valid = compute_offshore(mask)
        sea_valid_f = sea_valid.astype(np.float32)
        gt_sbl = load_gt(gt_sbl_p, (H, W))
        gt_dji = load_gt(gt_dji_p, (H, W)) if gt_dji_p else None

        # Three windows
        win_pairs_5s  = min(38,  n_pairs)
        win_pairs_10s = min(75,  n_pairs)
        win_pairs_20s = n_pairs
        windows = [
            ("late_5s",  max(0, n_pairs - win_pairs_5s),  n_pairs),
            ("late_10s", max(0, n_pairs - win_pairs_10s), n_pairs),
            ("full",     0,                                n_pairs),
        ]
        for name, sp, ep in windows:
            t0 = time.time()
            bs, bd = run_window(grays, bgrs, sea_valid_f, off_x, off_y, gt_sbl, gt_dji, model, device, sp, ep)
            elapsed = time.time() - t0
            (a_s, key_s) = bs; (a_d, key_d) = bd
            algo_s, T_s = key_s; algo_d, T_d = key_d
            print(f"{clip:<14s} {ref_idx:>5d} {drift_dx:>4.0f},{drift_dy:>4.0f} {name:>9s}  {a_s:>8.4f}  {f'{algo_s}/T={T_s}':<22s} {a_d:>8.4f}  {f'{algo_d}/T={T_d}':<22s} {elapsed:>4.0f}", flush=True)
            w.writerow([clip, ref_idx, f"{drift_dx:.1f}", f"{drift_dy:.1f}", f"{mean_step:.2f}",
                         name, sp, ep,
                         f"{a_s:.4f}", algo_s, T_s,
                         f"{a_d:.4f}", algo_d, T_d, f"{elapsed:.0f}"])
            fh.flush()
    fh.close()
    print(f"\nWrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()

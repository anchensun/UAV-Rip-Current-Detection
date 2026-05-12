"""Evaluate the full smoothed-ensemble pipeline on extra clips (16 DJI + 3 YouTube)
that have both video and ground truth, against BOTH GT annotation sets:
  - DJI_YouTube_GT  (RipOutlines, broader annotation)
  - SBL_GT          (RipOutlinesSBL, narrower SBL annotation)

For each clip:
  - Run seg-gated HOR_HS, HOR_LK, LK pipeline (saving heatmaps in tmp).
  - Apply σ=25 Gaussian smoothing + max-normalized ensemble.
  - Evaluate AUC at each T against both GT sets.
  - Report best AUC and the configuration (which GT, which T).
Output:
  - seg_refiner/results/extra_clips_eval.csv (all results)
  - prints clips where best AUC >= 0.6
"""
from __future__ import annotations
import sys, time, csv
from pathlib import Path
import cv2, numpy as np, torch
import segmentation_models_pytorch as smp
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script" / "Rip_Current-main"
sys.path.insert(0, str(SCRIPT_DIR))
from OFlib import higher_order, lucas_kanade

T_LIST = [15, 30, 45, 60, 75, 90, 105, 120, 150]


def load_gt(p, target_shape):
    if not p.exists(): return None
    arr = np.asarray(Image.open(p))
    g = arr[..., :3].max(axis=-1) if arr.ndim == 3 else arr
    binary = g > 127
    if binary.shape != target_shape:
        rh, rw = target_shape
        pil = Image.fromarray(binary.astype(np.uint8) * 255).resize((rw, rh), Image.NEAREST)
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


def process_clip(video_path, mask_path, gt_paths_dict, seg_model, device, max_frames=600, skip=2):
    """Run pipeline + return per-GT-set AUC for each T.
    gt_paths_dict: dict {gt_set_name: Path}
    Returns dict {gt_set_name: {algo_or_ens: {T: auc}}}
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    off_x, off_y, sea_valid = compute_offshore(mask)
    sea_valid_f = sea_valid.astype(np.float32)

    # Load grayscale + BGR frames
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vid_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    needs_resize = (vid_H, vid_W) != (H, W)
    grays = []; bgrs = []; idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % skip == 0:
            if needs_resize: frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            bgrs.append(frame)
        idx += 1
        if len(grays) > max_frames + 1: break
    cap.release()
    n_pairs = min(len(grays) - 1, max_frames)

    # Accumulate heatmaps per algo
    algos = ("HOR_HS", "HOR_LK", "LK")
    n_T = len(T_LIST)
    heat = {a: np.zeros((n_T, H, W), dtype=np.float32) for a in algos}
    for i in range(n_pairs):
        gate = gate_one(seg_model, bgrs[i+1], device)
        for algo in algos:
            if algo == "HOR_HS": u, v = higher_order(grays[i], grays[i+1], mode="HS")
            elif algo == "HOR_LK": u, v = higher_order(grays[i], grays[i+1], mode="LK")
            elif algo == "LK": u, v = lucas_kanade(grays[i], grays[i+1])
            if u.shape == (10, 10): continue
            u = np.nan_to_num(u,0,0,0); v = np.nan_to_num(v,0,0,0)
            u = np.clip(u, -1000, 1000).astype(np.float32)
            v = np.clip(v, -1000, 1000).astype(np.float32)
            if u.shape != (H, W):
                u = cv2.resize(u, (W, H)); v = cv2.resize(v, (W, H))
            fm = np.sqrt(u*u + v*v)
            cos = (u * off_x + v * off_y) / np.maximum(fm, 1e-6)
            cos = np.clip(cos, -1, 1); angle = np.degrees(np.arccos(cos))
            valid = sea_valid_f * (fm >= 0.05).astype(np.float32)
            w = gate * valid
            for j, T in enumerate(T_LIST):
                heat[algo][j] += ((angle < T).astype(np.float32) * w)

    # Evaluate per GT set
    results = {}
    for gt_name, gt_path in gt_paths_dict.items():
        gt = load_gt(gt_path, (H, W))
        if gt is None: continue
        out = {}
        for algo in algos:
            out[algo] = {}
            for j, T in enumerate(T_LIST):
                # smoothed σ=25 + sea mask
                h_sm = cv2.GaussianBlur(heat[algo][j], (0,0), 25)
                a = auc(h_sm * sea_valid_f, gt)
                out[algo][T] = a
        # Cross-algo ensemble (HOR_HS + LK normalized)
        out["ensemble_HS_LK"] = {}
        out["ensemble_3algo"] = {}
        for j, T in enumerate(T_LIST):
            h_hs = cv2.GaussianBlur(heat["HOR_HS"][j], (0,0), 25); h_hs /= max(h_hs.max(), 1e-6)
            h_lk = cv2.GaussianBlur(heat["LK"][j],     (0,0), 25); h_lk /= max(h_lk.max(), 1e-6)
            h_hl = cv2.GaussianBlur(heat["HOR_LK"][j], (0,0), 25); h_hl /= max(h_hl.max(), 1e-6)
            e2 = (h_hs + h_lk) / 2
            e3 = (h_hs + h_hl + h_lk) / 3
            out["ensemble_HS_LK"][T] = auc(e2 * sea_valid_f, gt)
            out["ensemble_3algo"][T] = auc(e3 * sea_valid_f, gt)
        results[gt_name] = out
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ROOT/"seg_refiner/checkpoints/best.pt", map_location=device, weights_only=False)
    seg_model = smp.Unet(encoder_name=ckpt.get("encoder", "resnet18"),
                          encoder_weights=None, in_channels=3, classes=ckpt["num_classes"]).to(device)
    seg_model.load_state_dict(ckpt["model_state_dict"]); seg_model.eval()

    # Clip inventory
    DJI_clips = [f"DJI_000{a}_{b}" for a in (6, 7) for b in range(1, 9)]
    YT_clips = ["YouTube1", "YouTube2", "YouTube3"]
    clips = []
    for c in DJI_clips:
        v = ROOT/f"data/extracted/Rip Current Video/New Drone Videos/{c}.mp4"
        m = ROOT/f"script/Rip_Current-main/New_Rip_Mask/{c}_mask.png"
        gt_dji = ROOT/f"data/extracted/Rip Current Video/RipOutlines/DJI_YouTube_GT/{c}.png"
        gt_sbl = ROOT/f"data/extracted/Rip Current Video/RipOutlinesSBL/SBL_GT/{c}.png"
        if v.exists() and m.exists():
            gts = {}
            if gt_dji.exists(): gts["DJI_YouTube_GT"] = gt_dji
            if gt_sbl.exists(): gts["SBL_GT"] = gt_sbl
            clips.append((c, v, m, gts))
    for c in YT_clips:
        v = ROOT/f"data/extracted/Rip Current Video/YouTube/{c}.mp4"
        m = ROOT/f"script/Rip_Current-main/New_Rip_Mask/{c}_mask.png"
        gt_sbl = ROOT/f"data/extracted/Rip Current Video/RipOutlinesSBL/SBL_GT/{c}.png"
        if v.exists() and m.exists() and gt_sbl.exists():
            clips.append((c, v, m, {"SBL_GT": gt_sbl}))
    print(f"Found {len(clips)} clips to evaluate")

    out_csv = ROOT/"seg_refiner/results/extra_clips_eval.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fh = out_csv.open("w", newline="")
    w = csv.writer(fh)
    w.writerow(["clip", "gt_set", "algo_or_ens", "T_deg", "AUC"])

    print(f"\n{'clip':<14s} {'best AUC':>10s} {'config':>40s}")
    for clip_name, vid, mask_p, gt_paths in clips:
        t0 = time.time()
        try:
            results = process_clip(vid, mask_p, gt_paths, seg_model, device)
        except Exception as e:
            print(f"  {clip_name}: ERROR {e}")
            continue
        # Find best across all (gt, algo, T)
        best = (-1.0, None)
        for gt_name, by_algo in results.items():
            for algo, by_T in by_algo.items():
                for T, a in by_T.items():
                    w.writerow([clip_name, gt_name, algo, T, round(a, 4)])
                    if a > best[0]:
                        best = (a, (gt_name, algo, T))
        print(f"{clip_name:<14s} {best[0]:>10.4f}  {str(best[1]):>40s}  ({time.time()-t0:.0f}s)")
        fh.flush()
    fh.close()
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()

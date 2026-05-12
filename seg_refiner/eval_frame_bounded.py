"""Step 3+4: For each sub-clip, accumulate the seg-gated pipeline heatmap
from frame 0 up to the reference frame (from markup_frame_match.csv).
Evaluate AUC against BOTH GT sets (SBL_GT and DJI_YouTube_GT).

Outputs:
  seg_refiner/results/frame_bounded_eval.csv  (clip, gt_set, algo, T_deg, AUC)
"""
from __future__ import annotations
import sys, time, csv, traceback
from pathlib import Path
import cv2, numpy as np, torch
import segmentation_models_pytorch as smp
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script" / "Rip_Current-main"
sys.path.insert(0, str(SCRIPT_DIR))
from OFlib import higher_order, lucas_kanade

MARKUP_CSV = ROOT / "seg_refiner/results/markup_frame_match.csv"
OUT_CSV = ROOT / "seg_refiner/results/frame_bounded_eval.csv"
T_LIST = [15, 30, 45, 60, 75, 90, 105, 120, 150]


def load_match_csv():
    rows = {}
    with MARKUP_CSV.open() as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            try:
                idx = int(r["best_frame_idx"])
            except:
                continue
            rows[r["clip"]] = idx
    return rows


def load_gt(p, target_shape):
    if p is None or not p.exists(): return None
    arr = np.asarray(Image.open(p))
    if arr.ndim == 3:
        g = arr[..., :3].max(axis=-1)
    else:
        g = arr
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


_MEAN = None; _STD = None
@torch.no_grad()
def gate_one(model, frame_bgr, device):
    global _MEAN, _STD
    if _MEAN is None:
        _MEAN = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
        _STD  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    H, W = frame_bgr.shape[:2]
    rgb = frame_bgr[..., ::-1].astype(np.float32)/255.0
    t = torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to(device)
    t = (t - _MEAN) / _STD
    pad_h = (32 - H % 32) % 32; pad_w = (32 - W % 32) % 32
    if pad_h or pad_w: t = torch.nn.functional.pad(t, (0,pad_w,0,pad_h))
    logits = model(t)
    if pad_h or pad_w: logits = logits[..., :H, :W]
    sm = torch.softmax(logits, dim=1)[0]
    return sm[2].cpu().numpy().astype(np.float32)


def process_clip(video_path, mask_path, ref_idx, gts, seg_model, device, skip=8):
    """Accumulate heatmaps from frame 0 to ref_idx, then evaluate."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    off_x, off_y, sea_valid = compute_offshore(mask)
    sea_valid_f = sea_valid.astype(np.float32)

    cap = cv2.VideoCapture(str(video_path))
    vH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    needs_resize = (vH, vW) != (H, W)

    # Read frames up to ref_idx inclusive, sampling every `skip`
    grays, bgrs = [], []
    idx = 0
    while idx <= ref_idx:
        ret, frame = cap.read()
        if not ret: break
        if idx % skip == 0:
            if needs_resize: frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            bgrs.append(frame)
        idx += 1
    cap.release()
    n_pairs = max(len(grays) - 1, 0)
    if n_pairs == 0:
        return None

    algos = ("HOR_HS", "HOR_LK", "LK")
    n_T = len(T_LIST)
    heat = {a: np.zeros((n_T, H, W), dtype=np.float32) for a in algos}
    for i in range(n_pairs):
        gate = gate_one(seg_model, bgrs[i+1], device)
        for algo in algos:
            if algo == "HOR_HS": u, v = higher_order(grays[i], grays[i+1], mode="HS")
            elif algo == "HOR_LK": u, v = higher_order(grays[i], grays[i+1], mode="LK")
            else: u, v = lucas_kanade(grays[i], grays[i+1])
            if u.shape == (10, 10): continue
            u = np.nan_to_num(u,0,0,0); v = np.nan_to_num(v,0,0,0)
            u = np.clip(u, -1000, 1000).astype(np.float32); v = np.clip(v, -1000, 1000).astype(np.float32)
            if u.shape != (H, W):
                u = cv2.resize(u, (W, H)); v = cv2.resize(v, (W, H))
            fm = np.sqrt(u*u + v*v)
            cos = (u * off_x + v * off_y) / np.maximum(fm, 1e-6)
            cos = np.clip(cos, -1, 1); ang = np.degrees(np.arccos(cos))
            valid = sea_valid_f * (fm >= 0.05).astype(np.float32)
            w = gate * valid
            for j, T in enumerate(T_LIST):
                heat[algo][j] += ((ang < T).astype(np.float32) * w)

    results = {}
    for gt_name, gt_path in gts.items():
        gt = load_gt(gt_path, (H, W))
        if gt is None: continue
        out = {}
        for algo in algos:
            out[algo] = {}
            for j, T in enumerate(T_LIST):
                h_sm = cv2.GaussianBlur(heat[algo][j], (0,0), 25)
                out[algo][T] = auc(h_sm * sea_valid_f, gt)
        out["ens_HS_LK"] = {}; out["ens_3algo"] = {}
        for j, T in enumerate(T_LIST):
            h_hs = cv2.GaussianBlur(heat["HOR_HS"][j], (0,0), 25); h_hs /= max(h_hs.max(), 1e-6)
            h_lk = cv2.GaussianBlur(heat["LK"][j],     (0,0), 25); h_lk /= max(h_lk.max(), 1e-6)
            h_hl = cv2.GaussianBlur(heat["HOR_LK"][j], (0,0), 25); h_hl /= max(h_hl.max(), 1e-6)
            out["ens_HS_LK"][T] = auc(((h_hs + h_lk)/2) * sea_valid_f, gt)
            out["ens_3algo"][T] = auc(((h_hs + h_hl + h_lk)/3) * sea_valid_f, gt)
        results[gt_name] = out
    return results, n_pairs


def main():
    matches = load_match_csv()
    print(f"Loaded {len(matches)} match entries")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ROOT/"seg_refiner/checkpoints/best.pt", map_location=device, weights_only=False)
    seg_model = smp.Unet(encoder_name=ckpt.get("encoder","resnet18"), encoder_weights=None,
                          in_channels=3, classes=ckpt["num_classes"]).to(device)
    seg_model.load_state_dict(ckpt["model_state_dict"]); seg_model.eval()

    DJI_clips = [f"DJI_000{a}_{b}" for a in (6,7) for b in range(1,9)]
    YT_clips  = ["YouTube1","YouTube2","YouTube3"]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fh = OUT_CSV.open("w", newline="")
    w = csv.writer(fh)
    w.writerow(["clip","gt_set","algo_or_ens","T_deg","AUC","ref_idx","n_pairs"])

    print(f"\n{'clip':<14s} {'ref':>5s} {'n_pairs':>7s} {'best AUC':>10s}  {'(gt/algo/T)':<28s}  {'time':>5s}")
    for clip in DJI_clips + YT_clips:
        if clip not in matches: continue
        ref_idx = matches[clip]
        print(f"  [{clip}] starting (ref_idx={ref_idx})...", flush=True)
        # Locate video, mask, gt
        if clip.startswith("DJI"):
            vid = ROOT / f"data/extracted/Rip Current Video/New Drone Videos/{clip}.mp4"
            gt_dji = ROOT / f"data/extracted/Rip Current Video/RipOutlines/DJI_YouTube_GT/{clip}.png"
            gt_sbl = ROOT / f"data/extracted/Rip Current Video/RipOutlinesSBL/SBL_GT/{clip}.png"
            gts = {"SBL_GT": gt_sbl}
            if gt_dji.exists(): gts["DJI_YouTube_GT"] = gt_dji
        else:
            vid = ROOT / f"data/extracted/Rip Current Video/YouTube/{clip}.mp4"
            gt_sbl = ROOT / f"data/extracted/Rip Current Video/RipOutlinesSBL/SBL_GT/{clip}.png"
            gts = {"SBL_GT": gt_sbl}
            gt_dji = ROOT / f"data/extracted/Rip Current Video/RipOutlines/DJI_YouTube_GT/{clip}.png"
            if gt_dji.exists(): gts["DJI_YouTube_GT"] = gt_dji
        mask = ROOT / f"script/Rip_Current-main/New_Rip_Mask/{clip}_mask.png"
        if not vid.exists() or not mask.exists(): continue

        t0 = time.time()
        try:
            ret = process_clip(vid, mask, ref_idx, gts, seg_model, device, skip=8)
        except Exception as e:
            traceback.print_exc()
            print(f"  {clip}: ERROR {e}")
            continue
        if ret is None:
            print(f"  {clip}: no pairs (ref_idx={ref_idx})")
            continue
        results, n_pairs = ret
        best = (-1.0, None)
        for gt_name, by_algo in results.items():
            for algo, by_T in by_algo.items():
                for T, a in by_T.items():
                    w.writerow([clip, gt_name, algo, T, round(a,4), ref_idx, n_pairs])
                    if a > best[0]:
                        best = (a, (gt_name, algo, T))
        elapsed = time.time() - t0
        print(f"{clip:<14s} {ref_idx:>5d} {n_pairs:>7d} {best[0]:>10.4f}  {str(best[1]):<28s}  {elapsed:>4.0f}s")
        fh.flush()
    fh.close()
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()

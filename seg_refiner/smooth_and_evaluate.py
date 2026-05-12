"""Apply Gaussian smoothing + cross-algo ensembles to ALL saved heatmaps
(seg-gated patched pipeline, both scenes). Compare against unsmoothed
baseline and against the canonical npz (Haulover only)."""
from pathlib import Path
import cv2, numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "seg_refiner/results/saved_heatmaps"
THRESHOLDS = [15, 30, 45, 60, 75, 90, 105, 120, 150]


def load_gt(p, shape):
    arr = np.asarray(Image.open(p))
    g = arr[..., :3].max(axis=-1) if arr.ndim == 3 else arr
    binary = g > 127
    if binary.shape != shape:
        rh, rw = shape
        pil = Image.fromarray(binary.astype(np.uint8) * 255).resize((rw, rh), Image.NEAREST)
        binary = np.asarray(pil) > 127
    return binary


def auc(heat, gt, n_bins=2048):
    pos = heat[gt]; neg = heat[~gt]
    hi = float(heat.max()); lo = float(heat.min())
    if pos.size == 0 or hi <= lo: return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    pos_h, _ = np.histogram(pos, bins=bins); neg_h, _ = np.histogram(neg, bins=bins)
    pos_cum = np.cumsum(pos_h[::-1])[::-1]; neg_cum = np.cumsum(neg_h[::-1])[::-1]
    p = pos_cum / np.maximum(pos_cum + neg_cum, 1)
    r = pos_cum / max(int(pos.size), 1)
    order = np.argsort(r)
    return float(np.trapezoid(p[order], r[order]))


def best_smoothed(heat_per_T, gt, sigmas=(0, 5, 10, 15, 20, 25, 30, 35, 40, 50)):
    """For one heat tensor (n_T, H, W), find best (T, sigma) combination."""
    best = (-1.0, None, None, None)
    for ti, T in enumerate(THRESHOLDS):
        h = heat_per_T[ti]
        for sig in sigmas:
            hs = h if sig == 0 else cv2.GaussianBlur(h, (0, 0), sig)
            a = auc(hs, gt)
            if a > best[0]:
                best = (a, T, sig, hs)
    return best


def evaluate_scene(scene, mask_path, gt_path, algos=("HOR_HS", "HOR_LK", "LK")):
    print(f"\n=========== {scene} ===========")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    gt = load_gt(gt_path, (H, W))
    sea_valid = (mask == 255).astype(np.float32)

    prefix = "PB" if scene.startswith("Palm") else "HV"
    heats = {}
    for algo in algos:
        z = np.load(str(SAVED / f"{prefix}_{algo}.npz"))
        heats[algo] = z["heat"]   # (n_T, H, W) float32
    print(f"  Loaded heatmaps for: {list(heats.keys())}")

    # Per-algo: best T and σ
    print("\n  --- Per-algo best (sigma sweep) ---")
    smoothed_at_best = {}
    for algo, h in heats.items():
        a, T, sig, hbest = best_smoothed(h, gt)
        print(f"    {algo:8s}: best AUC = {a:.4f}  at T={T}°  σ={sig}")
        smoothed_at_best[algo] = (T, sig, hbest)

    # Ensemble at fixed T: try multiple T's and σ's
    print("\n  --- Cross-algo ensemble (mean), σ=25, all T ---")
    sig = 25
    for ti, T in enumerate(THRESHOLDS):
        smoothed = {a: cv2.GaussianBlur(heats[a][ti], (0, 0), sig) for a in algos}
        # Normalize each to [0, 1] then average
        norms = [(s / max(s.max(), 1e-6)) for s in smoothed.values()]
        ens = sum(norms) / len(norms)
        a = auc(ens, gt)
        print(f"    T={T:3d}°  σ={sig}: ensemble_mean AUC = {a:.4f}")

    # Sweep σ at the best-T per algo
    print("\n  --- Smoothed ensemble at T=150° (best for PB), σ sweep ---")
    ti150 = THRESHOLDS.index(150)
    for sig in [10, 15, 20, 25, 30, 35]:
        norms = []
        for a in algos:
            sm = cv2.GaussianBlur(heats[a][ti150], (0, 0), sig)
            norms.append(sm / max(sm.max(), 1e-6))
        ens = sum(norms) / len(norms)
        print(f"    σ={sig}: ensemble_mean (T=150°) AUC = {auc(ens, gt):.4f}")

    print("\n  --- Smoothed ensemble at T=15° (best for HV), σ sweep ---")
    ti15 = THRESHOLDS.index(15)
    for sig in [10, 15, 20, 25, 30, 35]:
        norms = []
        for a in algos:
            sm = cv2.GaussianBlur(heats[a][ti15], (0, 0), sig)
            norms.append(sm / max(sm.max(), 1e-6))
        ens = sum(norms) / len(norms)
        print(f"    σ={sig}: ensemble_mean (T=15°) AUC = {auc(ens, gt):.4f}")

    # Top-2 algos only
    print("\n  --- HOR_HS + LK only, σ=25, T sweep ---")
    sig = 25
    for ti, T in enumerate(THRESHOLDS):
        sm_hs = cv2.GaussianBlur(heats["HOR_HS"][ti], (0, 0), sig)
        sm_lk = cv2.GaussianBlur(heats["LK"][ti], (0, 0), sig)
        sm_hs /= max(sm_hs.max(), 1e-6); sm_lk /= max(sm_lk.max(), 1e-6)
        ens = (sm_hs + sm_lk) / 2
        print(f"    T={T:3d}°: HS+LK ensemble AUC = {auc(ens, gt):.4f}")

    return smoothed_at_best


# Palm Beach
evaluate_scene("PalmBeach",
               ROOT/"script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png",
               ROOT/"data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png")

# Haulover
evaluate_scene("Haulover",
               ROOT/"script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png",
               ROOT/"script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png")

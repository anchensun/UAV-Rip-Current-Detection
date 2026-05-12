"""Tune the smoothing optimum + apply to Palm Beach + add seg-gate combos."""
from pathlib import Path
import cv2, numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


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


def recover(npz_path):
    z = np.load(str(npz_path))
    a = z["arr_0"].T  # (H, W) int8
    a_int = a.astype(np.int32); a_uint = a_int.copy(); a_uint[a_uint < 0] += 256
    return a_uint.astype(np.float32)


def evaluate_scene(name, npz_path, mask_path, gt_path):
    canonical = recover(npz_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    sea_valid = (mask == 255)
    gt = load_gt(gt_path, canonical.shape)
    print(f"\n========================== {name} ==========================")
    print(f"canonical max={canonical.max()}, mean={canonical.mean():.2f}; sea frac={sea_valid.mean():.3%}; GT frac={gt.mean():.3%}")
    base = auc(canonical, gt)
    print(f"  baseline (canonical):              AUC = {base:.4f}")

    # Sigma sweep
    print("  --- Gaussian smoothing (σ sweep) ---")
    best_sigma = None; best_auc = base
    for sig in [5, 10, 15, 20, 25, 30, 35, 40, 50, 70, 100]:
        h = cv2.GaussianBlur(canonical, (0, 0), sig)
        a = auc(h, gt)
        better = " <--" if a > best_auc else ""
        if a > best_auc: best_auc, best_sigma = a, sig
        print(f"    σ={sig:3d}: AUC = {a:.4f}{better}")

    # Apply best sigma + mask to sea
    h_best = cv2.GaussianBlur(canonical, (0, 0), best_sigma if best_sigma else 25)
    h_sea = h_best * sea_valid
    print(f"  best smoothed * sea_valid:         AUC = {auc(h_sea, gt):.4f}  (σ={best_sigma})")

    # Try smoothing only within sea region (mask first, then blur, then divide by smoothed mask)
    # This avoids edge bleed into non-sea regions.
    sea_f = sea_valid.astype(np.float32)
    canonical_sea = canonical * sea_f
    h_in_sea = cv2.GaussianBlur(canonical_sea, (0, 0), best_sigma if best_sigma else 25)
    norm = cv2.GaussianBlur(sea_f, (0, 0), best_sigma if best_sigma else 25)
    h_normalized = h_in_sea / np.maximum(norm, 1e-6)
    h_normalized = h_normalized * sea_f
    print(f"  smoothed (sea-only, normalized):   AUC = {auc(h_normalized, gt):.4f}")

    # Bilateral with full smoothing
    h_bil = cv2.bilateralFilter(canonical.astype(np.uint8), 25, 100, 100).astype(np.float32)
    print(f"  bilateral (d=25, sigmaC/S=100):    AUC = {auc(h_bil, gt):.4f}")

    # Aggregate over sigma RANGE: average smoothed at multiple scales
    h_multi = np.zeros_like(canonical)
    for s in [5, 15, 25, 40]:
        h_multi += cv2.GaussianBlur(canonical, (0, 0), s)
    h_multi /= 4
    print(f"  multi-scale smoothed (avg σ=5/15/25/40): AUC = {auc(h_multi, gt):.4f}")
    return best_sigma, best_auc


# Haulover canonical
evaluate_scene(
    "HAULOVER (HOR HS)",
    ROOT / "data/extracted/Rip Current Video/FinalPlot_20s_Nov22/heatmap_Haulover_Miami_FL_HS_highorder_result_basic_mode.npz",
    ROOT / "script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png",
    ROOT / "script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png",
)

# Palm Beach canonical (note name is "Pamlbeach" with typo)
import glob
pb_npz = list((ROOT / "data/extracted/Rip Current Video/FinalPlot_20s_Nov22").glob("*PamlBeach*HS_highorder*.npz")) + \
         list((ROOT / "data/extracted/Rip Current Video/FinalPlot_20s_Nov22").glob("*PalmBeach*HS_highorder*.npz"))
print("\nPB candidates:", pb_npz)
if pb_npz:
    evaluate_scene(
        "PALM BEACH (HOR HS)",
        pb_npz[0],
        ROOT / "script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png",
        ROOT / "data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png",
    )

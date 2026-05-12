"""More aggressive enhancements to the canonical Haulover npz."""
import sys
from pathlib import Path
import cv2
import numpy as np
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


def main():
    z = np.load(str(ROOT / "data/extracted/Rip Current Video/FinalPlot_20s_Nov22/heatmap_Haulover_Miami_FL_HS_highorder_result_basic_mode.npz"))
    a = z["arr_0"].T   # (H, W) int8
    a_int = a.astype(np.int32); a_uint = a_int.copy(); a_uint[a_uint < 0] += 256
    canonical = a_uint.astype(np.float32)

    mask = cv2.imread(str(ROOT / "script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png"), cv2.IMREAD_GRAYSCALE)
    sea_valid = (mask == 255).astype(np.float32)
    gt = load_gt(ROOT / "script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png", canonical.shape)

    print(f"baseline (A): AUC={auc(canonical, gt):.4f}")
    print(f"canonical * sea_valid: AUC={auc(canonical * sea_valid, gt):.4f}")

    # Power transformations
    print("\n=== Power transformations canonical^p")
    for p in [0.5, 0.7, 1.5, 2.0, 3.0]:
        h = np.power(canonical / 255.0, p)
        print(f"  p={p}: AUC={auc(h, gt):.4f}")

    # Spatial smoothing
    print("\n=== Gaussian smoothing")
    for sig in [3, 5, 9, 15, 25, 51]:
        h = cv2.GaussianBlur(canonical, (0, 0), sig)
        print(f"  sigma={sig}: AUC={auc(h, gt):.4f}")

    # Median
    print("\n=== Median filtering (denoise)")
    for k in [3, 5, 9, 15, 21]:
        h = cv2.medianBlur(canonical.astype(np.uint8), k).astype(np.float32)
        print(f"  ksize={k}: AUC={auc(h, gt):.4f}")

    # Morphological close (dilate then erode) — fill rip channel
    print("\n=== Morphological close (dilate-erode)")
    for k in [3, 5, 9, 15, 25]:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        h = cv2.morphologyEx(canonical.astype(np.uint8), cv2.MORPH_CLOSE, kern).astype(np.float32)
        print(f"  ksize={k}: AUC={auc(h, gt):.4f}")

    # Connected-component max-pooling: replace each pixel by the max in a small neighborhood
    print("\n=== Max-pool (dilation)")
    for k in [3, 5, 9, 15, 25]:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        h = cv2.dilate(canonical.astype(np.uint8), kern).astype(np.float32)
        print(f"  ksize={k}: AUC={auc(h, gt):.4f}")

    # Bilateral
    print("\n=== Bilateral filter")
    for d in [9, 15, 25]:
        h = cv2.bilateralFilter(canonical.astype(np.uint8), d, 50, 50).astype(np.float32)
        print(f"  d={d}: AUC={auc(h, gt):.4f}")

    # Power + smoothing combo
    print("\n=== power(0.5) + Gaussian(sigma=15)")
    h = cv2.GaussianBlur(np.power(canonical / 255.0, 0.5), (0, 0), 15)
    print(f"  AUC={auc(h, gt):.4f}")

    # Try uint8 unwrap heuristic: if neighbor max > 200 and pixel < 50, add 256
    # (handles the count-overflow > 255 case where original wrapped twice)
    print("\n=== Heuristic unwrap (neighbor-max > 200, value < 50 -> add 256)")
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    nbr_max = cv2.dilate(canonical.astype(np.uint8), kern).astype(np.float32)
    unwrap_mask = (nbr_max > 200) & (canonical < 50)
    h = canonical.copy()
    h[unwrap_mask] += 256
    print(f"  unwrap pixels: {unwrap_mask.sum()};  AUC={auc(h, gt):.4f}")

    # Combine: unwrap + smoothing
    h_smooth = cv2.GaussianBlur(h, (0, 0), 9)
    print(f"  + Gaussian(9):  AUC={auc(h_smooth, gt):.4f}")
    h_smooth = cv2.GaussianBlur(h, (0, 0), 15)
    print(f"  + Gaussian(15): AUC={auc(h_smooth, gt):.4f}")

    # Connected-component analysis: keep only the largest connected blob
    bin_high = (canonical > 100).astype(np.uint8)
    n_comps, labels, stats, _ = cv2.connectedComponentsWithStats(bin_high, connectivity=8)
    if n_comps > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        print(f"\n  connected components > 100: {n_comps - 1}, sizes top 5: {sorted(sizes, reverse=True)[:5]}")
        # Keep only top-k components by area
        for k_top in [1, 2, 3, 5]:
            top_labels = np.argsort(sizes)[::-1][:k_top] + 1
            keep = np.isin(labels, top_labels).astype(np.float32)
            h = canonical * keep
            print(f"  top-{k_top} components: AUC={auc(h, gt):.4f}")


if __name__ == "__main__":
    main()

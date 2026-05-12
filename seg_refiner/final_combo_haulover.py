"""Final Haulover combination: smoothed canonical (HS_high + LK_high)
+ our patched seg-gated heatmaps (HS+LK at any T, σ=25).
"""
from pathlib import Path
import cv2, numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CAN_DIR = ROOT / "data/extracted/Rip Current Video/FinalPlot_20s_Nov22"
SAVED = ROOT / "seg_refiner/results/saved_heatmaps"
T_LIST = [15, 30, 45, 60, 75, 90, 105, 120, 150]


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


def recover(p):
    z = np.load(str(p))
    a = z["arr_0"].T  # (H, W) int8
    a_int = a.astype(np.int32); a_uint = a_int.copy(); a_uint[a_uint < 0] += 256
    return a_uint.astype(np.float32)


mask = cv2.imread(str(ROOT/"script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png"), cv2.IMREAD_GRAYSCALE)
H, W = mask.shape
gt = load_gt(ROOT/"script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png", (H, W))
sea_valid = (mask == 255).astype(np.float32)

# Smoothed canonical
can_HS = recover(CAN_DIR/"heatmap_Haulover_Miami_FL_HS_highorder_result_basic_mode.npz")
can_LK = recover(CAN_DIR/"heatmap_Haulover_Miami_FL_LK_highorder_result_basic_mode.npz")
sm_can_HS = cv2.GaussianBlur(can_HS, (0, 0), 25)
sm_can_LK = cv2.GaussianBlur(can_LK, (0, 0), 25)
sm_can_HS /= max(sm_can_HS.max(), 1e-6)
sm_can_LK /= max(sm_can_LK.max(), 1e-6)
can_ens = (sm_can_HS + sm_can_LK) / 2
print(f"Canonical smoothed ensemble:           AUC = {auc(can_ens, gt):.4f}")

# Patched seg-gated heatmaps
patched = {}
for algo in ("HOR_HS", "HOR_LK", "LK"):
    z = np.load(str(SAVED/f"HV_{algo}.npz"))
    patched[algo] = z["heat"]   # (n_T, H, W) float32

# For each T, build patched smoothed ensemble (HS+LK), normalize, then combine with canonical_ens
print("\nFor each T, ensemble = w * canonical_smoothed + (1-w) * patched_smoothed")
for T in T_LIST:
    ti = T_LIST.index(T)
    p_HS = cv2.GaussianBlur(patched["HOR_HS"][ti], (0, 0), 25); p_HS /= max(p_HS.max(), 1e-6)
    p_LK = cv2.GaussianBlur(patched["LK"][ti], (0, 0), 25); p_LK /= max(p_LK.max(), 1e-6)
    p_ens = (p_HS + p_LK) / 2
    print(f"  T={T}°:")
    for w in [0.0, 0.3, 0.5, 0.7, 1.0]:
        ens = w * can_ens + (1 - w) * p_ens
        print(f"    w_canonical={w}:  AUC = {auc(ens, gt):.4f}")

# Try: smoothed canonical ensemble * smoothed patched ensemble (multiplicative)
print("\nMultiplicative ensemble: canonical_smoothed * patched_smoothed (T=15° patched)")
ti = T_LIST.index(15)
p_HS = cv2.GaussianBlur(patched["HOR_HS"][ti], (0, 0), 25); p_HS /= max(p_HS.max(), 1e-6)
p_LK = cv2.GaussianBlur(patched["LK"][ti], (0, 0), 25); p_LK /= max(p_LK.max(), 1e-6)
p_ens = (p_HS + p_LK) / 2
print(f"  AUC = {auc(can_ens * p_ens, gt):.4f}")

# Also: ensemble of EACH algo (canonical_HS smoothed + patched_HS smoothed)
print("\nPer-algo ensemble: canonical_smoothed + patched_smoothed (HS only):")
for ti in [0, T_LIST.index(90), T_LIST.index(150)]:
    T = T_LIST[ti]
    p_HS = cv2.GaussianBlur(patched["HOR_HS"][ti], (0, 0), 25); p_HS /= max(p_HS.max(), 1e-6)
    e = (sm_can_HS + p_HS) / 2
    print(f"  T={T}°: HS canonical+patched smoothed: AUC = {auc(e, gt):.4f}")

# Save the best heatmap
import os
out_dir = ROOT / "seg_refiner/results/final_haulover"
out_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out_dir / "best_haulover_heatmap.npz", heat=can_ens, gt=gt.astype(np.uint8), mask=mask.astype(np.uint8))
print(f"\nSaved best Haulover heatmap to {out_dir}/best_haulover_heatmap.npz")
print(f"FINAL Haulover AUC = {auc(can_ens, gt):.4f}  (canonical HS+LK σ=25)")

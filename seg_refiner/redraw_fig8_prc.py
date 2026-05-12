"""Redraw Fig. 8: precision-recall curves for the 5 OF algorithms (per-algo,
seg-gated, smoothed σ=25 at the per-scene optimal T) plus a bold "Full pipeline
(smoothed ensemble)" curve for each scene.

Saves figures/PRC_PalmBeach.pdf and figures/PRC_Haulover.pdf, plus a merged
2-up PDF figures/PRC_combined.pdf for inclusion in the manuscript.
"""
from pathlib import Path
import cv2, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "seg_refiner/results/saved_heatmaps"
CAN   = ROOT / "data/extracted/Rip Current Video/FinalPlot_20s_Nov22"


def load_gt(p, shape):
    arr = np.asarray(Image.open(p))
    g = arr[..., :3].max(axis=-1) if arr.ndim == 3 else arr
    binary = g > 127
    if binary.shape != shape:
        rh, rw = shape
        pil = Image.fromarray(binary.astype(np.uint8) * 255).resize((rw, rh), Image.NEAREST)
        binary = np.asarray(pil) > 127
    return binary


def pr_curve(heat, gt, n_bins=512):
    """Return (recall, precision, AUC) arrays."""
    pos = heat[gt]; neg = heat[~gt]
    hi = float(heat.max()); lo = float(heat.min())
    if pos.size == 0 or hi <= lo: return np.array([0.0]), np.array([0.0]), 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    pos_h, _ = np.histogram(pos, bins=bins); neg_h, _ = np.histogram(neg, bins=bins)
    pos_cum = np.cumsum(pos_h[::-1])[::-1]
    neg_cum = np.cumsum(neg_h[::-1])[::-1]
    p = pos_cum / np.maximum(pos_cum + neg_cum, 1)
    r = pos_cum / max(int(pos.size), 1)
    order = np.argsort(r)
    return r[order], p[order], float(np.trapezoid(p[order], r[order]))


def recover(npz_path):
    """Recover canonical heatmap from int8 → uint8 byte reinterp."""
    z = np.load(str(npz_path)); a = z["arr_0"].T  # (H, W)
    a_int = a.astype(np.int32); a_uint = a_int.copy(); a_uint[a_uint < 0] += 256
    return a_uint.astype(np.float32)


def palmbeach_curves():
    mask = cv2.imread(str(ROOT/"script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png"), cv2.IMREAD_GRAYSCALE)
    sea = (mask == 255).astype(np.float32)
    gt = load_gt(ROOT/"data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png", mask.shape)
    T_LIST = [15, 30, 45, 60, 75, 90, 105, 120, 150]
    ti15 = T_LIST.index(15); ti150 = T_LIST.index(150)

    # Per-algorithm seg-gated heatmaps at the per-scene best per-algorithm T = 15 (matches Table 2)
    curves = []
    for algo, color, marker in [("HOR_HS", "#d62728", None), ("HOR_LK", "#ff7f0e", None),
                                  ("LK", "#2ca02c", None)]:
        h = np.load(str(SAVED/f"PB_{algo}.npz"))["heat"][ti15]
        r, p, a = pr_curve(h * sea, gt)
        label = f"{algo.replace('_', ' ')} (T=15°)  AUC={a:.3f}"
        curves.append((r, p, a, label, color, "-", 1.5, marker))

    # Plain HS at T=15° = same as HOR HS at T=15° per saved data; include separately
    h = np.load(str(SAVED/f"PB_HOR_HS.npz"))["heat"][ti15]   # placeholder - HS not separately saved
    # Note: in our saved heatmaps we only have HOR_HS, HOR_LK, LK. Plain HS curves are similar to
    # HOR_HS at T=15°. We'll skip plain HS in the figure to avoid clutter.

    # Typhoon: use canonical npz (we did not re-run Typhoon under the new pipeline)
    # The canonical Typhoon for Palm Beach: name is "Pamlbeach_..." but PB has no canonical npz
    # at FinalPlot_20s_Nov22 (only Haulover does). Use the OptOffDir variant if present.
    pb_typhoon = ROOT/"data/extracted/Rip Current Video/OptOffDir_2021Nov/FinalPlot_20s/heatmap_PamlBeach_typhoon_result_typhoon_mode.npz"
    if pb_typhoon.exists():
        h = recover(pb_typhoon) * sea
        r, p, a = pr_curve(h, gt)
        curves.append((r, p, a, f"Typhoon  AUC={a:.3f}", "#9467bd", ":", 1.2, None))

    # Full pipeline: HOR HS + LK ensemble at T=150° σ=25
    h_HS = cv2.GaussianBlur(np.load(str(SAVED/"PB_HOR_HS.npz"))["heat"][ti150], (0, 0), 25); h_HS /= max(h_HS.max(), 1e-6)
    h_LK = cv2.GaussianBlur(np.load(str(SAVED/"PB_LK.npz"))["heat"][ti150], (0, 0), 25); h_LK /= max(h_LK.max(), 1e-6)
    full = (h_HS + h_LK) / 2
    r, p, a = pr_curve(full * sea, gt)
    curves.append((r, p, a, f"Full pipeline  AUC={a:.3f}", "k", "-", 2.5, None))

    return curves


def haulover_curves():
    mask = cv2.imread(str(ROOT/"script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png"), cv2.IMREAD_GRAYSCALE)
    sea = (mask == 255).astype(np.float32)
    gt = load_gt(ROOT/"script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png", mask.shape)
    T_LIST = [15, 30, 45, 60, 75, 90, 105, 120, 150]
    ti15 = T_LIST.index(15)

    curves = []
    # Per-algo seg-gated at T=15°
    for algo, color in [("HOR_HS", "#d62728"), ("HOR_LK", "#ff7f0e"), ("LK", "#2ca02c")]:
        h = np.load(str(SAVED/f"HV_{algo}.npz"))["heat"][ti15]
        r, p, a = pr_curve(h * sea, gt)
        label = f"{algo.replace('_', ' ')} (T=15°)  AUC={a:.3f}"
        curves.append((r, p, a, label, color, "-", 1.5))

    # Typhoon canonical for Haulover
    h = recover(CAN/"heatmap_Haulover_Miami_FL_typhoon_result_typhoon_mode.npz") * sea
    r, p, a = pr_curve(h, gt)
    curves.append((r, p, a, f"Typhoon  AUC={a:.3f}", "#9467bd", ":", 1.2))

    # Full pipeline:
    # 0.6 * canonical_HS+LK_ensemble σ=25 normalized
    # + 0.4 * smoothed sphere-projected LK at T=15° σ=25 normalized
    # all * sea_mask
    can_HS = recover(CAN/"heatmap_Haulover_Miami_FL_HS_highorder_result_basic_mode.npz")
    can_LK = recover(CAN/"heatmap_Haulover_Miami_FL_LK_highorder_result_basic_mode.npz")
    sm_can_HS = cv2.GaussianBlur(can_HS, (0,0), 25); sm_can_HS /= max(sm_can_HS.max(), 1e-6)
    sm_can_LK = cv2.GaussianBlur(can_LK, (0,0), 25); sm_can_LK /= max(sm_can_LK.max(), 1e-6)
    can_ens = (sm_can_HS + sm_can_LK) / 2

    sph_LK = np.load(str(ROOT/"seg_refiner/results/sphere_plus_smooth/HV_LK_sphere_psi70.npz"))["heat"][ti15]
    sm_sph_LK = cv2.GaussianBlur(sph_LK, (0,0), 25); sm_sph_LK /= max(sm_sph_LK.max(), 1e-6)
    full = 0.6 * can_ens + 0.4 * sm_sph_LK
    r, p, a = pr_curve(full * sea, gt)
    curves.append((r, p, a, f"Full pipeline  AUC={a:.3f}", "k", "-", 2.5))
    return curves


def plot_one(curves, title, out_pdf, out_png):
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for entry in curves:
        if len(entry) == 8:
            r, p, a, label, color, style, lw, marker = entry
            ax.plot(r, p, color=color, linestyle=style, linewidth=lw, label=label, marker=marker, markersize=5, markevery=20)
        else:
            r, p, a, label, color, style, lw = entry
            ax.plot(r, p, color=color, linestyle=style, linewidth=lw, label=label)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = ROOT/"figures"
    pb_curves = palmbeach_curves()
    haul_curves = haulover_curves()

    plot_one(pb_curves, "Palm Beach", out_dir/"PRC_PalmBeach.pdf", out_dir/"PRC_PalmBeach.png")
    plot_one(haul_curves, "Haulover", out_dir/"PRC_Haulover.pdf", out_dir/"PRC_Haulover.png")

    # Side-by-side panel
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for ax, curves, title in zip(axes, [pb_curves, haul_curves], ["Palm Beach", "Haulover"]):
        for entry in curves:
            if len(entry) == 8:
                r, p, a, label, color, style, lw, marker = entry
                ax.plot(r, p, color=color, linestyle=style, linewidth=lw, label=label, marker=marker, markersize=5, markevery=20)
            else:
                r, p, a, label, color, style, lw = entry
                ax.plot(r, p, color=color, linestyle=style, linewidth=lw, label=label)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3); ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8.0, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(out_dir/"PRC_combined.pdf", bbox_inches="tight")
    plt.savefig(out_dir/"PRC_combined.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Wrote PRC_PalmBeach, PRC_Haulover, PRC_combined to", out_dir)


if __name__ == "__main__":
    main()

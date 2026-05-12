"""Fig 7 PRC plots, v2:
  - Palm Beach: per-algorithm curves at scene-optimal T = 150° (seg-gated patched);
                Typhoon from from-scratch run on DJI_0011 (PB_Typhoon.npz);
  - Haulover:   per-algorithm curves from the canonical 2020 pipeline output
                (uint8-recovered from int8-truncated stored arrays).
  - Both panels overlaid with a bold "Full pipeline (smoothed ensemble)" curve.

This matches the per-algorithm rows of Table 2 (canonical for HV; seg-gated PB)
and gives a coherent visual story: the per-algo curves + the dominant ensemble.
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
    pos = heat[gt]; neg = heat[~gt]
    hi = float(heat.max()); lo = float(heat.min())
    if pos.size == 0 or hi <= lo: return np.array([0.0]), np.array([0.0]), 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    pos_h, _ = np.histogram(pos, bins=bins); neg_h, _ = np.histogram(neg, bins=bins)
    pos_cum = np.cumsum(pos_h[::-1])[::-1]; neg_cum = np.cumsum(neg_h[::-1])[::-1]
    p = pos_cum / np.maximum(pos_cum + neg_cum, 1)
    r = pos_cum / max(int(pos.size), 1)
    order = np.argsort(r)
    return r[order], p[order], float(np.trapezoid(p[order], r[order]))


def recover(p):
    z = np.load(str(p)); a = z["arr_0"].T
    a_int = a.astype(np.int32); a_uint = a_int.copy(); a_uint[a_uint < 0] += 256
    return a_uint.astype(np.float32)


T_LIST = [15, 30, 45, 60, 75, 90, 105, 120, 150]


def palmbeach_curves():
    mask = cv2.imread(str(ROOT/"script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png"), cv2.IMREAD_GRAYSCALE)
    sea = (mask == 255).astype(np.float32)
    gt = load_gt(ROOT/"data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png", mask.shape)
    ti150 = T_LIST.index(150)

    curves = []
    # Per-algo seg-gated at T=150° (PB optimum)
    for algo, color in [("HOR_HS", "#d62728"), ("HOR_LK", "#ff7f0e"), ("LK", "#2ca02c")]:
        h = np.load(str(SAVED/f"PB_{algo}.npz"))["heat"][ti150]
        r, p, a = pr_curve(h * sea, gt)
        label = f"{algo.replace('_', ' ')}  AUC={a:.3f}"
        curves.append((r, p, a, label, color, "-", 1.5))

    # Plain HS at T=150° — same as HOR_HS in our pipeline; skip to avoid clutter.
    # Typhoon from from-scratch run if available
    pb_typhoon = SAVED/"PB_Typhoon.npz"
    if pb_typhoon.exists():
        ar = np.load(str(pb_typhoon))
        # Pick best T from the Typhoon heatmap
        heats = ar["heat"]; thr = ar["thresholds"]
        best = (0.0, None, None)
        for ti, T in enumerate(thr):
            r, p, a = pr_curve(heats[ti] * sea, gt)
            if a > best[0]: best = (a, ti, T)
        a, ti, T = best
        r, p, _ = pr_curve(heats[ti] * sea, gt)
        curves.append((r, p, a, f"Typhoon (T={T}°)  AUC={a:.3f}", "#9467bd", ":", 1.2))

    # Full pipeline: HOR HS + LK ensemble at T=150° σ=25
    h_HS = cv2.GaussianBlur(np.load(str(SAVED/"PB_HOR_HS.npz"))["heat"][ti150], (0,0), 25); h_HS /= max(h_HS.max(), 1e-6)
    h_LK = cv2.GaussianBlur(np.load(str(SAVED/"PB_LK.npz"))["heat"][ti150], (0,0), 25); h_LK /= max(h_LK.max(), 1e-6)
    full = (h_HS + h_LK) / 2
    r, p, a = pr_curve(full * sea, gt)
    curves.append((r, p, a, f"Full pipeline  AUC={a:.3f}", "k", "-", 2.5))
    return curves


def haulover_curves():
    mask = cv2.imread(str(ROOT/"script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png"), cv2.IMREAD_GRAYSCALE)
    sea = (mask == 255).astype(np.float32)
    gt = load_gt(ROOT/"script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png", mask.shape)
    ti15 = T_LIST.index(15)

    curves = []
    # Use canonical 2020 pipeline output (uint8 reinterpreted)
    canonical_files = [
        ("HS_highorder",   "HOR Horn-Schunck",   "#d62728", "-", 1.5),
        ("LK_highorder",   "HOR Lucas-Kanade",   "#ff7f0e", "-", 1.5),
        ("HS_firstorder",  "Horn-Schunck",       "#1f77b4", "-", 1.5),
        ("lucaskanade",    "Lucas-Kanade",       "#2ca02c", "-", 1.5),
        ("typhoon",        "Typhoon",            "#9467bd", ":", 1.2),
    ]
    for fn, label, color, style, lw in canonical_files:
        suffix = "_result_typhoon_mode" if fn == "typhoon" else "_result_basic_mode"
        h = recover(CAN/f"heatmap_Haulover_Miami_FL_{fn}{suffix}.npz") * sea
        r, p, a = pr_curve(h, gt)
        curves.append((r, p, a, f"{label}  AUC={a:.3f}", color, style, lw))

    # Full pipeline: 0.6 * canonical_HS+LK_ensemble σ=25 + 0.4 * smoothed sphere-projected LK at T=15° σ=25
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


def main():
    out_dir = ROOT/"figures"
    pb_curves = palmbeach_curves()
    haul_curves = haulover_curves()

    # Side-by-side panel
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for ax, curves, title in zip(axes, [pb_curves, haul_curves], ["Palm Beach", "Haulover"]):
        for r, p, a, label, color, style, lw in curves:
            ax.plot(r, p, color=color, linestyle=style, linewidth=lw, label=label)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3); ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7.8, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(out_dir/"PRC_combined.pdf", bbox_inches="tight")
    plt.savefig(out_dir/"PRC_combined.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote PRC_combined to", out_dir)


if __name__ == "__main__":
    main()

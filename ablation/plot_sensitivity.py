"""Plot the angular tolerance sensitivity figure for the manuscript.

Two-panel figure:
  Left: per-clip AUC vs T for HOR_HS (16 lines).
  Right: cross-algorithm mean ± std AUC vs T for HOR_HS, HOR_LK, and YOLOv8.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIG_OUT = Path(__file__).resolve().parents[1] / "figures" / "AUC_vs_T_sensitivity.pdf"
FIG_OUT_PNG = FIG_OUT.with_suffix(".png")


def load_algo(dirpath: Path) -> tuple[list[str], list[int], np.ndarray]:
    """Return (clip_names, thresholds_T, AUC_matrix [n_clips × n_T])."""
    csvs = sorted(p for p in dirpath.glob("*.csv") if not p.stem.startswith("_"))
    clips = []
    rows_per_clip = {}
    for p in csvs:
        rows = list(csv.DictReader(p.open()))
        rows_per_clip[p.stem] = {int(r["T_deg"]): float(r["AUC"]) for r in rows}
        clips.append(p.stem)
    Ts = sorted({t for d in rows_per_clip.values() for t in d})
    M = np.array([[rows_per_clip[c].get(t, np.nan) for t in Ts] for c in clips])
    return clips, Ts, M


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plt.subplots_adjust(wspace=0.32)

    # Panel A: per-clip lines for HOR_HS
    clips_hs, Ts, M_hs = load_algo(RESULTS_DIR / "hor_hs")
    cmap = plt.cm.viridis(np.linspace(0, 1, len(clips_hs)))
    for i, clip in enumerate(clips_hs):
        axes[0].plot(Ts, M_hs[i], color=cmap[i], linewidth=1.2, alpha=0.7,
                     label=clip[-3:])
    axes[0].plot(Ts, np.nanmean(M_hs, axis=0), color="black", linewidth=2.5,
                 label="mean", marker="o", markersize=5)
    axes[0].axvline(60, linestyle=":", color="red", alpha=0.6,
                    label="±60° (manuscript text)")
    axes[0].axvline(90, linestyle="--", color="green", alpha=0.6,
                    label="±90° (code)")
    axes[0].set_xlabel("Angular tolerance T (degrees)")
    axes[0].set_ylabel("PR-AUC")
    axes[0].set_title("(a) HOR Horn-Schunck: per-clip AUC vs T (16 DJI clips)")
    axes[0].set_xticks(Ts)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)

    # Panel B: cross-algorithm comparison (mean ± std)
    _, _, M_lk = load_algo(RESULTS_DIR / "hor_lk")
    # Farneback: per-clip CSVs at top-level
    fb_files = sorted(p for p in RESULTS_DIR.glob("DJI_*.csv"))
    if fb_files:
        clips_fb = [p.stem for p in fb_files]
        rows_per_clip_fb = {}
        for p in fb_files:
            rows = list(csv.DictReader(p.open()))
            rows_per_clip_fb[p.stem] = {int(r["T_deg"]): float(r["AUC"]) for r in rows}
        Ts_fb = sorted({t for d in rows_per_clip_fb.values() for t in d})
        M_fb = np.array([[rows_per_clip_fb[c].get(t, np.nan) for t in Ts_fb] for c in clips_fb])
    else:
        Ts_fb = []
        M_fb = np.zeros((0, 0))

    # YOLOv8 — single number per clip; plot as horizontal line band
    yolo_path = RESULTS_DIR / "yolov8" / "per_clip_lowconf.csv"
    yolo_aucs = []
    if yolo_path.exists():
        for r in csv.DictReader(yolo_path.open()):
            yolo_aucs.append(float(r["AUC_PR"]))

    def plot_band(ax, T_vals, M, label, color, marker):
        m = np.nanmean(M, axis=0)
        s = np.nanstd(M, axis=0) / np.sqrt(M.shape[0])  # SEM
        ax.plot(T_vals, m, color=color, linewidth=2.0, marker=marker,
                markersize=6, label=f"{label} (n={M.shape[0]})")
        ax.fill_between(T_vals, m - s, m + s, color=color, alpha=0.18)

    plot_band(axes[1], Ts, M_hs, "HOR Horn-Schunck", "#d62728", "o")
    plot_band(axes[1], Ts, M_lk, "HOR Lucas-Kanade", "#1f77b4", "s")
    if M_fb.shape[0] > 0:
        plot_band(axes[1], Ts_fb, M_fb, "Farneback (proxy)", "#7f7f7f", "^")
    if yolo_aucs:
        ymean = np.mean(yolo_aucs); ysem = np.std(yolo_aucs) / np.sqrt(len(yolo_aucs))
        axes[1].axhline(ymean, linestyle="-.", color="#2ca02c", linewidth=1.5,
                        label=f"YOLOv8 baseline ({ymean:.3f}, n={len(yolo_aucs)})")
        axes[1].fill_between([min(Ts), max(Ts)], ymean - ysem, ymean + ysem,
                             color="#2ca02c", alpha=0.10)

    axes[1].axvline(60, linestyle=":", color="red", alpha=0.6)
    axes[1].axvline(90, linestyle="--", color="green", alpha=0.6)
    axes[1].annotate("manuscript ±60°", xy=(60, 0.04), xytext=(40, 0.045),
                     fontsize=8, color="red")
    axes[1].annotate("code ±90°", xy=(90, 0.04), xytext=(95, 0.045),
                     fontsize=8, color="green")
    axes[1].set_xlabel("Angular tolerance T (degrees)")
    axes[1].set_ylabel("Mean PR-AUC across clips (± SEM)")
    axes[1].set_title("(b) Cross-algorithm comparison")
    axes[1].set_xticks(Ts)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, bbox_inches="tight")
    plt.savefig(FIG_OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Wrote {FIG_OUT}")
    print(f"Wrote {FIG_OUT_PNG}")


if __name__ == "__main__":
    main()

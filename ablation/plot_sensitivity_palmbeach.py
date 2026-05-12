"""Plot the angular tolerance sensitivity figure for the Palm Beach scene
(DJI_0011 t=0-20s window, patched pipeline). Replaces the earlier 16-clip
sensitivity figure since the manuscript reverts to two-scene evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "palmbeach_dji0011_t0_20"
FIG_OUT = Path(__file__).resolve().parents[1] / "figures" / "AUC_vs_T_sensitivity.pdf"
FIG_OUT_PNG = FIG_OUT.with_suffix(".png")


def load(algo: str) -> tuple[list[int], list[float]]:
    p = RESULTS_DIR / f"{algo}.csv"
    rows = list(csv.DictReader(p.open()))
    Ts = [int(r["T_deg"]) for r in rows]
    aucs = [float(r["AUC"]) for r in rows]
    return Ts, aucs


def main():
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    algos = [
        ("HOR_HS", "HOR Horn-Schunck", "#d62728", "o"),
        ("HOR_LK", "HOR Lucas-Kanade", "#1f77b4", "s"),
        ("HS",     "Horn-Schunck",      "#9467bd", "^"),
        ("LK",     "Lucas-Kanade",      "#2ca02c", "v"),
    ]
    for key, label, color, marker in algos:
        Ts, aucs = load(key)
        ax.plot(Ts, aucs, marker=marker, color=color, linewidth=1.8,
                markersize=7, label=label)

    ax.axvline(60, linestyle=":", color="red", alpha=0.6, linewidth=1.2)
    ax.text(60, 0.575, r"$T=60^{\circ}$ (earlier preprint)",
            fontsize=8.5, color="red", ha="center", va="bottom")
    ax.axvline(90, linestyle="--", color="green", alpha=0.6, linewidth=1.2)
    ax.text(90, 0.555, r"$T=90^{\circ}$ (implementation)",
            fontsize=8.5, color="green", ha="center", va="bottom")

    ax.set_xlabel(r"Angular tolerance $T$ (degrees)")
    ax.set_ylabel("Precision-Recall AUC (Palm Beach)")
    ax.set_title("Sensitivity of PR-AUC to angular tolerance T\n"
                 "DJI 0011, 0–20 s window, Palm Beach annotation")
    ax.set_xticks([15, 30, 45, 60, 75, 90, 105, 120, 150])
    ax.set_ylim(0.27, 0.62)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", fontsize=9, ncol=2)

    plt.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, bbox_inches="tight")
    plt.savefig(FIG_OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Wrote {FIG_OUT}")
    print(f"Wrote {FIG_OUT_PNG}")


if __name__ == "__main__":
    main()

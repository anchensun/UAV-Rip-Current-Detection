"""Multi-scene angular-tolerance (T) sensitivity figure.

Question: is the default T = 90 deg an artifact of a single-scene parameter scan,
i.e. a value fitted to one dataset? This figure tests that across 18 evaluated
UAV scenes -- Palm Beach, Haulover, and 16 South Beach sub-clips (the sub-clips
are segments of two source captures, so the sweep spans four captures in total).
The PR-AUC-vs-T curve is a broad plateau rather than a sharp peak, and
T = 90 deg -- the physical offshore half-plane boundary -- sits inside it.
Performance is therefore insensitive to T over a wide band, which is inconsistent
with an over-tuned parameter. Note the corollary, reported in the paper: on the
per-scene-normalized median, tolerances below 60 deg score higher still, so
T = 90 deg is not the AUC-maximizing choice on this statistic.

All curves use the SAME optical-flow algorithm (HOR Horn-Schunck, patched 2020
pipeline, no CNN gating) for a fair cross-scene comparison. Inputs are the
already-computed sweep CSVs under ablation/results/ -- no re-run needed.

Output: figures/AUC_vs_T_multiscene.pdf (+ .png)
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "ablation" / "results"
FIGDIR = ROOT / "figures"

EVAL_SCENES = {
    # DJI_0011.MOV is the manuscript Palm Beach evaluation video (per coauthor
    # confirmation); the earlier ablation/results/palmbeach/ run used the
    # auxiliary capture DJI_0047-002.MP4 and is superseded.
    "Palm Beach": RES / "palmbeach_dji0011_t0_20" / "HOR_HS.csv",
    "Haulover":   RES / "haulover_dji0009_t0s" / "HOR_HS.csv",
}
CLIP_DIR = RES / "hor_hs"
CLIPS = [f"DJI_0006_{i}" for i in range(1, 9)] + [f"DJI_0007_{i}" for i in range(1, 9)]


def read_curve(path: Path):
    T, A = [], []
    with path.open() as f:
        for r in csv.DictReader(f):
            T.append(int(r["T_deg"])); A.append(float(r["AUC"]))
    return np.array(T), np.array(A)


def main():
    FIGDIR.mkdir(exist_ok=True)
    curves = {}          # name -> (T, AUC)
    for name, p in EVAL_SCENES.items():
        if p.exists():
            curves[name] = read_curve(p)
    for c in CLIPS:
        p = CLIP_DIR / f"{c}.csv"
        if p.exists():
            curves[c] = read_curve(p)

    # common T grid (intersection); all use the 9-point sweep
    Tref = np.array([15, 30, 45, 60, 75, 90, 105, 120, 150])
    argmaxT = []
    norm_stack = []
    for name, (T, A) in curves.items():
        # align to Tref
        idx = [list(T).index(t) if t in T else -1 for t in Tref]
        a = np.array([A[i] if i >= 0 else np.nan for i in idx])
        argmaxT.append(Tref[np.nanargmax(a)])
        norm_stack.append(a / np.nanmax(a) if np.nanmax(a) > 0 else a)
    norm_stack = np.array(norm_stack)
    med = np.nanmedian(norm_stack, axis=0)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [2.2, 1]})

    # Panel A: normalized AUC-vs-T, all scenes + median
    for name, (T, A) in curves.items():
        idx = [list(T).index(t) if t in T else -1 for t in Tref]
        a = np.array([A[i] if i >= 0 else np.nan for i in idx])
        a = a / np.nanmax(a)
        is_eval = name in EVAL_SCENES
        axL.plot(Tref, a, "-", lw=2.4 if is_eval else 0.9,
                 color=("#c0392b" if name == "Palm Beach" else
                        "#2471a3" if name == "Haulover" else "0.72"),
                 alpha=1.0 if is_eval else 0.6,
                 label=name if is_eval else None, zorder=3 if is_eval else 1)
    axL.plot(Tref, med, "k--", lw=2.6, label="median (18 scenes)", zorder=4)
    axL.axvline(90, color="#e67e22", ls=":", lw=2, zorder=2)
    axL.text(91, 0.36, r"$T=90^\circ$" + "\n(offshore\nhalf-plane)",
             color="#e67e22", fontsize=9, va="center")
    axL.set_xlabel(r"Angular tolerance $T$ (degrees)")
    axL.set_ylabel("PR-AUC normalized to per-scene maximum")
    axL.set_title("(a) AUC vs $T$ is a broad plateau across 18 scenes")
    axL.set_ylim(0.3, 1.03)
    axL.set_xticks(Tref)
    axL.legend(loc="lower left", fontsize=8, framealpha=0.95)
    axL.grid(alpha=0.25)

    # Panel B: raw AUC-vs-T for the two real-signal evaluation scenes (plateaus)
    for name, col in (("Palm Beach", "#c0392b"), ("Haulover", "#2471a3")):
        if name in curves:
            T, A = curves[name]
            axR.plot(T, A, "o-", lw=2.2, color=col, label=name)
    axR.axvline(90, color="#e67e22", ls=":", lw=2)
    axR.set_xlabel(r"Angular tolerance $T$ (degrees)")
    axR.set_ylabel("PR-AUC (HOR HS, no CNN)")
    axR.set_title(r"(b) Optima differ by scene: PB peaks at $120^\circ$, HV at $15^\circ$")
    axR.set_xticks([15, 45, 90, 120, 150])
    axR.legend(loc="best", fontsize=8)
    axR.grid(alpha=0.25)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"AUC_vs_T_multiscene.{ext}", dpi=150, bbox_inches="tight")
    print(f"scenes plotted: {len(curves)}")
    print(f"per-scene optimal T: {sorted(argmaxT)}")
    print(f"median normalized AUC at T=90: {med[list(Tref).index(90)]:.3f}")
    print(f"median normalized AUC range over T in [60,150]: "
          f"{med[3:].min():.3f}-{med[3:].max():.3f}")
    print(f"wrote {FIGDIR/'AUC_vs_T_multiscene.pdf'}")


if __name__ == "__main__":
    main()

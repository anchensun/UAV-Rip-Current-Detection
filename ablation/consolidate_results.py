"""Consolidate ablation/results/* into a single response-ready report:
  - Per-algorithm AUC vs T table (clip × T grid).
  - Best T per clip per algorithm.
  - Head-to-head best-AUC table: framework algorithms vs YOLOv8 deep baseline.
  - Summary statistics ready for the response-to-reviewers letter.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_per_clip_csv(csv_path: Path) -> dict[int, float]:
    """Load a clip's CSV (T_deg, AUC, ...) → {T: AUC}."""
    rows = list(csv.DictReader(csv_path.open()))
    return {int(r["T_deg"]): float(r["AUC"]) for r in rows}


def load_yolo_csv(csv_path: Path) -> dict[str, dict]:
    """Load YOLO per-clip CSV → {clip: full row}."""
    rows = list(csv.DictReader(csv_path.open()))
    return {r["clip"]: r for r in rows}


def report_algo(name: str, dirpath: Path, ks_log) -> dict:
    if not dirpath.exists():
        return {}
    rows_per_clip = {}
    for p in sorted(dirpath.glob("*.csv")):
        if p.stem.startswith("_"):
            continue
        rows_per_clip[p.stem] = load_per_clip_csv(p)
    if not rows_per_clip:
        return {}
    Ts = sorted({t for d in rows_per_clip.values() for t in d})
    clips = sorted(rows_per_clip)
    print(f"\n=== {name}  ({len(clips)} clips, {len(Ts)} thresholds) ===")
    header = f"{'T':>4}  " + "  ".join(f"{c[-3:]:>7}" for c in clips) + f"  {'mean':>7} {'med':>7}"
    print(header)
    ks_log.write(header + "\n")
    mean_per_T = {}
    median_per_T = {}
    for T in Ts:
        vals = [rows_per_clip[c].get(T, float("nan")) for c in clips]
        valid = [v for v in vals if not np.isnan(v)]
        if not valid:
            continue
        m = statistics.mean(valid)
        med = statistics.median(valid)
        mean_per_T[T] = m
        median_per_T[T] = med
        line = f"{T:>3}°  " + "  ".join(f"{v:>7.4f}" for v in vals) + f"  {m:>7.4f} {med:>7.4f}"
        print(line)
        ks_log.write(line + "\n")
    if mean_per_T:
        best_mean_T = max(mean_per_T, key=mean_per_T.get)
        best_med_T = max(median_per_T, key=median_per_T.get)
        print(f"  best mean T = {best_mean_T}° (mean AUC {mean_per_T[best_mean_T]:.4f})")
        print(f"  best median T = {best_med_T}° (median AUC {median_per_T[best_med_T]:.4f})")
    # Per-clip best
    per_clip_best = {}
    for c in clips:
        best = max(rows_per_clip[c], key=rows_per_clip[c].get)
        per_clip_best[c] = (best, rows_per_clip[c][best])
    return {
        "name": name,
        "clips": clips,
        "thresholds": Ts,
        "rows": {c: dict(rows_per_clip[c]) for c in clips},
        "mean_per_T": mean_per_T,
        "median_per_T": median_per_T,
        "best_mean_T": int(max(mean_per_T, key=mean_per_T.get)) if mean_per_T else None,
        "best_median_T": int(max(median_per_T, key=median_per_T.get)) if median_per_T else None,
        "per_clip_best": per_clip_best,
    }


def head_to_head(framework_results: dict, yolo_rows: dict) -> list[dict]:
    """Match clips across all algos and YOLO; report best-AUC per algo per clip."""
    common = sorted(set(framework_results.get("HOR_HS", {}).get("clips", [])) & set(yolo_rows.keys()))
    out = []
    for c in common:
        row = {"clip": c}
        for algo in ["HOR_HS", "HOR_LK", "FARNEBACK"]:
            r = framework_results.get(algo, {})
            if c in r.get("rows", {}):
                bestT = max(r["rows"][c], key=r["rows"][c].get)
                row[f"{algo}_best_T"] = bestT
                row[f"{algo}_best_AUC"] = r["rows"][c][bestT]
                row[f"{algo}_AUC_at_60"] = r["rows"][c].get(60)
                row[f"{algo}_AUC_at_90"] = r["rows"][c].get(90)
        if c in yolo_rows:
            row["yolo_AUC"] = float(yolo_rows[c]["AUC_PR"])
            row["yolo_best_IoU"] = float(yolo_rows[c]["best_IoU"])
        out.append(row)
    return out


def main():
    log_path = RESULTS_DIR / "_consolidated_report.txt"
    ks_log = log_path.open("w")
    ks_log.write(f"Consolidated ablation report\n{'=' * 60}\n")

    framework_results = {}
    framework_results["HOR_HS"] = report_algo("HOR_HS (manuscript best)", RESULTS_DIR / "hor_hs", ks_log)
    framework_results["HOR_LK"] = report_algo("HOR_LK", RESULTS_DIR / "hor_lk", ks_log)
    # Farneback results: per-clip CSVs at the top-level results dir (legacy).
    farneback_dir = RESULTS_DIR
    farneback_clips = {}
    for p in sorted(farneback_dir.glob("DJI_*.csv")):
        farneback_clips[p.stem] = load_per_clip_csv(p)
    if farneback_clips:
        Ts = sorted({t for d in farneback_clips.values() for t in d})
        framework_results["FARNEBACK"] = {
            "name": "Farneback (proxy)",
            "clips": sorted(farneback_clips),
            "thresholds": Ts,
            "rows": {c: dict(farneback_clips[c]) for c in farneback_clips},
            "mean_per_T": {T: statistics.mean(farneback_clips[c][T] for c in farneback_clips if T in farneback_clips[c]) for T in Ts},
            "median_per_T": {T: statistics.median(farneback_clips[c][T] for c in farneback_clips if T in farneback_clips[c]) for T in Ts},
        }
        framework_results["FARNEBACK"]["best_mean_T"] = int(max(framework_results["FARNEBACK"]["mean_per_T"], key=framework_results["FARNEBACK"]["mean_per_T"].get))
        framework_results["FARNEBACK"]["best_median_T"] = int(max(framework_results["FARNEBACK"]["median_per_T"], key=framework_results["FARNEBACK"]["median_per_T"].get))
        framework_results["FARNEBACK"]["per_clip_best"] = {
            c: (max(farneback_clips[c], key=farneback_clips[c].get), max(farneback_clips[c].values()))
            for c in farneback_clips
        }

    yolo_path = RESULTS_DIR / "yolov8" / "per_clip_lowconf.csv"
    yolo_rows = load_yolo_csv(yolo_path) if yolo_path.exists() else {}
    if yolo_rows:
        print(f"\n=== YOLOv8 (E1 weights, lowconf, max-agg) — {len(yolo_rows)} clips ===")
        ks_log.write("\n=== YOLOv8 ===\n")
        for c, r in sorted(yolo_rows.items()):
            line = f"  {c}  AUC={float(r['AUC_PR']):.4f}  best_IoU={float(r['best_IoU']):.4f}"
            print(line)
            ks_log.write(line + "\n")

    h2h = head_to_head(framework_results, yolo_rows)
    if h2h:
        print(f"\n=== HEAD-TO-HEAD (best AUC per clip per algorithm) ===")
        ks_log.write("\n=== HEAD-TO-HEAD ===\n")
        cols = ["clip", "HOR_HS_best_T", "HOR_HS_best_AUC", "HOR_LK_best_T", "HOR_LK_best_AUC",
                "FARNEBACK_best_T", "FARNEBACK_best_AUC", "yolo_AUC"]
        cols = [c for c in cols if any(c in r for r in h2h)]
        # Print
        header = "  ".join(f"{c:>14}" for c in cols)
        print(header)
        ks_log.write(header + "\n")
        for row in h2h:
            line = "  ".join(f"{str(row.get(c,'')):>14}" for c in cols)
            print(line)
            ks_log.write(line + "\n")
        # Also write CSV
        with (RESULTS_DIR / "_head_to_head.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in h2h:
                w.writerow({k: row.get(k, "") for k in cols})

    # Final json digest
    digest = {
        "framework_best_mean_T": {a: r.get("best_mean_T") for a, r in framework_results.items() if r},
        "framework_best_median_T": {a: r.get("best_median_T") for a, r in framework_results.items() if r},
        "yolo_mean_auc": (statistics.mean(float(r["AUC_PR"]) for r in yolo_rows.values()) if yolo_rows else None),
        "yolo_median_auc": (statistics.median(float(r["AUC_PR"]) for r in yolo_rows.values()) if yolo_rows else None),
        "yolo_best_clip": (max(yolo_rows.items(), key=lambda kv: float(kv[1]["AUC_PR"]))[0] if yolo_rows else None),
        "yolo_best_auc": (max(float(r["AUC_PR"]) for r in yolo_rows.values()) if yolo_rows else None),
        "n_clips_per_algo": {a: len(r.get("clips", [])) for a, r in framework_results.items() if r},
    }
    (RESULTS_DIR / "_digest.json").write_text(json.dumps(digest, indent=2))
    print(f"\nDigest:\n{json.dumps(digest, indent=2)}")
    print(f"\nReports: {log_path} + {RESULTS_DIR / '_head_to_head.csv'} + {RESULTS_DIR / '_digest.json'}")
    ks_log.close()


if __name__ == "__main__":
    main()

"""Read every CSV under seg_refiner/results/ and emit a master comparison
table covering all (config, scene, algo, T) AUC values produced by the
ablation stages.

Output: seg_refiner/MASTER_TABLE.md (Markdown) and .csv (raw)."""
from __future__ import annotations
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "seg_refiner" / "results"
ABLATION = ROOT / "ablation" / "results"

ROWS = []  # list of {scene, config, algo, T_deg, AUC}


def parse_dir(scene_dir, scene_name, config_name):
    if not scene_dir.exists(): return
    for f in sorted(scene_dir.glob("*.csv")):
        if f.name.startswith("_"): continue
        algo_from_name = f.stem  # e.g. "HOR_HS" or "HOR_HS_sea"
        with f.open() as h:
            for r in csv.DictReader(h):
                if "T_deg" not in r or "AUC" not in r: continue
                ROWS.append({"scene": scene_name, "config": config_name,
                             "algo": r.get("algo", algo_from_name),
                             "T_deg": int(r["T_deg"]), "AUC": float(r["AUC"])})


# Ablation stages
parse_dir(ABLATION / "palmbeach_dji0011_t0_20", "PalmBeach", "patched_only")
parse_dir(ABLATION / "haulover_dji0009_t0s",    "Haulover",  "patched_only")

# Seg-gate (no projection)
parse_dir(RESULTS / "palmbeach_sea",           "PalmBeach", "seagate_unet6c")
parse_dir(RESULTS / "haulover_dji0009_sea",    "Haulover",  "seagate_unet6c")

# Sphere ablation (HOR_HS only, varied tilt)
for psi in (0, 5, 15, 30):
    parse_dir(RESULTS / f"palmbeach_seasphere_sphere_psi{psi}",
              "PalmBeach", f"seagate+sphere_psi{psi}")
    parse_dir(RESULTS / f"haulover_dji0009_seasphere_sphere_psi{psi}",
              "Haulover", f"seagate+sphere_psi{psi}")

# Stage B: Haulover algo × gate sweep
sb = RESULTS / "stage_b_haulover"
if sb.exists():
    for f in sorted(sb.glob("*.csv")):
        algo, gate = f.stem.rsplit("_", 1) if "_" in f.stem else (f.stem, "?")
        # Stage B file names: HOR_HS_sea, HOR_HS_sea_x_nonfoam, HOR_HS_argmax
        # Easier: parse smartly
        nm = f.stem
        for g in ("sea_x_nonfoam", "argmax", "sea"):
            if nm.endswith("_" + g):
                algo = nm[: -(len(g) + 1)]; gate = g; break
        config = f"seagate({gate})"
        with f.open() as h:
            for r in csv.DictReader(h):
                if "T_deg" not in r: continue
                ROWS.append({"scene": "Haulover", "config": config,
                             "algo": algo, "T_deg": int(r["T_deg"]), "AUC": float(r["AUC"])})

# Stage D: PB sweep
sd = RESULTS / "stage_d_pbsweep"
if sd.exists():
    for f in sorted(sd.glob("*.csv")):
        config = f"seagate+sweep_{f.stem}"
        with f.open() as h:
            for r in csv.DictReader(h):
                ROWS.append({"scene": "PalmBeach", "config": config,
                             "algo": r.get("algo", "LK"),
                             "T_deg": int(r["T_deg"]), "AUC": float(r["AUC"])})

# Stage C ensemble dirs
for nm, scene in [("stage_c_palmbeach", "PalmBeach"), ("stage_c_haulover", "Haulover")]:
    f = RESULTS / nm / "ensemble_results.csv"
    if f.exists():
        with f.open() as h:
            for r in csv.DictReader(h):
                ROWS.append({"scene": scene, "config": "seagate+ensemble",
                             "algo": r["algo"],
                             "T_deg": int(r["T_deg"]), "AUC": float(r["AUC"])})

# Stage E ripseg
for nm, scene in [("stage_e_palmbeach", "PalmBeach"), ("stage_e_haulover", "Haulover")]:
    d = RESULTS / nm
    if d.exists():
        for f in sorted(d.glob("*.csv")):
            with f.open() as h:
                for r in csv.DictReader(h):
                    ROWS.append({"scene": scene, "config": "ripseg_binary",
                                 "algo": r.get("algo", f.stem),
                                 "T_deg": int(r["T_deg"]), "AUC": float(r["AUC"])})

# Write master CSV
out_csv = ROOT / "seg_refiner" / "MASTER_TABLE.csv"
with out_csv.open("w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=["scene", "config", "algo", "T_deg", "AUC"])
    w.writeheader(); w.writerows(ROWS)
print(f"Wrote {out_csv}  ({len(ROWS)} rows)")

# Pretty-print: best per (scene, config) at T=90 and overall best T
from collections import defaultdict
best_by = defaultdict(lambda: (-1.0, None))   # (scene, config, algo) -> (best_AUC, best_T)
for r in ROWS:
    k = (r["scene"], r["config"], r["algo"])
    if r["AUC"] > best_by[k][0]:
        best_by[k] = (r["AUC"], r["T_deg"])

# Markdown
out_md = ROOT / "seg_refiner" / "MASTER_TABLE.md"
with out_md.open("w") as h:
    h.write("# Master comparison table — all (scene, config, algo)\n\n")
    h.write("Best AUC across T thresholds for each (scene, config, algo) combination.\n\n")
    h.write("| Scene | Config | Algo | best AUC | at T |\n|---|---|---|---|---|\n")
    for (scene, config, algo), (auc, T) in sorted(best_by.items()):
        h.write(f"| {scene} | {config} | {algo} | {auc:.4f} | {T} |\n")
print(f"Wrote {out_md}")

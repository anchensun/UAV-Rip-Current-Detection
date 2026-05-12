"""Step 1: For each SBL Markup _finalplot.jpg, locate the matching frame in the
corresponding sub-clip video by SSIM/MSE on a low-resolution gray view of the
land/sky region (top half — sea region has the heatmap overlay).

Outputs: seg_refiner/results/markup_frame_match.csv
"""
from __future__ import annotations
import csv, os, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MARKUP_DIR = ROOT / "data/extracted/Rip Current Video/RipOutlinesSBL/SBL Markup"
VID_DIR = ROOT / "data/extracted/Rip Current Video/New Drone Videos"
YT_DIR = ROOT / "data/extracted/Rip Current Video/YouTube"
OUT_CSV = ROOT / "seg_refiner/results/markup_frame_match.csv"
OUT_DIAG_DIR = ROOT / "seg_refiner/results/markup_diag"
OUT_DIAG_DIR.mkdir(parents=True, exist_ok=True)


def match_clip(markup_path: Path, video_path: Path, sample_every: int = 5,
               crop_top_frac: float = 0.45) -> tuple[int, float, int]:
    """Return (best_frame_idx, best_score_mse_low, total_frames). Lower MSE = better."""
    mu = np.asarray(Image.open(markup_path).convert("RGB"), dtype=np.float32) / 255.0
    # Use top 45% (land/sky) — heatmap overlay covers the sea region below
    mh, mw = mu.shape[:2]
    crop_h = int(mh * crop_top_frac)
    mu_top = mu[:crop_h]
    # Downscale for speed
    target_w = 320
    target_h = int(crop_h * target_w / mw)
    mu_small = cv2.resize(mu_top, (target_w, target_h))
    mu_gray = cv2.cvtColor((mu_small * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu_gray -= mu_gray.mean()
    mu_norm = mu_gray / (mu_gray.std() + 1e-6)

    cap = cv2.VideoCapture(str(video_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best = (None, 1e18)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            # frame is BGR 1920x1080
            top = frame[: int(frame.shape[0] * crop_top_frac)]
            top_rs = cv2.resize(top, (target_w, target_h))
            top_gray = cv2.cvtColor(top_rs, cv2.COLOR_BGR2GRAY).astype(np.float32)
            top_gray -= top_gray.mean()
            top_norm = top_gray / (top_gray.std() + 1e-6)
            mse = float(((top_norm - mu_norm) ** 2).mean())
            if mse < best[1]:
                best = (idx, mse)
        idx += 1
    cap.release()
    return best[0], best[1], n_total


def save_diag(markup_path: Path, video_path: Path, best_idx: int, out_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return
    mu = np.asarray(Image.open(markup_path).convert("RGB"))
    mh, mw = mu.shape[:2]
    f_rs = cv2.resize(frame, (mw, mh))
    f_rgb = cv2.cvtColor(f_rs, cv2.COLOR_BGR2RGB)
    side = np.concatenate([mu, f_rgb], axis=1)
    Image.fromarray(side).save(out_path)


def main():
    rows = []
    DJI_clips = [f"DJI_000{a}_{b}" for a in (6, 7) for b in range(1, 9)]
    YT_clips = ["YouTube1", "YouTube2", "YouTube3"]
    targets = []
    for c in DJI_clips:
        mu = MARKUP_DIR / f"{c}_HS_highorder_finalplot.jpg"
        vid = VID_DIR / f"{c}.mp4"
        if mu.exists() and vid.exists():
            targets.append((c, mu, vid))
    for c in YT_clips:
        # Markup names vary slightly
        candidates = [
            MARKUP_DIR / f"{c}_HS_highorder_finalplot.jpg",
            MARKUP_DIR / f"{c}_HS_highorder_finalplotmarkup.jpg",
            MARKUP_DIR / f"{c}_Moment.jpg",
        ]
        mu = next((p for p in candidates if p.exists()), None)
        vid = YT_DIR / f"{c}.mp4"
        if mu is not None and vid.exists():
            targets.append((c, mu, vid))

    print(f"Matching {len(targets)} clips")
    print(f"{'clip':<14s} {'frame':>7s}  {'total':>7s}  {'frac':>5s}  {'norm_mse':>9s}")
    for clip, mu, vid in targets:
        try:
            best_idx, score, total = match_clip(mu, vid, sample_every=5)
        except Exception as e:
            print(f"  {clip}: ERROR {e}")
            rows.append((clip, str(mu.name), -1, -1.0, -1, "ERR"))
            continue
        diag = OUT_DIAG_DIR / f"{clip}_match.png"
        save_diag(mu, vid, best_idx, diag)
        frac = best_idx / max(total - 1, 1)
        print(f"{clip:<14s} {best_idx:>7d}  {total:>7d}  {frac:>5.2f}  {score:>9.4f}")
        rows.append((clip, str(mu.name), best_idx, score, total, str(diag.relative_to(ROOT))))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "markup_file", "best_frame_idx", "norm_mse", "total_frames", "diag_png"])
        w.writerows(rows)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()

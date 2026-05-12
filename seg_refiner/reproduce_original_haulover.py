"""Reproduce the EXACT original Haulover HOR HS pipeline from raw video.

This is a vectorized port of:
  - OptOffShoreDirection.OffshoreDir() — DetCoLine + Find_Dis_Matrix + np.gradient
  - OFlib.heatmap()                    — azimuthAngle + |off-d|<90 accumulator

Differences vs the patched-only run we did earlier:
  - Uses the original `azimuthAngle(...)`-based bearing comparison instead
    of cos-angle-via-dot-product. The two are NOT mathematically equivalent
    because the original (a) treats dx==dy==0 → angle=0, (b) treats axis-aligned
    cases via fall-through, and (c) returns bearing in [0, 360°) without
    wrap-around handling, so |off - d|<90 can MISS small angular differences
    near the 0/360 boundary.
  - Uses np.gradient(Dist_Matrix) in the original axis order (which the
    heatmap code consumes as off_dx, off_dy — these are actually
    (∂Dist/∂row, ∂Dist/∂col) but the heatmap treats them as (∂x, ∂y),
    a known axis-swap quirk in the 2020 codebase).
  - Uses the static manual sea mask (Haulover_Miami_FL_mask.png) for
    Find_Dis_Matrix, exactly matching DetCoLine's input.

We save the resulting (W, H)-shaped heatmap THREE ways to mirror the
manuscript's recovery story:
  - heat_int8.npz  (with the original int8 truncation bug)
  - heat_uint8.npz (byte-wise unsigned reinterpretation: matches manuscript 0.480)
  - heat_float.npz (the full untruncated accumulator — what the patched
                    pipeline ALSO produces)

Usage:
    python seg_refiner/reproduce_original_haulover.py \
        --video data/DJI_0009.MOV \
        --mask  script/Rip_Current-main/New_Rip_Mask/Haulover_Miami_FL_mask.png \
        --gt    script/Rip_Current-main/rip_curent_evaluation/workspace/annotations/Haulover_Miami_FL_annotation.png \
        --max-frames 600 --skip 2 --start-time 0 \
        --out-dir seg_refiner/results/reproduce_haulover
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script" / "Rip_Current-main"
sys.path.insert(0, str(SCRIPT_DIR))
from OFlib import higher_order
from HornSchunck import HS as HS_baseline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--mask", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--skip", type=int, default=2)
    p.add_argument("--start-time", type=float, default=0.0)
    p.add_argument("--algo", default="HOR_HS",
                   choices=["HOR_HS", "HOR_LK", "HS", "LK"])
    return p.parse_args()


# === Faithful port of OptOffShoreDirection.DetCoLine ===
def det_coline(img: np.ndarray) -> np.ndarray:
    """Median 51 + Gaussian 7 + Canny(100,150) + zero top 1/3 + force bottom row=255."""
    blur = cv2.medianBlur(img, 51)
    smooth = cv2.GaussianBlur(blur, (7, 7), 0)
    edges = cv2.Canny(smooth, 100, 150)
    sky_s = int(edges.shape[0] / 3)
    edges[:sky_s, :] = 0
    edges[-1, :] = 255
    return edges


# === Faithful (but vectorized) port of Find_Dis_Matrix ===
def find_dis_matrix(coline: np.ndarray, sea_mask: np.ndarray) -> np.ndarray:
    """Distance from each sea pixel to the nearest coline pixel (L2, full precision).
    Outside the sea mask the distance stays 0 (matches original)."""
    coline_zero = (coline == 0).astype(np.uint8)  # 0 at coline pixel, 1 elsewhere
    dist_full = cv2.distanceTransform(coline_zero, cv2.DIST_L2, 5).astype(np.float32)
    dist_sea = np.where(sea_mask == 255, dist_full, 0.0).astype(np.float32)
    return dist_sea


# === Vectorized azimuthAngle ===
def azimuth_field(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Vectorized bearing exactly matching the original azimuthAngle().
    Returns degrees in [0, 360) with the original's special-case handling
    for axis-aligned vectors (dx=0 → {0, 90, 270}; dy=0 with dx!=0 → 0)."""
    angle = np.zeros_like(dx, dtype=np.float32)
    # dx == 0
    z = dx == 0
    angle[z & (dy > 0)] = 90.0
    angle[z & (dy < 0)] = 270.0
    angle[z & (dy == 0)] = 0.0
    # dx > 0, dy > 0
    q1 = (dx > 0) & (dy > 0)
    angle[q1] = np.degrees(np.arctan(dx[q1] / dy[q1]))
    # dx > 0, dy < 0
    q2 = (dx > 0) & (dy < 0)
    angle[q2] = 90.0 + np.degrees(np.arctan(-dy[q2] / dx[q2]))
    # dx < 0, dy < 0
    q3 = (dx < 0) & (dy < 0)
    angle[q3] = 180.0 + np.degrees(np.arctan(dx[q3] / dy[q3]))
    # dx < 0, dy > 0
    q4 = (dx < 0) & (dy > 0)
    angle[q4] = 270.0 + np.degrees(np.arctan(dy[q4] / -dx[q4]))
    # dx != 0, dy == 0: angle remains 0 (matches original fall-through)
    return angle


def of_one(g_prev, g_curr, algo):
    if algo == "HOR_HS": u, v = higher_order(g_prev, g_curr, mode="HS")
    elif algo == "HOR_LK": u, v = higher_order(g_prev, g_curr, mode="LK")
    elif algo == "HS":   u, v = HS_baseline(g_prev, g_curr)
    elif algo == "LK":
        from OFlib import lucas_kanade
        u, v = lucas_kanade(g_prev, g_curr)
    if u.shape == (10, 10): return None
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(u, -1000, 1000).astype(np.float32), np.clip(v, -1000, 1000).astype(np.float32)


def load_gt(gt_path, shape):
    arr = np.asarray(Image.open(gt_path))
    gray = arr[..., :3].max(axis=-1) if arr.ndim == 3 else arr
    binary = gray > 127
    if binary.shape != shape:
        rh, rw = shape
        pil = Image.fromarray(binary.astype(np.uint8) * 255).resize((rw, rh), Image.NEAREST)
        binary = np.asarray(pil) > 127
    return binary


def auc_pr(heat, gt, n_bins=1024):
    pos = heat[gt]; neg = heat[~gt]
    hi = float(heat.max())
    if pos.size == 0 or hi <= 0: return 0.0
    bins = np.linspace(min(0, float(heat.min())), hi, n_bins + 1)
    pos_h, _ = np.histogram(pos, bins=bins); neg_h, _ = np.histogram(neg, bins=bins)
    pos_cum = np.cumsum(pos_h[::-1])[::-1]; neg_cum = np.cumsum(neg_h[::-1])[::-1]
    p = pos_cum / np.maximum(pos_cum + neg_cum, 1)
    r = pos_cum / max(int(pos.size), 1)
    order = np.argsort(r)
    return float(np.trapezoid(p[order], r[order]))


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    print(f"Mask {H}x{W}, sea frac {(mask==255).mean():.4%}")

    coline = det_coline(mask)
    dist_matrix = find_dis_matrix(coline, mask)
    print(f"CoLine (after Canny + skyline cut + bottom-row force): {(coline==255).sum()} pixels")
    print(f"Dist_Matrix sea-pixel range: min={dist_matrix[mask==255].min():.1f}, "
          f"max={dist_matrix[mask==255].max():.1f}, "
          f"mean={dist_matrix[mask==255].mean():.1f}")

    # The original axis order: u, v = np.gradient(Dist_Matrix) → [∂/∂row, ∂/∂col]
    # then heatmap consumes them as off_dx, off_dy (axis-swap quirk preserved here).
    off_u_axis_swap, off_v_axis_swap = np.gradient(dist_matrix)
    off_dx_field = off_u_axis_swap.astype(np.float32)
    off_dy_field = off_v_axis_swap.astype(np.float32)

    # Pre-compute offshore bearing (constant across frames).
    off_angle = azimuth_field(off_dx_field, off_dy_field)

    # Save offshore-direction visualization for debugging
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    mag = np.sqrt(off_dx_field**2 + off_dy_field**2)
    norm_mag = (mag / max(mag.max(), 1e-6) * 255).astype(np.uint8)
    vis[..., 0] = norm_mag
    cv2.imwrite(str(args.out_dir / "_offshore_mag.png"), vis)

    gt = load_gt(args.gt, (H, W))
    print(f"GT pos frac: {gt.mean():.4%}")

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video fps={fps:.1f}; algo={args.algo}")
    if args.start_time > 0: cap.set(cv2.CAP_PROP_POS_MSEC, args.start_time * 1000)
    vid_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vid_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    needs_resize = (vid_H, vid_W) != (H, W)

    # Pre-build the inverted "grandmask" used by motionmask (255 outside sea).
    grandmask_inv = (255 - mask).astype(np.uint8)

    # Pass 1: load grayscale frames AND keep BGR frames (for motionmask)
    grays = []
    bgrs = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % args.skip == 0:
            if needs_resize: frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            bgrs.append(frame)
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        idx += 1
        if len(grays) > args.max_frames + 1: break
    cap.release()
    n_pairs = min(len(grays) - 1, args.max_frames)
    print(f"Loaded {len(grays)} frames; {n_pairs} pairs")

    # The original heat is keyed [x, y] = [col, row] with shape (W, H).
    heat_native = np.zeros((W, H), dtype=np.int32)

    sea_pix_mask = (mask == 255)  # boolean (H, W)

    t_loop = time.time()
    n_proc = 0
    keep_frac_sum = 0.0
    for i in range(n_pairs):
        result = of_one(grays[i], grays[i + 1], args.algo)
        if result is None: continue
        u, v = result
        if u.shape != (H, W):
            u = cv2.resize(u, (W, H)); v = cv2.resize(v, (W, H))

        # === motionmask (faithful port, vectorized) ===
        frame_bgr = bgrs[i + 1]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0,0,50], np.uint8),    np.array([50,50,225], np.uint8))   # blue water
        m2 = cv2.inRange(hsv, np.array([0,0,150], np.uint8),   np.array([225,25,255], np.uint8))  # white foam
        m3 = cv2.inRange(hsv, np.array([0,0,0], np.uint8),     np.array([180,255,100], np.uint8)) # black shadow
        combined = cv2.add(cv2.add(cv2.add(m1, m3), grandmask_inv), m2)
        keep = (combined == 0)  # True at pixels NOT classified into any mask
        keep_frac_sum += float(keep.mean())

        # Per-pixel bearing of optical flow (using the original axis convention).
        d_angle = azimuth_field(u, v)

        # Original heatmap criterion: |off_angle - d_angle| < 90° AND (dx, dy) != (0, 0).
        # The original loop excludes a 2-px border. After motionmask, only "kept"
        # pixels have (u, v) ≠ 0; we mirror that explicitly.
        flow_active = (u != 0) | (v != 0)
        agree = (np.abs(off_angle - d_angle) < 90) & flow_active & sea_pix_mask & keep
        agree[:2, :] = False; agree[-2:, :] = False
        agree[:, :2] = False; agree[:, -2:] = False

        # heat[x, y] += 1: in (W, H) layout. We have agree shape (H, W). Transpose.
        heat_native += agree.T.astype(np.int32)

        n_proc += 1
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t_loop)
            eta = (n_pairs - i - 1) / max(rate, 1e-3)
            print(f"  {i+1}/{n_pairs} pairs, {rate:.2f}/s, ETA {eta:.0f}s, "
                  f"avg keep_frac={keep_frac_sum/(i+1):.3%}")
    print(f"Loop done in {time.time()-t_loop:.1f}s; processed {n_proc} pairs")
    print(f"heat_native: shape {heat_native.shape}, dtype {heat_native.dtype}, "
          f"min={heat_native.min()}, max={heat_native.max()}, "
          f"mean={heat_native.mean():.2f}")

    # Save 3 versions: int8 (with bug), uint8 reinterp, float
    heat_native_T = heat_native.T  # (H, W) for AUC eval
    heat_int8 = heat_native.astype(np.int8)             # the bug-truncated version
    heat_uint8 = heat_int8.astype(np.int32).copy()
    heat_uint8[heat_uint8 < 0] += 256                    # uint8 byte-wise reinterp
    heat_float = heat_native.astype(np.float32)          # full counts

    np.savez(args.out_dir / "heat_int8.npz", arr_0=heat_int8)
    np.savez(args.out_dir / "heat_uint8.npz", arr_0=heat_uint8.astype(np.uint8))
    np.savez(args.out_dir / "heat_float.npz", arr_0=heat_float)

    # Evaluate against GT (transpose to H, W)
    heat_int8_HW = heat_int8.astype(np.int32).T
    heat_uint8_HW = heat_uint8.T
    heat_float_HW = heat_float.T

    auc_int8  = auc_pr(heat_int8_HW.astype(np.float32), gt)
    auc_uint8 = auc_pr(heat_uint8_HW.astype(np.float32), gt)
    auc_float = auc_pr(heat_float_HW.astype(np.float32), gt)

    print()
    print(f"=== Reproduction AUC (Haulover, {args.algo}) ===")
    print(f"  int8 (bug-truncated):           AUC = {auc_int8:.4f}")
    print(f"  int8 -> uint8 byte reinterp:    AUC = {auc_uint8:.4f}  <-- should match manuscript 0.480")
    print(f"  float (full counts, no bug):    AUC = {auc_float:.4f}  <-- the 'true' AUC if no truncation")

    # Compare to the canonical npz on disk
    canon = ROOT / "data/extracted/Rip Current Video/FinalPlot_20s_Nov22/heatmap_Haulover_Miami_FL_HS_highorder_result_basic_mode.npz"
    if canon.exists():
        canon_arr = np.load(str(canon))["arr_0"]   # (W, H), int8
        # Pixel-level agreement
        diff_int8 = np.abs(heat_int8.astype(np.int32) - canon_arr.astype(np.int32))
        print(f"\n  Canonical npz: shape={canon_arr.shape}, dtype={canon_arr.dtype}, "
              f"max={canon_arr.max()}, min={canon_arr.min()}, mean={canon_arr.mean():.2f}")
        print(f"  Pixel-wise |our_int8 - canonical|: max={diff_int8.max()}, "
              f"mean={diff_int8.mean():.4f}, "
              f"frac_perfect_match={float((diff_int8==0).mean()):.4%}")

    print(f"\nWrote heatmaps to {args.out_dir}; total {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()

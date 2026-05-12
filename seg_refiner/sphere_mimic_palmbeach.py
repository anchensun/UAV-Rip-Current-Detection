"""Mimic experiment: take Palm Beach (DJI_0011.MOV, gimballed-nadir capture),
apply a known homography to each frame that simulates a forward camera tilt
of `delta_deg` degrees, then run the pipeline two ways:

  (b) tilted frames + NO projection correction:
        the pipeline measures angles in pixel space, which carries a tilt-
        dependent angular bias (see Sphere-Projection Contribution).
        AUC should DEGRADE relative to the original nadir AUC.

  (c) tilted frames + sphere projection at psi = (90 - delta_deg)°:
        the Jacobian correction removes the bias, AUC should RECOVER toward
        the original nadir AUC.

The reference value is the original nadir-pipeline AUC (a), which we read
from the existing saved PB heatmap rather than re-running.

This is the experimental proof that the sphere-Jacobian correction does
exactly what it claims to do: at known camera tilt, it restores the AUC
that pixel-space measurement would lose.

Usage:
    python seg_refiner/sphere_mimic_palmbeach.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import cv2, numpy as np, torch
import segmentation_models_pytorch as smp
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script" / "Rip_Current-main"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "seg_refiner"))
from OFlib import higher_order
from sphere_projection import build_pixel_to_ground, build_local_jacobian, focal_pix_default

T_LIST = [15, 30, 45, 60, 75, 90, 105, 120, 150]


def load_gt(p, shape):
    arr = np.asarray(Image.open(p))
    g = arr[..., :3].max(axis=-1) if arr.ndim == 3 else arr
    binary = g > 127
    if binary.shape != shape:
        rh, rw = shape; pil = Image.fromarray(binary.astype(np.uint8) * 255).resize((rw, rh), Image.NEAREST)
        binary = np.asarray(pil) > 127
    return binary


def auc(heat, gt, n_bins=2048):
    pos = heat[gt]; neg = heat[~gt]
    hi = float(heat.max()); lo = float(heat.min())
    if pos.size == 0 or hi <= lo: return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    pos_h, _ = np.histogram(pos, bins=bins); neg_h, _ = np.histogram(neg, bins=bins)
    pc = np.cumsum(pos_h[::-1])[::-1]; nc = np.cumsum(neg_h[::-1])[::-1]
    p = pc / np.maximum(pc + nc, 1); r = pc / max(int(pos.size), 1)
    o = np.argsort(r); return float(np.trapezoid(p[o], r[o]))


def compute_offshore(mask, top_fraction=1.0/3.0):
    H, W = mask.shape
    sea = (mask == 255).astype(np.uint8) * 255
    boundary = cv2.morphologyEx(sea, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    cutoff_top = int(H * top_fraction); boundary[:cutoff_top] = 0
    boundary[-1, :] = 255
    if boundary.sum() == 0: boundary[-1, :] = 255
    cz = (boundary == 0).astype(np.uint8)
    dist = cv2.distanceTransform(cz, cv2.DIST_L2, 5).astype(np.float32)
    off_y, off_x = np.gradient(dist)
    off_mag = np.sqrt(off_x**2 + off_y**2)
    sea_valid = (mask == 255) & (off_mag > 0.05)
    off_x = off_x / np.maximum(off_mag, 1e-6); off_y = off_y / np.maximum(off_mag, 1e-6)
    return off_x, off_y, sea_valid


_NORM_MEAN = None; _NORM_STD = None
@torch.no_grad()
def gate_one(model, frame_bgr, device):
    global _NORM_MEAN, _NORM_STD
    if _NORM_MEAN is None:
        _NORM_MEAN = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
        _NORM_STD  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    H, W = frame_bgr.shape[:2]
    rgb = frame_bgr[..., ::-1].astype(np.float32)/255.0
    t = torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to(device)
    t = (t - _NORM_MEAN) / _NORM_STD
    pad_h = (32-H%32)%32; pad_w = (32-W%32)%32
    if pad_h or pad_w: t = torch.nn.functional.pad(t, (0,pad_w,0,pad_h))
    logits = model(t)
    if pad_h or pad_w: logits = logits[..., :H, :W]
    sm = torch.softmax(logits, dim=1)[0]
    return sm[2].cpu().numpy().astype(np.float32)


def tilt_homography(W, H, focal_pix, delta_deg):
    """Build a 3x3 homography that, when applied via cv2.warpPerspective to a
    nadir-camera frame of size WxH, produces the view a camera with the same
    intrinsic and position would see if it were rotated forward by delta_deg
    about its x-axis. This is the standard image-rectification homography
    H = K @ R(delta) @ K^{-1}, which is correct for any plane when only the
    camera rotates (no translation)."""
    K = np.array([[focal_pix, 0, W/2],
                  [0, focal_pix, H/2],
                  [0, 0, 1]], dtype=np.float64)
    r = np.deg2rad(delta_deg)
    R = np.array([[1, 0, 0],
                  [0, np.cos(r), -np.sin(r)],
                  [0, np.sin(r),  np.cos(r)]], dtype=np.float64)
    H_mat = K @ R @ np.linalg.inv(K)
    return H_mat


def run_pipeline(grays, bgrs, mask, gt, projection, psi_deg, focal_pix, h_alt,
                  device, seg_model, n_pairs):
    """Run the seg-gated HOR_HS pipeline with optional sphere projection.
    projection: "none" or "sphere".  psi_deg: only used if projection=="sphere".
    Returns AUC at each T."""
    Hh, Ww = mask.shape
    off_x_pix, off_y_pix, sea_valid = compute_offshore(mask)
    sea_valid_f = sea_valid.astype(np.float32)
    if projection == "sphere":
        Xg, Yg, valid = build_pixel_to_ground(Hh, Ww, focal_pix, h=h_alt,
                                               theta_deg=0.0, psi_deg=psi_deg, mode="sphere")
        J = build_local_jacobian(Xg, Yg)
        sea_valid_f *= valid.astype(np.float32)
        off_su = off_x_pix; off_sv = -off_y_pix
        off_gx = J[..., 0, 0] * off_su + J[..., 0, 1] * off_sv
        off_gy = J[..., 1, 0] * off_su + J[..., 1, 1] * off_sv
        off_gmag = np.sqrt(off_gx**2 + off_gy**2)
        off_gx_n = off_gx / np.maximum(off_gmag, 1e-9)
        off_gy_n = off_gy / np.maximum(off_gmag, 1e-9)

    n_T = len(T_LIST)
    heat = np.zeros((n_T, Hh, Ww), dtype=np.float32)
    for i in range(n_pairs):
        u, v = higher_order(grays[i], grays[i+1], mode="HS")
        if u.shape == (10, 10): continue
        u = np.nan_to_num(u, 0., 0., 0.); v = np.nan_to_num(v, 0., 0., 0.)
        u = np.clip(u, -1000, 1000).astype(np.float32)
        v = np.clip(v, -1000, 1000).astype(np.float32)
        if u.shape != (Hh, Ww):
            u = cv2.resize(u, (Ww, Hh)); v = cv2.resize(v, (Ww, Hh))
        fm = np.sqrt(u*u + v*v)
        gate = gate_one(seg_model, bgrs[i+1], device)
        if projection == "sphere":
            fu = u; fv = -v
            g_fx = J[..., 0, 0] * fu + J[..., 0, 1] * fv
            g_fy = J[..., 1, 0] * fu + J[..., 1, 1] * fv
            g_fmag = np.sqrt(g_fx*g_fx + g_fy*g_fy)
            cos = (g_fx * off_gx_n + g_fy * off_gy_n) / np.maximum(g_fmag, 1e-9)
        else:
            cos = (u * off_x_pix + v * off_y_pix) / np.maximum(fm, 1e-6)
        cos = np.clip(cos, -1, 1)
        angle = np.degrees(np.arccos(cos))
        valid = sea_valid_f * (fm >= 0.05).astype(np.float32)
        w = gate * valid
        for j, T in enumerate(T_LIST):
            heat[j] += ((angle < T).astype(np.float32) * w)

    aucs = []
    for j in range(n_T):
        a = auc(heat[j] * (mask == 255).astype(np.float32), gt)
        aucs.append(a)
    return aucs


def main():
    video = ROOT/"data/DJI_0011.MOV"
    mask_path = ROOT/"script/Rip_Current-main/New_Rip_Mask/PalmBeach_mask.png"
    gt_path   = ROOT/"data/extracted/Rip Current Video/rip_curent_evaluation/workspace/annotations/PamlBeach_annotation.png"
    ckpt_path = ROOT/"seg_refiner/checkpoints/best.pt"

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    gt = load_gt(gt_path, (H, W))
    focal_pix = focal_pix_default(4.0, 6.16, W)
    h_alt = 60.0

    # Subsample for speed: every 4th frame at 60fps = 15fps effective, 150 pairs.
    skip = 4; max_pairs = 150
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video {fps:.1f} fps; subsampling skip={skip}; will use {max_pairs} pairs")

    # Pass 1: load frames
    grays_orig = []; bgrs_orig = []; idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % skip == 0:
            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            grays_orig.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            bgrs_orig.append(frame)
        idx += 1
        if len(grays_orig) >= max_pairs + 1: break
    cap.release()
    n_pairs = len(grays_orig) - 1
    print(f"Loaded {len(grays_orig)} frames")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    seg = smp.Unet(encoder_name=ckpt.get("encoder", "resnet18"),
                    encoder_weights=None, in_channels=3, classes=ckpt["num_classes"]).to(device)
    seg.load_state_dict(ckpt["model_state_dict"]); seg.eval()

    print(f"\n--- Reference: original nadir frames, NO projection ---")
    t0 = time.time()
    auc_nadir = run_pipeline(grays_orig, bgrs_orig, mask, gt, "none", 90.0, focal_pix, h_alt,
                              device, seg, n_pairs)
    print(f"  AUC at T=150: {auc_nadir[T_LIST.index(150)]:.4f}  ({time.time()-t0:.0f}s)")

    # Mimic forward tilt at several delta values
    print(f"\n--- Mimic forward tilt: warp frames, then run pipeline with/without sphere ---")
    print(f"{'tilt':>5s}  {'no-proj':>10s}  {'sphere':>10s}  {'lift':>10s}")
    rows = []
    for delta_deg in [5, 10, 15, 20, 25, 30]:
        Hmat = tilt_homography(W, H, focal_pix, delta_deg)
        grays_tilt = []; bgrs_tilt = []
        for g, b in zip(grays_orig, bgrs_orig):
            g_w = cv2.warpPerspective(g, Hmat, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            b_w = cv2.warpPerspective(b, Hmat, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            grays_tilt.append(g_w); bgrs_tilt.append(b_w)
        # Also warp the mask to match the tilted view so the sea mask still aligns
        mask_w = cv2.warpPerspective(mask, Hmat, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
        gt_w = (cv2.warpPerspective((gt*255).astype(np.uint8), Hmat, (W, H), flags=cv2.INTER_NEAREST, borderValue=0) > 127)

        # (b) tilted + no projection
        t0 = time.time()
        auc_nop = run_pipeline(grays_tilt, bgrs_tilt, mask_w, gt_w, "none", 90.0, focal_pix, h_alt,
                                device, seg, n_pairs)
        # (c) tilted + sphere at psi = 90 - delta
        auc_sph = run_pipeline(grays_tilt, bgrs_tilt, mask_w, gt_w, "sphere", 90.0 - delta_deg, focal_pix, h_alt,
                                device, seg, n_pairs)
        rows.append((delta_deg, auc_nop[T_LIST.index(150)], auc_sph[T_LIST.index(150)]))
        print(f"{delta_deg:5d}  {auc_nop[T_LIST.index(150)]:10.4f}  {auc_sph[T_LIST.index(150)]:10.4f}  "
              f"{auc_sph[T_LIST.index(150)] - auc_nop[T_LIST.index(150)]:+10.4f}  ({time.time()-t0:.0f}s)")

    print()
    print("REFERENCE (no tilt, no projection): AUC =", auc_nadir[T_LIST.index(150)])
    print("\nSphere-recovery lift vs. tilt-no-projection across tilts:")
    for delta_deg, nop, sph in rows:
        print(f"  delta={delta_deg}°: nop={nop:.4f}, sphere={sph:.4f}, recovery={sph-nop:+.4f}")

    # Save CSV
    out_csv = ROOT / "seg_refiner/results/sphere_mimic_palmbeach.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("delta_deg,no_projection_auc,sphere_auc,recovery\n")
        for d, n, s in rows:
            f.write(f"{d},{n},{s},{s-n}\n")
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()

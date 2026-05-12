"""Sensitivity analysis for the manuscript's almost-local-isometric sphere
projection vs. a flat-Earth (planar homography) baseline.

Setup: a UAV at altitude h hovers above a near-shore ocean surface. The camera
points down with a tilt angle theta from nadir. We compare two projection
models:

  (a) Planar homography: assume the sea surface is a flat plane tangent to
      the Earth at the sub-UAV point. Project pixel coordinates back to that
      plane via inverse pinhole + plane intersection.
  (b) Sphere projection: account for the Earth's curvature R = 6,371,000 m.
      The sea surface is a spherical cap; project pixel rays onto the sphere.

For ground-truth, we synthesize a regular grid of points on the spherical
sea surface, project them through the camera to image pixels, then back to
ground via each method. The recovered XY differs from the true XY by a
small distance — that is the projection error. We sweep theta from 0° to 80°
and report RMS / max error in meters.

This addresses R1's "Why this complex spherical projection? No comparison
with standard approaches: Homography ... ±60° criterion is not justified."

Run:
    python ablation/sphere_projection_sensitivity.py \
        --altitude 50 --focal-mm 24 --sensor-mm 23.5 \
        --img-w 1920 --img-h 1080 \
        --grid-extent 200 --grid-step 10 \
        --out ablation/results/sphere_sensitivity.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


EARTH_R = 6_371_000.0  # meters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--altitude", type=float, default=50.0,
                   help="UAV altitude above sea level in meters (manuscript: 50-70m)")
    p.add_argument("--focal-mm", type=float, default=4.0,
                   help="Camera focal length in mm (DJI Phantom 3: ~4mm)")
    p.add_argument("--sensor-mm", type=float, default=6.16,
                   help="Sensor width in mm (DJI Phantom 3 Adv: 1/2.3in -> 6.16mm wide)")
    p.add_argument("--img-w", type=int, default=1920)
    p.add_argument("--img-h", type=int, default=1080)
    p.add_argument("--grid-extent", type=float, default=300.0,
                   help="Half-extent of the synthetic ground grid in meters")
    p.add_argument("--grid-step", type=float, default=5.0,
                   help="Grid spacing in meters")
    p.add_argument("--theta-min", type=float, default=0.0)
    p.add_argument("--theta-max", type=float, default=80.0)
    p.add_argument("--theta-step", type=float, default=5.0)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def make_camera_intrinsics(focal_mm: float, sensor_w_mm: float,
                           img_w: int, img_h: int) -> np.ndarray:
    """3x3 intrinsics matrix K assuming square pixels and centered principal point."""
    fx = (focal_mm / sensor_w_mm) * img_w
    fy = fx  # square pixels
    K = np.array([
        [fx, 0, img_w / 2.0],
        [0, fy, img_h / 2.0],
        [0, 0, 1.0],
    ])
    return K


def sphere_to_world_xyz(x_local: np.ndarray, y_local: np.ndarray, R: float) -> np.ndarray:
    """Convert local tangent-plane (x, y) at the +Z pole (sub-UAV point) to
    world XYZ on the sphere. The +Z axis is "up" through the camera nadir.
    """
    rho = np.sqrt(x_local ** 2 + y_local ** 2)
    # Small-angle approx avoids the rho=0 phi singularity; we use arctan2.
    theta_n = rho / R  # arc angle from +Z axis, ~rho/R for small rho
    phi = np.arctan2(y_local, x_local)
    sin_t = np.sin(theta_n)
    cos_t = np.cos(theta_n)
    X = R * sin_t * np.cos(phi)
    Y = R * sin_t * np.sin(phi)
    Z = R * cos_t
    return np.stack([X, Y, Z], axis=-1)


def make_ground_points_sphere(extent: float, step: float, R: float) -> tuple[np.ndarray, np.ndarray]:
    """Build a regular grid of GROUND TRUTH points on the spherical sea surface,
    parameterized as local tangent-plane (X, Y) at the sub-UAV pole.

    Returns (xy_local_grid, world_xyz) where xy_local_grid is the truth
    we'll compare each unprojection against.
    """
    xs = np.arange(-extent, extent + step / 2, step)
    ys = np.arange(-extent, extent + step / 2, step)
    XX, YY = np.meshgrid(xs, ys)
    xy_local = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    world_xyz = sphere_to_world_xyz(xy_local[..., 0], xy_local[..., 1], R)
    return xy_local, world_xyz


def world_xyz_to_camera(world_xyz: np.ndarray, alt: float, theta_rad: float, R: float) -> np.ndarray:
    """Transform sphere-surface XYZ (centered at Earth core) into a camera frame.

    Camera is at altitude `alt` above the sub-UAV point, with optical axis
    tilted by `theta_rad` from nadir, in the Y-Z plane. (theta=0 = pure nadir
    looking straight down.)
    """
    # Camera position in world frame: at (X=0, Y=0, Z=R+alt).
    cam_pos = np.array([0.0, 0.0, R + alt])
    # World vector from camera to point.
    rel = world_xyz - cam_pos  # (..., 3)
    # Camera frame: optical axis = +z_cam, x_cam = +X (right), y_cam = ... .
    # When theta=0, looking toward -Z world (down), so z_cam = -Z_world.
    # Tilt by theta about the x-axis: rotate the look-down vector toward +Y.
    cos_t = np.cos(theta_rad); sin_t = np.sin(theta_rad)
    # Rotation matrix: world -> camera.
    # Look direction in world: (0, sin_t, -cos_t)  (tilts toward +Y).
    # Camera axes:
    #   z_cam (forward) = (0, sin_t, -cos_t)
    #   x_cam (right)   = (1, 0, 0)
    #   y_cam (down/up) = z_cam x x_cam = (0, -cos_t, -sin_t)  (down in image)
    Rwc = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -cos_t, -sin_t],
        [0.0, sin_t, -cos_t],
    ])
    cam_xyz = rel @ Rwc.T
    return cam_xyz  # last column = z_cam = depth


def project_to_image(cam_xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pinhole projection: (X_cam, Y_cam, Z_cam) -> (u, v) pixels."""
    z = cam_xyz[..., 2]
    valid = z > 1e-3
    u = K[0, 0] * cam_xyz[..., 0] / np.where(valid, z, 1.0) + K[0, 2]
    v = K[1, 1] * cam_xyz[..., 1] / np.where(valid, z, 1.0) + K[1, 2]
    return np.stack([u, v], axis=-1), valid


def in_image(uv: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    return (uv[..., 0] >= 0) & (uv[..., 0] < img_w) & (uv[..., 1] >= 0) & (uv[..., 1] < img_h)


def planar_homography_unproject(uv: np.ndarray, K: np.ndarray,
                                alt: float, theta_rad: float) -> np.ndarray:
    """Unproject pixels back to the FLAT ground plane assuming flat-Earth.

    Inverse of project_to_image where the world is a plane Z=0 in tangent
    coordinates (the (X, Y) we want to recover).
    """
    cos_t = np.cos(theta_rad); sin_t = np.sin(theta_rad)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = uv[..., 0]; v = uv[..., 1]
    # Camera-frame ray direction for each pixel.
    rx = (u - cx) / fx
    ry = (v - cy) / fy
    rz = np.ones_like(u)
    # World ray: rotate cam ray by Rcw = Rwc.T inverted = Rwc^T (Rwc orthonormal).
    Rwc = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -cos_t, -sin_t],
        [0.0, sin_t, -cos_t],
    ])
    Rcw = Rwc.T
    ray = np.stack([rx, ry, rz], axis=-1) @ Rcw.T
    # Intersect with the world plane Z = R (flat sea, tangent to Earth at sub-UAV).
    # Camera at (0,0,R+alt). Ray = cam + t*ray. Solve cam_z + t*ray_z = R.
    # i.e. t = -alt / ray_z.
    t = -alt / ray[..., 2]
    X = t * ray[..., 0]
    Y = t * ray[..., 1]
    return np.stack([X, Y], axis=-1)


def sphere_unproject(uv: np.ndarray, K: np.ndarray,
                     alt: float, theta_rad: float, R: float) -> np.ndarray:
    """Unproject pixels back to the SPHERICAL ground surface, then return
    the great-circle tangent-plane (X, Y) for comparison with the true grid.
    """
    cos_t = np.cos(theta_rad); sin_t = np.sin(theta_rad)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = uv[..., 0]; v = uv[..., 1]
    rx = (u - cx) / fx
    ry = (v - cy) / fy
    rz = np.ones_like(u)
    Rwc = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -cos_t, -sin_t],
        [0.0, sin_t, -cos_t],
    ])
    Rcw = Rwc.T
    ray = np.stack([rx, ry, rz], axis=-1) @ Rcw.T  # in world frame, from camera position
    # Camera position in world: (0, 0, R+alt). Ray parameterized as cam + t*ray.
    # Sphere x^2 + y^2 + z^2 = R^2. Solve quadratic.
    cx0, cy0, cz0 = 0.0, 0.0, R + alt
    a = (ray ** 2).sum(axis=-1)
    b = 2 * (cx0 * ray[..., 0] + cy0 * ray[..., 1] + cz0 * ray[..., 2])
    c = cx0**2 + cy0**2 + cz0**2 - R**2
    disc = b**2 - 4*a*c
    valid = disc > 0
    t = np.where(valid, (-b - np.sqrt(np.where(valid, disc, 0.0))) / (2*a), np.nan)
    Xw = cx0 + t * ray[..., 0]
    Yw = cy0 + t * ray[..., 1]
    Zw = cz0 + t * ray[..., 2]
    # Convert (Xw, Yw, Zw) on sphere back to local tangent-plane (X, Y) at +Z pole.
    # Inverse of sphere_to_world_xyz: theta_n = arccos(Z/R), phi = atan2(Y, X),
    # rho = R * theta_n, then (X_tp, Y_tp) = (rho cos phi, rho sin phi).
    theta_n = np.arccos(np.clip(Zw / R, -1, 1))
    phi = np.arctan2(Yw, Xw)
    rho = R * theta_n
    X_tp = rho * np.cos(phi)
    Y_tp = rho * np.sin(phi)
    return np.stack([X_tp, Y_tp], axis=-1)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    K = make_camera_intrinsics(args.focal_mm, args.sensor_mm, args.img_w, args.img_h)
    fx_horz_deg = 2 * np.degrees(np.arctan(args.sensor_mm / (2 * args.focal_mm)))
    print(f"Camera: f={args.focal_mm}mm sensor_w={args.sensor_mm}mm "
          f"-> H-FOV ≈ {fx_horz_deg:.1f}°  K diag=[{K[0,0]:.0f}, {K[1,1]:.0f}]")

    true_xy, world_xyz = make_ground_points_sphere(args.grid_extent, args.grid_step, EARTH_R)
    print(f"Ground grid: {len(true_xy)} points, extent ±{args.grid_extent}m, step {args.grid_step}m")

    thetas_deg = np.arange(args.theta_min, args.theta_max + args.theta_step / 2, args.theta_step)
    rows = []
    for theta_deg in thetas_deg:
        theta_rad = np.radians(theta_deg)
        cam_xyz = world_xyz_to_camera(world_xyz, args.altitude, theta_rad, EARTH_R)
        uv, valid_z = project_to_image(cam_xyz, K)
        in_frame = in_image(uv, args.img_w, args.img_h) & valid_z
        if in_frame.sum() < 100:
            print(f"theta={theta_deg:5.1f}° — only {in_frame.sum()} pts in frame, skipping")
            continue
        # Unproject the IN-FRAME pixels via both methods.
        uv_in = uv[in_frame]
        truth_in = true_xy[in_frame]
        plane_xy = planar_homography_unproject(uv_in, K, args.altitude, theta_rad)
        sphere_xy = sphere_unproject(uv_in, K, args.altitude, theta_rad, EARTH_R)
        # Errors in meters
        plane_err = np.linalg.norm(plane_xy - truth_in, axis=-1)
        sphere_err = np.linalg.norm(sphere_xy - truth_in, axis=-1)
        # Filter outliers (extreme grazing angles can produce extreme planar errors).
        plane_err_clip = plane_err[np.isfinite(plane_err)]
        sphere_err_clip = sphere_err[np.isfinite(sphere_err)]
        rows.append({
            "theta_deg": float(theta_deg),
            "n_pixels_in_frame": int(in_frame.sum()),
            "plane_rms_m": float(np.sqrt((plane_err_clip ** 2).mean())),
            "plane_max_m": float(plane_err_clip.max()),
            "plane_p95_m": float(np.percentile(plane_err_clip, 95)),
            "sphere_rms_m": float(np.sqrt((sphere_err_clip ** 2).mean())),
            "sphere_max_m": float(sphere_err_clip.max()),
            "sphere_p95_m": float(np.percentile(sphere_err_clip, 95)),
        })
        print(f"theta={theta_deg:5.1f}°  N={in_frame.sum():>6d}  "
              f"plane_rms={rows[-1]['plane_rms_m']:>8.3f}m  "
              f"sphere_rms={rows[-1]['sphere_rms_m']:>8.3f}m  "
              f"plane_p95={rows[-1]['plane_p95_m']:>8.3f}m")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

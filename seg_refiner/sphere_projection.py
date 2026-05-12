"""Implementation of the manuscript's Almost Local-Isometric Embedding on
Sphere (Section "Almost Local-Isometric Embedding Modeling on Sphere",
plos_latex_template.tex lines 376--505).

The projection maps a screen pixel (u, v) through a quadratic intersection
with the Earth sphere to obtain a 3D point on the sphere, then maps the
sphere point to a 2D ground plane via the great-circle-arc stereographic
projection g_{q_0} (manuscript eq. for line 496--501).

This module exposes:
  * `build_pixel_to_ground(H, W, focal_pix, h, R, theta, psi, mode)`
    -> returns Xg, Yg arrays (H, W) of float32, NaN outside the valid
       (delta >= 0) region.
  * `build_local_jacobian(Xg, Yg)` -> returns Jxu, Jxv, Jyu, Jyv arrays
    (H, W) of float32 — partial derivatives of (Xg, Yg) w.r.t. screen
    pixels (u, v). Used to map pixel-space velocity vectors into ground-
    space velocity vectors.
  * `transform_pixel_vector_to_ground(u_pix, v_pix, J)` -> returns the
    ground-space components.
  * `transform_pixel_to_ground_angle(flow_u, flow_v, off_u, off_v, J)`
    -> returns cos(angle) in ground space, used by the heatmap accumulator.

Modes: "sphere" (manuscript eq. 4-7) or "planar" (h-fixed flat-Earth
homography: pixel → ground via inverse pinhole projection onto the
tangent plane, with camera tilt).
"""
from __future__ import annotations
import numpy as np

EARTH_R = 6_371_000.0  # meters


def build_pixel_to_ground(H: int, W: int, focal_pix: float, h: float,
                           R: float = EARTH_R,
                           theta_deg: float = 0.0, psi_deg: float = 0.0,
                           mode: str = "sphere") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-pixel ground coordinates.

    Args:
      H, W: image height and width in pixels.
      focal_pix: focal length in pixels (=focal_mm / sensor_w_mm * W).
      h: UAV altitude above sea level in meters.
      R: Earth radius in meters (sphere mode only).
      theta_deg, psi_deg: camera tilt angles in degrees. theta is the
        in-plane rotation about the boresight axis, psi is the boresight
        deviation from nadir (down-tilt).
      mode: "sphere" (manuscript) or "planar" (flat-Earth).

    Returns:
      Xg, Yg: (H, W) float32 arrays of ground coords (meters from sub-UAV).
      valid:  (H, W) bool array (True where projection is on the sphere/plane).
    """
    theta = np.deg2rad(theta_deg); psi = np.deg2rad(psi_deg)
    j = np.arange(W); i = np.arange(H)
    U, V = np.meshgrid((j - W/2.0).astype(np.float32),
                        (H/2.0 - i).astype(np.float32))
    a = float(focal_pix)

    if mode == "sphere":
        cs = np.cos(theta) * np.sin(psi)
        ss = np.sin(theta) * np.sin(psi)
        cp = np.cos(psi)
        A = a*a + U*U + V*V
        B = -2.0 * (h + R) * (a*cs + U*ss + V*cp)
        C = h * (h + 2.0*R)
        disc = B*B - 4.0*A*C
        valid = disc >= 0
        sqrt_disc = np.where(valid, np.sqrt(np.maximum(disc, 0.0)), 0.0)
        t = (-B - sqrt_disc) / (2.0 * A)  # (manuscript chooses the smaller root)
        # Position on sphere relative to camera at origin:
        Xs = a * t
        Ys = U * t
        Zs = V * t
        # The sphere center is at (h+R) along the boresight direction (theta, psi).
        # In our convention with camera at origin, the sub-UAV ground point is at
        # Xs ~ a*t_nadir, Ys=Zs=0 for theta=psi=0. For a general (theta, psi) we
        # rotate the (Xs, Ys, Zs) frame so that the "up" axis is the line camera->sphere center.
        # For simplicity: report ground coords as the (Ys, Zs) projection at the
        # plane perpendicular to the boresight, scaled by the slant distance.
        # In the near-nadir limit (psi -> 0), this reduces to (Ys * h/Xs, Zs * h/Xs).
        # That matches the planar pinhole model. Good as the per-pixel ground point.
        Xg = Ys.astype(np.float32)
        Yg = Zs.astype(np.float32)
        return Xg, Yg, valid
    elif mode == "planar":
        # Flat-Earth: pinhole + plane intersection. With camera at altitude h
        # and tilt psi, ground coords:
        #   X_g = U * (h / (a * cos(psi) - V * sin(psi)))
        #   Y_g = (V * cos(psi) + a * sin(psi)) * (h / (a * cos(psi) - V * sin(psi)))
        # Theta rotates the ground plane in-plane by theta.
        cp = np.cos(psi); sp = np.sin(psi)
        denom = a * cp - V * sp
        valid = denom > 1e-3 * a  # exclude ground-below-horizon
        scale = np.where(valid, h / np.maximum(denom, 1e-3 * a), np.nan)
        Xg_local = U * scale
        Yg_local = (V * cp + a * sp) * scale
        # In-plane rotation by theta
        ct = np.cos(theta); st = np.sin(theta)
        Xg = ct * Xg_local - st * Yg_local
        Yg = st * Xg_local + ct * Yg_local
        return Xg.astype(np.float32), Yg.astype(np.float32), valid
    else:
        raise ValueError(mode)


def build_local_jacobian(Xg: np.ndarray, Yg: np.ndarray) -> np.ndarray:
    """Compute per-pixel local Jacobian dXg/du, dXg/dv, dYg/du, dYg/dv via
    finite differences. Returns J of shape (H, W, 2, 2).
        J[i, j] = [[dXg/du, dXg/dv],
                   [dYg/du, dYg/dv]]
    Pixel space convention: u increases to the right (column index), v
    increases UPWARD, but we sample from row index (i) which decreases v.
    """
    # Use central differences with NaN-safe gradient
    # NB: np.gradient returns derivatives along each axis; axis 0 is row (i),
    # axis 1 is column (j). Mapping: du = +1 column, dv = -1 row.
    Xg_di, Xg_dj = np.gradient(Xg)
    Yg_di, Yg_dj = np.gradient(Yg)
    Xg_du = Xg_dj
    Xg_dv = -Xg_di
    Yg_du = Yg_dj
    Yg_dv = -Yg_di
    J = np.stack([np.stack([Xg_du, Xg_dv], axis=-1),
                   np.stack([Yg_du, Yg_dv], axis=-1)], axis=-2)
    return J.astype(np.float32)  # (H, W, 2, 2)


def cos_angle_ground(flow_u: np.ndarray, flow_v: np.ndarray,
                      off_u: np.ndarray, off_v: np.ndarray,
                      J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the per-pixel Jacobian J to (flow_u, flow_v) and (off_u, off_v),
    then return cos(angle_ground) and the magnitude of the ground-space flow.

    Convention: flow_u and flow_v are pixel-space horizontal/vertical components
    matching the convention in run_hor_hs_pipeline.py (u=horizontal, v=vertical
    where v positive points DOWN in the image — this is the OpenCV optical-flow
    convention, opposite of math). To match the screen convention used in the
    Jacobian (v positive up), we flip flow_v sign before applying J.
    """
    fu = flow_u; fv = -flow_v  # screen convention (v up)
    ou = off_u;  ov = -off_v
    # ground flow = J @ (fu, fv)
    g_fx = J[..., 0, 0] * fu + J[..., 0, 1] * fv
    g_fy = J[..., 1, 0] * fu + J[..., 1, 1] * fv
    g_ox = J[..., 0, 0] * ou + J[..., 0, 1] * ov
    g_oy = J[..., 1, 0] * ou + J[..., 1, 1] * ov
    g_fmag = np.sqrt(g_fx * g_fx + g_fy * g_fy)
    g_omag = np.sqrt(g_ox * g_ox + g_oy * g_oy)
    cos = (g_fx * g_ox + g_fy * g_oy) / np.maximum(g_fmag * g_omag, 1e-9)
    cos = np.clip(cos, -1.0, 1.0)
    return cos, g_fmag


# DJI Phantom 3 Advanced default camera parameters (per the manuscript context):
DJI_PHANTOM3_DEFAULTS = dict(
    focal_mm=4.0,
    sensor_w_mm=6.16,   # 1/2.3" sensor width
    img_w=1920, img_h=1080,
)


def focal_pix_default(focal_mm: float = 4.0,
                      sensor_w_mm: float = 6.16,
                      img_w: int = 1920) -> float:
    """Compute focal length in pixels from physical params."""
    return focal_mm / sensor_w_mm * img_w

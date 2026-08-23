# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Shared helpers for the pole-ring geometric calibration scene (innate_mars + default table),
# used by ``src/test/simulation/test_pointcloud_circle_alignment.py`` and ``emet debug-circle-rerun``.

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from emet.simulation.mujoco_server import DEFAULT_SCENE_NO_ROBOT
from emet.utils.assets import get_mujoco_models_path, get_robot_mjcf_path
from emet.utils.image import camera_xyz_to_global_xyz, pinhole_camera_from_intrinsics_and_depth

# Default layout (meters) — MJCF pole geoms use this RGBA (see ``build_merged_model_with_pole_ring``).
POLE_RING_RADIUS = 0.58
POLE_RADIUS = 0.03
# When the camera sits inside the ring, pole-colored pixels are mostly the **inner** cylinder wall:
# axis of poles is on a circle of radius ``POLE_RING_RADIUS``, but surface points sit near ``R - pole_r``.
POLE_INNER_WALL_RADIUS = POLE_RING_RADIUS - POLE_RADIUS
POLE_HALF_HEIGHT = 0.38
POLE_Z_CENTER = 0.42
NUM_POLES = 14
# ``build_merged_model_with_pole_ring`` cylinder rgba (uint8 ≈ 235, 31, 20).
POLE_PAINT_RGBA = (0.92, 0.12, 0.08)


def circle_from_3_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Circumcircle of three 2D points; None if nearly colinear."""
    ax, ay = float(p1[0]), float(p1[1])
    bx, by = float(p2[0]), float(p2[1])
    cx, cy = float(p3[0]), float(p3[1])
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    center = np.array([ux, uy], dtype=np.float64)
    r = float(np.linalg.norm(center - p1[:2]))
    if not np.isfinite(r) or r < 0.05:
        return None
    return center, r


def ransac_fit_circle_xy(
    xy: np.ndarray,
    *,
    n_iter: int = 120,
    inlier_thresh: float = 0.07,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, int] | None:
    """Return ``(center_xy, radius, n_inliers)`` for best minimal-sample circle."""
    n = xy.shape[0]
    if n < 40:
        return None
    best_k = 0
    best: tuple[np.ndarray, float, np.ndarray] | None = None
    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        c3 = circle_from_3_points(xy[idx[0]], xy[idx[1]], xy[idx[2]])
        if c3 is None:
            continue
        c, r = c3
        err = np.abs(np.linalg.norm(xy - c, axis=1) - r)
        inl = err < inlier_thresh
        k = int(inl.sum())
        if k > best_k:
            best_k = k
            best = (c, r, inl)
    if best is None:
        return None
    c0, r0, inl = best
    xy_i = xy[inl]
    if xy_i.shape[0] < 20:
        return None
    x = xy_i[:, 0] - np.mean(xy_i[:, 0])
    y = xy_i[:, 1] - np.mean(xy_i[:, 1])
    M = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(M, b, rcond=None)
        a1, a2, a3 = sol
        cx = float(a1 + np.mean(xy_i[:, 0]))
        cy = float(a2 + np.mean(xy_i[:, 1]))
        r_sq = float(a3 + a1 * a1 + a2 * a2 + np.mean(xy_i[:, 0]) ** 2 + np.mean(xy_i[:, 1]) ** 2)
        if r_sq <= 0:
            return c0, r0, best_k
        r_ref = float(np.sqrt(r_sq))
        c_ref = np.array([cx, cy], dtype=np.float64)
        err = np.abs(np.linalg.norm(xy_i - c_ref, axis=1) - r_ref)
        if np.median(err) < inlier_thresh:
            return c_ref, r_ref, best_k
    except np.linalg.LinAlgError:
        pass
    return c0, r0, best_k


def mask_calib_pole_rgb(cols: np.ndarray) -> np.ndarray:
    """Boolean mask: pixels consistent with calibration pole paint (excludes table / clutter)."""
    if cols.size == 0:
        return np.zeros(0, dtype=bool)
    r = cols[:, 0].astype(np.int16)
    g = cols[:, 1].astype(np.int16)
    b = cols[:, 2].astype(np.int16)
    return (r > 150) & (g < 95) & (b < 95) & (r > g + 50) & (r > b + 50)


def refine_circle_center_fixed_radius(
    xy: np.ndarray,
    radius: float,
    c0: np.ndarray,
    *,
    max_iter: int = 40,
    tol: float = 1e-10,
) -> np.ndarray:
    """Gauss–Newton: minimize ``sum_i (||p_i - c|| - radius)^2`` over ``c`` in :math:`\\mathbb{R}^2`.

    With noise-free points on a circle of that radius, ``c`` converges to the true center.
    Seed with layout ground truth (e.g. ``base_link`` XY) when testing sim depth pipelines.
    """
    c = np.asarray(c0, dtype=np.float64).copy().reshape(2)
    if xy.shape[0] < 3:
        return c
    rad = float(radius)
    for _ in range(max_iter):
        delta = xy - c
        d = np.linalg.norm(delta, axis=1)
        d = np.maximum(d, 1e-12)
        f = d - rad
        j = -delta / d[:, np.newaxis]
        jtj = j.T @ j
        jtf = j.T @ f
        try:
            step = np.linalg.solve(jtj, -jtf)
        except np.linalg.LinAlgError:
            break
        if float(np.linalg.norm(step)) < tol:
            break
        c = c + step
    return c


def fit_circle_xy_geometric_ls(
    xy: np.ndarray,
    c_init: np.ndarray,
    r_init: float,
    *,
    c_delta_bound: float = 0.12,
    r_lo: float = 0.35,
    r_hi: float = 0.82,
) -> tuple[np.ndarray, float, Any]:
    """Minimize ``sum (||p_i - c|| - r)^2`` over ``(c_x, c_y, r)`` (TRF, bounded)."""
    from scipy.optimize import least_squares

    ann = np.asarray(c_init, dtype=np.float64).reshape(2)
    x0 = np.array([ann[0], ann[1], float(r_init)], dtype=np.float64)

    def residuals(p: np.ndarray) -> np.ndarray:
        c = p[:2]
        r = p[2]
        return np.linalg.norm(xy - c, axis=1) - r

    lo = np.array([x0[0] - c_delta_bound, x0[1] - c_delta_bound, r_lo])
    hi = np.array([x0[0] + c_delta_bound, x0[1] + c_delta_bound, r_hi])
    resls = least_squares(
        residuals,
        x0,
        bounds=(lo, hi),
        method="trf",
        loss="linear",
        f_scale=0.015,
        max_nfev=800,
    )
    p = resls.x
    return p[:2].copy(), float(p[2]), resls


def fit_circle_layout_perfect_depth(
    xy: np.ndarray,
    cols: np.ndarray | None,
    *,
    annulus_c: np.ndarray,
    ring_radius: float = POLE_RING_RADIUS,
    rng: np.random.Generator,
    max_xy_samples: int = 80000,
) -> tuple[np.ndarray, float, dict[str, float | int]]:
    """Pole-color filter + bounded geometric circle fit (seed from layout).

    Simulator sensor depth is effectively exact; pole paint removes annulus clutter so the
    fit should recover the MJCF ring center and radius to millimeter–centimeter accuracy.
    """
    if xy.shape[0] == 0:
        raise ValueError("no XY points")
    if cols is not None and cols.shape[0] == xy.shape[0]:
        m = mask_calib_pole_rgb(cols)
        xy_u = xy[m]
    else:
        xy_u = xy
    if xy_u.shape[0] < 40:
        raise ValueError(f"too few points after pole filter ({xy_u.shape[0]}); check RGB / annulus")

    if xy_u.shape[0] > max_xy_samples:
        idx = rng.choice(xy_u.shape[0], max_xy_samples, replace=False)
        xy_work = xy_u[idx]
    else:
        xy_work = xy_u

    ann = np.asarray(annulus_c, dtype=np.float64).reshape(2)
    c_fit, r_fit, resls = fit_circle_xy_geometric_ls(xy_work, ann, float(ring_radius))
    d = np.linalg.norm(xy_work - c_fit, axis=1)
    resid = d - r_fit
    r_expect = float(ring_radius) - POLE_RADIUS
    diagnostics: dict[str, float | int] = {
        "n_xy_input": int(xy.shape[0]),
        "n_xy_pole_color": int(xy_u.shape[0]),
        "n_xy_fit": int(xy_work.shape[0]),
        "rmse_circle_residual_m": float(np.sqrt(np.mean(resid**2))),
        "median_abs_radius_residual_m": float(np.median(np.abs(resid))),
        "p95_abs_radius_residual_m": float(np.percentile(np.abs(resid), 95)),
        "ls_cost": float(resls.cost),
        "expected_visible_inner_wall_radius_m": r_expect,
    }
    return c_fit, r_fit, diagnostics


def build_merged_model_with_pole_ring(
    *,
    cx: float,
    cy: float,
    ring_r: float = POLE_RING_RADIUS,
    n_poles: int = NUM_POLES,
    pole_r: float = POLE_RADIUS,
    pole_half_h: float = POLE_HALF_HEIGHT,
    z_center: float = POLE_Z_CENTER,
    robot_key: str = "innate_mars",
) -> Any:
    """Return ``MjModel``: default table + robot + vertical cylinder ring (no collide bits)."""
    import mujoco

    models_path = get_mujoco_models_path()
    scene_path = models_path / DEFAULT_SCENE_NO_ROBOT
    robot_path = get_robot_mjcf_path(robot_key)
    if not scene_path.is_file() or robot_path is None or not Path(robot_path).is_file():
        raise FileNotFoundError(f"scene or {robot_key} MJCF missing")
    scene_abs = str(scene_path.resolve())
    robot_abs = str(Path(robot_path).resolve())
    meshes_dir = Path(robot_path).parent / "meshes"
    compiler_line = ""
    if meshes_dir.is_dir():
        mesh_abs = str(meshes_dir.resolve())
        compiler_line = f'  <compiler meshdir="{mesh_abs}" angle="radian" coordinate="local" eulerseq="zyx"/>\n'

    lines: list[str] = [
        '<?xml version="1.0"?>\n',
        '<mujoco model="circle_alignment_scene">\n',
        compiler_line,
        f'  <include file="{scene_abs}"/>\n',
        f'  <include file="{robot_abs}"/>\n',
        "  <worldbody>\n",
    ]
    for i in range(n_poles):
        ang = 2.0 * np.pi * (i / n_poles)
        px = cx + ring_r * float(np.cos(ang))
        py = cy + ring_r * float(np.sin(ang))
        pr, pg, pb = POLE_PAINT_RGBA[0], POLE_PAINT_RGBA[1], POLE_PAINT_RGBA[2]
        lines.append(
            f'    <body name="align_pole_{i}" pos="{px:.5f} {py:.5f} {z_center:.5f}">\n'
            f'      <geom type="cylinder" size="{pole_r:.4f} {pole_half_h:.4f}" '
            f'rgba="{pr:.3f} {pg:.3f} {pb:.3f} 1" contype="0" conaffinity="0"/>\n'
            "    </body>\n"
        )
    lines.append("  </worldbody>\n</mujoco>\n")
    wrapper = "".join(lines)
    robot_dir = str(Path(robot_path).parent)
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".xml", prefix="circle_align_", dir=robot_dir)
    try:
        os.close(fd)
        Path(path).write_text(wrapper)
        return mujoco.MjModel.from_xml_path(path)
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def sample_annulus_points_head_camera(
    server: Any,
    *,
    stride: int = 6,
    z_lo: float = 0.18,
    z_hi: float = 0.95,
    annulus_c: np.ndarray,
    annulus_r: float,
    annulus_half_width: float,
    pole_color_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """MuJoCo RGB-D + OpenCV camera→world; returns ``(xyz_world Nx3, rgb_uint8 Nx3)``.

    If ``pole_color_only``, keep only pixels matching calibration pole paint (see
    :func:`mask_calib_pole_rgb`) so the annulus is not diluted by the table / floor.
    """
    spec = server._spec
    cam = spec.camera_names[0]
    with server._mj_lock:
        rgb, depth, K = server._primary_rgb_and_depth(cam)
        pose = server._camera_pose_world(cam)
    if depth is None or K is None or pose is None or rgb is None:
        z = np.zeros((0, 3), dtype=np.float64)
        c = np.zeros((0, 3), dtype=np.uint8)
        return z, c
    cam_model = pinhole_camera_from_intrinsics_and_depth(np.asarray(K, dtype=np.float64), depth)
    xyz_c = cam_model.depth_to_xyz(np.asarray(depth, dtype=np.float32))
    xyz_w = camera_xyz_to_global_xyz(xyz_c, np.asarray(pose, dtype=np.float64))
    sl = slice(None, None, stride)
    pts = xyz_w[sl, sl, :].reshape(-1, 3)
    cols = np.asarray(rgb[sl, sl, :], dtype=np.uint8).reshape(-1, 3)
    mask = np.isfinite(pts).all(axis=1) & (pts[:, 2] > z_lo) & (pts[:, 2] < z_hi)
    xy = pts[:, :2]
    rad = np.linalg.norm(xy - annulus_c.reshape(1, 2), axis=1)
    mask &= (rad > annulus_r - annulus_half_width) & (rad < annulus_r + annulus_half_width)
    pts_m = pts[mask]
    cols_m = cols[mask]
    if pole_color_only and pts_m.shape[0] > 0:
        pc = mask_calib_pole_rgb(cols_m)
        pts_m = pts_m[pc]
        cols_m = cols_m[pc]
    return pts_m, cols_m

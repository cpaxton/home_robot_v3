# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Sensor-depth reprojection; never fill missing samples with scene geometry."""

import numpy as np


def rendered_camera_intrinsics(fovy, width, height):
    f = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
    return np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def register_depth(depth, depth_k, rgb_k, rgb_from_depth, output_shape):
    """Project measured CV-camera depth with a nearest-sample z buffer.

    Holes, outside-image points and points behind the RGB camera remain zero.
    This is spatial registration, not exposure synchronization or hole filling.
    """
    depth = np.asarray(depth, dtype=float)
    kd, kr, transform = (np.asarray(a, dtype=float) for a in (depth_k, rgb_k, rgb_from_depth))
    if (
        depth.ndim != 2
        or kd.shape != (3, 3)
        or kr.shape != (3, 3)
        or transform.shape != (4, 4)
        or not np.isfinite(kd).all()
        or not np.isfinite(kr).all()
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1])
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-6)
        or not (kd[0, 0] > 0 and kd[1, 1] > 0 and kr[0, 0] > 0 and kr[1, 1] > 0)
    ):
        raise ValueError("invalid_depth_registration_calibration")
    h, w = output_shape
    if h <= 0 or w <= 0:
        raise ValueError("invalid_depth_registration_shape")
    y, x = np.where(np.isfinite(depth) & (depth > 0))
    rays = np.stack([x, y, np.ones(len(x))], axis=-1) @ np.linalg.inv(kd).T
    points = (rays * depth[y, x, None]) @ transform[:3, :3].T + transform[:3, 3]
    valid = np.isfinite(points).all(axis=-1) & (points[:, 2] > 0)
    points = points[valid]
    projected = points @ kr.T
    pixels = np.rint(projected[:, :2] / projected[:, 2, None])
    valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    pixels = pixels[valid].astype(int)
    aligned = np.full(h * w, np.inf)
    np.minimum.at(aligned, pixels[:, 1] * w + pixels[:, 0], points[valid, 2])
    aligned[~np.isfinite(aligned)] = 0.0
    return aligned.reshape(h, w).astype(np.float32)

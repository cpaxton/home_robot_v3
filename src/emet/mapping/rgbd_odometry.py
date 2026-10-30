# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Small-step RGB-D odometry with explicit registration failure.

Commands may initialize registration; they never replace its measured result.
Open3D estimates source-camera -> target-camera, so world poses compose with
the inverse. RGB/depth must be registered and depth must be in metres.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OdometryReceipt:
    ok: bool
    status: str
    timestamp_ns: int
    camera_pose: np.ndarray | None
    overlap: float = 0.0
    depth_rmse_m: float | None = None
    color_mae: float | None = None


def registration_quality(source_rgb, source_depth, target_rgb, target_depth, k, transform):
    """Independent sampled reprojection check, including occlusions as misses."""
    v, u = np.mgrid[0 : source_depth.shape[0] : 4, 0 : source_depth.shape[1] : 4]
    z = source_depth[v, u]
    valid = np.isfinite(z) & (z > 0.1) & (z < 6.0)
    u, v, z = u[valid], v[valid], z[valid]
    if len(z) < 64:
        return 0.0, None, None
    xyz = np.column_stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z))
    xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    positive = xyz[:, 2] > 0.1
    projected = xyz @ k.T
    pixels = np.rint(projected[:, :2] / np.maximum(projected[:, 2:3], 1e-9)).astype(int)
    h, w = target_depth.shape
    inside = positive & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    indices = np.flatnonzero(inside)
    x, y = pixels[inside].T
    measured = target_depth[y, x]
    residual = measured - xyz[inside, 2]
    matched = np.isfinite(measured) & (measured > 0.1) & (abs(residual) <= 0.08)
    overlap = float(matched.sum() / len(z))
    if not matched.any():
        return overlap, None, None
    src = source_rgb[v[indices[matched]], u[indices[matched]]].astype(float) / 255
    dst = target_rgb[y[matched], x[matched]].astype(float) / 255
    return overlap, float(np.sqrt(np.mean(residual[matched] ** 2))), float(np.mean(abs(src - dst)))


class RGBDOdometry:
    """Retain only accepted camera poses; a lost track requires explicit reset.

    This is local odometry, without loop closure or global relocalization. A
    successful receipt is a checked estimate, not proof of ground-truth arrival.
    """

    def __init__(self, camera_k, initial_pose, *, min_overlap=0.4, max_color_mae=0.15):
        self.k = np.asarray(camera_k, dtype=float).copy()
        self.pose = np.asarray(initial_pose, dtype=float).copy()
        if (
            self.k.shape != (3, 3)
            or self.pose.shape != (4, 4)
            or not np.isfinite(self.k).all()
            or not np.isfinite(self.pose).all()
            or min(self.k[0, 0], self.k[1, 1]) <= 0
            or not np.allclose(self.pose[3], [0, 0, 0, 1])
            or not np.allclose(self.pose[:3, :3].T @ self.pose[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(self.pose[:3, :3]), 1)
        ):
            raise ValueError("Finite calibrated intrinsics and right-handed camera pose required")
        if not 0 < min_overlap <= 1 or not 0 < max_color_mae <= 1:
            raise ValueError("Registration thresholds must lie in (0, 1]")
        self.min_overlap, self.max_color_mae = min_overlap, max_color_mae
        self.previous = None
        self.timestamp_ns = 0
        self.lost = False

    def update(self, rgb, depth, timestamp_ns, *, source_to_target_guess=None):
        import open3d as o3d

        rgb, depth = np.asarray(rgb), np.asarray(depth)
        if (
            rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
            or depth.shape != rgb.shape[:2]
            or not np.isfinite(depth).all()
            or (depth < 0).any()
            or not isinstance(timestamp_ns, (int, np.integer))
            or timestamp_ns <= self.timestamp_ns
        ):
            raise ValueError("Registered uint8 RGB, finite metric depth, and a fresh timestamp required")
        if self.previous is not None and depth.shape != self.previous[1].shape:
            raise ValueError("Camera dimensions changed within odometry track")
        if self.lost:
            return OdometryReceipt(False, "tracking_lost", timestamp_ns, None)
        if np.count_nonzero((depth > 0.1) & (depth < 6)) < 64:
            self.lost = True
            return OdometryReceipt(False, "insufficient_depth", timestamp_ns, None)

        def image(color, distance):
            return o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color)),
                o3d.geometry.Image(np.ascontiguousarray(distance, dtype=np.float32)),
                depth_scale=1.0,
                depth_trunc=6.0,
                convert_rgb_to_intensity=True,
            )

        initialized = self.previous is None
        target = image(rgb, depth)
        overlap, error, color_error = 1.0, 0.0, 0.0
        if self.previous is not None:
            guess = np.eye(4) if source_to_target_guess is None else np.asarray(source_to_target_guess, dtype=float)
            if guess.shape != (4, 4) or not np.isfinite(guess).all():
                raise ValueError("Finite source-to-target initial transform required")
            h, w = depth.shape
            intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, self.k[0, 0], self.k[1, 1], self.k[0, 2], self.k[1, 2])
            ok, transform, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                self.previous[2],
                target,
                intrinsic,
                guess,
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                o3d.pipelines.odometry.OdometryOption(depth_max=6.0),
            )
            bounded = (
                np.isfinite(transform).all()
                and np.linalg.norm(transform[:3, 3]) <= 0.35
                and np.trace(transform[:3, :3]) >= 1 + 2 * np.cos(np.deg2rad(15))
            )
            if ok and bounded:
                overlap, error, color_error = registration_quality(
                    self.previous[0], self.previous[1], rgb, depth, self.k, transform
                )
            if (
                not ok
                or not bounded
                or overlap < self.min_overlap
                or color_error is None
                or color_error > self.max_color_mae
            ):
                self.lost = True
                return OdometryReceipt(False, "registration_rejected", timestamp_ns, None, overlap, error, color_error)
            self.pose = self.pose @ np.linalg.inv(transform)
        self.previous = (rgb.copy(), depth.copy(), target)
        self.timestamp_ns = int(timestamp_ns)
        return OdometryReceipt(
            True,
            "initialized" if initialized else "tracked",
            int(timestamp_ns),
            self.pose.copy(),
            overlap,
            error,
            color_error,
        )

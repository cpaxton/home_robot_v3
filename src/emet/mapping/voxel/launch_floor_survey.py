# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Register a measured launch-area point cloud to a fresh odometry frame."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LaunchFloorSurvey:
    points: np.ndarray
    observation_scope_id: str
    registered_at: float
    expires_at: float
    max_anchor_residual_m: float

    def points_at(self, scope_id, now):
        if scope_id != self.observation_scope_id or not self.registered_at <= now < self.expires_at:
            return None
        return self.points


def register_launch_floor_survey(
    depth_points, survey_anchors, odom_anchors, *,
    observation_scope_id, survey_captured_at, anchors_captured_at, now,
    start_xy, radius_m, max_age_s, max_anchor_error_m,
):
    """Register identified 3D anchors visible to both depth sensors.

    Inputs are measured in gravity-aligned frames. Fit only XY translation
    and yaw, and check the full 3D anchor residual against the calibration
    bound. Survey capture precedes robot placement. Missing returns stay
    absent, and points outside the local launch envelope are discarded.
    """
    values = [survey_captured_at, anchors_captured_at, now, radius_m, max_age_s, max_anchor_error_m]
    if (not observation_scope_id or not np.isfinite(values).all()
            or not survey_captured_at <= anchors_captured_at <= now < survey_captured_at + max_age_s
            or min(radius_m, max_age_s, max_anchor_error_m) <= 0):
        raise ValueError("launch_survey_timing_or_calibration_invalid")
    points, source, target, center = (
        np.asarray(value, dtype=float) for value in (depth_points, survey_anchors, odom_anchors, start_xy)
    )
    if (source.ndim != 2 or source.shape[1] != 3 or len(source) < 3 or source.shape != target.shape
            or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("launch_survey_anchors_missing")
    if points.ndim != 2 or points.shape[1] != 3 or center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("launch_survey_points_invalid")
    source_center, target_center = source[:, :2].mean(axis=0), target[:, :2].mean(axis=0)
    source_offsets = source[:, :2] - source_center
    target_offsets = target[:, :2] - target_center
    u, values, vh = np.linalg.svd(source_offsets.T @ target_offsets)
    if values[-1] <= max_anchor_error_m**2 or np.linalg.det(u @ vh) < 0:
        raise ValueError("launch_survey_anchor_geometry_invalid")
    rotation = u @ vh
    translation = target_center - source_center @ rotation
    aligned_anchors = np.column_stack((source[:, :2] @ rotation + translation, source[:, 2]))
    residual = float(np.linalg.norm(aligned_anchors - target, axis=1).max())
    if residual > max_anchor_error_m:
        raise ValueError("launch_survey_anchor_misaligned")
    points = points[np.isfinite(points).all(axis=1)]
    xy = points[:, :2] @ rotation + translation
    local = np.linalg.norm(xy - center, axis=1) <= radius_m
    registered = np.column_stack((xy[local], points[local, 2]))
    if not len(registered):
        raise ValueError("launch_survey_empty")
    # An owned, immutable buffer keeps evidence fixed during map checks.
    registered = np.frombuffer(registered.tobytes(), dtype=float).reshape(-1, 3)
    return LaunchFloorSurvey(registered, observation_scope_id, now, survey_captured_at + max_age_s, residual)

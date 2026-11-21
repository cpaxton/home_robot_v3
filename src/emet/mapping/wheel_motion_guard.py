# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Compare wheel odometry with independently measured, synchronized chassis motion."""

from dataclasses import dataclass

import numpy as np

from emet.core.navigation_result import NavigationPoseUncertainty


@dataclass(frozen=True)
class WheelMotionCheck:
    status: str
    position_disagreement_m: float | None = None
    heading_disagreement_rad: float | None = None

    @property
    def ok(self):
        return self.status == "agrees"


class WheelMotionGuard:
    """Compare displacement in each source's own initial base frame.

    A guard belongs to one motion. Once a sample fails, later samples cannot
    automatically reauthorize motion.
    """

    def __init__(self, tolerance: NavigationPoseUncertainty, *, max_age_s: float):
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("positive finite wheel motion sample age required")
        self._tolerance = tolerance
        self._max_age_ns = round(max_age_s * 1e9)
        self._last_ns = 0
        self._origin = None
        self._failure = None

    def update(self, wheel_xyt, reference_xyt, *, wheel_timestamp_ns, reference_timestamp_ns, now_ns):
        if self._failure is not None:
            return self._failure
        try:
            poses = np.asarray([wheel_xyt, reference_xyt], dtype=float)
        except (ValueError, TypeError, OverflowError):
            poses = None
        if poses is None or poses.shape != (2, 3) or not np.isfinite(poses).all():
            self._failure = WheelMotionCheck("independent_pose_unavailable")
            return self._failure
        if (
            not all(type(value) is int for value in (wheel_timestamp_ns, reference_timestamp_ns, now_ns))
            or wheel_timestamp_ns != reference_timestamp_ns or wheel_timestamp_ns <= self._last_ns
            or not 0 <= now_ns - wheel_timestamp_ns <= self._max_age_ns
        ):
            self._failure = WheelMotionCheck("wheel_motion_alignment_unavailable")
            return self._failure
        self._last_ns = wheel_timestamp_ns
        if self._origin is None:
            self._origin = poses.copy()
        deltas = poses - self._origin
        c, s = np.cos(self._origin[:, 2]), np.sin(self._origin[:, 2])
        local_x = c * deltas[:, 0] + s * deltas[:, 1]
        local_y = -s * deltas[:, 0] + c * deltas[:, 1]
        xy = float(np.hypot(local_x[0] - local_x[1], local_y[0] - local_y[1]))
        yaw_delta = deltas[0, 2] - deltas[1, 2]
        yaw = float(abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))))
        if xy > self._tolerance.xy_m or yaw > self._tolerance.yaw_rad:
            self._failure = WheelMotionCheck("wheel_motion_disagreement", xy, yaw)
            return self._failure
        return WheelMotionCheck("agrees", xy, yaw)

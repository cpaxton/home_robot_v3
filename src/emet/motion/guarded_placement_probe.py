# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Local force guard for a single, short vertical placement probe."""

from dataclasses import dataclass

import numpy as np

MAX_TRAVEL_M = 0.010
MAX_SPEED_M_S = 0.005
TIMEOUT_S = 3.0
MAX_SAMPLE_AGE_S = 0.125
MAX_LATERAL_DRIFT_M = 0.001


@dataclass(frozen=True)
class ProbeCalibration:
    calibration_id: str
    axial_sign: int
    contact_n: float
    overload_n: float
    lateral_n: float
    torque_nm: float

    def __post_init__(self):
        values = (self.contact_n, self.overload_n, self.lateral_n, self.torque_nm)
        if (
            not self.calibration_id or self.axial_sign not in (-1, 1)
            or not np.isfinite(values).all() or min(values) <= 0
            or self.overload_n <= self.contact_n
        ):
            raise ValueError("invalid_probe_calibration")


class GuardedPlacementProbe:
    """Filter fresh world-frame wrench samples and hold on every terminal result.

    The robot owns the bounded trajectory and the stop. A result confirms local
    contact only. The placement executor owns the payload and gripper release.
    """

    def __init__(self, calibration, *, start_xyz, baseline, started_at, contact_band_m):
        self.calibration = calibration
        self.start_xyz = np.asarray(start_xyz, dtype=float).copy()
        self.baseline = np.asarray(baseline, dtype=float).copy()
        self.lower, self.upper = map(float, contact_band_m)
        if (
            self.start_xyz.shape != (3,) or self.baseline.shape != (6,)
            or not np.isfinite(np.r_[self.start_xyz, self.baseline, started_at]).all()
            or not 0 <= self.lower <= self.upper <= MAX_TRAVEL_M
        ):
            raise ValueError("invalid_probe_start")
        self.started_at = self.last_sample_at = float(started_at)
        self.last_travel_m = 0.0
        self.last_wrench_delta = np.zeros(6)
        self.contact_samples = 0
        self.status = "descending"
        self.reason = None

    def stop(self, reason):
        if self.status in ("descending", "holding"):
            self.status = "contact" if reason == "support_contact" else "stopped"
            self.reason = reason
        return self.status

    def update(self, *, now, sampled_at, xyz, wrench, retention_ok=True):
        if self.status not in ("descending", "holding"):
            return self.status
        if not np.isfinite(now) or not 0 <= now - self.started_at < TIMEOUT_S:
            return self.stop("probe_timeout")
        if not retention_ok:
            return self.stop("probe_payload_tracking_lost")
        if not np.isfinite(sampled_at) or not 0 <= now - sampled_at <= MAX_SAMPLE_AGE_S:
            return self.stop("probe_feedback_stale")
        if sampled_at < self.last_sample_at:
            return self.stop("probe_feedback_out_of_order")
        if sampled_at == self.last_sample_at:
            return "holding"

        xyz, wrench = np.asarray(xyz, dtype=float), np.asarray(wrench, dtype=float)
        if xyz.shape != (3,) or wrench.shape != (6,) or not np.isfinite(np.r_[xyz, wrench]).all():
            return self.stop("probe_feedback_invalid")
        displacement = xyz - self.start_xyz
        travel = -float(displacement[2])
        dt = sampled_at - self.last_sample_at
        if (
            np.linalg.norm(displacement[:2]) > MAX_LATERAL_DRIFT_M
            or travel < -MAX_LATERAL_DRIFT_M or travel > MAX_TRAVEL_M
            or travel - self.last_travel_m > MAX_SPEED_M_S * dt + 0.00025
        ):
            return self.stop("probe_tracking_deviation")
        self.last_sample_at, self.last_travel_m = sampled_at, travel

        self.last_wrench_delta = wrench - self.baseline
        force, torque = np.split(self.last_wrench_delta, 2)
        cal = self.calibration
        axial = force[2] * cal.axial_sign
        if abs(axial) >= cal.overload_n or np.linalg.norm(force[:2]) >= cal.lateral_n:
            return self.stop("probe_unexpected_force")
        if np.linalg.norm(torque) >= cal.torque_nm or axial <= -cal.contact_n:
            return self.stop("probe_unexpected_force")
        if axial >= cal.contact_n:
            if not self.lower <= travel <= self.upper:
                return self.stop("probe_contact_outside_band")
            self.contact_samples += 1
            if self.contact_samples == 3:
                return self.stop("support_contact")
            self.status = "holding"
            return self.status
        if self.status == "holding":
            return self.stop("probe_contact_unstable")
        if travel >= self.upper:
            return self.stop("probe_no_contact")
        return self.status

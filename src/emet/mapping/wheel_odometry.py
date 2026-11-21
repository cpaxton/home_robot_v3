# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Differential-drive odometry in the first wheel sample's local base frame."""

import numpy as np

from emet.controller.manipulation.encoder_capture import WheelCapture
from emet.controller.manipulation.native_inputs import ObservationUnavailable


class DifferentialDriveOdometry:
    """Integrate unwrapped wheel angles, assuming rolling contact and a flat floor."""

    def __init__(self, *, wheel_radius_m: float, track_width_m: float):
        calibration = np.asarray([wheel_radius_m, track_width_m], dtype=float)
        if not np.isfinite(calibration).all() or np.any(calibration <= 0):
            raise ValueError("wheel radius and track width must be finite and positive")
        self._radius, self._track = calibration
        self._xyt = np.zeros(3)
        self._previous: WheelCapture | None = None

    def update(self, sample: WheelCapture) -> np.ndarray:
        """Return local x, y, yaw at this sample. Rejected updates preserve the estimate."""
        previous = self._previous
        if previous is not None:
            if (sample.capture.calibration_id != previous.capture.calibration_id
                    or sample.capture.earliest <= previous.capture.latest):
                raise ObservationUnavailable("wheel_odometry_capture_out_of_order")
            left = self._radius * (sample.left_rad - previous.left_rad)
            right = self._radius * (sample.right_rad - previous.right_rad)
            distance = (left + right) / 2
            yaw_change = (right - left) / self._track
            # Integrate the arc at constant curvature between encoder samples.
            midpoint = self._xyt[2] + yaw_change / 2
            chord = distance * np.sinc(yaw_change / (2 * np.pi))
            pose = self._xyt + [chord * np.cos(midpoint), chord * np.sin(midpoint), yaw_change]
            if not np.isfinite(pose).all():
                raise ObservationUnavailable("invalid_wheel_feedback")
            self._xyt = pose
        self._previous = sample
        return self._xyt.copy()

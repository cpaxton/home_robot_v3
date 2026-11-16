"""Latched divergence detection for stationary, base-only heading contracts.

This monitor never supplies a corrective action. It requires both wrong-way
commands and measured wrong-way motion; braking, moving references, translation
and hand-constrained whole-body maneuvers are outside its applicability.
"""

import numpy as np


def wrapped_yaw(value):
    return float(np.arctan2(np.sin(value), np.cos(value)))


class TurnProgressGuard:
    def __init__(self, interval=0.05):
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("positive monitor cadence required")
        self.interval = interval
        self.reset()

    def reset(self):
        """Only a trusted controller/episode reset may clear a latched fault."""
        self.fault = None
        self.evidence = None
        self.previous = None
        self.bad_since = None
        self.bad_yaw = None
        self.bad_target = None

    def update(self, *, time, yaw, target, command, eligible, tolerance):
        if self.fault:
            return self.fault
        if not np.isfinite([time, yaw, target, command, tolerance]).all() or tolerance <= 0:
            self.fault = "base_turn_invalid_feedback"
            return self.fault
        current = (float(time), float(yaw), float(target))
        previous = self.previous
        if previous is not None and time == previous[0]:
            return None  # apply/observe can read the same physics instant twice.
        if previous is None:
            self.previous = current
            return None
        dt = time - previous[0]
        if dt < 0:
            self.fault = "base_turn_clock_reversed"
            return self.fault
        if dt < self.interval - 1e-8:
            return None
        self.previous = current
        error = wrapped_yaw(target - yaw)
        direction = np.sign(error)
        measured = wrapped_yaw(yaw - previous[1]) / dt
        stationary = abs(wrapped_yaw(target - previous[2])) <= 1e-6
        wrong_way = (
            eligible
            and stationary
            and 2 * tolerance < abs(error) < np.pi - 0.05
            and direction * command < -0.0005
            and direction * measured < -0.0005
        )
        if not wrong_way:
            self.bad_since = self.bad_yaw = self.bad_target = None
            return None
        if self.bad_since is None:
            self.bad_since, self.bad_yaw = previous[:2]
            self.bad_target = target
        if abs(wrapped_yaw(target - self.bad_target)) > 1e-5:
            self.bad_since = self.bad_yaw = self.bad_target = None
            return None
        duration = time - self.bad_since
        away = -direction * wrapped_yaw(yaw - self.bad_yaw)
        if duration >= 0.5 and away >= 0.0025:
            self.fault = "base_turn_divergence"
            self.evidence = {
                "duration_s": float(duration),
                "away_rad": float(away),
                "error_rad": error,
                "command_rad_s": float(command),
                "measured_rad_s": float(measured),
            }
        return self.fault

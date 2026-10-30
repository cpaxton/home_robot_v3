"""Experimental final motor-command smoothing, independent of policy or task."""

import numpy as np


def smooth_velocity(velocity, acceleration, target, interval, speed_limits, acceleration_limits, jerk_limits):
    """Critically damp velocity requests with bounded acceleration and jerk.

    A 6 rad/s second-order filter damps rapidly alternating feedback corrections.
    Hard speed limits take priority if a saturated transient reaches a boundary;
    emergency stops bypass the filter. Limits are simulator experiment settings,
    not robot manufacturer jerk ratings. Zero-time observations cannot advance it.
    """
    if interval <= 0:
        return velocity.copy(), acceleration.copy()
    jerk = np.clip(36 * (target - velocity) - 12 * acceleration, -jerk_limits, jerk_limits)
    next_acceleration = np.clip(acceleration + interval * jerk, -acceleration_limits, acceleration_limits)
    next_velocity = np.clip(velocity + interval * next_acceleration, -speed_limits, speed_limits)
    # Keep the state consistent if a hard velocity boundary wins over smoothing.
    return next_velocity, (next_velocity - velocity) / interval

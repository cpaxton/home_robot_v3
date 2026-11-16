"""Exact one-axis, jerk-limited braking of an announced drive reference."""

import numpy as np

from emet.rl.manipulation.dynamic_trajectory import ArmTrajectory, Segment
from emet.rl.manipulation.jerk_reference import polynomial_peaks


def braking_path(velocity, acceleration, speed_limit, acceleration_limit, jerk_limit):
    """Stop without reversing, matching initial velocity and acceleration.

    This is a kinematic reference primitive, not a physical emergency stop.
    Impossible boundaries are rejected. In particular, outward acceleration at
    the speed limit, or too much deceleration to stop without reversing, cannot
    be repaired by declaring the initial state at rest.
    """
    if not np.isfinite([velocity, acceleration, speed_limit, acceleration_limit, jerk_limit]).all() or min(
        speed_limit, acceleration_limit, jerk_limit
    ) <= 0:
        raise ValueError("finite braking boundaries and positive derivative limits required")
    if abs(velocity) > speed_limit + 1e-10 or abs(acceleration) > acceleration_limit + 1e-10:
        raise ValueError("braking boundary exceeds native reference limits")
    if abs(velocity) < 1e-12 and abs(acceleration) < 1e-12:
        return None, None
    sign = np.sign(velocity)
    if sign == 0:
        raise ValueError("cannot monotonically brake zero velocity with nonzero acceleration")
    v, a, j = abs(velocity), sign * acceleration, jerk_limit
    peak = min(acceleration_limit, float(np.sqrt(j * v + a * a / 2)))
    first = (a + peak) / j
    plateau = (v + (a * a - 2 * peak * peak) / (2 * j)) / peak
    if first < -1e-10 or plateau < -1e-10:
        raise ValueError("braking boundary requires velocity reversal")
    q, elapsed, parts = 0.0, 0.0, []
    for duration, jerk in ((max(0.0, first), -j), (max(0.0, plateau), 0.0), (peak / j, j)):
        if duration < 1e-12:
            continue
        coefficients = np.zeros((6, 1))
        coefficients[:4, 0] = sign * np.array([q, v, a / 2, jerk / 6])
        part = Segment(elapsed, duration, coefficients)
        peaks = polynomial_peaks(part)
        if any(float(x[0]) > limit + 1e-9 for x, limit in zip(
            peaks, (speed_limit, acceleration_limit, jerk_limit), strict=True
        )):
            raise ValueError("cannot brake inside native reference derivative limits")
        position, vel, acc, _ = part.sample(elapsed + duration)
        q, v, a = (sign * float(x[0]) for x in (position, vel, acc))
        parts.append(part)
        elapsed += duration
    if abs(v) > 1e-9 or abs(a) > 1e-9:
        raise ValueError("braking endpoint did not reach rest")
    path = ArmTrajectory(parts)
    return path, {
        "format": "announced_axis_braking_v1",
        "initial_velocity": float(velocity), "initial_acceleration": float(acceleration),
        "speed": speed_limit, "acceleration": acceleration_limit, "jerk": jerk_limit,
        "duration_s": path.end,
        "jerk_limit_source": "experimental_reference_limit_not_a_hardware_rating",
        "dynamic_feasibility_verified": False,
    }

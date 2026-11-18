"""Conservative cubic-geometry timing without an artificial unit-phase floor."""

import numpy as np

from emet.rl.manipulation.jerk_reference import compile_joint_path, polynomial_peaks


def retime_derivative_bounds(
    first, second, third, speed, acceleration, jerk, *, minimum_duration=0.05, mode="bounded_v2"
):
    """Choose a bounded duration directly, allowing references shorter than 3.9s.

    A unit-time rest-to-rest quintic has phase peaks 1.875, <=5.774, and 60.
    Bound each chain-rule derivative, then scale by T, T², T³ respectively.
    These remain conservative independent maxima, not dynamic certificates.
    """
    arrays = [np.asarray(v, float) for v in (first, second, third, speed, acceleration, jerk)]
    first, second, third, speed, acceleration, jerk = arrays
    if (
        first.ndim != 1
        or not len(first)
        or any(a.shape != first.shape or not np.isfinite(a).all() for a in arrays)
        or any(np.any(a < 0) for a in arrays[:3])
        or any(np.any(a <= 0) for a in arrays[3:])
        or isinstance(minimum_duration, (bool, np.bool_))
        or not np.isfinite(minimum_duration)
        or minimum_duration < 0.05
        or mode not in {"legacy_v1", "bounded_v2"}
    ):
        raise ValueError("finite matching nonnegative derivatives and positive physical limits required")
    if mode == "legacy_v1":
        scale = 1.0
        for _ in range(20):
            path, _ = compile_joint_path([[0.0], [1.0]], [1 / scale], [1 / scale**2], [1 / scale**3])
            v, a, j = (float(x[0]) for x in polynomial_peaks(path.segments[0]))
            bounds = np.array([first * v, second * v * v + first * a, third * v**3 + 3 * second * v * a + first * j])
            ratios = np.max(bounds / np.array([speed, acceleration, jerk]), axis=1)
            factor = max(ratios[0], np.sqrt(ratios[1]), np.cbrt(ratios[2]))
            if factor <= 1 + 1e-9:
                return path, bounds
            scale *= factor * 1.01
        raise ValueError("reference timing did not converge")
    v, a, j = 1.875, 5.774, 60.0
    unit = np.array([first * v, second * v * v + first * a, third * v**3 + 3 * second * v * a + first * j])
    duration = (
        max(
            minimum_duration,
            float(np.max(unit[0] / speed)),
            float(np.sqrt(np.max(unit[1] / acceleration))),
            float(np.cbrt(np.max(unit[2] / jerk))),
        )
        * 1.001
    )
    path, _ = compile_joint_path([[0.0], [1.0]], [v / duration], [a / duration**2], [j / duration**3])
    bounds = unit / np.array([path.end, path.end**2, path.end**3])[:, None]
    return path, bounds

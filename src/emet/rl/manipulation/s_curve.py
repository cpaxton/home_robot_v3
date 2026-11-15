"""Path-preserving seven-phase jerk-limited trapezoidal velocity references.

Short moves omit cruise/constant-acceleration phases. True polyline corners
retain rest-to-rest stops. Certificates include the existing phase governor.
"""

import numpy as np

from emet.rl.manipulation.dynamic_trajectory import ArmTrajectory, Segment


def _phases(v, a, j):
    tj = min(a / j, np.sqrt(v / j))
    ta = max(0.0, v / (j * tj) - tj)
    peak = j * tj * (tj + ta)
    distance = peak * (2 * tj + ta)
    if distance <= 1:
        cruise = (1 - distance) / peak
    else:
        tj = a / j
        ta = (-3 * tj + np.sqrt(tj * tj + 4 / a)) / 2
        if ta < 0:
            tj, ta = np.cbrt(1 / (2 * j)), 0.0
        cruise = 0.0
    return [(tj, j), (ta, 0), (tj, -j), (cruise, 0), (tj, -j), (ta, 0), (tj, j)]


def compile_s_curve(path, speed, acceleration, jerk, *, phase_acceleration=0, phase_jerk=0):
    from emet.rl.manipulation.jerk_reference import polynomial_peaks, simplify_collinear

    points = simplify_collinear(path)
    width = points.shape[1]
    limits = tuple(np.asarray(x, dtype=float) for x in (speed, acceleration, jerk))
    if (
        not width
        or any(x.shape != (width,) or not np.isfinite(x).all() or np.any(x <= 0) for x in limits)
        or not np.isfinite([phase_acceleration, phase_jerk]).all()
        or min(phase_acceleration, phase_jerk) < 0
    ):
        raise ValueError("positive derivative limits and finite phase bounds required")
    if len(points) == 1:
        points = np.repeat(points, 2, axis=0)
    parts, records, elapsed = [], [], 0.0
    for start, end in zip(points[:-1], points[1:], strict=True):
        delta = end - start
        active = abs(delta) > 1e-14
        if not active.any():
            c = np.zeros((6, width))
            c[0] = start
            parts.append(Segment(elapsed, 0.05, c))
            elapsed += 0.05
            continue
        v, a, j = (float(np.min(x[active] / abs(delta[active]))) for x in limits)
        phases = _phases(v, a, j)
        scale = 1.0
        for _ in range(200):
            trial, q, velocity, accel, time = [], 0.0, 0.0, 0.0, elapsed
            for duration, force in phases:
                h, jj = duration * scale, force / scale**3
                if h <= 1e-14:
                    continue
                c = np.zeros((6, width))
                c[:4] = [start + delta * q, delta * velocity, delta * accel / 2, delta * jj / 6]
                trial.append(Segment(time, h, c))
                q += velocity * h + accel * h**2 / 2 + jj * h**3 / 6
                velocity += accel * h + jj * h**2 / 2
                accel += jj * h
                time += h
            peaks = tuple(np.max([polynomial_peaks(p)[i] for p in trial], axis=0) for i in range(3))
            vv, aa, jj = peaks
            bounds = (vv, aa + vv * phase_acceleration, jj + 3 * aa * phase_acceleration + vv * phase_jerk)
            if all(np.all(x <= y * (1 + 1e-9)) for x, y in zip(bounds, limits, strict=True)):
                break
            scale *= 1.1
        else:
            raise ValueError("cannot meet governed S-curve derivative bounds")
        if not np.allclose(trial[-1].sample(time)[0], end, atol=1e-8, rtol=0):
            raise ValueError("S-curve integration missed its endpoint")
        parts.extend(trial)
        elapsed = time
        records.append(
            {
                "nominal_peaks": [x.tolist() for x in peaks],
                "governed_bounds": [x.tolist() for x in bounds],
                "phase_durations": [x[0] * scale for x in phases],
            }
        )
    return ArmTrajectory(parts), {
        "input_vertices": len(path),
        "retained_vertices": len(points),
        "geometry": "original_polyline_without_redundant_collinear_vertices",
        "interpolation": "jerk_limited_s_curve_trapezoidal_velocity_c2",
        "speed": limits[0].tolist(),
        "acceleration": limits[1].tolist(),
        "jerk": limits[2].tolist(),
        "jerk_limit_source": "explicit_experimental_reference_limit_not_a_hardware_rating",
        "phase_rate_bound": 1,
        "phase_acceleration_bound": phase_acceleration,
        "phase_jerk_bound": phase_jerk,
        "segments": records,
        "duration_s": elapsed,
        "dynamic_feasibility_verified": False,
    }

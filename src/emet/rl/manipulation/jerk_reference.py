"""Path-preserving nominal references with explicit derivative bounds.

These are kinematic reference limits, not contact/dynamic feasibility proofs.
No Ruckig installation or online IK is required. Only redundant collinear
planner vertices are removed; real corners retain a zero-speed/acceleration stop.
"""

from dataclasses import dataclass

import numpy as np

from emet.rl.manipulation.dynamic_trajectory import ArmTrajectory, Segment


def polynomial_peaks(part):
    """Exact extrema of each polynomial derivative over the closed interval."""
    from numpy.polynomial.polynomial import polyder, polyroots, polyval

    result = []
    for order in (1, 2, 3):
        peaks = []
        for coefficients in part.coefficients.T:
            derivative = polyder(coefficients, order)
            roots = polyroots(polyder(derivative))
            points = [0.0, part.duration]
            points += [float(r.real) for r in roots if abs(r.imag) < 1e-8 and 0 < r.real < part.duration]
            peaks.append(float(np.max(abs(polyval(points, derivative)))))
        result.append(np.array(peaks))
    return tuple(result)


def simplify_collinear(path, tolerance=1e-10):
    """Preserve the original polyline, including reversals and true corners."""
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("finite nonempty joint path required")
    kept = [points[0].copy()]
    for point in points[1:]:
        if np.max(abs(point - kept[-1])) <= tolerance:
            continue
        kept.append(point.copy())
        while len(kept) >= 3:
            a, b, c = kept[-3:]
            span = c - a
            norm = float(span @ span)
            if norm <= tolerance**2:
                break
            fraction = float((b - a) @ span / norm)
            if not 0 < fraction < 1 or np.max(abs(b - (a + fraction * span))) > tolerance:
                break
            kept.pop(-2)
    return np.array(kept)


def compile_joint_path(path, speed, acceleration, jerk, *, phase_acceleration=0, phase_jerk=0):
    """Rest-to-rest quintics, bounded also under the declared phase governor.

    q_dot=q_s*r; q_ddot=q_ss*r²+q_s*r_dot;
    q_jerk=q_sss*r³+3*q_ss*r*r_dot+q_s*r_ddot, for 0<=r<=1.
    Duration grows until conservative bounds on all three derivatives pass.
    """
    points = simplify_collinear(path)
    width = points.shape[1]
    speed, acceleration, jerk = (np.asarray(v, dtype=float) for v in (speed, acceleration, jerk))
    if (
        width == 0
        or any(v.shape != (width,) or not np.isfinite(v).all() or np.any(v <= 0) for v in (speed, acceleration, jerk))
        or not np.isfinite([phase_acceleration, phase_jerk]).all()
        or min(phase_acceleration, phase_jerk) < 0
    ):
        raise ValueError("positive derivative limits and finite phase bounds required")
    if len(points) == 1:
        points = np.repeat(points, 2, axis=0)
    parts, elapsed, certificates = [], 0.0, []
    for start, end in zip(points[:-1], points[1:], strict=True):
        delta = end - start
        duration = max(
            0.05,
            float(np.max(1.875 * abs(delta) / speed)),
            float(np.max(np.sqrt(5.774 * abs(delta) / acceleration))),
            float(np.max(np.cbrt(60 * abs(delta) / jerk))),
        )
        for _ in range(200):
            coefficients = np.zeros((6, width))
            coefficients[0] = start
            coefficients[3:] = [10 * delta / duration**3, -15 * delta / duration**4, 6 * delta / duration**5]
            part = Segment(elapsed, duration, coefficients)
            v, a, j = polynomial_peaks(part)
            bounds = (v, a + v * phase_acceleration, j + 3 * a * phase_acceleration + v * phase_jerk)
            if all(
                np.all(actual <= limit * (1 + 1e-10))
                for actual, limit in zip(bounds, (speed, acceleration, jerk), strict=True)
            ):
                break
            duration *= 1.1
        else:
            raise ValueError("cannot meet reference derivative limits")
        parts.append(part)
        certificates.append(
            {"nominal_peaks": [x.tolist() for x in (v, a, j)], "governed_bounds": [x.tolist() for x in bounds]}
        )
        elapsed += duration
    return ArmTrajectory(parts), {
        "input_vertices": len(path),
        "retained_vertices": len(points),
        "geometry": "original_polyline_without_redundant_collinear_vertices",
        "interpolation": "rest_to_rest_quintic_c2",
        "speed": speed.tolist(),
        "acceleration": acceleration.tolist(),
        "jerk": jerk.tolist(),
        "jerk_limit_source": "explicit_experimental_reference_limit_not_a_hardware_rating",
        "phase_rate_bound": 1,
        "phase_acceleration_bound": phase_acceleration,
        "phase_jerk_bound": phase_jerk,
        "segments": certificates,
        "duration_s": elapsed,
        "dynamic_feasibility_verified": False,
    }


@dataclass
class SmoothPhase:
    """Exact two-pole phase-rate filter; rate is C1 despite request jumps.

    Both filter states stay in [0,1]. Phase acceleration is bounded by 1/tau,
    jerk by 2/tau². Integrating the analytic solution avoids Euler overshoot.
    Predictions hold the last requested rate and never predict future feedback.
    """

    end: float
    tau: float = 0.5
    progress: float = 0.0
    first: float = 0.0
    rate: float = 0.0
    request: float = 0.0
    previous_progress: float = 0.0
    last_interval: float = 0.0

    def __post_init__(self):
        if not np.isfinite([self.end, self.tau]).all() or min(self.end, self.tau) <= 0:
            raise ValueError("positive finite phase duration and filter time required")
        if any(v != 0 for v in (self.progress, self.first, self.rate, self.request)):
            raise ValueError("phase must initialize from rest")

    @property
    def acceleration(self):
        return (self.first - self.rate) / self.tau

    @property
    def jerk(self):
        return (self.request - 2 * self.first + self.rate) / self.tau**2

    def predict(self, interval):
        if not np.isfinite(interval) or interval < 0:
            raise ValueError("nonnegative finite prediction interval required")
        if interval == 0:
            return self.progress, self.first, self.rate
        h, u, tau = float(interval), self.request, self.tau
        decay = np.exp(-h / tau)
        first = u + (self.first - u) * decay
        rate = u + ((self.rate - u) + (self.first - u) * h / tau) * decay
        distance = u * h + (self.rate - u) * tau * (-np.expm1(-h / tau))
        distance += (self.first - u) * tau * (1 - (1 + h / tau) * decay)
        return min(self.end, self.progress + max(0.0, distance)), first, rate

    def advance(self, request, interval):
        if not np.isfinite(interval) or interval < 0:
            raise ValueError("nonnegative finite phase interval required")
        if not np.isfinite(request) or not 0 <= request <= 1:
            raise ValueError("phase rate request must be in [0,1]")
        self.request = float(request)
        self.previous_progress, self.last_interval = self.progress, float(interval)
        self.progress, self.first, self.rate = self.predict(interval)
        return self.progress

    def preview_progress(self, offset):
        if not np.isfinite(offset):
            raise ValueError("finite preview offset required")
        if offset >= 0:
            return self.predict(offset)[0]
        return float(np.interp(offset, [-max(self.last_interval, 1e-12), 0], [self.previous_progress, self.progress]))

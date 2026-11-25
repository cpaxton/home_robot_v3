"""Detect sustained elbow tracking loss during an inward startup stroke."""

import math

ELBOW_LOAD_SOFT_PERCENT = 35.0
ELBOW_LOAD_STOP_PERCENT = 40.0


class ElbowProgressGuard:
    """Require fresh inward movement while a streamed target leads the elbow."""

    def __init__(self, *, window_s=1.0, demand_rad=math.radians(0.5), encoder_step_rad=math.tau / 4096):
        self.window_s = window_s
        self.demand_rad = demand_rad
        self.min_progress_rad = 2 * encoder_step_rad
        self.started_at = None
        self.best_rad = None

    def check(self, *, now, measured_rad, target_rad):
        if not all(map(math.isfinite, (now, measured_rad, target_rad))):
            raise RuntimeError("invalid elbow progress feedback")
        if measured_rad - target_rad <= self.demand_rad:
            self.started_at = None
            self.best_rad = None
            return
        if self.best_rad is None or self.best_rad - measured_rad >= self.min_progress_rad - 1e-12:
            self.best_rad = measured_rad
            self.started_at = now
        elif now - self.started_at >= self.window_s:
            raise RuntimeError("elbow stopped progressing toward the commanded target")


class ElbowLoadGuard:
    """Stop on sustained elevated load or an immediate upper ceiling.

    XL430 load is a proxy. The thresholds are in the published percent units.
    """

    def __init__(self):
        self.elevated_since = None

    def check(self, *, now, load_percent):
        if not math.isfinite(now) or not math.isfinite(load_percent):
            raise RuntimeError("invalid elbow load feedback")
        load = abs(load_percent)
        if load >= ELBOW_LOAD_STOP_PERCENT:
            raise RuntimeError("elbow load reached the immediate ceiling")
        if load <= ELBOW_LOAD_SOFT_PERCENT:
            self.elevated_since = None
            return
        if self.elevated_since is None:
            self.elevated_since = now
        elif now - self.elevated_since >= 0.25:
            raise RuntimeError("elbow load stayed above the sustained ceiling")


def pace_elbow_target(*, measured_rad, requested_rad, load_percent):
    """Limit inward position lead as load rises during the startup stroke.

    Preserve the requested direction. Above 35%, reduce lead continuously
    from half a degree to zero at the immediate stop threshold.
    """
    if not all(map(math.isfinite, (measured_rad, requested_rad, load_percent))):
        raise RuntimeError("invalid elbow target pacing feedback")
    load = abs(load_percent)
    if requested_rad >= measured_rad or load <= ELBOW_LOAD_SOFT_PERCENT:
        return requested_rad
    scale = elbow_load_speed_scale(load)
    return max(requested_rad, measured_rad - math.radians(0.5) * scale)


def elbow_load_speed_scale(load_percent):
    """Return the remaining elbow velocity authority under the load proxy."""
    if not math.isfinite(load_percent):
        raise RuntimeError("invalid elbow load feedback")
    band = ELBOW_LOAD_STOP_PERCENT - ELBOW_LOAD_SOFT_PERCENT
    return min(1.0, max(0.0, (ELBOW_LOAD_STOP_PERCENT - abs(load_percent)) / band))


def check_reach_envelope(
    *,
    measured_rad,
    start_rad,
    goal_rad,
    hand_error_m,
    previous_hand_error_m,
    clearance_verified,
    nominal_slack_rad=0.009,
    maximum_slack_rad=math.radians(2),
    hand_slack_m=0.003,
):
    """Allow modest tracking error inside a separately checked clearance envelope."""
    measured, start, goal = tuple(measured_rad), tuple(start_rad), tuple(goal_rad)
    if len(measured) != 6 or len(start) != 6 or len(goal) != 6:
        raise ValueError("expected six measured and commanded Mars joint angles")
    if not all(map(math.isfinite, (*measured, *start, *goal, hand_error_m, previous_hand_error_m))):
        raise RuntimeError("invalid reach feedback")
    excess = max(max(min(a, b) - q, q - max(a, b), 0) for q, a, b in zip(measured, start, goal, strict=True))
    if excess <= nominal_slack_rad:
        return
    if (
        excess > maximum_slack_rad
        or clearance_verified is not True
        or hand_error_m > previous_hand_error_m + hand_slack_m
    ):
        raise RuntimeError("arm left the checked reach envelope")

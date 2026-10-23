"""Fail-closed measured-pose boundary for a supervised whole-body experiment.

The caller must check fresh odometry before EVERY command and independently
enforce a robot-side command watchdog. This guard is not a force limiter.
"""

import math
import sys
from collections.abc import Callable, Sequence


class MotionBoundary:
    """Latch violations relative to the initial pose; never recenter on updates."""

    def __init__(
        self,
        start: Sequence[float],
        *,
        translation_limit_m: float,
        translation_tolerance_m: float = 0.02,
        yaw_limit_rad: float = math.pi / 4,
        yaw_tolerance_rad: float = math.radians(2),
        max_state_age_s: float = 0.2,
    ):
        self.start = self._pose(start)
        values = (translation_limit_m, translation_tolerance_m, yaw_limit_rad,
                  yaw_tolerance_rad, max_state_age_s)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("boundary limits must be finite and nonnegative")
        if yaw_limit_rad + yaw_tolerance_rad >= math.pi or max_state_age_s <= 0:
            raise ValueError("invalid yaw boundary or state age")
        self.translation_limit_m = translation_limit_m
        self.translation_tolerance_m = translation_tolerance_m
        self.yaw_limit_rad = yaw_limit_rad
        self.yaw_tolerance_rad = yaw_tolerance_rad
        self.max_state_age_s = max_state_age_s
        self.fault: str | None = None

    @staticmethod
    def _pose(pose):
        if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
            raise ValueError("base pose must contain finite x, y, yaw")
        return tuple(float(v) for v in pose)

    def check(self, pose: Sequence[float], *, state_age_s: float) -> str | None:
        if self.fault is not None:
            return self.fault
        try:
            x, y, yaw = self._pose(pose)
            if not math.isfinite(state_age_s) or not 0 <= state_age_s <= self.max_state_age_s:
                raise ValueError("missing or stale base odometry")
            distance = math.hypot(x - self.start[0], y - self.start[1])
            angle = abs(math.remainder(yaw - self.start[2], 2 * math.pi))
            if angle > self.yaw_limit_rad + self.yaw_tolerance_rad:
                self.fault = f"base yaw boundary exceeded: {math.degrees(angle):.2f} deg"
            elif distance > self.translation_limit_m + self.translation_tolerance_m:
                self.fault = f"base translation boundary exceeded: {distance:.3f} m"
        except (TypeError, ValueError, OverflowError) as exc:
            self.fault = f"invalid base feedback: {exc}"
        return self.fault

    def enforce(self, pose, *, state_age_s: float, stop: Callable[[], bool]) -> None:
        """Request whole-body stop, report confirmation, then exit nonzero.

        ``stop`` must stop wheels AND arm and confirm measured stopping. It must
        have its own timeout. A fault cannot be cleared by another pose update.
        """
        reason = self.check(pose, state_age_s=state_age_s)
        if reason is None:
            return
        confirmed = False
        try:
            confirmed = stop() is True
        except Exception as exc:
            print(f"ERROR: whole-body stop failed: {exc}", file=sys.stderr, flush=True)
        finally:
            print(f"ERROR: {reason}; stop_confirmed={confirmed}", file=sys.stderr, flush=True)
        raise SystemExit(2)

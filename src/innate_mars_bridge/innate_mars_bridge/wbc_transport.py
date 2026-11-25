"""Robot-side command ownership, expiry and measured-stop supervision for Mars."""

import math
import threading
import time

import numpy as np


class MarsWatchdogTransport:
    """Supervise a single arm-position and wheel-twist command stream."""

    def __init__(self, *, feedback, owns_topics, publish_arm, publish_base, clock=time.monotonic):
        self._read = feedback
        self._owns = owns_topics
        self._arm = publish_arm
        self._base = publish_base
        self._clock = clock
        self._lock = threading.Lock()
        self._deadline = None
        self._stop_position = None
        self._stop_started = None
        self._stop_quiet_since = None
        self._stop_confirmed = False
        self._fault = None
        self._last_tick = -math.inf

    def capabilities(self):
        with self._lock:
            sample = self._read()
            ready = bool(
                self._fault is None and self._clock() - self._last_tick <= 0.05 and self._owns() and self._fresh(sample)
            )
            return {
                "watchdog_arm_position_and_base_twist": ready,
                "robot_watchdog_s": 0.2,
                "measured_whole_body_stop": ready,
                "feedback_si_units_verified": sample.get("feedback_si_units_verified") is True,
                "motor_read_validity_verified": sample.get("motor_read_validity_verified") is True,
                "elbow_load_percent_verified": sample.get("elbow_load_percent_verified") is True,
            }

    def _can_rearm(self):
        return (
            self._stop_confirmed
            and self._clock() - self._last_tick <= 0.05
            and self._owns()
            and self._fresh(self._read())
        )

    def can_rearm(self):
        with self._lock:
            return self._can_rearm()

    def rearm(self):
        """Explicit new stream after a measured stop, with current ownership."""
        with self._lock:
            if not self._can_rearm():
                raise RuntimeError("Mars transport requires a fresh confirmed stop before rearming")
            self._fault = None
            self._stop_started = None
            self._stop_position = None
            self._stop_quiet_since = None
            self._stop_confirmed = False

    def feedback(self):
        return self._read()

    def _fresh(self, sample):
        now = self._clock()
        for key, size in (("joint_position", 5), ("joint_velocity", 5), ("base_twist", 6)):
            vector = np.asarray(sample.get(key), dtype=float)
            if vector.shape != (size,) or not np.isfinite(vector).all():
                return False
        grip = sample.get("gripper_target_rad")
        if not isinstance(grip, (int, float)) or not math.isfinite(grip):
            return False
        return (
            sample.get("valid") is True
            and sample.get("arm_healthy") is True
            and sample.get("feedback_si_units_verified") is True
            and sample.get("motor_read_validity_verified") is True
            and all(
                0 <= now - sample.get(key, -math.inf) <= 0.2
                for key in ("arm_captured_monotonic", "odom_captured_monotonic")
            )
        )

    def send_targets(self, *, arm_position_rad, base_twist_si, valid_for_s):
        arm = np.asarray(arm_position_rad, dtype=float)
        base = np.asarray(base_twist_si, dtype=float)
        if arm.shape != (6,) or base.shape != (2,) or not np.isfinite(np.r_[arm, base]).all():
            raise ValueError("complete six-axis arm and two-axis base targets required")
        if not math.isfinite(valid_for_s) or not 0 < valid_for_s <= 0.2:
            raise ValueError("Mars packet lifetime must be in (0, 0.2] seconds")
        with self._lock:
            if self._fault is not None or self._stop_started is not None:
                raise RuntimeError("Mars transport requires a new ownership session")
            if self._deadline is not None and self._clock() >= self._deadline:
                self._begin_stop("command watchdog expired")
                raise RuntimeError(self._fault)
            if self._clock() - self._last_tick > 0.05 or not self._owns() or not self._fresh(self._read()):
                self._begin_stop("Mars watchdog, command ownership, or verified feedback lost")
                raise RuntimeError(self._fault)
            self._deadline = self._clock() + valid_for_s
            try:
                self._arm(arm)
                self._base(base)
            except Exception:
                self._begin_stop("Mars target publication failed")
                raise

    def _begin_stop(self, reason):
        self._fault = self._fault or reason
        self._deadline = None
        self._stop_started = self._clock()
        self._stop_quiet_since = None
        self._stop_position = None
        self._base(np.zeros(2))
        sample = self._read()
        if self._owns() and self._fresh(sample):
            arm = np.asarray(sample["joint_position"], dtype=float)
            grip = sample["gripper_target_rad"]
            if arm.shape == (5,) and np.isfinite(np.r_[arm, grip]).all():
                self._stop_position = arm.copy()
                self._arm(np.r_[arm, grip])

    def tick(self):
        """Run from a robot-local timer, independently of the client stream."""
        with self._lock:
            now = self._clock()
            self._last_tick = now
            if self._deadline is not None and (
                now >= self._deadline or not self._owns() or not self._fresh(self._read())
            ):
                self._begin_stop("Mars watchdog stopped the command stream")
            if self._stop_started is None or self._stop_confirmed or now - self._stop_started >= 0.75:
                return
            self._base(np.zeros(2))
            sample = self._read()
            if (
                self._stop_position is None
                or not self._owns()
                or not self._fresh(sample)
                or min(sample["arm_captured_monotonic"], sample["odom_captured_monotonic"]) <= self._stop_started
                or np.max(abs(np.asarray(sample["joint_position"]) - self._stop_position)) > 0.02
                or np.max(abs(np.asarray(sample["joint_velocity"]))) >= 0.03
                or np.linalg.norm(sample["base_twist"]) >= 0.03
            ):
                self._stop_quiet_since = None
                return
            if self._stop_quiet_since is None:
                self._stop_quiet_since = now
            self._stop_confirmed = now - self._stop_quiet_since >= 0.15

    def stop_and_confirm(self, *, timeout_s=0.75):
        with self._lock:
            if self._stop_started is None:
                self._begin_stop("session stopped")
        until = self._clock() + timeout_s
        while self._clock() < until:
            self.tick()
            with self._lock:
                if self._stop_confirmed:
                    return True
            time.sleep(0.01)
        return False

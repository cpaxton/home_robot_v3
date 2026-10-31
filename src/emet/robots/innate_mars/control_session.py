"""Opt-in boundary for a future verified Mars whole-body velocity driver.

No ROS/network code or automatic hardware activation. The current bridge does
not satisfy this interface. Driver callbacks must implement a robot-side
watchdog and bounded measured whole-body stop; loopback tests are not hardware
verification. This free-space session does not authorize contact or grip release.
"""

import math
import time

import numpy as np

from emet.robots.innate_mars.motion_boundary import MotionBoundary


class GuardedControlSession:
    def __init__(self, *, driver, profile, enabled=False):
        self.driver, self.profile = driver, profile
        self.enabled = enabled
        self.guard = None
        self.fault = None
        self.last_velocity = np.zeros(2 + profile.arm_size)
        self.last_command_time = None

    def start(self):
        if self.guard is not None or self.fault is not None:
            raise RuntimeError("session cannot be restarted or recentered")
        if not self.enabled:
            raise RuntimeError("hardware motion is disabled")
        caps = self.driver.capabilities()
        if (
            caps.get("whole_body_velocity") is not True
            or caps.get("measured_whole_body_stop") is not True
            or not isinstance(caps.get("robot_watchdog_s"), (int, float))
            or isinstance(caps.get("robot_watchdog_s"), bool)
            or not math.isfinite(caps["robot_watchdog_s"])
            or not 0 < caps["robot_watchdog_s"] <= 0.2
        ):
            raise RuntimeError("verified velocity, measured-stop and robot watchdog capabilities required")
        feedback = self.driver.feedback()
        age = self._age(feedback)
        # Initial feedback must be fresh before capturing the immutable origin.
        self.guard = MotionBoundary(feedback["base_pose"], translation_limit_m=0, translation_tolerance_m=0.05)
        self.guard.enforce(feedback["base_pose"], state_age_s=age, stop=self._stop)
        try:
            if np.any(abs(self._arm_velocity(feedback)) >= np.minimum(0.03, self.profile.speed)):
                raise ValueError("arm must be stationary before starting a session")
        except Exception as exc:
            self.fault = str(exc)
            confirmed = self._stop()
            raise RuntimeError(f"{self.fault}; stop_confirmed={confirmed}") from exc
        self.last_command_time = time.monotonic()

    @staticmethod
    def _age(feedback):
        timestamp = float(feedback.get("captured_monotonic", float("nan")))
        return time.monotonic() - timestamp

    def _arm_velocity(self, feedback):
        measured = np.asarray(feedback["joint_velocity"], dtype=float).copy()
        if measured.shape != (self.profile.arm_size,) or not np.isfinite(measured).all():
            raise ValueError("invalid arm feedback")
        return measured

    def _stop(self):
        self.last_velocity[:] = 0
        return self.driver.stop_and_confirm(timeout_s=0.75) is True

    def stop(self):
        self.fault = self.fault or "session stopped"
        return self._stop()

    def command(self, velocity, *, base_target, target_visible=True):
        """One physical SI velocity packet, preceded by fresh boundary checks.

        driver.feedback() must cover arm AND odometry capture, not a republished
        telemetry timestamp. Caller supplies base + arm velocities in profile
        order; gripper release is not part of this interface.
        """
        if self.guard is None or self.fault is not None:
            raise RuntimeError("session is not active")
        try:
            feedback = self.driver.feedback()
            self.guard.enforce(feedback["base_pose"], state_age_s=self._age(feedback), stop=self._stop)
            measured = self._arm_velocity(feedback)
            if np.any(abs(measured) > self.profile.speed):
                raise ValueError("measured arm velocity exceeds robot profile limits")
            if not target_visible:
                self.fault = "target lost"
                return self._stop()
            target = np.asarray(base_target, dtype=float)
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError("invalid base target")
            yaw = abs(math.remainder(target[2] - self.guard.start[2], 2 * math.pi))
            if np.linalg.norm(target[:2] - self.guard.start[:2]) > 1e-6 or yaw > self.guard.yaw_limit_rad:
                raise ValueError("target outside hold-XY / yaw experiment range")
            packet = np.asarray(velocity, dtype=float)
            limits = np.r_[self.profile.base_speed, self.profile.base_yaw_speed, self.profile.speed]
            if packet.shape != limits.shape or not np.isfinite(packet).all() or np.any(abs(packet) > limits):
                raise ValueError("velocity outside robot profile limits")
            now = time.monotonic()
            elapsed = now - self.last_command_time
            if not 0 <= elapsed <= 0.2:
                raise ValueError("command stream interrupted; explicit new session required")
            # Measured overspeed faults above. Slew from the last accepted
            # command, rather than treating noisy/disturbed feedback as a new
            # command. Restarting after a watchdog gap is prohibited above.
            acceleration = np.r_[
                self.profile.base_acceleration, self.profile.base_yaw_acceleration, self.profile.acceleration
            ]
            bounded = self.last_velocity + np.clip(
                packet - self.last_velocity, -acceleration * elapsed, acceleration * elapsed
            )
            self.driver.command_velocity(bounded.copy(), valid_for_s=0.2)
            self.last_velocity = bounded
            self.last_command_time = now
            return True
        except SystemExit:
            self.fault = self.guard.fault
            raise
        except Exception as exc:
            self.fault = str(exc)
            confirmed = self._stop()
            raise RuntimeError(f"{self.fault}; stop_confirmed={confirmed}") from exc

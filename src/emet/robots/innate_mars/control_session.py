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
from emet.robots.innate_mars.reach_progress import ElbowLoadGuard, elbow_load_speed_scale


class GuardedControlSession:
    def __init__(
        self,
        *,
        driver,
        profile,
        enabled=False,
        translation_limit_m=0.0,
        translation_tolerance_m=0.05,
        adaptive_elbow_load=False,
    ):
        if not all(math.isfinite(v) and v >= 0 for v in (translation_limit_m, translation_tolerance_m)):
            raise ValueError("translation limits must be finite and nonnegative")
        self.driver, self.profile = driver, profile
        self.enabled = enabled
        self._translation_limit_m = translation_limit_m
        self._translation_tolerance_m = translation_tolerance_m
        self.guard = None
        self.fault = None
        self.last_velocity = np.zeros(2 + profile.arm_size)
        self.last_command_time = None
        if adaptive_elbow_load and profile.robot != "innate_mars":
            raise ValueError("adaptive elbow load pacing requires Mars")
        self._elbow_load_guard = ElbowLoadGuard() if adaptive_elbow_load else None
        self.elbow_speed_scale = 1.0

    def start(self):
        if self.guard is not None or self.fault is not None:
            raise RuntimeError("session cannot be restarted or recentered")
        if not self.enabled:
            raise RuntimeError("hardware motion is disabled")
        caps = self.driver.capabilities()
        if (
            caps.get("whole_body_velocity") is not True
            or caps.get("measured_whole_body_stop") is not True
            or caps.get("feedback_si_units_verified") is not True
            or caps.get("motor_read_validity_verified") is not True
            or not isinstance(caps.get("robot_watchdog_s"), (int, float))
            or isinstance(caps.get("robot_watchdog_s"), bool)
            or not math.isfinite(caps["robot_watchdog_s"])
            or not 0 < caps["robot_watchdog_s"] <= 0.2
        ):
            raise RuntimeError(
                "verified velocity, SI/read validity, measured-stop and robot watchdog capabilities required"
            )
        if self._elbow_load_guard and caps.get("elbow_load_percent_verified") is not True:
            raise RuntimeError("adaptive elbow pacing requires verified XL430 load feedback")
        try:
            feedback = self.driver.feedback()
            age = self._age(feedback)
            if not 0 <= age <= 0.2:
                raise ValueError("missing or stale arm/odometry capture")
            # Capture the immutable origin only after validating both sources.
            self.guard = MotionBoundary(
                feedback["base_pose"],
                translation_limit_m=self._translation_limit_m,
                translation_tolerance_m=self._translation_tolerance_m,
            )
            self.guard.enforce(feedback["base_pose"], state_age_s=age, stop=self._stop)
            self._update_elbow_load(feedback)
            if np.any(abs(self._arm_velocity(feedback)) >= np.minimum(0.03, self.profile.speed)):
                raise ValueError("arm must be stationary before starting a session")
            twist = self._base_velocity(feedback)
            if np.linalg.norm(twist[:3]) >= 0.03 or np.linalg.norm(twist[3:]) >= 0.05:
                raise ValueError("base must be stationary before starting a session")
        except SystemExit:
            # Feedback/age validation can abort before the boundary exists.
            # Preserve that original termination and latch a nonempty fault.
            self.fault = getattr(self.guard, "fault", None) or "session start aborted"
            raise
        except Exception as exc:
            self.fault = str(exc)
            confirmed = self._stop()
            raise RuntimeError(f"{self.fault}; stop_confirmed={confirmed}") from exc
        self.last_command_time = time.monotonic()

    @staticmethod
    def _age(feedback):
        if feedback.get("valid") is not True:
            raise ValueError("valid motor/odometry acquisition required")
        now = time.monotonic()
        ages = [
            now - float(feedback.get(k, float("nan"))) for k in ("arm_captured_monotonic", "odom_captured_monotonic")
        ]
        if any(not math.isfinite(age) or age < 0 for age in ages):
            raise ValueError("invalid arm/odometry capture time in driver monotonic clock")
        return max(ages)

    @staticmethod
    def _base_velocity(feedback):
        measured = np.asarray(feedback.get("base_twist"), dtype=float).copy()
        if measured.shape != (6,) or not np.isfinite(measured).all():
            raise ValueError("complete finite six-component base twist required")
        return measured

    def _arm_velocity(self, feedback):
        measured = np.asarray(feedback["joint_velocity"], dtype=float).copy()
        if measured.shape != (self.profile.arm_size,) or not np.isfinite(measured).all():
            raise ValueError("invalid arm feedback")
        return measured

    def _update_elbow_load(self, feedback):
        if self._elbow_load_guard is not None:
            if feedback.get("arm_healthy") is not True:
                raise RuntimeError("Mars arm fault or missing health feedback")
            load = float(feedback.get("elbow_load_percent", math.nan))
            self._elbow_load_guard.check(now=feedback["arm_captured_monotonic"], load_percent=load)
            self.elbow_speed_scale = elbow_load_speed_scale(load)

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
        return self._command(velocity, base_target=base_target, target_visible=target_visible, normalized=False)

    def command_action(self, action, *, base_target, target_visible=True):
        """Send one normalized WBC action as base and arm SI velocities.

        The final grip channel is validated with the action and omitted from
        the driver packet. Grip remains under the hardware executor's control.
        """
        return self._command(action, base_target=base_target, target_visible=target_visible, normalized=True)

    def _command(self, velocity, *, base_target, target_visible, normalized):
        if self.guard is None or self.fault is not None:
            raise RuntimeError("session is not active")
        try:
            feedback = self.driver.feedback()
            self.guard.enforce(feedback["base_pose"], state_age_s=self._age(feedback), stop=self._stop)
            self._update_elbow_load(feedback)
            measured = self._arm_velocity(feedback)
            if np.any(abs(measured) > self.profile.speed):
                raise ValueError("measured arm velocity exceeds robot profile limits")
            twist = self._base_velocity(feedback)
            if (
                np.linalg.norm(twist[:3]) > self.profile.base_speed
                or np.linalg.norm(twist[3:]) > self.profile.base_yaw_speed
            ):
                raise ValueError("measured base velocity exceeds robot profile limits")
            if not target_visible:
                self.fault = "target lost"
                return self._stop()
            target = np.asarray(base_target, dtype=float)
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError("invalid base target")
            yaw = abs(math.remainder(target[2] - self.guard.start[2], 2 * math.pi))
            travel = np.linalg.norm(target[:2] - self.guard.start[:2])
            if travel > self.guard.translation_limit_m + 1e-6 or yaw > self.guard.yaw_limit_rad:
                raise ValueError("target outside the session base envelope")
            limits = np.r_[self.profile.base_speed, self.profile.base_yaw_speed, self.profile.speed]
            packet = np.asarray(velocity, dtype=float)
            if normalized:
                if packet.shape != (len(limits) + 1,) or not np.isfinite(packet).all() or np.any(abs(packet) > 1):
                    raise ValueError("invalid normalized WBC action")
                packet = packet[:-1] * limits
            if packet.shape != limits.shape or not np.isfinite(packet).all() or np.any(abs(packet) > limits):
                raise ValueError("velocity outside robot profile limits")
            if self._elbow_load_guard is not None:
                packet = packet.copy()
                packet[2 + 2] *= self.elbow_speed_scale
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

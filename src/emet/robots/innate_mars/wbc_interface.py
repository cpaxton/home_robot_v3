"""Run a Mars WBC policy through a verified arm-position and base-twist transport."""

import math

import numpy as np

from emet.robots.innate_mars.control_session import GuardedControlSession
from emet.robots.innate_mars.startup_recovery import JOINT_NAMES, shoulder_lower_limit


class PositionTargetDriver:
    """Translate one bounded SI velocity packet into measured-relative targets.

    The robot-side transport supplies source-timed feedback and publishes both
    targets under one ownership lease with a watchdog. It owns the confirmed
    stop. This adapter never commands a gripper phase change.
    """

    def __init__(self, transport, published_limits, *, control_period_s):
        self.transport = transport
        self.limits = np.asarray([published_limits[name] for name in JOINT_NAMES], dtype=float)
        if (
            self.limits.shape != (6, 2)
            or not np.isfinite(self.limits).all()
            or np.any(self.limits[:, 0] >= self.limits[:, 1])
        ):
            raise ValueError("finite published limits required for the Mars arm and gripper")
        if not math.isfinite(control_period_s) or not 0 < control_period_s <= 0.1:
            raise ValueError("expected a bounded Mars control period")
        self.period = control_period_s
        self._feedback = None
        self.last_arm_target = None

    def capabilities(self):
        caps = self.transport.capabilities()
        return {**caps, "whole_body_velocity": caps.get("watchdog_arm_position_and_base_twist") is True}

    def feedback(self):
        self._feedback = self.transport.feedback()
        return self._feedback

    def command_velocity(self, value, *, valid_for_s):
        if self._feedback is None:
            raise RuntimeError("measured Mars joint position required")
        measured = np.asarray(self._feedback.get("joint_position"), dtype=float)
        grip = float(self._feedback.get("gripper_target_rad", math.nan))
        if measured.shape != (5,) or not np.isfinite(measured).all() or not math.isfinite(grip):
            raise RuntimeError("fresh five-axis position and standing gripper target required")
        lower, upper = self.limits.T
        if np.any(measured < lower[:5]) or np.any(measured > upper[:5]) or not lower[5] <= grip <= upper[5]:
            raise RuntimeError("measured Mars arm or standing grip is outside its published limits")
        arm_step = np.asarray(value[2:], dtype=float) * self.period
        target = np.clip(measured + arm_step, lower[:5], upper[:5])

        # Innate applies this shoulder/yaw coupling to position targets. If
        # yaw would force an unrequested shoulder motion, defer yaw while the
        # controller brings the shoulder into range on subsequent ticks.
        minimum_shoulder = shoulder_lower_limit(target[0], lower[1])
        if target[1] < minimum_shoulder:
            target[0] = measured[0]
            if target[1] < shoulder_lower_limit(target[0], lower[1]):
                raise RuntimeError("shoulder target needs clearance before yaw")
        self.transport.send_targets(
            arm_position_rad=np.r_[target, grip],
            base_twist_si=np.asarray(value[:2], dtype=float).copy(),
            valid_for_s=valid_for_s,
        )

        self.last_arm_target = target.copy()

    def stop_and_confirm(self, *, timeout_s):
        return self.transport.stop_and_confirm(timeout_s=timeout_s)


class MarsWBCInterface:
    """One policy update per observation, using the Mars guarded transport."""

    def __init__(self, *, transport, policy, profile, published_limits, **session_options):
        if profile.robot != "innate_mars":
            raise ValueError("expected the Mars actuation profile")
        self.policy = policy
        self.driver = PositionTargetDriver(transport, published_limits, control_period_s=profile.dt)
        self.session = GuardedControlSession(driver=self.driver, profile=profile, **session_options)

    def start(self):
        self.session.start()

    def step(self, observation, *, base_target, target_visible=True):
        if not target_visible:
            return self.session.command_action(
                np.zeros(2 + self.session.profile.arm_size + 1), base_target=base_target, target_visible=False
            )
        if self.session.guard is None or self.session.fault is not None:
            raise RuntimeError("session is not active")
        try:
            action = self.policy(observation)
        except Exception as exc:
            if not self.session.stop():
                raise RuntimeError("policy inference failed and Mars stop was unconfirmed") from exc
            raise
        return self.session.command_action(action, base_target=base_target)

    def stop(self):
        return self.session.stop()

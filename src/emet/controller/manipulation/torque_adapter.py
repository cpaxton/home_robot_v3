# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Actuator adapters for the torque-based WBC contract.

The policy always emits joint torques. How a robot realizes those torques is an
actuator property, not a policy property:

- **position-servo adapter** (default, Mars-class hardware): convert torque to a
  bounded position-target offset and, when available, a motor current limit.
- **torque adapter**: publish torque/current directly for hardware that
  declares true torque control, without changing the policy contract.

The position-servo mapping is intentionally simple and fails closed: it
requires an identified effective stiffness and maximum lead for every joint.
It is a model-based feedforward, not a torque sensor or a stability proof.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from emet.controller.manipulation.wbc_contract import (
    EmbodimentProfile,
    GripperCommand,
    JointSpec,
    WbcAction,
    WbcObservation,
    clamp_action,
)


@dataclass(frozen=True)
class ServoCommand:
    """Robot-facing command produced by an actuator adapter."""

    base_velocity: tuple[float, ...]
    joint_positions: dict[str, float]
    current_limits: dict[str, float] = field(default_factory=dict)
    commanded_torques: dict[str, float] = field(default_factory=dict)
    grippers: dict[str, GripperCommand] = field(default_factory=dict)
    clipped: bool = False

    def __post_init__(self) -> None:
        if not all(np.isfinite(float(v)) for v in self.base_velocity):
            raise ValueError("base_velocity contains a non-finite value")
        if not all(np.isfinite(float(v)) for v in self.joint_positions.values()):
            raise ValueError("joint_positions contains a non-finite value")
        if not all(np.isfinite(float(v)) for v in self.current_limits.values()):
            raise ValueError("current_limits contains a non-finite value")
        if not all(np.isfinite(float(v)) for v in self.commanded_torques.values()):
            raise ValueError("commanded_torques contains a non-finite value")


Feedforward = Mapping[str, float] | Callable[[WbcObservation], Mapping[str, float]]


class ActuatorAdapter(Protocol):
    """Common interface for position-servo, current, or torque adapters."""

    profile: EmbodimentProfile

    def command(self, action: WbcAction, observation: WbcObservation) -> ServoCommand:
        """Map one policy action to a robot-facing command."""


class TorqueAdapter:
    """Direct torque adapter for hardware that realizes joint torque natively.

    The policy contract is unchanged. This adapter only accepts joints whose
    ``control_mode`` is ``torque``; a position-servo or current-position joint
    must use :class:`PositionServoAdapter` instead. Gravity/friction terms are
    optional identified feedforwards and are added before the torque limit.
    """

    def __init__(
        self,
        profile: EmbodimentProfile,
        *,
        gravity_torques: Feedforward | None = None,
        friction_torques: Feedforward | None = None,
    ) -> None:
        self.profile = profile
        self._gravity = gravity_torques
        self._friction = friction_torques
        known = set(profile.controlled_joint_names)
        for source in (gravity_torques, friction_torques):
            if isinstance(source, Mapping):
                extra = set(source) - known
                if extra:
                    raise ValueError(f"feedforward torques reference unknown joints: {sorted(extra)}")
        for group in profile.joint_groups:
            for joint in group.joints:
                if joint.control_mode != "torque":
                    raise ValueError(f"{joint.name} does not declare true torque control; use PositionServoAdapter")
                if joint.current_limit is not None and (joint.torque_constant is None or joint.torque_constant <= 0):
                    raise ValueError(
                        f"{joint.name} declares current_limit but no positive torque_constant; "
                        "the current limit cannot be enforced"
                    )

    def command(self, action: WbcAction, observation: WbcObservation) -> ServoCommand:
        action.validate(self.profile)
        observation.validate(self.profile)
        clipped_action = clamp_action(action, self.profile)
        clipped = clipped_action != action
        action = clipped_action
        gravity = self._resolve(self._gravity, observation, "gravity_torques")
        friction = self._resolve(self._friction, observation, "friction_torques")
        currents: dict[str, float] = {}
        torques: dict[str, float] = {}
        if self.profile.torso is not None:
            torso_obs = observation.torso
            if torso_obs is None:
                raise ValueError("observation is missing the torso group required by the profile")
            for joint, tau in zip(self.profile.torso.joints, action.torso_torques, strict=True):
                self._joint(joint, tau, gravity, friction, currents, torques)
        for arm_spec in self.profile.arms:
            group = self.profile.joint_group(arm_spec.joint_group)
            arm_action = action.arm(arm_spec.name)
            for joint, tau in zip(group.joints, arm_action.joint_torques, strict=True):
                self._joint(joint, tau, gravity, friction, currents, torques)

        grippers: dict[str, GripperCommand] = {}
        for arm_spec in self.profile.arms:
            command = action.arm(arm_spec.name).gripper
            if command is not None:
                grippers[arm_spec.name] = command
        return ServoCommand(
            base_velocity=action.base_velocity,
            joint_positions={},
            current_limits=currents,
            commanded_torques=torques,
            grippers=grippers,
            clipped=clipped,
        )

    def _resolve(
        self,
        source: Feedforward | None,
        observation: WbcObservation,
        label: str,
    ) -> Mapping[str, float]:
        if source is None:
            return {}
        resolved = source(observation) if callable(source) else source
        if not isinstance(resolved, Mapping):
            raise ValueError(f"{label} must return a mapping of joint name to torque")
        extra = set(resolved) - set(self.profile.controlled_joint_names)
        if extra:
            raise ValueError(f"{label} references unknown joints: {sorted(extra)}")
        return resolved

    def _joint(
        self,
        joint: JointSpec,
        tau: float,
        gravity: Mapping[str, float],
        friction: Mapping[str, float],
        currents: dict[str, float],
        torques: dict[str, float],
    ) -> None:
        total = float(tau) + float(gravity.get(joint.name, 0.0)) + float(friction.get(joint.name, 0.0))
        total = float(np.clip(total, -joint.torque_limit, joint.torque_limit))
        current_limit = joint.current_limit
        if current_limit is not None and joint.torque_constant is not None:
            torque_at_current_limit = current_limit * joint.torque_constant * joint.gear_ratio
            total = float(np.clip(total, -torque_at_current_limit, torque_at_current_limit))
            currents[joint.name] = float(min(abs(total) / (joint.torque_constant * joint.gear_ratio), current_limit))
        torques[joint.name] = total


class PositionServoAdapter:
    """Torque -> bounded position-target offset for position-controlled servos.

    For each joint::

        tau_total = clip(tau_cmd + gravity + friction + kd_eff * qd, +/-tau_limit)
        tau_total = clip(tau_total, +/-current_limit * kt * gear_ratio)  # if known
        q_cmd     = clip(q_meas + clip(tau_total / kp_eff, +/-max_lead), position_limit)
        i_limit   = |tau_total| / (kt * gear_ratio)                     # if known

    ``kp_eff`` and ``kd_eff`` are identified effective position-loop gains,
    not datasheet PID values. ``kd_eff * qd`` compensates the servo's own
    velocity damping so the policy's torque is the net requested torque.
    ``max_lead`` bounds the position error the policy can request, which is
    the primary safety limit. Base velocity is clipped to the profile envelope;
    base acceleration remains the executor/robot-side stream's responsibility.
    """

    def __init__(
        self,
        profile: EmbodimentProfile,
        *,
        gravity_torques: Feedforward | None = None,
        friction_torques: Feedforward | None = None,
    ) -> None:
        self.profile = profile
        self._gravity = gravity_torques
        self._friction = friction_torques
        known = set(profile.controlled_joint_names)
        for source in (gravity_torques, friction_torques):
            if isinstance(source, Mapping):
                extra = set(source) - known
                if extra:
                    raise ValueError(f"feedforward torques reference unknown joints: {sorted(extra)}")
        for group in profile.joint_groups:
            for joint in group.joints:
                if joint.control_mode == "torque":
                    raise ValueError(
                        f"{joint.name} declares true torque control; use a torque adapter, not PositionServoAdapter"
                    )
                if joint.current_limit is not None and (joint.torque_constant is None or joint.torque_constant <= 0):
                    raise ValueError(
                        f"{joint.name} declares current_limit but no positive torque_constant; "
                        "the current limit cannot be enforced"
                    )

    def command(self, action: WbcAction, observation: WbcObservation) -> ServoCommand:
        action.validate(self.profile)
        observation.validate(self.profile)
        clipped_action = clamp_action(action, self.profile)
        clipped = clipped_action != action
        action = clipped_action
        gravity = self._resolve(self._gravity, observation, "gravity_torques")
        friction = self._resolve(self._friction, observation, "friction_torques")
        positions: dict[str, float] = {}
        currents: dict[str, float] = {}
        torques: dict[str, float] = {}
        if self.profile.torso is not None:
            torso_obs = observation.torso
            if torso_obs is None:
                raise ValueError("observation is missing the torso group required by the profile")
            for joint, tau, q, qd in zip(
                self.profile.torso.joints,
                action.torso_torques,
                torso_obs.position,
                torso_obs.velocity,
                strict=True,
            ):
                self._joint(joint, tau, q, qd, gravity, friction, positions, currents, torques)
        for arm_spec in self.profile.arms:
            group = self.profile.joint_group(arm_spec.joint_group)
            arm_obs = observation.arm(arm_spec.name)
            arm_action = action.arm(arm_spec.name)
            for joint, tau, q, qd in zip(
                group.joints,
                arm_action.joint_torques,
                arm_obs.joint_position,
                arm_obs.joint_velocity,
                strict=True,
            ):
                self._joint(joint, tau, q, qd, gravity, friction, positions, currents, torques)

        grippers: dict[str, GripperCommand] = {}
        for arm_spec in self.profile.arms:
            command = action.arm(arm_spec.name).gripper
            if command is not None:
                grippers[arm_spec.name] = command
        return ServoCommand(
            base_velocity=action.base_velocity,
            joint_positions=positions,
            current_limits=currents,
            commanded_torques=torques,
            grippers=grippers,
            clipped=clipped,
        )

    def _resolve(
        self,
        source: Feedforward | None,
        observation: WbcObservation,
        label: str,
    ) -> Mapping[str, float]:
        if source is None:
            return {}
        resolved = source(observation) if callable(source) else source
        if not isinstance(resolved, Mapping):
            raise ValueError(f"{label} must return a mapping of joint name to torque")
        extra = set(resolved) - set(self.profile.controlled_joint_names)
        if extra:
            raise ValueError(f"{label} references unknown joints: {sorted(extra)}")
        return resolved

    def _joint(
        self,
        joint: JointSpec,
        tau: float,
        q: float,
        qd: float,
        gravity: Mapping[str, float],
        friction: Mapping[str, float],
        positions: dict[str, float],
        currents: dict[str, float],
        torques: dict[str, float],
    ) -> None:
        if joint.kp_eff is None or joint.kp_eff <= 0:
            raise ValueError(f"{joint.name} requires an identified positive kp_eff for position-servo control")
        if joint.max_lead is None or joint.max_lead <= 0:
            raise ValueError(f"{joint.name} requires a positive max_lead for position-servo control")

        total = (
            float(tau)
            + float(gravity.get(joint.name, 0.0))
            + float(friction.get(joint.name, 0.0))
            + float(joint.kd_eff or 0.0) * float(qd)
        )
        total = float(np.clip(total, -joint.torque_limit, joint.torque_limit))

        current_limit = joint.current_limit
        has_current_command = joint.control_mode != "position" and current_limit is not None
        if has_current_command and joint.torque_constant is not None:
            torque_at_current_limit = current_limit * joint.torque_constant * joint.gear_ratio
            total = float(np.clip(total, -torque_at_current_limit, torque_at_current_limit))

        lead = float(np.clip(total / joint.kp_eff, -joint.max_lead, joint.max_lead))
        lo, hi = joint.position_limit
        positions[joint.name] = float(np.clip(q + lead, lo, hi))
        torques[joint.name] = total
        if has_current_command and joint.torque_constant is not None:
            current = abs(total) / (joint.torque_constant * joint.gear_ratio)
            currents[joint.name] = float(min(current, current_limit))


__all__ = ["ActuatorAdapter", "PositionServoAdapter", "ServoCommand", "TorqueAdapter"]

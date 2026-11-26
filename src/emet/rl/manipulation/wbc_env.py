# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Gymnasium adapter for the whole-body torque contract.

The policy sees a fixed-width, profile-derived observation vector and emits a
normalized action vector. The environment decodes that vector through the
contract, sends a robot-facing command through an actuator adapter, and scores
the resulting contract observation with :func:`reach_reward`. Physics or robot
I/O lives behind :class:`WbcBackend`; this module owns no simulator imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.torque_adapter import ActuatorAdapter, PositionServoAdapter, ServoCommand
from emet.controller.manipulation.wbc_contract import (
    CONTACT_WRENCH_DIM,
    EmbodimentProfile,
    JointSpec,
    ReachSpec,
    WbcAction,
    WbcObservation,
    action_from_normalized_vector,
    active_action_mask,
    reach_reward,
)

OBSERVATION_LAYOUT_VERSION = 3

# Measured contact channels are normalized by fixed physical scales so a
# checkpoint carries an interpretable force range. The values are clipped only
# to keep rare impacts from dominating a normalized actor input.
CONTACT_FORCE_SCALE_N = 50.0
CONTACT_TORQUE_SCALE_NM = 10.0
GRIPPER_FORCE_SCALE_N = 20.0
CONTACT_OBSERVATION_CLIP = 5.0


class WbcBackend(Protocol):
    """Minimal physics/robot backend consumed by :class:`WbcGymEnv`."""

    profile: EmbodimentProfile

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> WbcObservation:
        """Return the initial contract observation."""

    def step(self, command: ServoCommand, dt: float) -> tuple[WbcObservation, bool, dict[str, Any]]:
        """Apply one robot-facing command and return the next observation."""

    def load_hard_case(self, case: Any) -> WbcObservation | None:
        """Start from a recorded failure posture and target, if supported.

        The default is unsupported; backends with a direct plant can reproduce
        the recorded arm posture and return the resulting observation.
        """
        return None

    def close(self) -> None:
        """Release simulator/robot resources."""


@dataclass(frozen=True)
class ObservationLayout:
    """Fixed-width ordering for one embodiment's contract observation."""

    profile: EmbodimentProfile
    fields: tuple[tuple[str, int], ...]
    slices: dict[str, slice]
    occupancy_shape: tuple[int, int, int] | None = None

    @property
    def size(self) -> int:
        return sum(width for _, width in self.fields)

    def encode(self, observation: WbcObservation, *, normalize: bool = True) -> np.ndarray:
        return encode_observation(observation, self.profile, layout=self, normalize=normalize)


def _normalized_joint(value: float, joint: JointSpec) -> float:
    low, high = joint.position_limit
    midpoint = 0.5 * (low + high)
    half_range = 0.5 * (high - low)
    return (float(value) - midpoint) / half_range


def _normalized_gripper(value: float, low: float, high: float) -> float:
    midpoint = 0.5 * (low + high)
    half_range = 0.5 * (high - low)
    return (float(value) - midpoint) / half_range


def wbc_observation_layout(
    profile: EmbodimentProfile,
    *,
    occupancy_shape: tuple[int, int, int] | None = None,
) -> ObservationLayout:
    """Build the canonical scalar layout for one embodiment.

    Optional measured fields (torque, gripper velocity/force, targets) keep a
    fixed width and are paired with presence flags. Occupancy is only included
    when the caller declares a crop shape; its flattening order is C order.
    """
    shape = None
    if occupancy_shape is not None:
        shape = tuple(int(v) for v in occupancy_shape)
        if len(shape) != 3 or any(v <= 0 for v in shape):
            raise ValueError("occupancy_shape must contain three positive integers")
    fields: list[tuple[str, int]] = [
        ("base_pose", 3),
        ("base_twist", profile.base.action_dim),
        ("has_base_target", 1),
        ("base_target", 3),
        ("base_target_mask", 3),
        ("base_position_error", 2),
        ("base_heading_error", 1),
    ]
    if profile.torso is not None:
        n = len(profile.torso.joints)
        fields.extend(
            [
                ("torso_position", n),
                ("torso_velocity", n),
                ("has_torso_torque", 1),
                ("torso_torque", n),
                ("has_torso_target", 1),
                ("torso_target", n),
                ("torso_target_mask", n),
            ]
        )
    if profile.head is not None:
        n = len(profile.head.joints)
        fields.extend(
            [
                ("head_position", n),
                ("head_velocity", n),
                ("has_head_torque", 1),
                ("head_torque", n),
            ]
        )
    for arm_spec in profile.arms:
        group = profile.joint_group(arm_spec.joint_group)
        n = len(group.joints)
        prefix = f"arm.{arm_spec.name}"
        fields.extend(
            [
                (f"{prefix}.joint_position", n),
                (f"{prefix}.joint_velocity", n),
                (f"{prefix}.has_joint_torque", 1),
                (f"{prefix}.joint_torque", n),
                (f"{prefix}.ee_pose", 7),
                (f"{prefix}.ee_position_error", 3),
                (f"{prefix}.ee_orientation_error", 3),
                (f"{prefix}.has_target", 1),
                (f"{prefix}.ee_target", 7),
                (f"{prefix}.target_mask", 2),
            ]
        )
        if arm_spec.gripper is not None:
            m = arm_spec.gripper.dof
            fields.extend(
                [
                    (f"{prefix}.gripper_position", m),
                    (f"{prefix}.has_gripper_velocity", 1),
                    (f"{prefix}.gripper_velocity", m),
                    (f"{prefix}.has_gripper_force", 1),
                    (f"{prefix}.gripper_force", m),
                ]
            )
        fields.extend(
            [
                (f"{prefix}.has_contact_state", 1),
                (f"{prefix}.contact_flag", 1),
                (f"{prefix}.contact_wrench", CONTACT_WRENCH_DIM),
            ]
        )
    if shape is not None:
        fields.extend([("has_occupancy", 1), ("occupancy", int(np.prod(shape)))])

    slices: dict[str, slice] = {}
    offset = 0
    for name, width in fields:
        slices[name] = slice(offset, offset + width)
        offset += width
    return ObservationLayout(profile=profile, fields=tuple(fields), slices=slices, occupancy_shape=shape)


def encode_observation(
    observation: WbcObservation,
    profile: EmbodimentProfile,
    *,
    layout: ObservationLayout | None = None,
    occupancy_shape: tuple[int, int, int] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Encode one contract observation into the fixed training layout."""
    if layout is None:
        layout = wbc_observation_layout(profile, occupancy_shape=occupancy_shape)
    if layout.profile != profile:
        raise ValueError("observation layout profile mismatch")
    if layout.occupancy_shape is None and observation.occupancy is not None:
        raise ValueError("observation has occupancy but the layout does not declare a crop shape")
    observation.validate(profile)
    values = np.zeros(layout.size, dtype=np.float32)

    def put(name: str, value: Any) -> None:
        data = np.asarray(value, dtype=np.float32).reshape(-1)
        target = layout.slices[name]
        if data.size != target.stop - target.start:
            raise ValueError(f"{name} has {data.size} values, expected {target.stop - target.start}")
        values[target] = data

    base_limits = np.asarray(profile.base.velocity_limits, dtype=np.float64)
    put("base_pose", observation.base_pose)
    base_twist = np.asarray(observation.base_twist, dtype=np.float64)
    put("base_twist", base_twist / base_limits if normalize else base_twist)
    put("has_base_target", [observation.base_target is not None])
    put("base_target", [0.0, 0.0, 0.0] if observation.base_target is None else observation.base_target)
    put("base_target_mask", [float(v) for v in observation.base_target_mask])
    if observation.base_target is None:
        base_position_error = (0.0, 0.0)
        base_heading_error = 0.0
    else:
        base_position_error = (
            observation.base_target[0] - observation.base_pose[0],
            observation.base_target[1] - observation.base_pose[1],
        )
        base_heading_error = float(
            np.arctan2(
                np.sin(observation.base_target[2] - observation.base_pose[2]),
                np.cos(observation.base_target[2] - observation.base_pose[2]),
            )
        )
    put("base_position_error", base_position_error)
    put("base_heading_error", [base_heading_error])

    if profile.torso is not None:
        assert observation.torso is not None
        group = observation.torso
        joints = profile.torso.joints
        put(
            "torso_position",
            [_normalized_joint(q, j) if normalize else q for q, j in zip(group.position, joints, strict=True)],
        )
        put(
            "torso_velocity",
            [qd / j.velocity_limit if normalize else qd for qd, j in zip(group.velocity, joints, strict=True)],
        )
        put("has_torso_torque", [bool(group.torque)])
        torque = group.torque or (0.0,) * len(joints)
        put(
            "torso_torque",
            [tau / j.torque_limit if normalize else tau for tau, j in zip(torque, joints, strict=True)],
        )
        put("has_torso_target", [bool(group.target)])
        target = group.target or (0.0,) * len(joints)
        put("torso_target", [_normalized_joint(q, j) if normalize else q for q, j in zip(target, joints, strict=True)])
        put("torso_target_mask", [float(v) for v in group.effective_target_mask or (False,) * len(joints)])

    if profile.head is not None:
        assert observation.head is not None
        group = observation.head
        joints = profile.head.joints
        put(
            "head_position",
            [_normalized_joint(q, j) if normalize else q for q, j in zip(group.position, joints, strict=True)],
        )
        put(
            "head_velocity",
            [qd / j.velocity_limit if normalize else qd for qd, j in zip(group.velocity, joints, strict=True)],
        )
        put("has_head_torque", [bool(group.torque)])
        torque = group.torque or (0.0,) * len(joints)
        put(
            "head_torque",
            [tau / j.torque_limit if normalize else tau for tau, j in zip(torque, joints, strict=True)],
        )

    for arm_spec in profile.arms:
        arm = observation.arm(arm_spec.name)
        group = profile.joint_group(arm_spec.joint_group)
        prefix = f"arm.{arm_spec.name}"
        put(
            f"{prefix}.joint_position",
            [
                _normalized_joint(q, j) if normalize else q
                for q, j in zip(arm.joint_position, group.joints, strict=True)
            ],
        )
        put(
            f"{prefix}.joint_velocity",
            [
                qd / j.velocity_limit if normalize else qd
                for qd, j in zip(arm.joint_velocity, group.joints, strict=True)
            ],
        )
        put(f"{prefix}.has_joint_torque", [bool(arm.joint_torque)])
        torque = arm.joint_torque or (0.0,) * len(group.joints)
        put(
            f"{prefix}.joint_torque",
            [tau / j.torque_limit if normalize else tau for tau, j in zip(torque, group.joints, strict=True)],
        )
        put(f"{prefix}.ee_pose", arm.ee_pose)
        if arm.ee_target is None:
            position_error = (0.0, 0.0, 0.0)
            orientation_error = (0.0, 0.0, 0.0)
        else:
            position_error = tuple(float(arm.ee_target[i] - arm.ee_pose[i]) for i in range(3))
            relative = Rotation.from_quat(arm.ee_target[3:]) * Rotation.from_quat(arm.ee_pose[3:]).inv()
            orientation_error = tuple(float(v) for v in relative.as_rotvec())
        put(f"{prefix}.ee_position_error", position_error)
        put(f"{prefix}.ee_orientation_error", orientation_error)
        put(f"{prefix}.has_target", [arm.ee_target is not None])
        put(f"{prefix}.ee_target", (0.0,) * 7 if arm.ee_target is None else arm.ee_target)
        put(f"{prefix}.target_mask", [float(v) for v in arm.target_mask])
        if arm_spec.gripper is not None:
            assert arm.gripper is not None
            gripper = arm.gripper
            limits = arm_spec.gripper.command_limits
            if limits is not None and normalize:
                position = [
                    _normalized_gripper(value, low, high)
                    for value, (low, high) in zip(gripper.position, limits, strict=True)
                ]
            else:
                position = gripper.position
            put(f"{prefix}.gripper_position", position)
            put(f"{prefix}.has_gripper_velocity", [bool(gripper.velocity)])
            put(f"{prefix}.gripper_velocity", gripper.velocity or (0.0,) * arm_spec.gripper.dof)
            put(f"{prefix}.has_gripper_force", [bool(gripper.force)])
            force = gripper.force or (0.0,) * arm_spec.gripper.dof
            if normalize:
                force = [
                    float(np.clip(value / GRIPPER_FORCE_SCALE_N, -CONTACT_OBSERVATION_CLIP, CONTACT_OBSERVATION_CLIP))
                    for value in force
                ]
            put(f"{prefix}.gripper_force", force)

        contact = arm.contact
        put(f"{prefix}.has_contact_state", [contact is not None])
        put(f"{prefix}.contact_flag", [bool(contact.in_contact) if contact is not None else False])
        if contact is None:
            wrench = (0.0,) * CONTACT_WRENCH_DIM
        elif normalize:
            wrench = tuple(
                float(np.clip(value / scale, -CONTACT_OBSERVATION_CLIP, CONTACT_OBSERVATION_CLIP))
                for value, scale in zip(
                    contact.wrench,
                    (CONTACT_FORCE_SCALE_N,) * 3 + (CONTACT_TORQUE_SCALE_NM,) * 3,
                    strict=True,
                )
            )
        else:
            wrench = contact.wrench
        put(f"{prefix}.contact_wrench", wrench)

    if layout.occupancy_shape is not None:
        put("has_occupancy", [observation.occupancy is not None])
        if observation.occupancy is None:
            occupancy = np.zeros(layout.occupancy_shape, dtype=np.float32)
        else:
            occupancy = np.asarray(observation.occupancy)
            if occupancy.shape != layout.occupancy_shape:
                raise ValueError(f"occupancy shape {occupancy.shape} != layout {layout.occupancy_shape}")
            occupancy = occupancy.astype(np.float32)
            # Accept binary crops (0/1) as well as the common 0..255 encoding.
            if occupancy.size and float(occupancy.max()) > 1.0:
                occupancy = occupancy / 255.0
        put("occupancy", occupancy.reshape(-1))
    return values


class WbcGymEnv(gym.Env):
    """Contract-native Gymnasium environment over a physics/robot backend.

    ``action`` is the normalized action vector from
    :func:`normalized_action_vector`. Rewards come from :func:`reach_reward`
    on the next contract observation. A successful contract is held for
    ``success_hold_s`` before termination so a one-step overshoot cannot end
    the episode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        backend: WbcBackend,
        *,
        adapter: ActuatorAdapter | None = None,
        spec: ReachSpec = ReachSpec(),
        horizon_s: float = 6.0,
        control_dt: float = 0.05,
        success_hold_s: float = 0.25,
        collision_force_n: float = 10.0,
        occupancy_shape: tuple[int, int, int] | None = None,
        normalize_observations: bool = True,
        hold_base: bool = False,
    ) -> None:
        self.backend = backend
        self.profile = backend.profile
        if not np.isfinite([horizon_s, control_dt, success_hold_s, collision_force_n]).all():
            raise ValueError("environment timings and collision threshold must be finite")
        if horizon_s <= 0 or control_dt <= 0 or success_hold_s < 0 or collision_force_n < 0:
            raise ValueError("environment timings must be positive and thresholds nonnegative")
        if success_hold_s > horizon_s:
            raise ValueError("success hold cannot exceed the episode horizon")
        if adapter is None:
            adapter = getattr(backend, "preferred_adapter", None)
        self.adapter = adapter if adapter is not None else PositionServoAdapter(self.profile)
        if self.adapter.profile != self.profile:
            raise ValueError("adapter profile does not match backend profile")
        self.spec = spec
        self.horizon_s = float(horizon_s)
        self.control_dt = float(control_dt)
        self.success_hold_s = float(success_hold_s)
        self.collision_force_n = float(collision_force_n)
        self.normalize_observations = bool(normalize_observations)
        self.hold_base = bool(hold_base)
        self.layout = wbc_observation_layout(self.profile, occupancy_shape=occupancy_shape)
        self.action_space = spaces.Box(-1.0, 1.0, (self.profile.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.layout.size,), dtype=np.float32)
        self._observation: WbcObservation | None = None
        self._previous_action: WbcAction | None = None
        self._elapsed = 0.0
        self._success_time = 0.0
        self._done = True

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        observation = self.backend.reset(seed=seed, options=options)
        observation.validate(self.profile)
        if self.hold_base and observation.base_target is not None:
            raise ValueError("hold_base=True requires observations without a base target")
        self._observation = observation
        self._previous_action = None
        self._elapsed = 0.0
        self._success_time = 0.0
        self._done = False
        return self.encode(observation), {
            "profile_fingerprint": self.profile.fingerprint,
            **self.contact_info(observation),
        }

    def load_hard_case(self, case: Any):
        """Replace the current reset with a recorded hard-case posture."""
        observation = self.backend.load_hard_case(case)
        if observation is None:
            return None
        observation.validate(self.profile)
        if self.hold_base and observation.base_target is not None:
            raise ValueError("hold_base=True requires observations without a base target")
        self._observation = observation
        self._previous_action = None
        self._elapsed = 0.0
        self._success_time = 0.0
        self._done = False
        return self.encode(observation), {
            "profile_fingerprint": self.profile.fingerprint,
            "hard_case": getattr(case, "name", ""),
            **self.contact_info(observation),
        }

    def encode(self, observation: WbcObservation) -> np.ndarray:
        return self.layout.encode(observation, normalize=self.normalize_observations)

    @staticmethod
    def contact_info(observation: WbcObservation) -> dict[str, bool]:
        """Report measured contact availability without exposing GT contact."""

        states = [arm.contact for arm in observation.arms if arm.contact is not None]
        return {
            "contact_state_available": len(states) == len(observation.arms),
            "contact_flag": any(state.in_contact for state in states),
        }

    def step(self, action: np.ndarray):
        if self._done or self._observation is None:
            raise RuntimeError("reset required")
        vector = np.asarray(action, dtype=np.float64).reshape(-1)
        if vector.shape != self.action_space.shape or not np.isfinite(vector).all():
            raise ValueError("invalid normalized whole-body action")
        if self.hold_base:
            if self._observation.base_target is not None:
                raise ValueError("hold_base=True requires observations without a base target")
            vector = vector.copy()
            vector[: self.profile.base.action_dim] = 0.0
        contract_action = action_from_normalized_vector(vector, self.profile)
        command = self.adapter.command(contract_action, self._observation)
        next_observation, backend_terminated, info = self.backend.step(command, self.control_dt)
        next_observation.validate(self.profile)
        collision = bool(info.get("collision", False)) or float(info.get("contact_force", 0.0)) > self.collision_force_n
        reward, components = reach_reward(
            contract_action,
            next_observation,
            profile=self.profile,
            collision=collision,
            prev_action=self._previous_action,
            spec=self.spec,
        )
        self._elapsed += self.control_dt
        success = components["success"] > 0.5
        self._success_time = self._success_time + self.control_dt if success else 0.0
        terminated = bool(backend_terminated) or (success and self._success_time >= self.success_hold_s)
        truncated = (not terminated) and self._elapsed >= self.horizon_s
        action_mask = active_action_mask(next_observation, self.profile)
        if self.hold_base:
            action_mask[: self.profile.base.action_dim] = False
        info = {
            **info,
            "profile_fingerprint": self.profile.fingerprint,
            "reward_terms": components,
            "action_mask": action_mask,
            "success": success,
            "collision": collision,
            **self.contact_info(next_observation),
        }
        self._observation = next_observation
        self._previous_action = contract_action
        self._done = terminated or truncated
        return self.encode(next_observation), float(reward), terminated, truncated, info

    def close(self) -> None:
        self.backend.close()


__all__ = [
    "OBSERVATION_LAYOUT_VERSION",
    "ObservationLayout",
    "WbcBackend",
    "WbcGymEnv",
    "encode_observation",
    "wbc_observation_layout",
]

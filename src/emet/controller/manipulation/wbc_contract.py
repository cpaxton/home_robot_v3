# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Frozen multi-arm contract for whole-body RL (schema v2).

The contract is deliberately **embodiment-first**:

- Every robot declares a :class:`EmbodimentProfile` with a base, one or more
  named arms, optional torso/head groups, and optional grippers. Joint order is
  explicit; there is no implicit "one hand, one gripper" assumption.
- Observations are real-world-derivable only: joint encoders, odometry, EE pose
  from forward kinematics, and a local 3D occupancy crop from depth. No GT object
  pose.
- Actions are base velocity plus **per-arm joint torques**. Torque is the policy
  contract; the actuator adapter decides how a specific robot realizes it
  (position servo + current limit, current control, or true torque).
- Rewards are computed from the same signals and aggregate over active arms.

The schema is versioned. A v1 single-arm flat checkpoint is not silently
reinterpreted as v2; loaders must reject a schema or embodiment mismatch.

See ``docs/plans/whole-body-rl-control.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np

SCHEMA_VERSION = 2

# EE poses are (position xyz, quaternion xyzw).
EE_POSE_DIM = 7
BASE_POSE_DIM = 3
TARGET_MASK_DIM = 2  # (position, orientation)

JointKind = Literal["revolute", "prismatic"]
GroupRole = Literal["arm", "torso", "head"]
BaseKind = Literal["planar", "holonomic", "fixed"]
GripperMode = Literal["position", "force", "velocity"]
ControlMode = Literal["position", "current_position", "torque"]


def _require_finite(value: float, name: str, *, positive: bool = False, nonnegative: bool = False) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_vector(values: tuple[float, ...], size: int, name: str) -> None:
    if len(values) != size:
        raise ValueError(f"{name} must have length {size}, got {len(values)}")
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError(f"{name} contains a non-finite value")


@dataclass(frozen=True)
class JointSpec:
    """One actuated joint in a named group.

    ``kp_eff`` / ``max_lead`` / ``torque_constant`` / ``current_limit`` are
    actuator-model fields used by the position-servo adapter. They are optional
    because a profile may be built before bench identification; the adapter
    fails closed if a required field is missing.
    """

    name: str
    position_limit: tuple[float, float]
    velocity_limit: float
    torque_limit: float
    kind: JointKind = "revolute"
    control_mode: ControlMode = "position"
    kp_eff: float | None = None  # effective position-loop stiffness, N*m/rad
    kd_eff: float | None = None  # effective damping, N*m*s/rad
    torque_constant: float | None = None  # N*m/A at the motor
    gear_ratio: float = 1.0
    current_limit: float | None = None  # A at the motor
    max_lead: float | None = None  # rad or m of position-target lead

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("joint name must be a non-empty string")
        if self.kind not in ("revolute", "prismatic"):
            raise ValueError(f"unknown joint kind: {self.kind!r}")
        if self.control_mode not in ("position", "current_position", "torque"):
            raise ValueError(f"unknown control mode: {self.control_mode!r}")
        lo, hi = self.position_limit
        _require_finite(lo, f"{self.name}.position_limit[0]")
        _require_finite(hi, f"{self.name}.position_limit[1]")
        if lo >= hi:
            raise ValueError(f"{self.name}.position_limit must be increasing")
        _require_finite(self.velocity_limit, f"{self.name}.velocity_limit", positive=True)
        _require_finite(self.torque_limit, f"{self.name}.torque_limit", positive=True)
        _require_finite(self.gear_ratio, f"{self.name}.gear_ratio", positive=True)
        for value, label, positive in (
            (self.kp_eff, "kp_eff", True),
            (self.kd_eff, "kd_eff", False),
            (self.torque_constant, "torque_constant", True),
            (self.current_limit, "current_limit", True),
            (self.max_lead, "max_lead", True),
        ):
            if value is not None:
                _require_finite(value, f"{self.name}.{label}", positive=positive, nonnegative=not positive)


@dataclass(frozen=True)
class JointGroupSpec:
    """A named ordered group of joints (one arm, torso, or head)."""

    name: str
    joints: tuple[JointSpec, ...]
    role: GroupRole = "arm"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("joint group name must be a non-empty string")
        if self.role not in ("arm", "torso", "head"):
            raise ValueError(f"unknown group role: {self.role!r}")
        if not self.joints:
            raise ValueError(f"{self.name} must contain at least one joint")
        names = [j.name for j in self.joints]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name} contains duplicate joint names")

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.joints)


@dataclass(frozen=True)
class BaseSpec:
    """Mobile base action interface.

    ``planar`` is (vx, omega) for a differential-drive base; ``holonomic`` is
    (vx, vy, omega) for a swerve/omni base; ``fixed`` has no base action.

    ``acceleration_limits`` are executor-owned. ``clamp_action`` and the
    position-servo adapter clip the velocity envelope only; the robot-side
    command stream owns slew/acceleration limits.
    """

    name: str = "base"
    kind: BaseKind = "planar"
    velocity_limits: tuple[float, ...] = (0.5, 1.0)
    acceleration_limits: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("base name must be a non-empty string")
        if self.kind not in ("planar", "holonomic", "fixed"):
            raise ValueError(f"unknown base kind: {self.kind!r}")
        expected = self.action_dim
        if len(self.velocity_limits) != expected:
            raise ValueError(f"{self.name}.velocity_limits must have length {expected}")
        for i, value in enumerate(self.velocity_limits):
            _require_finite(value, f"{self.name}.velocity_limits[{i}]", positive=True)
        if self.acceleration_limits is not None:
            if len(self.acceleration_limits) != expected:
                raise ValueError(f"{self.name}.acceleration_limits must have length {expected}")
            for i, value in enumerate(self.acceleration_limits):
                _require_finite(value, f"{self.name}.acceleration_limits[{i}]", positive=True)

    @property
    def action_dim(self) -> int:
        return {"planar": 2, "holonomic": 3, "fixed": 0}[self.kind]


@dataclass(frozen=True)
class GripperSpec:
    """Gripper action interface for one arm.

    ``command_mode`` describes the policy's gripper channel. A parallel jaw is
    one value; a multi-finger hand can declare more. Gripper commands are *not*
    part of the arm torque vector.
    """

    name: str
    joint_names: tuple[str, ...]
    command_mode: GripperMode = "position"
    force_limit: float | None = None
    # Per-joint command bounds. None means the gripper command is passed through
    # (the executor/servo remains responsible for its own limit).
    command_limits: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("gripper name must be a non-empty string")
        if not self.joint_names:
            raise ValueError(f"{self.name} must contain at least one joint")
        if self.command_mode not in ("position", "force", "velocity"):
            raise ValueError(f"unknown gripper command mode: {self.command_mode!r}")
        if self.force_limit is not None:
            _require_finite(self.force_limit, f"{self.name}.force_limit", positive=True)
        if self.command_limits is not None:
            if len(self.command_limits) != len(self.joint_names):
                raise ValueError(f"{self.name}.command_limits must match gripper DOF")
            for i, (lo, hi) in enumerate(self.command_limits):
                _require_finite(lo, f"{self.name}.command_limits[{i}][0]")
                _require_finite(hi, f"{self.name}.command_limits[{i}][1]")
                if lo >= hi:
                    raise ValueError(f"{self.name}.command_limits[{i}] must be increasing")

    @property
    def dof(self) -> int:
        return len(self.joint_names)


@dataclass(frozen=True)
class ArmSpec:
    """One named arm and its optional gripper."""

    name: str
    joint_group: str
    ee_link: str
    target_frame: str = "base_link"
    gripper: GripperSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("arm name must be a non-empty string")
        if not isinstance(self.joint_group, str) or not self.joint_group:
            raise ValueError(f"{self.name}.joint_group must be a non-empty string")
        if not isinstance(self.ee_link, str) or not self.ee_link:
            raise ValueError(f"{self.name}.ee_link must be a non-empty string")
        if not isinstance(self.target_frame, str) or not self.target_frame:
            raise ValueError(f"{self.name}.target_frame must be a non-empty string")


@dataclass(frozen=True)
class EmbodimentProfile:
    """Canonical joint/arm layout for one robot or robot family.

    ``joint_groups`` contains the actuated groups referenced by ``arms`` and
    ``torso_group``/``head_group``. Arm order in ``arms`` defines the canonical
    multi-arm ordering for actions, observations, and rewards.
    """

    name: str
    base: BaseSpec
    joint_groups: tuple[JointGroupSpec, ...]
    arms: tuple[ArmSpec, ...]
    torso_group: str | None = None
    head_group: str | None = None
    schema_version: int = SCHEMA_VERSION
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("profile name must be a non-empty string")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported profile schema version: {self.schema_version}")
        if not self.arms:
            raise ValueError(f"{self.name} must declare at least one arm")
        group_names = [g.name for g in self.joint_groups]
        if len(set(group_names)) != len(group_names):
            raise ValueError(f"{self.name} contains duplicate joint-group names")
        groups = {g.name: g for g in self.joint_groups}
        arm_names = [a.name for a in self.arms]
        if len(set(arm_names)) != len(arm_names):
            raise ValueError(f"{self.name} contains duplicate arm names")
        referenced: set[str] = set()
        for arm in self.arms:
            group = groups.get(arm.joint_group)
            if group is None:
                raise ValueError(f"{self.name}.{arm.name} references unknown group {arm.joint_group!r}")
            if group.role != "arm":
                raise ValueError(f"{self.name}.{arm.name} group {group.name!r} is not an arm")
            referenced.add(group.name)
        for role, group_name in (("torso", self.torso_group), ("head", self.head_group)):
            if group_name is None:
                continue
            group = groups.get(group_name)
            if group is None:
                raise ValueError(f"{self.name} references unknown {role} group {group_name!r}")
            if group.role != role:
                raise ValueError(f"{self.name}.{role}_group {group.name!r} is not a {role}")
            referenced.add(group.name)
        unused = set(groups) - referenced
        if unused:
            raise ValueError(f"{self.name} declares unused joint groups: {sorted(unused)}")
        all_joints = [j.name for g in self.joint_groups for j in g.joints]
        for arm in self.arms:
            if arm.gripper is not None:
                all_joints.extend(arm.gripper.joint_names)
        if len(set(all_joints)) != len(all_joints):
            raise ValueError(f"{self.name} contains duplicate joint names across groups/grippers")

    def joint_group(self, name: str) -> JointGroupSpec:
        for group in self.joint_groups:
            if group.name == name:
                return group
        raise KeyError(f"unknown joint group: {name!r}")

    def arm(self, name: str) -> ArmSpec:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise KeyError(f"unknown arm: {name!r}")

    @property
    def arm_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.arms)

    @property
    def torso(self) -> JointGroupSpec | None:
        return None if self.torso_group is None else self.joint_group(self.torso_group)

    @property
    def head(self) -> JointGroupSpec | None:
        return None if self.head_group is None else self.joint_group(self.head_group)

    @property
    def action_dim(self) -> int:
        total = self.base.action_dim
        if self.torso is not None:
            total += len(self.torso.joints)
        for arm in self.arms:
            total += len(self.joint_group(arm.joint_group).joints)
            if arm.gripper is not None:
                total += arm.gripper.dof
        return total

    @property
    def controlled_joint_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.torso is not None:
            names.extend(self.torso.joint_names)
        for arm in self.arms:
            names.extend(self.joint_group(arm.joint_group).joint_names)
        return tuple(names)

    def meta(self, key: str, default: str = "") -> str:
        return dict(self.metadata).get(key, default)


@dataclass(frozen=True)
class GripperCommand:
    """Policy command for one gripper."""

    values: tuple[float, ...]
    mode: GripperMode = "position"

    def __post_init__(self) -> None:
        if self.mode not in ("position", "force", "velocity"):
            raise ValueError(f"unknown gripper command mode: {self.mode!r}")
        if not self.values:
            raise ValueError("gripper command must contain at least one value")
        if not all(math.isfinite(float(v)) for v in self.values):
            raise ValueError("gripper command contains a non-finite value")


@dataclass(frozen=True)
class JointGroupObservation:
    """Measured state for a torso/head group."""

    name: str
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    torque: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("joint group observation name must be non-empty")
        if len(self.position) != len(self.velocity):
            raise ValueError(f"{self.name} position/velocity length mismatch")
        if self.torque and len(self.torque) != len(self.position):
            raise ValueError(f"{self.name} torque length mismatch")
        if not all(math.isfinite(float(v)) for v in (*self.position, *self.velocity, *self.torque)):
            raise ValueError(f"{self.name} observation contains a non-finite value")


@dataclass(frozen=True)
class GripperObservation:
    """Measured gripper state. ``force`` is optional and may be a proxy."""

    position: tuple[float, ...]
    velocity: tuple[float, ...] = ()
    force: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.position:
            raise ValueError("gripper observation must contain at least one value")
        if self.velocity and len(self.velocity) != len(self.position):
            raise ValueError("gripper position/velocity length mismatch")
        if self.force and len(self.force) != len(self.position):
            raise ValueError("gripper position/force length mismatch")
        if not all(math.isfinite(float(v)) for v in (*self.position, *self.velocity, *self.force)):
            raise ValueError("gripper observation contains a non-finite value")


@dataclass(frozen=True)
class ArmObservation:
    """Measured state and task target for one arm."""

    name: str
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    joint_torque: tuple[float, ...] = ()
    ee_pose: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    ee_target: tuple[float, ...] | None = None
    target_mask: tuple[bool, bool] = (True, True)
    gripper: GripperObservation | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("arm observation name must be non-empty")
        if len(self.joint_position) != len(self.joint_velocity):
            raise ValueError(f"{self.name} position/velocity length mismatch")
        if self.joint_torque and len(self.joint_torque) != len(self.joint_position):
            raise ValueError(f"{self.name} torque length mismatch")
        if len(self.ee_pose) != EE_POSE_DIM:
            raise ValueError(f"{self.name}.ee_pose must have length {EE_POSE_DIM}")
        if self.ee_target is not None and len(self.ee_target) != EE_POSE_DIM:
            raise ValueError(f"{self.name}.ee_target must have length {EE_POSE_DIM}")
        if len(self.target_mask) != TARGET_MASK_DIM:
            raise ValueError(f"{self.name}.target_mask must have length {TARGET_MASK_DIM}")
        if self.ee_target is not None and not any(self.target_mask):
            raise ValueError(f"{self.name}.target_mask must enable position or orientation")
        values = (*self.joint_position, *self.joint_velocity, *self.joint_torque, *self.ee_pose)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f"{self.name} observation contains a non-finite value")
        if self.ee_target is not None and not all(math.isfinite(float(v)) for v in self.ee_target):
            raise ValueError(f"{self.name}.ee_target contains a non-finite value")


@dataclass(frozen=True, eq=False)
class WbcObservation:
    """Whole-body observation for a multi-arm robot."""

    base_pose: tuple[float, float, float]
    base_twist: tuple[float, ...]
    arms: tuple[ArmObservation, ...]
    occupancy: np.ndarray | None
    torso: JointGroupObservation | None = None
    head: JointGroupObservation | None = None
    failure_mode: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported observation schema version: {self.schema_version}")
        if len(self.base_pose) != BASE_POSE_DIM or not all(math.isfinite(float(v)) for v in self.base_pose):
            raise ValueError("base_pose must contain three finite values")
        if not all(math.isfinite(float(v)) for v in self.base_twist):
            raise ValueError("base_twist contains a non-finite value")
        if not self.arms:
            raise ValueError("observation must contain at least one arm")
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("duplicate arm observations")
        if self.occupancy is not None:
            occupancy = np.asarray(self.occupancy)
            if occupancy.ndim != 3:
                raise ValueError("occupancy must be a 3D array")

    def arm(self, name: str) -> ArmObservation:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise KeyError(f"unknown arm observation: {name!r}")

    def validate(self, profile: EmbodimentProfile) -> None:
        if {a.name for a in self.arms} != set(profile.arm_names):
            raise ValueError(
                f"observation arms {sorted(a.name for a in self.arms)} do not match profile {sorted(profile.arm_names)}"
            )
        _require_vector(self.base_twist, profile.base.action_dim, "base_twist")
        for arm_spec in profile.arms:
            arm = self.arm(arm_spec.name)
            group = profile.joint_group(arm_spec.joint_group)
            _require_vector(arm.joint_position, len(group.joints), f"{arm.name}.joint_position")
            _require_vector(arm.joint_velocity, len(group.joints), f"{arm.name}.joint_velocity")
            if arm.joint_torque:
                _require_vector(arm.joint_torque, len(group.joints), f"{arm.name}.joint_torque")
            if arm_spec.gripper is not None:
                if arm.gripper is None:
                    raise ValueError(f"{arm.name} is missing gripper observation")
                _require_vector(arm.gripper.position, arm_spec.gripper.dof, f"{arm.name}.gripper.position")
            elif arm.gripper is not None:
                raise ValueError(f"{arm.name} declares a gripper observation but the profile has no gripper")
        if profile.torso is None:
            if self.torso is not None:
                raise ValueError("observation has torso state but the profile has no torso")
        else:
            if self.torso is None:
                raise ValueError(f"observation is missing torso group {profile.torso.name!r}")
            if self.torso.name != profile.torso.name:
                raise ValueError(f"torso observation name {self.torso.name!r} != profile {profile.torso.name!r}")
            _require_vector(self.torso.position, len(profile.torso.joints), f"{self.torso.name}.position")
            _require_vector(self.torso.velocity, len(profile.torso.joints), f"{self.torso.name}.velocity")
            if self.torso.torque:
                _require_vector(self.torso.torque, len(profile.torso.joints), f"{self.torso.name}.torque")
        if self.head is not None:
            if profile.head is None:
                raise ValueError("observation has head state but the profile has no head")
            if self.head.name != profile.head.name:
                raise ValueError(f"head observation name {self.head.name!r} != profile {profile.head.name!r}")
            _require_vector(self.head.position, len(profile.head.joints), f"{self.head.name}.position")
            _require_vector(self.head.velocity, len(profile.head.joints), f"{self.head.name}.velocity")
            if self.head.torque:
                _require_vector(self.head.torque, len(profile.head.joints), f"{self.head.name}.torque")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_pose": list(self.base_pose),
            "base_twist": list(self.base_twist),
            "arms": [
                {
                    "name": a.name,
                    "joint_position": list(a.joint_position),
                    "joint_velocity": list(a.joint_velocity),
                    "joint_torque": list(a.joint_torque),
                    "ee_pose": list(a.ee_pose),
                    "ee_target": None if a.ee_target is None else list(a.ee_target),
                    "target_mask": list(a.target_mask),
                    "gripper": None
                    if a.gripper is None
                    else {
                        "position": list(a.gripper.position),
                        "velocity": list(a.gripper.velocity),
                        "force": list(a.gripper.force),
                    },
                }
                for a in self.arms
            ],
            "torso": None
            if self.torso is None
            else {
                "name": self.torso.name,
                "position": list(self.torso.position),
                "velocity": list(self.torso.velocity),
                "torque": list(self.torso.torque),
            },
            "head": None
            if self.head is None
            else {
                "name": self.head.name,
                "position": list(self.head.position),
                "velocity": list(self.head.velocity),
                "torque": list(self.head.torque),
            },
            "occupancy": None if self.occupancy is None else np.asarray(self.occupancy, dtype=np.uint8).tolist(),
            "failure_mode": self.failure_mode,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WbcObservation:
        arms = tuple(
            ArmObservation(
                name=str(a["name"]),
                joint_position=tuple(float(x) for x in a["joint_position"]),
                joint_velocity=tuple(float(x) for x in a["joint_velocity"]),
                joint_torque=tuple(float(x) for x in a.get("joint_torque", ())),
                ee_pose=tuple(float(x) for x in a.get("ee_pose", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))),
                ee_target=None if a.get("ee_target") is None else tuple(float(x) for x in a["ee_target"]),
                target_mask=tuple(bool(x) for x in a.get("target_mask", (True, True))),
                gripper=None
                if a.get("gripper") is None
                else GripperObservation(
                    position=tuple(float(x) for x in a["gripper"]["position"]),
                    velocity=tuple(float(x) for x in a["gripper"].get("velocity", ())),
                    force=tuple(float(x) for x in a["gripper"].get("force", ())),
                ),
            )
            for a in d["arms"]
        )

        def group(key: str) -> JointGroupObservation | None:
            raw = d.get(key)
            if raw is None:
                return None
            return JointGroupObservation(
                name=str(raw["name"]),
                position=tuple(float(x) for x in raw["position"]),
                velocity=tuple(float(x) for x in raw["velocity"]),
                torque=tuple(float(x) for x in raw.get("torque", ())),
            )

        occupancy = d.get("occupancy")
        return cls(
            schema_version=int(d.get("schema_version", 1)),
            base_pose=tuple(float(x) for x in d["base_pose"]),
            base_twist=tuple(float(x) for x in d["base_twist"]),
            arms=arms,
            occupancy=None if occupancy is None else np.asarray(occupancy, dtype=np.uint8),
            torso=group("torso"),
            head=group("head"),
            failure_mode=str(d.get("failure_mode", "")),
        )


@dataclass(frozen=True)
class ArmAction:
    """Torque and optional gripper command for one arm."""

    name: str
    joint_torques: tuple[float, ...]
    gripper: GripperCommand | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("arm action name must be non-empty")
        if not self.joint_torques:
            raise ValueError(f"{self.name} must contain at least one joint torque")
        if not all(math.isfinite(float(v)) for v in self.joint_torques):
            raise ValueError(f"{self.name} joint torques contain a non-finite value")


@dataclass(frozen=True)
class WbcAction:
    """Whole-body action: base velocity + per-arm joint torques + gripper commands."""

    base_velocity: tuple[float, ...]
    arms: tuple[ArmAction, ...]
    torso_torques: tuple[float, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported action schema version: {self.schema_version}")
        if not self.arms:
            raise ValueError("action must contain at least one arm")
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("duplicate arm actions")
        if not all(math.isfinite(float(v)) for v in self.base_velocity):
            raise ValueError("base_velocity contains a non-finite value")
        if self.torso_torques and not all(math.isfinite(float(v)) for v in self.torso_torques):
            raise ValueError("torso_torques contains a non-finite value")

    def arm(self, name: str) -> ArmAction:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise KeyError(f"unknown arm action: {name!r}")

    def validate(self, profile: EmbodimentProfile) -> None:
        if {a.name for a in self.arms} != set(profile.arm_names):
            raise ValueError(
                f"action arms {sorted(a.name for a in self.arms)} do not match profile {sorted(profile.arm_names)}"
            )
        _require_vector(self.base_velocity, profile.base.action_dim, "base_velocity")
        if profile.torso is None:
            if self.torso_torques:
                raise ValueError("action has torso torques but the profile has no torso")
        else:
            _require_vector(self.torso_torques, len(profile.torso.joints), "torso_torques")
        for arm_spec in profile.arms:
            action = self.arm(arm_spec.name)
            group = profile.joint_group(arm_spec.joint_group)
            _require_vector(action.joint_torques, len(group.joints), f"{arm_spec.name}.joint_torques")
            if arm_spec.gripper is not None:
                if action.gripper is None:
                    raise ValueError(f"{arm_spec.name} is missing a gripper command")
                _require_vector(action.gripper.values, arm_spec.gripper.dof, f"{arm_spec.name}.gripper.values")
                if action.gripper.mode != arm_spec.gripper.command_mode:
                    raise ValueError(
                        f"{arm_spec.name} gripper mode {action.gripper.mode!r} != profile mode "
                        f"{arm_spec.gripper.command_mode!r}"
                    )
            elif action.gripper is not None:
                raise ValueError(f"{arm_spec.name} declares a gripper command but the profile has no gripper")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_velocity": list(self.base_velocity),
            "torso_torques": list(self.torso_torques),
            "arms": [
                {
                    "name": a.name,
                    "joint_torques": list(a.joint_torques),
                    "gripper": None
                    if a.gripper is None
                    else {"values": list(a.gripper.values), "mode": a.gripper.mode},
                }
                for a in self.arms
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WbcAction:
        return cls(
            schema_version=int(d.get("schema_version", 1)),
            base_velocity=tuple(float(x) for x in d["base_velocity"]),
            torso_torques=tuple(float(x) for x in d.get("torso_torques", ())),
            arms=tuple(
                ArmAction(
                    name=str(a["name"]),
                    joint_torques=tuple(float(x) for x in a["joint_torques"]),
                    gripper=None
                    if a.get("gripper") is None
                    else GripperCommand(
                        values=tuple(float(x) for x in a["gripper"]["values"]),
                        mode=cast(GripperMode, a["gripper"].get("mode", "position")),
                    ),
                )
                for a in d["arms"]
            ),
        )


def action_vector(action: WbcAction, profile: EmbodimentProfile) -> np.ndarray:
    """Flatten a multi-arm action in profile order (base, torso, arm torque/gripper)."""
    action.validate(profile)
    values: list[float] = list(action.base_velocity)
    if profile.torso is not None:
        values.extend(action.torso_torques)
    for arm_spec in profile.arms:
        values.extend(action.arm(arm_spec.name).joint_torques)
        if arm_spec.gripper is not None:
            values.extend(action.arm(arm_spec.name).gripper.values)  # type: ignore[union-attr]
    return np.asarray(values, dtype=np.float64)


def action_from_vector(vector: np.ndarray | tuple[float, ...], profile: EmbodimentProfile) -> WbcAction:
    """Rebuild a multi-arm action from :func:`action_vector`."""
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.size != profile.action_dim:
        raise ValueError(f"action vector has length {values.size}, expected {profile.action_dim}")
    if not np.isfinite(values).all():
        raise ValueError("action vector contains a non-finite value")
    offset = 0
    base = tuple(float(x) for x in values[: profile.base.action_dim])
    offset += profile.base.action_dim
    torso: tuple[float, ...] = ()
    if profile.torso is not None:
        n = len(profile.torso.joints)
        torso = tuple(float(x) for x in values[offset : offset + n])
        offset += n
    arms: list[ArmAction] = []
    for arm_spec in profile.arms:
        group = profile.joint_group(arm_spec.joint_group)
        n = len(group.joints)
        torques = tuple(float(x) for x in values[offset : offset + n])
        offset += n
        gripper = None
        if arm_spec.gripper is not None:
            m = arm_spec.gripper.dof
            gripper = GripperCommand(
                values=tuple(float(x) for x in values[offset : offset + m]),
                mode=arm_spec.gripper.command_mode,
            )
            offset += m
        arms.append(ArmAction(name=arm_spec.name, joint_torques=torques, gripper=gripper))
    return WbcAction(base_velocity=base, arms=tuple(arms), torso_torques=torso)


def clamp_action(action: WbcAction, profile: EmbodimentProfile) -> WbcAction:
    """Clip a policy action to profile-level base/torque/gripper limits.

    This is the shared action-boundary safety clamp. Every actuator adapter
    should call it before mapping to hardware so a raw policy output cannot
    exceed the robot's declared envelope. Actuator-specific limits (current,
    position lead, thermal derating) remain the adapter/executor's job.
    """
    action.validate(profile)
    base_velocity = tuple(
        float(np.clip(value, -limit, limit))
        for value, limit in zip(action.base_velocity, profile.base.velocity_limits, strict=True)
    )
    torso = (
        tuple(
            float(np.clip(torque, -joint.torque_limit, joint.torque_limit))
            for joint, torque in zip(profile.torso.joints, action.torso_torques, strict=True)
        )
        if profile.torso is not None
        else ()
    )
    arms = []
    for arm_spec in profile.arms:
        arm_action = action.arm(arm_spec.name)
        group = profile.joint_group(arm_spec.joint_group)
        torques = tuple(
            float(np.clip(torque, -joint.torque_limit, joint.torque_limit))
            for joint, torque in zip(group.joints, arm_action.joint_torques, strict=True)
        )
        gripper = arm_action.gripper
        if gripper is not None and arm_spec.gripper is not None and arm_spec.gripper.command_limits is not None:
            values = tuple(
                float(np.clip(value, low, high))
                for value, (low, high) in zip(gripper.values, arm_spec.gripper.command_limits, strict=True)
            )
            gripper = replace(gripper, values=values)
        arms.append(replace(arm_action, joint_torques=torques, gripper=gripper))
    return replace(action, base_velocity=base_velocity, torso_torques=torso, arms=tuple(arms))


@dataclass(frozen=True)
class ReachSpec:
    """Reward weights and tolerances for one or more active reaching targets."""

    pos_weight: float = 1.0
    quat_weight: float = 0.25
    collision_penalty: float = 10.0
    action_penalty: float = 0.01
    jerk_penalty: float = 0.0
    success_bonus: float = 20.0
    pos_tolerance_m: float = 0.02
    quat_tolerance: float = 0.05
    arm_weights: tuple[float, ...] = ()
    success_mode: Literal["all", "any"] = "all"

    def __post_init__(self) -> None:
        for value, label, positive in (
            (self.pos_weight, "pos_weight", False),
            (self.quat_weight, "quat_weight", False),
            (self.collision_penalty, "collision_penalty", False),
            (self.action_penalty, "action_penalty", False),
            (self.jerk_penalty, "jerk_penalty", False),
            (self.success_bonus, "success_bonus", False),
            (self.pos_tolerance_m, "pos_tolerance_m", True),
            (self.quat_tolerance, "quat_tolerance", True),
        ):
            _require_finite(value, label, positive=positive, nonnegative=not positive)
        if self.success_mode not in ("all", "any"):
            raise ValueError(f"unknown success_mode: {self.success_mode!r}")
        for i, weight in enumerate(self.arm_weights):
            _require_finite(weight, f"arm_weights[{i}]", nonnegative=True)


def _ee_pos_err(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return float(np.linalg.norm(np.asarray(a[:3], dtype=np.float64) - np.asarray(b[:3], dtype=np.float64)))


def _ee_quat_err(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    qa = np.asarray(a[3:7], dtype=np.float64)
    qb = np.asarray(b[3:7], dtype=np.float64)
    qa = qa / (np.linalg.norm(qa) + 1e-12)
    qb = qb / (np.linalg.norm(qb) + 1e-12)
    return float(1.0 - abs(float(np.dot(qa, qb))))


def reach_reward(
    action: WbcAction,
    next_obs: WbcObservation,
    *,
    profile: EmbodimentProfile,
    collision: bool = False,
    prev_action: WbcAction | None = None,
    spec: ReachSpec = ReachSpec(),
) -> tuple[float, dict[str, float]]:
    """Dense reward for a one- or multi-arm reaching step.

    Active arms are those with ``ee_target is not None``. Position/orientation
    terms honor each arm's ``target_mask``. ``success_mode`` controls whether all
    or any active arm must satisfy the tolerance. Action and jerk penalties use
    per-joint torque limits, so values are comparable across robots.
    """
    action.validate(profile)
    next_obs.validate(profile)
    weights = spec.arm_weights or (1.0,) * len(profile.arms)
    if len(weights) != len(profile.arms):
        raise ValueError(f"arm_weights must have length {len(profile.arms)}")

    pos_terms: dict[str, float] = {}
    quat_terms: dict[str, float] = {}
    successes: dict[str, bool] = {}
    active: list[str] = []
    for arm_spec in profile.arms:
        obs = next_obs.arm(arm_spec.name)
        if obs.ee_target is None:
            continue
        active.append(arm_spec.name)
        pos_err = _ee_pos_err(obs.ee_pose, obs.ee_target)
        quat_err = _ee_quat_err(obs.ee_pose, obs.ee_target)
        if not obs.target_mask[0]:
            pos_err = 0.0
        if not obs.target_mask[1]:
            quat_err = 0.0
        pos_terms[arm_spec.name] = pos_err
        quat_terms[arm_spec.name] = quat_err
        successes[arm_spec.name] = pos_err <= spec.pos_tolerance_m and quat_err <= spec.quat_tolerance
    if not active:
        raise ValueError("reach_reward requires at least one arm with ee_target")

    weight_sum = sum(weights[profile.arm_names.index(name)] for name in active)
    if weight_sum <= 0:
        raise ValueError("active arm_weights must contain a positive weight")
    pos_err = sum(weights[profile.arm_names.index(name)] * pos_terms[name] for name in active) / weight_sum
    quat_err = sum(weights[profile.arm_names.index(name)] * quat_terms[name] for name in active) / weight_sum

    action_norm_sq = 0.0
    for i, value in enumerate(action.base_velocity):
        action_norm_sq += (value / profile.base.velocity_limits[i]) ** 2
    if profile.torso is not None:
        for joint, tau in zip(profile.torso.joints, action.torso_torques, strict=True):
            action_norm_sq += (tau / joint.torque_limit) ** 2
    for arm_spec in profile.arms:
        group = profile.joint_group(arm_spec.joint_group)
        arm_action = action.arm(arm_spec.name)
        for joint, tau in zip(group.joints, arm_action.joint_torques, strict=True):
            action_norm_sq += (tau / joint.torque_limit) ** 2

    jerk = 0.0
    if prev_action is not None:
        prev_action.validate(profile)
        if {a.name for a in prev_action.arms} != set(profile.arm_names):
            raise ValueError("prev_action arms do not match the profile")
        if profile.torso is not None:
            for joint, tau, prev in zip(
                profile.torso.joints, action.torso_torques, prev_action.torso_torques, strict=True
            ):
                jerk += ((tau - prev) / joint.torque_limit) ** 2
        for arm_spec in profile.arms:
            group = profile.joint_group(arm_spec.joint_group)
            arm_action = action.arm(arm_spec.name)
            prev_arm = prev_action.arm(arm_spec.name)
            for joint, tau, prev in zip(group.joints, arm_action.joint_torques, prev_arm.joint_torques, strict=True):
                jerk += ((tau - prev) / joint.torque_limit) ** 2

    success = all(successes.values()) if spec.success_mode == "all" else any(successes.values())
    reward = -spec.pos_weight * pos_err - spec.quat_weight * quat_err - spec.action_penalty * action_norm_sq
    reward -= spec.jerk_penalty * jerk
    if collision:
        reward -= spec.collision_penalty
    if success:
        reward += spec.success_bonus

    components: dict[str, float] = {
        "pos_err": pos_err,
        "quat_err": quat_err,
        "action_norm_sq": action_norm_sq,
        "jerk": jerk,
        "collision": float(collision),
        "success": float(success),
    }
    for name in active:
        components[f"arm.{name}.pos_err"] = pos_terms[name]
        components[f"arm.{name}.quat_err"] = quat_terms[name]
        components[f"arm.{name}.success"] = float(successes[name])
    return float(reward), components


__all__ = [
    "ArmAction",
    "ArmObservation",
    "ArmSpec",
    "BaseSpec",
    "EE_POSE_DIM",
    "EmbodimentProfile",
    "GripperCommand",
    "GripperObservation",
    "GripperSpec",
    "JointGroupObservation",
    "JointGroupSpec",
    "JointSpec",
    "ReachSpec",
    "SCHEMA_VERSION",
    "ControlMode",
    "WbcAction",
    "WbcObservation",
    "action_from_vector",
    "action_vector",
    "clamp_action",
    "reach_reward",
]

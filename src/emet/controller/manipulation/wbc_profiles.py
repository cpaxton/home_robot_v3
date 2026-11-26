# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Example multi-arm embodiment profiles for the WBC v2 contract.

These profiles are schema examples, not calibrated hardware limits. They use
published joint ranges where available and clearly-marked placeholder torque /
correspondence values elsewhere. Replace the actuator fields with bench
measurements before deploying a policy.

The examples cover the four current families:

- ``example_mars_profile``: one 5-DOF arm + gripper, planar base.
- ``example_sourccey_profile``: lift + two 5-DOF arms + grippers, planar base.
- ``example_rby1_profile``: 4-DOF torso + two 6-DOF arms + two-finger grippers,
  holonomic base.
- ``example_stretch_profile``: one telescoping 8-DOF arm + gripper, planar base.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from emet.controller.manipulation.wbc_contract import (
    ArmSpec,
    BaseSpec,
    ControlMode,
    EmbodimentProfile,
    GripperSpec,
    JointGroupSpec,
    JointKind,
    JointSpec,
)

_EXAMPLE = "example_placeholder"


def _joint(
    name: str,
    *,
    position_limit: tuple[float, float],
    velocity_limit: float,
    torque_limit: float,
    kp_eff: float,
    max_lead: float,
    kind: JointKind = "revolute",
    control_mode: ControlMode = "position",
) -> JointSpec:
    return JointSpec(
        name=name,
        position_limit=position_limit,
        velocity_limit=velocity_limit,
        torque_limit=torque_limit,
        kind=kind,
        control_mode=control_mode,
        kp_eff=kp_eff,
        max_lead=max_lead,
    )


def example_mars_profile() -> EmbodimentProfile:
    """Innate Mars: planar base, one 5-DOF arm, one gripper.

    Joint ranges match the published Maurice/Innate arm limits used by
    ``arm_motion.py``. Torque limits and ``kp_eff`` are placeholders.
    """
    arm = JointGroupSpec(
        name="primary_arm",
        role="arm",
        joints=(
            _joint(
                "joint1",
                position_limit=(-1.5708, 1.5708),
                velocity_limit=2.0,
                torque_limit=1.5,
                kp_eff=1000.0,
                max_lead=0.05,
            ),
            _joint(
                "joint2",
                position_limit=(-1.5708, 1.22),
                velocity_limit=2.0,
                torque_limit=1.5,
                kp_eff=1000.0,
                max_lead=0.05,
            ),
            _joint(
                "joint3",
                position_limit=(-1.5708, 1.7453),
                velocity_limit=2.0,
                torque_limit=1.5,
                kp_eff=1000.0,
                max_lead=0.05,
            ),
            _joint(
                "joint4",
                position_limit=(-1.9199, 1.7453),
                velocity_limit=2.0,
                torque_limit=1.5,
                kp_eff=1000.0,
                max_lead=0.05,
                control_mode="current_position",
            ),
            _joint(
                "joint5",
                position_limit=(-1.5708, 1.5708),
                velocity_limit=2.0,
                torque_limit=1.5,
                kp_eff=1000.0,
                max_lead=0.05,
                control_mode="current_position",
            ),
        ),
    )
    return EmbodimentProfile(
        name="example_innate_mars",
        base=BaseSpec(name="base", kind="planar", velocity_limits=(0.5, 1.0)),
        joint_groups=(arm,),
        arms=(
            ArmSpec(
                name="primary",
                joint_group="primary_arm",
                ee_link="ee_link",
                gripper=GripperSpec(
                    name="primary_gripper",
                    joint_names=("joint6",),
                    command_limits=((-0.6, 0.85),),
                ),
            ),
        ),
        metadata=(("limits_source", _EXAMPLE), ("robot", "innate_mars")),
    )


def mars_sim_profile() -> EmbodimentProfile:
    """Mars profile matched to the retained MuJoCo position actuators.

    This is a simulator profile, not a hardware identification. The example
    profile intentionally keeps placeholder actuator fields; training against
    MuJoCo must use the plant's actual ``kp``/``kv``/effort limits or the
    torque-to-position-lead mapping is wrong by more than an order of magnitude.
    """
    from emet.rl.manipulation.profiles import PROFILES
    from emet.rl.profile import MARS

    base = example_mars_profile()
    groups = []
    for group in base.joint_groups:
        joints = tuple(
            replace(
                joint,
                kp_eff=MARS.arm_kp,
                kd_eff=MARS.arm_kv,
                torque_limit=MARS.arm_effort_limit,
                velocity_limit=MARS.max_joint_speed,
                max_lead=MARS.max_target_lead,
            )
            for joint in group.joints
        )
        groups.append(replace(group, joints=joints))
    metadata = dict(base.metadata)
    metadata.update(
        {
            "limits_source": "mujoco_position_servo",
            "actuator_model": "retained_mujoco_kp80_kv4_effort2",
        }
    )
    # World-frame EE goals let the sampled base goal actually assist reach.
    grip_open = PROFILES["innate_mars"].gripper_open
    arms = tuple(
        replace(
            arm,
            target_frame="world",
            gripper=(
                arm.gripper if arm.gripper is None else replace(arm.gripper, command_limits=((-grip_open, grip_open),))
            ),
        )
        for arm in base.arms
    )
    return replace(
        base,
        base=replace(base.base, velocity_limits=(MARS.max_v, MARS.max_w)),
        joint_groups=tuple(groups),
        arms=arms,
        metadata=tuple(metadata.items()),
    )


def stretch_sim_profile() -> EmbodimentProfile:
    """Stretch 3 profile matched to the retained MuJoCo position/tendon actuators.

    The contract uses the controller's five motion axes (``lift``, virtual
    telescope ``arm``, and three wrist joints) plus the gripper. The virtual
    ``arm`` position is the sum of the four telescope joints in the backend.
    """
    joints = (
        JointSpec(
            "lift",
            (0.0, 1.1),
            0.1,
            70.0,
            kind="prismatic",
            kp_eff=400.0,
            kd_eff=100.0,
            max_lead=0.01,
        ),
        JointSpec(
            "arm",
            (0.0, 0.52),
            0.1,
            56.0,
            kind="prismatic",
            kp_eff=150.0,
            kd_eff=10.0,
            max_lead=0.01,
        ),
        JointSpec("wrist_yaw", (-1.39, 4.42), 0.6, 20.0, kp_eff=20.0, kd_eff=0.0, max_lead=0.05),
        JointSpec("wrist_pitch", (-1.57, 0.56), 0.6, 50.0, kp_eff=50.0, kd_eff=0.0, max_lead=0.05),
        JointSpec("wrist_roll", (-3.14, 3.14), 0.6, 20.0, kp_eff=20.0, kd_eff=0.0, max_lead=0.05),
    )
    arm = JointGroupSpec(name="stretch_arm", role="arm", joints=joints)
    return EmbodimentProfile(
        name="stretch_sim",
        base=BaseSpec(name="base", kind="planar", velocity_limits=(0.12, 0.5)),
        joint_groups=(arm,),
        arms=(
            ArmSpec(
                name="primary",
                joint_group="stretch_arm",
                ee_link="link_grasp_center",
                target_frame="world",
                gripper=GripperSpec(
                    name="primary_gripper",
                    joint_names=("joint_gripper_slide",),
                    command_limits=((-0.02, 0.04),),
                ),
            ),
        ),
        metadata=(("limits_source", "mujoco_servo"), ("robot", "stretch")),
    )


def sourccey_sim_profile() -> EmbodimentProfile:
    """Sourccey Mk.V profile matched to the packaged MuJoCo MJCF."""
    lift = JointGroupSpec(
        name="lift",
        role="torso",
        joints=(
            JointSpec(
                "lift",
                (-0.315, -0.0142),
                0.1,
                3000.0,
                kind="prismatic",
                kp_eff=30000.0,
                kd_eff=3000.0,
                max_lead=0.005,
            ),
        ),
    )

    limits = {
        "left": (
            (-1.570796, 1.570796),
            (-2.094395, 2.094395),
            (-1.570796, 1.570796),
            (-1.570796, 1.570796),
            (-3.141593, 3.141593),
        ),
        "right": (
            (-1.570796, 1.570796),
            (-2.094395, 1.919862),
            (-1.570796, 1.570796),
            (-1.570796, 1.570796),
            (-3.141593, 3.141593),
        ),
    }
    suffixes = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

    def arm(side: str) -> JointGroupSpec:
        return JointGroupSpec(
            name=f"{side}_arm",
            role="arm",
            joints=tuple(
                JointSpec(
                    f"{side}_{suffix}",
                    limits[side][index],
                    1.0,
                    100.0,
                    kp_eff=500.0,
                    kd_eff=35.0,
                    max_lead=0.05,
                )
                for index, suffix in enumerate(suffixes)
            ),
        )

    return EmbodimentProfile(
        name="sourccey_sim",
        base=BaseSpec(name="base", kind="planar", velocity_limits=(0.3, 0.8)),
        joint_groups=(lift, arm("left"), arm("right")),
        arms=(
            ArmSpec(
                name="left",
                joint_group="left_arm",
                ee_link="Gripper_Base_v1_1",
                target_frame="world",
                gripper=GripperSpec(
                    name="left_gripper",
                    joint_names=("left_gripper",),
                    command_limits=((-0.087266, 1.047198),),
                ),
            ),
            ArmSpec(
                name="right",
                joint_group="right_arm",
                ee_link="Gripper_Base_v1_2",
                target_frame="world",
                gripper=GripperSpec(
                    name="right_gripper",
                    joint_names=("right_gripper",),
                    command_limits=((-0.087266, 1.047198),),
                ),
            ),
        ),
        torso_group="lift",
        metadata=(("limits_source", "mujoco_servo"), ("robot", "sourccey")),
    )


def rby1_sim_profile() -> EmbodimentProfile:
    """RB-Y1 / Galaxea R1 proxy profile matched to the packaged MJCF.

    The base is holonomic in the contract. The direct-MuJoCo plant must map
    ``(vx, vy, yaw_rate)`` through the three steer/wheel modules; the RoboCasa
    adapter must use the same profile and base command semantics.
    """
    torso = JointGroupSpec(
        name="torso",
        role="torso",
        joints=(
            JointSpec("torso_joint1", (-1.1345, 1.8326), 1.0, 100.0, kp_eff=150.0, kd_eff=50.0, max_lead=0.05),
            JointSpec("torso_joint2", (-2.7925, 2.5307), 1.0, 100.0, kp_eff=150.0, kd_eff=50.0, max_lead=0.05),
            JointSpec("torso_joint3", (-1.8326, 1.5708), 1.0, 100.0, kp_eff=150.0, kd_eff=50.0, max_lead=0.05),
            JointSpec("torso_joint4", (-3.0543, 3.0543), 1.0, 100.0, kp_eff=150.0, kd_eff=50.0, max_lead=0.05),
        ),
    )
    arm_limits = (
        (-2.8798, 2.8798),
        (0.0, 3.2289),
        (-3.3161, 0.0),
        (-2.8798, 2.8798),
        (-1.6581, 1.6581),
        (-2.8798, 2.8798),
    )
    arm_torque = (40.0, 40.0, 40.0, 7.0, 7.0, 7.0)
    arm_kp = (100.0, 100.0, 100.0, 40.0, 40.0, 40.0)
    arm_kd = (30.0, 30.0, 30.0, 10.0, 10.0, 10.0)

    def arm(side: str) -> JointGroupSpec:
        return JointGroupSpec(
            name=f"{side}_arm",
            role="arm",
            joints=tuple(
                JointSpec(
                    f"{side}_arm_joint{index + 1}",
                    arm_limits[index],
                    1.0,
                    arm_torque[index],
                    kp_eff=arm_kp[index],
                    kd_eff=arm_kd[index],
                    max_lead=0.05,
                )
                for index in range(6)
            ),
        )

    return EmbodimentProfile(
        name="rby1_sim",
        base=BaseSpec(name="base", kind="holonomic", velocity_limits=(0.4, 0.4, 0.8)),
        joint_groups=(torso, arm("left"), arm("right")),
        arms=(
            ArmSpec(
                name="left",
                joint_group="left_arm",
                ee_link="left_arm_link6",
                target_frame="world",
                gripper=GripperSpec(
                    name="left_gripper",
                    joint_names=("left_gripper_finger_joint1", "left_gripper_finger_joint2"),
                    command_limits=((0.0, 0.05), (0.0, 0.05)),
                ),
            ),
            ArmSpec(
                name="right",
                joint_group="right_arm",
                ee_link="right_arm_link6",
                target_frame="world",
                gripper=GripperSpec(
                    name="right_gripper",
                    joint_names=("right_gripper_finger_joint1", "right_gripper_finger_joint2"),
                    command_limits=((0.0, 0.05), (0.0, 0.05)),
                ),
            ),
        ),
        torso_group="torso",
        metadata=(("limits_source", "mujoco_servo"), ("robot", "rby1"), ("sim_model", "galaxea_r1_proxy")),
    )


def pandaomron_sim_profile() -> EmbodimentProfile:
    """PandaOmron / Omron LD-60 profile used by RoboCasa.

    RoboCasa composes the Omron holonomic base, a 1-DOF torso height, one
    7-DOF Panda arm, and a two-finger gripper. Joint names are canonical here;
    the RoboCasa adapter is responsible for the ``robot0_`` prefixes.
    """
    torso = JointGroupSpec(
        name="torso",
        role="torso",
        joints=(
            JointSpec(
                "joint_torso_height",
                (0.0, 0.34),
                0.2,
                2000.0,
                kind="prismatic",
                kp_eff=1000.0,
                kd_eff=200.0,
                max_lead=0.02,
            ),
        ),
    )
    panda_limits = (
        (-2.8973, 2.8973),
        (-1.7628, 1.7628),
        (-2.8973, 2.8973),
        (-3.0718, -0.0698),
        (-2.8973, 2.8973),
        (-0.0175, 3.7525),
        (-2.8973, 2.8973),
    )
    panda_torque = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
    arm = JointGroupSpec(
        name="panda_arm",
        role="arm",
        joints=tuple(
            JointSpec(
                f"joint{index + 1}",
                panda_limits[index],
                2.0,
                panda_torque[index],
                kp_eff=1000.0,
                kd_eff=200.0,
                max_lead=0.05,
            )
            for index in range(7)
        ),
    )
    return EmbodimentProfile(
        name="pandaomron_sim",
        base=BaseSpec(name="base", kind="holonomic", velocity_limits=(0.5, 0.5, 1.0)),
        joint_groups=(torso, arm),
        arms=(
            ArmSpec(
                name="right",
                joint_group="panda_arm",
                ee_link="right_hand",
                target_frame="world",
                gripper=GripperSpec(
                    name="right_gripper",
                    joint_names=("panda_finger_joint1", "panda_finger_joint2"),
                    command_limits=((0.0, 0.04), (0.0, 0.04)),
                ),
            ),
        ),
        torso_group="torso",
        metadata=(
            ("limits_source", "robocasa_robot"),
            ("robot", "pandaomron"),
            ("sim_model", "omron_ld60_panda"),
        ),
    )


def example_sourccey_profile() -> EmbodimentProfile:
    """Sourccey: planar base, lift torso, two 5-DOF arms, two grippers."""
    lift = JointGroupSpec(
        name="lift",
        role="torso",
        joints=(
            _joint(
                "lift",
                position_limit=(0.0, 0.20),
                velocity_limit=0.1,
                torque_limit=20.0,
                kp_eff=200.0,
                max_lead=0.01,
                kind="prismatic",
            ),
        ),
    )

    def arm(side: str) -> JointGroupSpec:
        return JointGroupSpec(
            name=f"{side}_arm",
            role="arm",
            joints=tuple(
                _joint(
                    f"{side}_{suffix}",
                    position_limit=(-math.pi, math.pi),
                    velocity_limit=1.0,
                    torque_limit=10.0,
                    kp_eff=50.0,
                    max_lead=0.05,
                )
                for suffix in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
            ),
        )

    return EmbodimentProfile(
        name="example_sourccey",
        base=BaseSpec(name="base", kind="planar", velocity_limits=(0.5, 1.0)),
        joint_groups=(lift, arm("left"), arm("right")),
        arms=(
            ArmSpec(
                name="left",
                joint_group="left_arm",
                ee_link="left_Gripper-Finger",
                gripper=GripperSpec(
                    name="left_gripper",
                    joint_names=("left_gripper",),
                    command_limits=((0.0, 1.0),),
                ),
            ),
            ArmSpec(
                name="right",
                joint_group="right_arm",
                ee_link="right_Gripper-Finger",
                gripper=GripperSpec(
                    name="right_gripper",
                    joint_names=("right_gripper",),
                    command_limits=((0.0, 1.0),),
                ),
            ),
        ),
        torso_group="lift",
        metadata=(("limits_source", _EXAMPLE), ("robot", "sourccey")),
    )


def example_rby1_profile() -> EmbodimentProfile:
    """RB-Y1 proxy: holonomic base, 4-DOF torso, two 6-DOF arms, two-finger grippers."""
    torso = JointGroupSpec(
        name="torso",
        role="torso",
        joints=tuple(
            _joint(
                f"torso_joint{i}",
                position_limit=(-math.pi, math.pi),
                velocity_limit=1.0,
                torque_limit=100.0,
                kp_eff=1000.0,
                max_lead=0.05,
            )
            for i in range(1, 5)
        ),
    )

    def arm(side: str) -> JointGroupSpec:
        return JointGroupSpec(
            name=f"{side}_arm",
            role="arm",
            joints=tuple(
                _joint(
                    f"{side}_arm_joint{i}",
                    position_limit=(-math.pi, math.pi),
                    velocity_limit=1.0,
                    torque_limit=40.0,
                    kp_eff=500.0,
                    max_lead=0.05,
                )
                for i in range(1, 7)
            ),
        )

    return EmbodimentProfile(
        name="example_rby1",
        base=BaseSpec(name="base", kind="holonomic", velocity_limits=(0.5, 0.5, 1.0)),
        joint_groups=(torso, arm("left"), arm("right")),
        arms=(
            ArmSpec(
                name="left",
                joint_group="left_arm",
                ee_link="left_ee",
                gripper=GripperSpec(
                    name="left_gripper",
                    joint_names=("left_gripper_finger_joint1", "left_gripper_finger_joint2"),
                    command_limits=((0.0, 0.05), (0.0, 0.05)),
                ),
            ),
            ArmSpec(
                name="right",
                joint_group="right_arm",
                ee_link="right_ee",
                gripper=GripperSpec(
                    name="right_gripper",
                    joint_names=("right_gripper_finger_joint1", "right_gripper_finger_joint2"),
                    command_limits=((0.0, 0.05), (0.0, 0.05)),
                ),
            ),
        ),
        torso_group="torso",
        metadata=(("limits_source", _EXAMPLE), ("robot", "rby1")),
    )


def example_stretch_profile() -> EmbodimentProfile:
    """Stretch 3: planar base, one telescoping 8-DOF arm, one gripper.

    The lift/telescope/wrist chain is represented as one arm group so the policy
    has one EE target; the gripper is a separate action channel.
    """
    joints = (
        _joint(
            "joint_lift",
            position_limit=(0.0, 1.1),
            velocity_limit=0.2,
            torque_limit=30.0,
            kp_eff=200.0,
            max_lead=0.01,
            kind="prismatic",
        ),
        _joint(
            "joint_arm_l0",
            position_limit=(0.0, 0.13),
            velocity_limit=0.2,
            torque_limit=10.0,
            kp_eff=200.0,
            max_lead=0.01,
            kind="prismatic",
        ),
        _joint(
            "joint_arm_l1",
            position_limit=(0.0, 0.13),
            velocity_limit=0.2,
            torque_limit=10.0,
            kp_eff=200.0,
            max_lead=0.01,
            kind="prismatic",
        ),
        _joint(
            "joint_arm_l2",
            position_limit=(0.0, 0.13),
            velocity_limit=0.2,
            torque_limit=10.0,
            kp_eff=200.0,
            max_lead=0.01,
            kind="prismatic",
        ),
        _joint(
            "joint_arm_l3",
            position_limit=(0.0, 0.13),
            velocity_limit=0.2,
            torque_limit=10.0,
            kp_eff=200.0,
            max_lead=0.01,
            kind="prismatic",
        ),
        _joint(
            "joint_wrist_yaw",
            position_limit=(-math.pi, math.pi),
            velocity_limit=1.0,
            torque_limit=2.0,
            kp_eff=100.0,
            max_lead=0.05,
        ),
        _joint(
            "joint_wrist_pitch",
            position_limit=(-math.pi, math.pi),
            velocity_limit=1.0,
            torque_limit=2.0,
            kp_eff=100.0,
            max_lead=0.05,
        ),
        _joint(
            "joint_wrist_roll",
            position_limit=(-math.pi, math.pi),
            velocity_limit=1.0,
            torque_limit=2.0,
            kp_eff=100.0,
            max_lead=0.05,
        ),
    )
    arm = JointGroupSpec(name="stretch_arm", role="arm", joints=joints)
    return EmbodimentProfile(
        name="example_stretch",
        base=BaseSpec(name="base", kind="planar", velocity_limits=(0.5, 1.0)),
        joint_groups=(arm,),
        arms=(
            ArmSpec(
                name="primary",
                joint_group="stretch_arm",
                ee_link="link_grasp_center",
                gripper=GripperSpec(
                    name="primary_gripper",
                    joint_names=("joint_gripper_slide",),
                    command_limits=((-0.020, 0.040),),
                ),
            ),
        ),
        metadata=(("limits_source", _EXAMPLE), ("robot", "stretch")),
    )


_ACTUATOR_FIELDS = frozenset(
    {
        "position_limit",
        "velocity_limit",
        "torque_limit",
        "kind",
        "control_mode",
        "kp_eff",
        "kd_eff",
        "torque_constant",
        "gear_ratio",
        "current_limit",
        "max_lead",
    }
)


def apply_actuator_identification(
    profile: EmbodimentProfile,
    identification: dict[str, Any],
) -> EmbodimentProfile:
    """Return a profile with bench-identified per-joint actuator fields applied.

    Identification JSON shape::

        {"embodiment": "example_innate_mars",
         "source": "bench-2026-12-01",
         "joints": {"joint3": {"kp_eff": 12.3, "max_lead": 0.02}}}

    Only joint-model fields are accepted. The function fails closed on unknown
    joints/fields or a mismatched embodiment so a bench file cannot silently
    alter an unrelated robot.
    """
    if not isinstance(identification, dict):
        raise ValueError("actuator identification must be an object")
    embodiment = identification.get("embodiment")
    if embodiment is not None and str(embodiment) != profile.name:
        raise ValueError(f"actuator identification embodiment {embodiment!r} != profile {profile.name!r}")
    joints = identification.get("joints")
    if not isinstance(joints, dict) or not joints:
        raise ValueError("actuator identification must contain a non-empty joints object")
    unknown = set(joints) - set(profile.controlled_joint_names)
    if unknown:
        raise ValueError(f"actuator identification references unknown joints: {sorted(unknown)}")

    groups = []
    for group in profile.joint_groups:
        updated = []
        for joint in group.joints:
            raw = joints.get(joint.name)
            if raw is None:
                updated.append(joint)
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"actuator identification for {joint.name} must be an object")
            extra = set(raw) - _ACTUATOR_FIELDS
            if extra:
                raise ValueError(f"unknown actuator fields for {joint.name}: {sorted(extra)}")
            updated.append(replace(joint, **raw))
        groups.append(replace(group, joints=tuple(updated)))

    metadata = dict(profile.metadata)
    source = identification.get("source")
    if source:
        metadata["actuator_identification"] = str(source)
    return replace(profile, joint_groups=tuple(groups), metadata=tuple(metadata.items()))


def load_actuator_identification(path: str | Path, profile: EmbodimentProfile) -> EmbodimentProfile:
    """Load a JSON identification file and apply it to *profile*."""
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return apply_actuator_identification(profile, data)


_SIM_PROFILE_FACTORIES = {
    "innate_mars": mars_sim_profile,
    "mars": mars_sim_profile,
    "stretch": stretch_sim_profile,
    "sourccey": sourccey_sim_profile,
    "rby1": rby1_sim_profile,
    "rb_y1": rby1_sim_profile,
    "pandaomron": pandaomron_sim_profile,
    "panda_omron": pandaomron_sim_profile,
}


def sim_profile(robot: str) -> EmbodimentProfile:
    """Return the sim-matched profile for a supported robot key."""
    key = robot.lower().replace("-", "_")
    try:
        return _SIM_PROFILE_FACTORIES[key]()
    except KeyError as exc:
        raise KeyError(f"no WBC sim profile for robot {robot!r}") from exc


__all__ = [
    "apply_actuator_identification",
    "example_mars_profile",
    "example_rby1_profile",
    "example_sourccey_profile",
    "example_stretch_profile",
    "load_actuator_identification",
    "mars_sim_profile",
    "pandaomron_sim_profile",
    "rby1_sim_profile",
    "sim_profile",
    "sourccey_sim_profile",
    "stretch_sim_profile",
]

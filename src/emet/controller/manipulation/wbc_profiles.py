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

import math

from emet.controller.manipulation.wbc_contract import (
    ArmSpec,
    BaseSpec,
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
) -> JointSpec:
    return JointSpec(
        name=name,
        position_limit=position_limit,
        velocity_limit=velocity_limit,
        torque_limit=torque_limit,
        kind=kind,
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
            ),
            _joint(
                "joint5",
                position_limit=(-1.5708, 1.5708),
                velocity_limit=2.0,
                torque_limit=1.5,
                kp_eff=1000.0,
                max_lead=0.05,
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
                gripper=GripperSpec(name="primary_gripper", joint_names=("joint6",)),
            ),
        ),
        metadata=(("limits_source", _EXAMPLE), ("robot", "innate_mars")),
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
                gripper=GripperSpec(name="left_gripper", joint_names=("left_gripper",)),
            ),
            ArmSpec(
                name="right",
                joint_group="right_arm",
                ee_link="right_Gripper-Finger",
                gripper=GripperSpec(name="right_gripper", joint_names=("right_gripper",)),
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
                ),
            ),
            ArmSpec(
                name="right",
                joint_group="right_arm",
                ee_link="right_ee",
                gripper=GripperSpec(
                    name="right_gripper",
                    joint_names=("right_gripper_finger_joint1", "right_gripper_finger_joint2"),
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
                gripper=GripperSpec(name="primary_gripper", joint_names=("joint_gripper_slide",)),
            ),
        ),
        metadata=(("limits_source", _EXAMPLE), ("robot", "stretch")),
    )


__all__ = [
    "example_mars_profile",
    "example_rby1_profile",
    "example_sourccey_profile",
    "example_stretch_profile",
]

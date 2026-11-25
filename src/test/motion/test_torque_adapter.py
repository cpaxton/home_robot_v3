# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Position-servo adapter for the torque WBC contract."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from emet.controller.manipulation.torque_adapter import PositionServoAdapter
from emet.controller.manipulation.wbc_contract import (
    ArmAction,
    ArmObservation,
    GripperCommand,
    GripperObservation,
    WbcAction,
    WbcObservation,
)
from emet.controller.manipulation.wbc_profiles import example_mars_profile, example_sourccey_profile


def _profile_with_actuator(**changes):
    profile = example_mars_profile()
    group = profile.joint_group("primary_arm")
    joints = tuple(replace(joint, **changes) for joint in group.joints)
    return replace(profile, joint_groups=(replace(group, joints=joints),))


def _mars_action(profile, torques):
    arm = profile.arm("primary")
    return WbcAction(
        base_velocity=(0.1, 0.0),
        arms=(
            ArmAction(
                name="primary",
                joint_torques=tuple(torques),
                gripper=GripperCommand(values=(0.2,)),
            ),
        ),
    )


def _mars_observation(profile, q=0.0):
    group = profile.joint_group("primary_arm")
    return WbcObservation(
        base_pose=(0.0, 0.0, 0.0),
        base_twist=(0.0, 0.0),
        arms=(
            ArmObservation(
                name="primary",
                joint_position=(q,) * len(group.joints),
                joint_velocity=(0.0,) * len(group.joints),
                gripper=GripperObservation(position=(0.0,)),
            ),
        ),
        occupancy=None,
    )


def test_torque_maps_to_bounded_position_lead():
    profile = _profile_with_actuator(kp_eff=100.0, max_lead=0.02)
    adapter = PositionServoAdapter(profile)
    command = adapter.command(_mars_action(profile, (10.0, 0.0, 0.0, 0.0, 0.0)), _mars_observation(profile))
    assert command.joint_positions["joint1"] == pytest.approx(0.015)
    assert command.joint_positions["joint2"] == 0.0
    assert command.commanded_torques["joint1"] == pytest.approx(1.5)  # clipped by torque_limit
    assert command.base_velocity == (0.1, 0.0)


def test_torque_limit_and_current_limit_clip():
    profile = _profile_with_actuator(
        kp_eff=100.0,
        max_lead=0.02,
        torque_constant=0.1,
        current_limit=1.0,
        gear_ratio=1.0,
    )
    adapter = PositionServoAdapter(profile)
    command = adapter.command(_mars_action(profile, (10.0, 0.0, 0.0, 0.0, 0.0)), _mars_observation(profile))
    # current limit 1 A * 0.1 N*m/A = 0.1 N*m, below the 1.5 N*m torque limit.
    assert command.commanded_torques["joint1"] == pytest.approx(0.1)
    assert command.joint_positions["joint1"] == pytest.approx(0.001)
    assert command.current_limits["joint1"] == pytest.approx(1.0)


def test_gravity_feedforward_offsets_position_target():
    profile = _profile_with_actuator(kp_eff=100.0, max_lead=0.02)
    adapter = PositionServoAdapter(profile, gravity_torques={"joint1": 1.0})
    command = adapter.command(_mars_action(profile, (0.0,) * 5), _mars_observation(profile))
    assert command.joint_positions["joint1"] == pytest.approx(0.01)


def test_callable_gravity_feedforward_receives_observation():
    profile = _profile_with_actuator(kp_eff=100.0, max_lead=0.02)
    seen = []

    def gravity(observation):
        seen.append(observation)
        return {"joint1": 1.0}

    adapter = PositionServoAdapter(profile, gravity_torques=gravity)
    command = adapter.command(_mars_action(profile, (0.0,) * 5), _mars_observation(profile))
    assert seen
    assert command.joint_positions["joint1"] == pytest.approx(0.01)


def test_missing_kp_fails_closed():
    profile = _profile_with_actuator(kp_eff=None)
    adapter = PositionServoAdapter(profile)
    with pytest.raises(ValueError, match="kp_eff"):
        adapter.command(_mars_action(profile, (0.1,) * 5), _mars_observation(profile))


def test_missing_max_lead_fails_closed():
    profile = _profile_with_actuator(max_lead=None)
    adapter = PositionServoAdapter(profile)
    with pytest.raises(ValueError, match="max_lead"):
        adapter.command(_mars_action(profile, (0.1,) * 5), _mars_observation(profile))


def test_unknown_feedforward_joint_is_rejected():
    profile = example_mars_profile()
    with pytest.raises(ValueError, match="unknown joints"):
        PositionServoAdapter(profile, gravity_torques={"not_a_joint": 1.0})


def test_dual_arm_adapter_commands_all_groups():
    profile = example_sourccey_profile()
    arms = []
    for arm_spec in profile.arms:
        group = profile.joint_group(arm_spec.joint_group)
        arms.append(
            ArmAction(
                name=arm_spec.name,
                joint_torques=(0.5,) * len(group.joints),
                gripper=GripperCommand(values=(0.1,) * arm_spec.gripper.dof),
            )
        )
    action = WbcAction(base_velocity=(0.0, 0.0), arms=tuple(arms), torso_torques=(0.2,))
    arms_obs = []
    for arm_spec in profile.arms:
        group = profile.joint_group(arm_spec.joint_group)
        arms_obs.append(
            ArmObservation(
                name=arm_spec.name,
                joint_position=(0.0,) * len(group.joints),
                joint_velocity=(0.0,) * len(group.joints),
                gripper=GripperObservation(position=(0.0,) * arm_spec.gripper.dof),
            )
        )
    observation = WbcObservation(
        base_pose=(0.0, 0.0, 0.0),
        base_twist=(0.0, 0.0),
        arms=tuple(arms_obs),
        torso=None,
        occupancy=np.zeros((4, 4, 4), dtype=np.uint8),
    )
    # Add the torso observation required by the profile.
    from emet.controller.manipulation.wbc_contract import JointGroupObservation

    observation = replace(
        observation,
        torso=JointGroupObservation(name="lift", position=(0.0,), velocity=(0.0,)),
    )
    command = PositionServoAdapter(profile).command(action, observation)
    assert set(command.joint_positions) == {
        "lift",
        "left_shoulder_pan",
        "left_shoulder_lift",
        "left_elbow_flex",
        "left_wrist_flex",
        "left_wrist_roll",
        "right_shoulder_pan",
        "right_shoulder_lift",
        "right_elbow_flex",
        "right_wrist_flex",
        "right_wrist_roll",
    }
    assert command.joint_positions["lift"] == pytest.approx(0.001)  # 0.2 / 200
    assert set(command.grippers) == {"left", "right"}

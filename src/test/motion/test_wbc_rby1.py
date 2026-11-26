# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Smoke tests for the direct-MuJoCo RB-Y1 / Galaxea R1 WBC backend."""

from __future__ import annotations

import numpy as np
import pytest


def _zero_command(profile):
    from emet.controller.manipulation.torque_adapter import ServoCommand
    from emet.controller.manipulation.wbc_contract import GripperCommand

    return ServoCommand(
        base_velocity=(0.0, 0.0, 0.0),
        joint_positions=dict.fromkeys(profile.controlled_joint_names, 0.0),
        grippers={
            arm.name: GripperCommand(values=(0.04,) * arm.gripper.dof, mode=arm.gripper.command_mode)
            for arm in profile.arms
        },
    )


@pytest.mark.sim
def test_rby1_backend_reset_observation_matches_profile():
    from emet.rl.manipulation.wbc_env import active_action_mask, wbc_observation_layout
    from emet.rl.manipulation.wbc_rby1 import Rby1WbcBackend

    backend = Rby1WbcBackend(seed=0, base_goal_probability=1.0)
    try:
        profile = backend.profile
        assert profile.name == "rby1_sim"
        assert profile.base.kind == "holonomic"
        assert profile.action_dim == 23

        observation = backend.reset(seed=0)
        observation.validate(profile)
        assert len(observation.base_twist) == 3
        assert np.isfinite(observation.base_pose).all()
        assert np.linalg.norm(observation.base_pose[:2]) < 1e-9
        assert observation.base_target is not None
        assert observation.base_target_mask == (True, True, True)

        assert observation.torso is not None
        assert observation.torso.name == profile.torso.name
        assert len(observation.torso.position) == 4
        assert len(observation.torso.velocity) == 4
        assert len(observation.torso.torque) == 4
        assert len(observation.torso.target) == 4

        assert tuple(arm.name for arm in observation.arms) == ("left", "right")
        for arm_spec in profile.arms:
            arm = observation.arm(arm_spec.name)
            assert arm.ee_target is not None
            assert len(arm.joint_position) == 6
            assert len(arm.joint_velocity) == 6
            assert len(arm.joint_torque) == 6
            assert len(arm.ee_pose) == 7
            assert len(arm.ee_target) == 7
            assert arm.gripper is not None
            assert len(arm.gripper.position) == 2
            assert len(arm.gripper.velocity) == 2

        layout = wbc_observation_layout(profile)
        encoded = layout.encode(observation)
        assert encoded.shape == (layout.size,)
        assert np.isfinite(encoded).all()
        mask = active_action_mask(observation, profile)
        assert mask.shape == (profile.action_dim,)
        assert mask.all()
    finally:
        backend.close()


@pytest.mark.sim
def test_rby1_backend_zero_servo_command_steps_are_finite():
    from emet.rl.manipulation.wbc_rby1 import Rby1WbcBackend

    backend = Rby1WbcBackend(seed=1, base_goal_probability=0.0)
    try:
        observation = backend.reset(seed=1)
        command = _zero_command(backend.profile)
        for _ in range(2):
            observation, terminated, info = backend.step(command, backend.control_dt)
            observation.validate(backend.profile)
            assert not terminated
            assert info["invalid_physics"] is False
            assert len(observation.base_twist) == 3
            assert np.isfinite(observation.base_twist).all()
            assert observation.torso is not None
            assert np.isfinite(observation.torso.position).all()
            assert np.isfinite(observation.torso.velocity).all()
            assert np.isfinite(observation.torso.torque).all()
            for arm in observation.arms:
                assert np.isfinite(arm.joint_position).all()
                assert np.isfinite(arm.joint_velocity).all()
                assert np.isfinite(arm.joint_torque).all()
                assert np.isfinite(arm.ee_pose).all()
                assert arm.gripper is not None
                assert np.isfinite(arm.gripper.position).all()
                assert np.isfinite(arm.gripper.velocity).all()
        # A zero base command must not drive the plant away from the origin.
        assert abs(observation.base_pose[0]) < 1e-3
        assert abs(observation.base_pose[1]) < 1e-3
    finally:
        backend.close()


@pytest.mark.sim
def test_rby1_teacher_action_is_finite_normalized_and_deterministic():
    from emet.controller.manipulation.wbc_contract import action_from_normalized_vector
    from emet.rl.manipulation.wbc_rby1 import Rby1WbcBackend

    backend = Rby1WbcBackend(seed=2, base_goal_probability=1.0, arm_goal_probability=1.0)
    try:
        backend.reset(seed=2)
        action = backend.teacher_action()
        assert action.shape == (backend.profile.action_dim,)
        assert action.dtype == np.float32
        assert np.isfinite(action).all()
        assert np.abs(action).max() <= 1.0 + 1e-6
        # The action should carry real base and arm commands, not a null vector.
        assert np.abs(action).max() > 0.1

        decoded = action_from_normalized_vector(action, backend.profile)
        decoded.validate(backend.profile)
        assert np.isfinite(decoded.base_velocity).all()

        repeated = backend.teacher_action()
        np.testing.assert_array_equal(action, repeated)
    finally:
        backend.close()


@pytest.mark.sim
def test_rby1_teacher_drives_base_toward_forward_goal():
    from emet.controller.manipulation.torque_adapter import PositionServoAdapter
    from emet.controller.manipulation.wbc_contract import action_from_normalized_vector
    from emet.rl.manipulation.wbc_rby1 import Rby1WbcBackend

    backend = Rby1WbcBackend(seed=3, base_goal_probability=1.0, arm_goal_probability=0.0)
    try:
        observation = backend.reset(seed=3)
        assert observation.base_target is not None
        assert all(arm.ee_target is None for arm in observation.arms)
        target = np.asarray(observation.base_target)
        start = np.asarray(observation.base_pose)
        adapter = PositionServoAdapter(backend.profile)
        initial_distance = float(np.linalg.norm(target[:2] - start[:2]))
        assert initial_distance > 0.04

        for _ in range(100):
            action = action_from_normalized_vector(backend.teacher_action(), backend.profile)
            command = adapter.command(action, observation)
            observation, terminated, info = backend.step(command, backend.control_dt)
            assert info["invalid_physics"] is False
            if terminated:
                break

        end = np.asarray(observation.base_pose)
        final_distance = float(np.linalg.norm(target[:2] - end[:2]))
        assert final_distance < 0.03
        assert float(np.linalg.norm(end[:2] - start[:2])) > 0.03
    finally:
        backend.close()

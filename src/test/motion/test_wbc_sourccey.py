# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Direct-MuJoCo Sourccey whole-body backend tests."""

from __future__ import annotations

import numpy as np
import pytest

from emet.controller.manipulation.torque_adapter import ServoCommand
from emet.controller.manipulation.wbc_profiles import sourccey_sim_profile
from emet.rl.manipulation.wbc_env import WbcGymEnv, wbc_observation_layout


@pytest.mark.sim
def test_sourccey_backend_reset_observation_and_action_shapes():
    from emet.rl.manipulation.wbc_sourccey import SourcceyWbcBackend

    backend = SourcceyWbcBackend(seed=0)
    profile = backend.profile
    assert profile == sourccey_sim_profile()
    assert profile.action_dim == 15

    observation = backend.reset(seed=0)
    observation.validate(profile)
    layout = wbc_observation_layout(profile)
    encoded = layout.encode(observation)
    assert encoded.shape == (layout.size,)
    assert np.isfinite(encoded).all()
    assert len(observation.arms) == len(profile.arms)

    action = backend.teacher_action()
    assert action.shape == (profile.action_dim,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    backend.close()


@pytest.mark.sim
def test_sourccey_backend_zero_servo_commands_are_finite():
    from emet.rl.manipulation.wbc_sourccey import SourcceyWbcBackend

    backend = SourcceyWbcBackend(seed=0, base_goal_probability=0.0)
    backend.reset(seed=0)
    zero = ServoCommand(base_velocity=(0.0, 0.0), joint_positions={})
    for _ in range(2):
        observation, terminated, info = backend.step(zero, backend.control_dt)
        observation.validate(backend.profile)
        assert not terminated
        assert not info["invalid_physics"]
        assert info["model_fingerprint"]
        assert np.isfinite(info["elapsed_s"])
        assert np.isfinite(observation.base_pose).all()
        assert np.isfinite(observation.base_twist).all()
        for arm in observation.arms:
            assert np.isfinite(arm.joint_position).all()
            assert np.isfinite(arm.joint_velocity).all()
            assert np.isfinite(arm.joint_torque).all()
            assert np.isfinite(arm.ee_pose).all()
            assert arm.gripper is not None
            assert np.isfinite(arm.gripper.position).all()
            assert np.isfinite(arm.gripper.velocity).all()
    backend.close()


@pytest.mark.sim
def test_sourccey_teacher_action_is_deterministic():
    from emet.rl.manipulation.wbc_sourccey import SourcceyWbcBackend

    backend = SourcceyWbcBackend(seed=4, base_goal_probability=0.0)
    backend.reset(seed=4)
    first = backend.teacher_action()
    second = backend.teacher_action()
    assert first.shape == (backend.profile.action_dim,)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    backend.close()


@pytest.mark.sim
def test_sourccey_base_velocity_moves_planar_base():
    from emet.rl.manipulation.wbc_sourccey import SourcceyWbcBackend

    backend = SourcceyWbcBackend(seed=0, base_goal_probability=0.0)
    start = backend.reset(seed=0)
    command = ServoCommand(base_velocity=(0.2, 0.0), joint_positions={})
    for _ in range(10):
        observation, terminated, _ = backend.step(command, backend.control_dt)
        assert not terminated
    assert observation.base_pose[0] - start.base_pose[0] > 0.02
    assert abs(observation.base_pose[1] - start.base_pose[1]) < 0.01
    assert abs(observation.base_pose[2] - start.base_pose[2]) < 0.01
    assert observation.base_twist[0] > 0.05
    backend.close()


@pytest.mark.sim
def test_sourccey_teacher_reaches_sampled_arm_target():
    from emet.rl.manipulation.wbc_sourccey import SourcceyWbcBackend

    backend = SourcceyWbcBackend(seed=0, base_goal_probability=0.0)
    env = WbcGymEnv(backend, horizon_s=4.0, control_dt=0.05, success_hold_s=0.1)
    env.reset(seed=0)
    assert backend._target_q
    assert all(target is not None for target in backend._target_ee.values())

    info = {"success": 0.0}
    for _ in range(60):
        _, _, terminated, truncated, info = env.step(backend.teacher_action())
        if terminated or truncated:
            break
    assert info["success"], info
    assert info["reward_terms"]["pos_err"] < 0.02
    env.close()


@pytest.mark.sim
def test_sourccey_single_arm_goal_disables_the_other_arm():
    from emet.rl.manipulation.wbc_sourccey import SourcceyWbcBackend

    backend = SourcceyWbcBackend(
        seed=0,
        arm_goal_probability=1.0,
        base_goal_probability=0.0,
        active_arms=("right",),
    )
    try:
        observation = backend.reset(seed=0)
        assert observation.arm("left").ee_target is None
        assert observation.arm("left").target_mask == (False, False)
        assert observation.arm("right").ee_target is not None
        assert backend._target_q["left"] == pytest.approx(backend._start_q["left"])
    finally:
        backend.close()
    with pytest.raises(ValueError, match="unknown active arms"):
        SourcceyWbcBackend(seed=0, active_arms=("middle",))

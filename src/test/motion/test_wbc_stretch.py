# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for the direct-MuJoCo Stretch whole-body-control backend."""

from __future__ import annotations

import numpy as np
import pytest

from emet.controller.manipulation.torque_adapter import ServoCommand
from emet.controller.manipulation.wbc_contract import GripperCommand
from emet.controller.manipulation.wbc_profiles import stretch_sim_profile
from emet.rl.manipulation.wbc_env import WbcGymEnv, WbcPerturbation, wbc_observation_layout


def _backend(**kwargs):
    from emet.rl.manipulation.wbc_stretch import StretchWbcBackend

    return StretchWbcBackend(**kwargs)


def _zero_command() -> ServoCommand:
    profile = stretch_sim_profile()
    return ServoCommand(
        base_velocity=(0.0, 0.0),
        joint_positions=dict.fromkeys(profile.controlled_joint_names, 0.0),
        grippers={profile.arms[0].name: GripperCommand(values=(0.0,))},
    )


def _ee_errors(observation) -> tuple[float, float]:
    arm = observation.arm("primary")
    target = np.asarray(arm.ee_target, dtype=np.float64)
    measured = np.asarray(arm.ee_pose, dtype=np.float64)
    position = float(np.linalg.norm(target[:3] - measured[:3]))
    tq = target[3:] / np.linalg.norm(target[3:])
    mq = measured[3:] / np.linalg.norm(measured[3:])
    orientation = float(2.0 * np.arccos(min(1.0, abs(float(np.dot(tq, mq))))))
    return position, orientation


@pytest.mark.sim
def test_backend_loads_and_reset_observation_validates():
    backend = _backend(seed=0)
    profile = stretch_sim_profile()
    assert backend.profile == profile

    observation = backend.reset(seed=0)
    observation.validate(profile)

    assert len(observation.base_pose) == 3
    assert len(observation.base_twist) == profile.base.action_dim
    assert observation.torso is None
    assert observation.head is None
    arm = observation.arm("primary")
    assert len(arm.joint_position) == 5
    assert len(arm.joint_velocity) == 5
    assert len(arm.joint_torque) == 5
    assert len(arm.ee_pose) == 7
    assert arm.ee_target is not None and len(arm.ee_target) == 7
    assert arm.gripper is not None and len(arm.gripper.position) == 1

    layout = wbc_observation_layout(profile)
    vector = layout.encode(observation)
    assert vector.shape == (layout.size,)
    assert np.isfinite(vector).all()

    target_q = np.asarray(backend._target_q, dtype=np.float64)
    group = profile.joint_group(profile.arms[0].joint_group)
    limits = np.asarray([joint.position_limit for joint in group.joints], dtype=np.float64)
    assert np.all(target_q >= limits[:, 0]) and np.all(target_q <= limits[:, 1])

    # The optional base goal must be consistent with the observation.
    assert (observation.base_target is not None) == (backend._target_base is not None)
    backend.close()


@pytest.mark.sim
def test_step_zero_command_twice_stays_finite():
    backend = _backend(seed=0)
    profile = stretch_sim_profile()
    backend.reset(seed=0)

    for _ in range(2):
        observation, terminated, info = backend.step(_zero_command(), backend.control_dt)
        observation.validate(profile)
        assert np.isfinite(observation.base_pose).all()
        assert np.isfinite(observation.base_twist).all()
        arm = observation.arm("primary")
        assert np.isfinite(arm.joint_position).all()
        assert np.isfinite(arm.joint_velocity).all()
        assert np.isfinite(arm.joint_torque).all()
        assert np.isfinite(arm.ee_pose).all()
        assert np.isfinite(arm.gripper.position).all()
        assert np.isfinite(arm.gripper.velocity).all()
        assert not terminated
        assert info["model_fingerprint"]
        assert np.isfinite(info["elapsed_s"])

    with pytest.raises(ValueError, match="dt"):
        backend.step(_zero_command(), 0.5 * backend.control_dt)
    backend.close()


@pytest.mark.sim
def test_teacher_action_is_bounded_and_deterministic():
    backend = _backend(seed=0, base_goal_probability=0.0)
    backend.reset(seed=0)

    action = backend.teacher_action()
    assert action.shape == (backend.profile.action_dim,)
    assert np.isfinite(action).all()
    assert np.all(np.abs(action) <= 1.0)
    np.testing.assert_array_equal(action, backend.teacher_action())
    backend.close()


@pytest.mark.sim
def test_teacher_reaches_sampled_arm_target_within_60_steps():
    backend = _backend(seed=0, base_goal_probability=0.0)
    env = WbcGymEnv(backend, horizon_s=3.0, control_dt=0.05, success_hold_s=0.1)
    env.reset(seed=0)

    initial = env._observation
    assert initial.arm("primary").ee_target is not None
    initial_position, initial_orientation = _ee_errors(initial)
    target_q = np.asarray(backend._target_q, dtype=np.float64)
    initial_joint_error = float(np.linalg.norm(target_q - np.asarray(initial.arm("primary").joint_position)))

    for _ in range(60):
        _, _, terminated, truncated, _ = env.step(backend.teacher_action())
        if terminated or truncated:
            break

    final = env._observation
    final_position, final_orientation = _ee_errors(final)
    final_joint_error = float(np.linalg.norm(target_q - np.asarray(final.arm("primary").joint_position)))

    # The plant is a weakly damped position servo, so 3 s is not always enough
    # for a 0.02 m / 0.05 rad contract success. The teacher must still make
    # unambiguous progress toward the sampled five-axis target.
    assert final_joint_error < 0.5 * initial_joint_error
    assert final_orientation < 0.5 * initial_orientation
    assert final_position < 0.1
    assert final_position < initial_position
    env.close()


@pytest.mark.sim
def test_stretch_payload_changes_joint_torque_and_closes_gripper():
    baseline = _backend(seed=0, base_goal_probability=0.0)
    baseline.reset(seed=0)
    baseline_bias = float(np.linalg.norm(baseline.data.qfrc_bias))
    baseline.close()

    loaded = _backend(seed=0, base_goal_probability=0.0)
    loaded.set_perturbation(WbcPerturbation(payload_mass=0.8, gripper_close=True))
    loaded.reset(seed=0)
    loaded_bias = float(np.linalg.norm(loaded.data.qfrc_bias))
    assert loaded.model.body_mass[loaded._payload_body] == pytest.approx(0.8)
    assert loaded_bias > baseline_bias
    action = loaded.teacher_action()
    assert action[-1] < 0.0
    loaded.close()

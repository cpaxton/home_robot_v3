# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Direct-MuJoCo PandaOmron contract smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.rl.manipulation.wbc_env import WbcGymEnv, WbcPerturbation
from emet.rl.manipulation.wbc_pandaomron import PandaOmronWbcBackend


@pytest.mark.sim
def test_pandaomron_reset_observation_and_action_shapes():
    backend = PandaOmronWbcBackend(seed=0, base_goal_probability=0.0)
    env = WbcGymEnv(backend, horizon_s=3.0, control_dt=0.05, success_hold_s=0.1)
    observation, info = env.reset(seed=0)
    assert observation.shape == env.observation_space.shape
    assert backend.profile.action_dim == 13
    assert env.action_space.shape == (13,)
    assert info["profile_fingerprint"] == pandaomron_sim_profile().fingerprint
    env.close()


@pytest.mark.sim
def test_pandaomron_zero_servo_commands_are_finite():
    backend = PandaOmronWbcBackend(seed=0, base_goal_probability=0.0)
    env = WbcGymEnv(backend, horizon_s=3.0, control_dt=0.05, success_hold_s=0.1)
    env.reset(seed=0)
    for _ in range(2):
        observation, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert "reward_terms" in info
        if terminated or truncated:
            break
    env.close()


@pytest.mark.sim
def test_pandaomron_teacher_action_shape_and_bounds():
    backend = PandaOmronWbcBackend(seed=0, base_goal_probability=0.0)
    backend.reset(seed=0)
    action = backend.teacher_action()
    assert action.shape == (13,)
    assert np.isfinite(action).all()
    assert np.all(np.abs(action) <= 1.0)
    backend.close()


@pytest.mark.sim
def test_pandaomron_reports_measured_contact_state():
    backend = PandaOmronWbcBackend(seed=0, base_goal_probability=0.0)
    observation = backend.reset(seed=0)
    arm = observation.arm("right")
    assert arm.contact is not None
    assert arm.contact.source == "ft_sensor"
    assert len(arm.contact.wrench) == 6
    assert np.isfinite(arm.contact.wrench).all()
    assert arm.gripper is not None
    assert len(arm.gripper.force) == 2
    assert np.isfinite(arm.gripper.force).all()
    backend.close()


def test_robocasa_target_and_base_frames_share_the_reset_origin():
    from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend

    backend = RoboCasaPandaBackend(None, base_backoff=0.0)
    backend._origin_xy = np.asarray((1.0, 2.0))
    backend._origin_yaw = np.pi / 2
    backend._world_base_pose = lambda: (2.0, 2.0, np.pi / 2)  # type: ignore[method-assign]
    assert np.allclose(backend._base_pose(), (0.0, -1.0, 0.0), atol=1e-9)
    local = backend._local_pose((2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0))
    assert np.allclose(local[:3], (0.0, -1.0, 1.0), atol=1e-9)
    world, _ = backend._world_pose(local)
    assert np.allclose(world, (2.0, 2.0, 1.0), atol=1e-9)


@pytest.mark.sim
def test_pandaomron_env_reports_contact_availability():
    backend = PandaOmronWbcBackend(seed=0, base_goal_probability=0.0)
    env = WbcGymEnv(backend, horizon_s=3.0, control_dt=0.05, success_hold_s=0.1)
    _, info = env.reset(seed=0)
    assert info["contact_state_available"] is True
    _, _, _, _, info = env.step(np.zeros(env.action_space.shape))
    assert info["contact_state_available"] is True
    assert np.isfinite(info["contact_force"])
    assert np.isfinite(info["hand_contact_force"])
    env.close()


@pytest.mark.sim
def test_pandaomron_payload_mass_changes_measured_wrench():
    baseline = PandaOmronWbcBackend(seed=0, base_goal_probability=0.0)
    baseline_observation = baseline.reset(seed=0)
    baseline_wrench = abs(float(baseline_observation.arm("right").contact.wrench[2]))
    baseline.close()

    loaded = PandaOmronWbcBackend(seed=0, base_goal_probability=0.0, torso_range=(0.20, 0.32))
    loaded.set_perturbation(WbcPerturbation(payload_mass=1.0, gripper_close=True))
    observation = loaded.reset(seed=0)
    loaded_wrench = abs(float(observation.arm("right").contact.wrench[2]))
    assert loaded.model.body_mass[loaded._payload_body] == pytest.approx(1.0)
    assert loaded_wrench > baseline_wrench + 2.0
    action = loaded.teacher_action()
    assert action[-2] < -0.5
    assert action[-1] > 0.5
    loaded.close()

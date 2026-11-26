# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Direct-MuJoCo PandaOmron contract smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.rl.manipulation.wbc_env import WbcGymEnv
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

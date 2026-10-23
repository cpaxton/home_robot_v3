"""Temporal policy and sustained tracking-failure regression tests."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")
from emet.rl.env import EnvConfig, MarsReachEnv
from emet.rl.profile import MARS

pytestmark = pytest.mark.sim


def test_history_cadence_goal_pose_and_episode_reset():
    env = MarsReachEnv()
    try:
        first, _ = env.reset(seed=42)
        assert first["history"].shape == (20, 111)
        np.testing.assert_array_equal(first["history"][0], first["history"][-1])
        again, _ = env.control.observe()
        np.testing.assert_array_equal(first["history"], again["history"])
        next_obs, *_ = env.step(np.full(7, 0.1))
        np.testing.assert_array_equal(next_obs["history"][:-1], first["history"][1:])
        np.testing.assert_allclose(
            next_obs["history"][-1], np.concatenate([next_obs[k] for k in ("state", "goal", "obstacles", "poses")])
        )
        env.set_goal({"arm": {"joint1": 0.3}})
        d = mujoco.MjData(env.model)
        d.qpos[:] = env.data.qpos
        d.qpos[env.control.arm_q] = env.control.goal.arm
        mujoco.mj_forward(env.model, d)
        np.testing.assert_allclose(env.control.goal_pose[:3], d.xpos[env.model.body("ee_link").id])
        reset, _ = env.reset(seed=17)
        np.testing.assert_array_equal(reset["history"][0], reset["history"][-1])
    finally:
        env.close()


def test_stalled_joint_bounds_lead_detects_and_allows_retreat():
    env = MarsReachEnv()
    try:
        env.reset(seed=42)
        c = env.control
        env.set_goal({"arm": {"joint1": 0.5}})
        # Fault injection: encoder stays stationary despite repeated commands.
        for _tick in range(12):
            c.accept_action([0, 0, 1, 0, 0, 0, 0])
            for _ in range(25):
                env.data.time += MARS.physics_dt
                c.apply()
                assert np.max(np.abs(c.targets - c.arm_positions())) <= MARS.max_target_lead + 1e-9
        assert c.obstruction[2]
        assert c.targets[0] == pytest.approx(c.arm_positions()[0])
        c.accept_action([0, 0, -1, 0, 0, 0, 0])
        assert not c.obstruction[2]
        assert c.targets[0] < c.arm_positions()[0]
        c.reset()
        assert not c.obstruction.any()
    finally:
        env.close()


def test_sim_effort_limits_and_contact_metrics():
    env = MarsReachEnv(EnvConfig(contact_training=True))
    try:
        env.reset(seed=42)
        c = env.control
        assert env.model.actuator_forcelimited[c.arm_act].all()
        np.testing.assert_allclose(
            env.model.actuator_forcerange[c.arm_act], np.tile([-MARS.arm_effort_limit, MARS.arm_effort_limit], (5, 1))
        )
        _, _, _, _, info = env.step(np.ones(7))
        assert info["peak_contact_force"] >= 0
        assert info["contact_impulse"] >= 0
    finally:
        env.close()


@pytest.mark.parametrize("allow_contact,force_limit", [(False, 30.0), (True, 30.0), (True, 1.0)])
def test_box_contact_stops_or_unloads(allow_contact, force_limit):
    from emet.rl.control import baseline_action

    env = MarsReachEnv(
        EnvConfig(
            stage=3,
            obstacle_count=1,
            contact_training=allow_contact,
            unsafe_contact_force=force_limit,
            horizon_seconds=3,
        )
    )
    try:
        env.reset(seed=42)
        c = env.control
        base = c.base_pose()
        forward = np.array([np.cos(base[2]), np.sin(base[2])])
        env.data.mocap_pos[0] = [*(base[:2] + 0.65 * forward), 0.2]
        mujoco.mj_forward(env.model, env.data)
        assert not c.observe()[1]["collision"]
        env.set_goal({"base": [*(base[:2] + forward), float(base[2])]})
        peak = 0.0
        for _ in range(60):
            _, _, terminated, truncated, info = env.step(baseline_action(c))
            peak = max(peak, info["peak_contact_force"])
            if terminated or truncated:
                break
        assert peak > 0
        if force_limit == 1:
            assert terminated and info["unsafe_contact"]
        elif not allow_contact:
            assert terminated and info["collision"]
        else:
            assert truncated and not terminated
            assert info["obstruction"] and c.last_action[0] == 0
            assert info["peak_contact_force"] < 1.0  # Unloads the box after detection.
    finally:
        env.close()

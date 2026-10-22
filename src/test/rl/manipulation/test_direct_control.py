"""Direct actions must not silently retain the teacher as a runtime prior."""

import numpy as np
import pytest

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv, running_cost


def test_direct_zero_action_stays_zero_despite_a_distant_goal_and_stop_latches():
    env = ManipulationEnv(ManipulationConfig(robot="innate_mars", scenario="reach", action_mode="direct_velocity"))
    try:
        env.reset(seed=1800010)
        c = env.control
        c.set_reference(np.clip(c.positions() + 0.1, c.limits[:, 0], c.limits[:, 1]))
        c.set_pose_targets(c.base_pose() + [0.2, 0.1, 0.3], np.ones(3, bool), c.ee_pose(), True)
        c.velocity[:] = 0
        c.base_command[:] = 0
        c.accept(np.zeros(env.action_space.shape))
        np.testing.assert_array_equal(c.velocity, 0)
        np.testing.assert_array_equal(c.base_command, 0)
        c.accept(np.ones(env.action_space.shape))
        assert np.max(abs(c.velocity)) <= max(c.profile.acceleration) * c.profile.dt + 1e-9
        assert abs(c.base_command[0]) <= c.profile.base_acceleration * c.profile.dt + 1e-9
        c.stop()
        c.accept(np.ones(env.action_space.shape))
        np.testing.assert_array_equal(c.velocity, 0)
        np.testing.assert_array_equal(c.base_command, 0)
    finally:
        env.close()


def test_direct_teacher_matches_residual_teacher_commands():
    env = ManipulationEnv(ManipulationConfig(robot="innate_mars", scenario="reach"))
    try:
        env.reset(seed=1800011)
        c = env.control
        action = c.teacher_action()
        target = c.target.copy()
        c.accept(np.zeros(env.action_space.shape))
        expected = c.velocity.copy(), c.base_command.copy(), c.target.copy()
        c.target[:] = target
        c.velocity[:] = 0
        c.base_command[:] = 0
        c.action_mode = "direct_velocity"
        c.accept(action)
        for actual, wanted in zip((c.velocity, c.base_command, c.target), expected, strict=True):
            np.testing.assert_allclose(actual, wanted, atol=1e-7)
    finally:
        env.close()


def test_base_tolerance_does_not_erase_hand_tracking_cost():
    env = ManipulationEnv(ManipulationConfig(robot="innate_mars", scenario="reach", base_xy_tolerance=0.05))
    try:
        env.reset(seed=1800012)
        c = env.control
        c.set_pose_targets(c.base_pose() + [0.04, 0, 0], np.ones(3, bool), c.ee_pose(), True)
        assert env._tracking_cost() == pytest.approx(0, abs=1e-7)
        c.hand_goal[0] += 0.01
        assert env._tracking_cost() == pytest.approx(0.01, abs=1e-7)
    finally:
        env.close()


def test_persistent_tracking_and_settled_motion_are_penalized():
    args = {"dt": 0.05, "force": 0, "jerk": np.zeros(5), "action": np.zeros(8), "rotation": 0, "translation": 0}
    hold = running_cost(**args)
    assert running_cost(**args, tracking=0.02) > hold
    assert running_cost(**args, motion=0.1) > hold


def test_direct_bundle_declares_independent_action_semantics():
    from emet.rl.manipulation.profiles import MARS

    legacy = MARS.contract()
    direct = MARS.contract("direct_velocity")
    assert "residual" in legacy["action"]
    assert "residual" not in direct["action"]
    assert direct["base_feedback"] == direct["arm_feedback"] == "policy_only"
    with pytest.raises(ValueError):
        MARS.contract("unknown")


def test_hand_contract_cannot_be_loosened_to_hide_tracking_failure():
    for args in ({"hand_position_tolerance": 0.02}, {"hand_orientation_tolerance": 0.2}):
        with pytest.raises(ValueError):
            ManipulationConfig(**args)


def test_dataset_aggregation_restores_curriculum_and_labels_learner_states():
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from emet.rl.manipulation.imitation import initialize_direct_actor

    class FeedbackTask(gym.Env):
        observation_space = gym.spaces.Dict({"history": gym.spaces.Box(-10, 10, (2, 3), dtype=np.float32)})
        action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        target_scale = 0.25

        def set_target_scale(self, scale):
            self.target_scale = scale

        def teacher_action(self):
            return np.clip(-self.state[:2], -1, 1).astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.state = np.full(3, 0.5, dtype=np.float32)
            return {"history": np.repeat(self.state[None], 2, axis=0)}, {}

        def step(self, action):
            self.state[:2] += action * 0.1
            return {"history": np.repeat(self.state[None], 2, axis=0)}, 0, False, False, {}

    torch.set_num_threads(1)
    env = VecNormalize(DummyVecEnv([FeedbackTask]), norm_reward=False)
    try:
        model = PPO("MultiInputPolicy", env, n_steps=8, batch_size=4, device="cpu", seed=1)
        report = initialize_direct_actor(model, env, 20, rounds=2, epochs=1)
        assert env.get_attr("target_scale") == [0.25]
        assert report["samples"] == 20
        assert [r["teacher_probability"] for r in report["rounds"]] == [1, 0]
        assert report["runtime_teacher"] is False
    finally:
        env.close()


def test_residual_accept_advances_base_teacher_once_and_direct_never(monkeypatch):
    env = ManipulationEnv(ManipulationConfig(robot="innate_mars", scenario="reach"))
    try:
        env.reset(seed=1800013)
        calls = []

        def base_teacher():
            calls.append(True)
            return np.zeros(2)

        monkeypatch.setattr(env.control, "base_feedback", base_teacher)
        env.control.accept(np.zeros(env.action_space.shape))
        assert len(calls) == 1
        env.control.action_mode = "direct_velocity"
        env.control.accept(np.zeros(env.action_space.shape))
        assert len(calls) == 1
    finally:
        env.close()

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


@pytest.mark.sim
@pytest.mark.parametrize("released", [False, True])
def test_retention_cost_uses_current_error_while_peak_evidence_remains_latched(monkeypatch, released):
    from emet.rl.manipulation import env as implementation

    env = ManipulationEnv(
        ManipulationConfig(robot="innate_mars", scenario="reach", retarget_probability=0, reach_push_probability=0)
    )
    try:
        env.reset(seed=1800013)
        env._released = released
        errors = iter([(0.006, 0.06), (0.001, 0.01)])
        monkeypatch.setattr(implementation, "retention_error", lambda *_: next(errors))
        costs = []

        def capture(**terms):
            costs.append(terms)
            return 0.0

        monkeypatch.setattr(implementation, "running_cost", capture)
        for _ in range(2):
            env.step(np.zeros(env.action_space.shape))
        assert [c["rotation"] for c in costs] == ([0, 0] if released else [0.06, 0.01])
        assert [c["translation"] for c in costs] == ([0, 0] if released else [0.006, 0.001])
        assert env.max_rotation == (0 if released else 0.06)
        assert env.max_translation == (0 if released else 0.006)
        assert not env.retention_lost
    finally:
        env.close()


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


@pytest.mark.sim
def test_explicit_hold_stays_quiet_and_new_goal_resumes_without_clearing_faults():
    from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv

    e = ManipulationEnv(
        ManipulationConfig(
            robot="innate_mars",
            scenario="reach",
            action_mode="direct_velocity",
            trajectory_preview=True,
            randomization_scale=0,
            horizon=5,
        )
    )
    try:
        e.reset(seed=250000)
        c = e.control
        c.hold_position()
        assert c.holding and not c.stopped
        velocities = []
        for step in range(60):
            _, _, done, truncated, info = e.step(np.zeros(e.action_space.shape))
            assert not done and not truncated, info
            if step >= 40:
                velocities.append(e.last_base_velocity.copy())
        peak = np.max(np.abs(velocities), axis=0)
        assert peak[0] < 0.0005, peak
        assert peak[1] < 0.001, peak
        c.accept(np.ones(e.action_space.shape))
        np.testing.assert_array_equal(c.base_command, 0)
        grip = c.grip_target
        c.set_pose_targets(c.base_goal, c.base_mask, c.hand_goal)
        assert c.holding
        hand = c.hand_goal.copy()
        hand[0] += 0.01
        c.set_pose_targets(c.base_goal, c.base_mask, hand)
        assert not c.holding and not c.stopped
        assert c.grip_target == grip
        c.stop()
        with pytest.raises(ValueError):
            c.hold_position()
        c.set_pose_targets(c.base_goal, c.base_mask, c.hand_goal)
        assert c.stopped
    finally:
        e.close()


@pytest.mark.sim
def test_normal_hold_allows_only_executor_authorized_grip_release():
    e = ManipulationEnv(ManipulationConfig(robot="innate_mars", scenario="reach", randomization_scale=0))
    try:
        e.reset(seed=250000)
        c = e.control
        c.grip_reference = c.grip_target = e.profile.gripper_closed
        c.hold_position()
        arm = c.target.copy()
        c.accept(np.ones(e.action_space.shape))
        assert c.grip_target == e.profile.gripper_closed
        c.set_reference(c.positions(), phase="release", release=True)
        c.accept(np.ones(e.action_space.shape))
        assert c.grip_target > e.profile.gripper_closed
        assert c.holding and not c.stopped
        np.testing.assert_array_equal(c.target, arm)
        np.testing.assert_array_equal(c.base_command, 0)
        np.testing.assert_array_equal(c.velocity, 0)
        c.stop()
        grip = c.grip_target
        c.accept(np.ones(e.action_space.shape))
        assert c.grip_target == grip and c.stopped
    finally:
        e.close()


@pytest.mark.sim
def test_release_rebases_obstructed_jaw_once_and_never_through_fault():
    e = ManipulationEnv(ManipulationConfig(robot="innate_mars", scenario="reach", randomization_scale=0))
    try:
        e.reset(seed=250000)
        c = e.control
        e.data.qpos[c.grip_q] = 0.28  # Measured obstructed jaw, not the closed command.
        c.grip_reference = c.grip_target = e.profile.gripper_closed
        c.set_reference(c.positions(), phase="release", release=True)
        assert c.grip_target == pytest.approx(0.28)
        c.accept(np.zeros(e.action_space.shape))
        opening_target = c.grip_target
        assert 0.28 < opening_target <= 0.28 + e.profile.gripper_speed * e.profile.grip_rate_fraction * e.profile.dt
        c.set_reference(c.positions(), phase="release", release=True)
        assert c.grip_target == opening_target
        c.stop()
        fault_target = c.grip_target
        e.data.qpos[c.grip_q] = 0.30
        c.set_reference(c.positions(), phase="release", release=True)
        assert c.grip_target == fault_target
        assert c.stopped
    finally:
        e.close()


@pytest.mark.sim
def test_arm_contact_hold_keeps_base_active_and_resumes_on_new_arm_goal():
    e = ManipulationEnv(
        ManipulationConfig(robot="innate_mars", scenario="reach", action_mode="direct_velocity", randomization_scale=0)
    )
    try:
        e.reset(seed=250000)
        c = e.control
        c.set_pose_targets(c.base_pose(), np.ones(3, dtype=bool), c.ee_pose(), True)
        c.hold_arm_position()
        arm = c.target.copy()
        c.accept(np.ones(e.action_space.shape))
        assert c.arm_holding and not c.holding and not c.stopped
        np.testing.assert_array_equal(c.target, arm)
        np.testing.assert_array_equal(c.velocity, 0)
        assert np.any(c.base_command != 0)
        goal = c.reference.copy()
        goal[1] += 0.01
        c.set_reference(goal)
        assert not c.arm_holding
        c.stop()
        with pytest.raises(ValueError):
            c.hold_arm_position()
    finally:
        e.close()

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.feedback_policy import FeedbackActorCriticPolicy, GoalFeedbackFeatures
from emet.rl.manipulation.normalization import CommandVecNormalize
from emet.rl.manipulation.observation_layout import history_layout, preview_layout
from emet.rl.manipulation.policy import ManipulationPolicy, export
from emet.rl.manipulation.profiles import MARS


class Commands(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)
    observation_space = gym.spaces.Dict(
        {
            "history": gym.spaces.Box(-np.inf, np.inf, (20, 101), dtype=np.float32),
            "preview": gym.spaces.Box(-np.inf, np.inf, (7, 18), dtype=np.float32),
        }
    )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = equilibrium()
        return obs, {}

    def step(self, action):
        obs, _ = self.reset()
        obs["preview"][:, 1] = 0.05
        return obs, -float(np.sum((action - 0.1) ** 2)), False, False, {}


def equilibrium():
    rng = np.random.default_rng(14)
    obs = {"history": rng.normal(size=(20, 101)).astype(np.float32), "preview": np.zeros((7, 18), dtype=np.float32)}
    h, _ = history_layout(MARS, retention_context=True)
    p, _ = preview_layout()
    obs["history"][:, h["proprio/velocities"]] = 0
    obs["history"][:, h["base_velocity"]] = 0
    obs["preview"][:, p["base_mask"]] = 1
    obs["preview"][:, p["hand_mask"]] = 1
    obs["preview"][:, p["hand_error"].start + 3 : p["hand_error"].stop] = [1, 0, 0, 1, 0, 0]
    return obs


@pytest.fixture
def feedback():
    torch.set_num_threads(1)
    cfg = ManipulationConfig(
        robot="innate_mars",
        action_mode="direct_velocity",
        trajectory_preview=True,
        categorical_features_unscaled=True,
        feedback_policy=True,
        retention_context=True,
    )
    env = CommandVecNormalize(DummyVecEnv([Commands]), config=cfg, norm_reward=False)
    model = PPO(
        FeedbackActorCriticPolicy,
        env,
        device="cpu",
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={
            "features_extractor_class": GoalFeedbackFeatures,
            "features_extractor_kwargs": {"robot": "innate_mars", "retention_context": True},
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        },
    )
    yield cfg, env, model
    env.close()


def test_motion_equilibrium_is_independent_of_learned_gains_and_absolute_context(feedback):
    _, env, model = feedback
    with torch.no_grad():
        for parameter in model.policy.action_net.parameters():
            parameter.normal_(mean=0, std=0.3)
    obs = equilibrium()
    action = model.predict(env.normalize_obs(obs), deterministic=True)[0]
    np.testing.assert_array_equal(action[:-1], np.zeros(7))
    assert np.isfinite(action).all() and np.max(abs(action)) <= 1
    # Goal updates and actual motion can activate the learned feedback.
    p, _ = preview_layout()
    obs["preview"][4, p["base_error"].start] = 0.02
    moved = model.predict(env.normalize_obs(obs), deterministic=True)[0]
    assert np.any(abs(moved[:-1]) > 1e-5)
    obs = equilibrium()
    h, _ = history_layout(MARS, retention_context=True)
    obs["history"][-1, h["base_velocity"].start + 4] = 0.03
    assert np.any(abs(model.predict(env.normalize_obs(obs), deterministic=True)[0][:-1]) > 1e-5)


def test_feedback_policy_ppo_export_and_original_bundle_loader_roundtrip(feedback, tmp_path):
    cfg, env, model = feedback
    model.learn(16)
    assert model.policy.action_net.gain.weight.detach().abs().sum() > 0
    export(model, env, tmp_path, cfg, "feedback-test", steps=16)
    policy = ManipulationPolicy(tmp_path, robot="innate_mars", fingerprint="feedback-test")
    obs = equilibrium()
    np.testing.assert_allclose(policy(obs), model.predict(env.normalize_obs(obs), deterministic=True)[0], atol=1e-7)
    np.testing.assert_array_equal(policy(obs)[:-1], np.zeros(7))


def test_feedback_mode_rejects_running_command_normalization():
    with pytest.raises(ValueError, match="feedback policy"):
        ManipulationConfig(feedback_policy=True)


@pytest.mark.sim
def test_explicit_angular_units_transfer_preserves_actor_and_critic(feedback, tmp_path):
    from emet.rl.manipulation.initialization import _model, transfer_actor

    cfg, env, source = feedback
    with torch.no_grad():
        source.policy.action_net.gain.weight.normal_(std=0.03)
        source.policy.action_net.gain.bias.normal_(std=0.03)
    _, fingerprint = _model(cfg.robot, {})
    export(source, env, tmp_path, cfg, fingerprint, steps=0)
    new_cfg = replace(cfg, feedback_orientation_scale=0.15, wheel_feedback_gain=0.5)
    new_env = CommandVecNormalize(DummyVecEnv([Commands]), config=new_cfg, norm_reward=False)
    try:
        target = PPO(
            FeedbackActorCriticPolicy,
            new_env,
            device="cpu",
            n_steps=8,
            batch_size=4,
            policy_kwargs={
                "features_extractor_class": GoalFeedbackFeatures,
                "features_extractor_kwargs": {"robot": cfg.robot, "retention_context": True, "orientation_scale": 0.15},
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            },
        )
        details = transfer_actor(target, new_env, tmp_path, new_cfg, fingerprint)
        assert details["feedback_orientation_scale_change"]["function_preserving"]
        obs = equilibrium()
        p, _ = preview_layout()
        obs["preview"][:, p["hand_error"].start + 4] = 0.15
        old_obs, new_obs = env.normalize_obs(obs), new_env.normalize_obs(obs)
        np.testing.assert_allclose(
            source.predict(old_obs, deterministic=True)[0], target.predict(new_obs, deterministic=True)[0], atol=1e-6
        )
        with torch.no_grad():
            old_value = source.policy.predict_values(source.policy.obs_to_tensor(old_obs)[0])
            new_value = target.policy.predict_values(target.policy.obs_to_tensor(new_obs)[0])
        torch.testing.assert_close(old_value, new_value, atol=1e-6, rtol=1e-5)
        assert not target.policy.optimizer.state_dict()["state"]
    finally:
        new_env.close()


def test_yaw_audit_distinguishes_correct_error_from_negative_learned_gain(feedback, tmp_path):
    from emet.rl.manipulation.yaw_audit import feedback_terms

    cfg, env, model = feedback
    head = model.policy.action_net
    with torch.no_grad():
        head.zero_output()
        head.gain.bias[head.feedback_width + 2] = -1
        head.gain.bias[head.feedback_width + 14] = -1
    export(model, env, tmp_path, cfg, "feedback-test", steps=0)
    policy = ManipulationPolicy(tmp_path)
    obs = equilibrium()
    h, _ = history_layout(MARS, retention_context=True)
    obs["history"][:, h["base_pose"]] = 0
    obs["history"][:, h["base_goal"]] = [0, 0, 0.04]
    obs["preview"][:, 3] = 0.04
    action, row = feedback_terms(policy, obs)
    assert row["world_yaw_error_rad"] > 0 and row["normalized_yaw_error"] > 0
    assert row["yaw_gain_current"] < 0 and row["yaw_gain_preview"] < 0
    assert action[1] < 0
    np.testing.assert_allclose(action, policy(obs), atol=1e-7)

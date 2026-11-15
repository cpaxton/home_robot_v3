"""Ignored requests must not create saturated history or fictitious PPO credit."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from emet.rl.manipulation.command_policy import ExecutorCommandVecNormalize, ExecutorFeedbackPolicy, ExecutorGaussian
from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.feedback_policy import GoalFeedbackFeatures
from emet.rl.manipulation.observation_layout import history_layout
from emet.rl.manipulation.policy import ManipulationPolicy, export
from emet.rl.manipulation.profiles import MARS


class Requests(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)
    observation_space = gym.spaces.Dict(
        {
            "history": gym.spaces.Box(
                -np.inf, np.inf, (20, history_layout(MARS, retention_context=True)[1]), dtype=np.float32
            ),
            "preview": gym.spaces.Box(-np.inf, np.inf, (7, 18), dtype=np.float32),
        }
    )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = {k: np.zeros(space.shape, dtype=np.float32) for k, space in self.observation_space.spaces.items()}
        fields, _ = history_layout(MARS, retention_context=True)
        obs["history"][:, fields["phase"].start] = 1
        obs["history"][:, fields["base_mask"]] = 1
        obs["preview"][:, 1] = 0.01
        obs["preview"][:, 4:7] = 1
        return obs, {}

    def step(self, action):
        assert np.array_equal(action[2:], np.zeros(6))
        return self.reset()[0], float(action[0]), False, False, {}


def config():
    return ManipulationConfig(
        robot="innate_mars",
        action_mode="direct_velocity",
        feedback_policy=True,
        categorical_features_unscaled=True,
        trajectory_preview=True,
        retention_context=True,
        restoring_yaw_feedback=True,
    )


def model(env, *, shared=True):
    torch.set_num_threads(1)
    return PPO(
        ExecutorFeedbackPolicy,
        env,
        device="cpu",
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        seed=19,
        policy_kwargs={
            "features_extractor_class": GoalFeedbackFeatures,
            "features_extractor_kwargs": {"robot": "innate_mars", "retention_context": True},
            "net_arch": {"pi": [16, 16], "vf": [16, 16]},
            "restoring_yaw_feedback": True,
            "turn_xy_tolerance": config().base_xy_tolerance,
            "share_features_extractor": shared,
        },
    )


def test_ignored_dimensions_do_not_change_likelihood_entropy_or_gradients():
    means = torch.zeros(2, 8, requires_grad=True)
    log_std = torch.zeros(8, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.float32)
    distribution = ExecutorGaussian(8, mask).proba_distribution(means, log_std)
    actions = torch.ones(2, 8) * 0.2 * mask
    expected = distribution.log_prob(actions)
    torch.testing.assert_close(distribution.log_prob(actions + 100 * (1 - mask)), expected)
    entropy = 0.5 * np.log(2 * np.pi * np.e)
    torch.testing.assert_close(distribution.entropy(), mask.sum(dim=1) * entropy)
    distribution.log_prob(actions).sum().backward()
    torch.testing.assert_close(means.grad[mask == 0], torch.zeros_like(means.grad[mask == 0]))
    assert means.grad[mask == 1].abs().sum() > 0 and log_std.grad[-1] == 0
    for _ in range(10):
        assert torch.count_nonzero(distribution.sample() * (1 - mask)) == 0
    assert torch.count_nonzero(distribution.mode() * (1 - mask)) == 0


def test_request_units_remain_finite_after_constant_zero_component_history():
    env = ExecutorCommandVecNormalize(DummyVecEnv([Requests]), config=config(), norm_reward=False)
    try:
        for _ in range(20):
            env.reset()
        fields, _ = history_layout(MARS, retention_context=True)
        source = env.get_original_obs()
        columns = env.component_zero_columns
        source["history"][..., columns] = 0.01
        normalized = env.normalize_obs(source)
        np.testing.assert_allclose(normalized["history"][..., columns], 0.01, rtol=0, atol=1e-8)
        np.testing.assert_array_equal(env.obs_rms["history"].mean[..., columns], 0)
        arm = fields["proprio/previous_action"].start + 2
        assert env.obs_rms["history"].var[-1, arm] < 0.01
    finally:
        env.close()


def test_mask_tracks_visible_base_hand_phase_and_retention_without_disabling_free_base():
    env = ExecutorCommandVecNormalize(DummyVecEnv([Requests]), config=config(), norm_reward=False)
    try:
        actor = model(env)
        obs = env.reset()
        fields, _ = history_layout(MARS, retention_context=True)

        def mask():
            return actor.policy.executor_mask(actor.policy.obs_to_tensor(env.normalize_obs(obs))[0]).numpy()[0]

        np.testing.assert_array_equal(mask(), [1, 1, 0, 0, 0, 0, 0, 0])
        obs["history"][..., fields["hand_mask"]] = 1
        obs["history"][..., fields["base_mask"]] = 0
        np.testing.assert_array_equal(mask(), [1, 1, 1, 1, 1, 1, 1, 0])
        obs["history"][..., fields["phase"]] = 0
        obs["history"][..., fields["phase"].start + 3] = 1
        obs["history"][..., fields["retain_object_command"]] = 1
        np.testing.assert_array_equal(mask(), np.ones(8))
        obs["history"][..., fields["retain_object_command"]] = 0
        assert mask()[-1] == 0
        for phase in [0, 1, 2, 5]:
            obs["history"][..., fields["phase"]] = 0
            obs["history"][..., fields["phase"].start + phase] = 1
            obs["history"][..., fields["retain_object_command"]] = 1
            assert mask()[-1] == 0
    finally:
        env.close()


@pytest.mark.parametrize("shared", [True, False])
def test_actual_ppo_updates_only_effective_head_rows_and_roundtrips(tmp_path, shared):
    env = ExecutorCommandVecNormalize(DummyVecEnv([Requests]), config=config(), norm_reward=False)
    try:
        actor = model(env, shared=shared)
        before = actor.policy.action_net.gain.weight.detach().clone().reshape(7, -1)
        actor.learn(16)
        after = actor.policy.action_net.gain.weight.detach().reshape(7, -1)
        assert not torch.equal(before[:2], after[:2])
        torch.testing.assert_close(before[2:], after[2:], rtol=0, atol=0)
        assert actor.policy.action_net.grip.weight.abs().sum() == 0
        obs = env.reset()
        deterministic = actor.predict(obs, deterministic=True)[0]
        sampled = actor.predict(obs, deterministic=False)[0]
        np.testing.assert_array_equal(deterministic[:, 2:], 0)
        np.testing.assert_array_equal(sampled[:, 2:], 0)
        export(actor, env, tmp_path, config(), "mask-test", steps=16)
        restored = ManipulationPolicy(tmp_path)
        raw = {k: v[0] for k, v in env.get_original_obs().items()}
        np.testing.assert_allclose(restored(raw), deterministic[0], rtol=0, atol=1e-7)
    finally:
        env.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_policy", "unknown"),
        ("training_policy", []),
        ("retention_context", False),
        ("categorical_features_unscaled", False),
    ],
)
def test_masked_training_rejects_unsupported_observation_contract(tmp_path, field, value):
    import json
    from pathlib import Path

    from emet.rl.manipulation.mastery import load_suite

    spec = load_suite(
        Path(__file__).resolve().parents[4] / "configs/benchmarks/mars_tracking_mastery_command_history_v1.json"
    )
    if field == "training_policy":
        spec[field] = value
    else:
        spec["config"][field] = value
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        load_suite(path)


@pytest.mark.sim
def test_native_action_history_contains_zero_ignored_actor_requests():
    from pathlib import Path

    from emet.rl.manipulation.mastery import MasteryEnv, load_suite

    spec = load_suite(
        Path(__file__).resolve().parents[4] / "configs/benchmarks/mars_tracking_mastery_command_history_v1.json"
    )
    base = MasteryEnv(spec, stages=["base", "arm"], split="train")
    env = ExecutorCommandVecNormalize(DummyVecEnv([lambda: base]), config=base.config, norm_reward=False)
    try:
        actor = model(env)
        with torch.no_grad():
            actor.policy.action_net.grip.bias.fill_(1)
            actor.policy.log_std.fill_(np.log(0.01))
        fields, _ = history_layout(MARS, retention_context=True)
        for case_id in ["train:base:hold", "train:arm:pitch_up"]:
            obs, _ = base.reset(options={"case_id": case_id})
            action = actor.predict(env.normalize_obs(obs), deterministic=False)[0]
            assert action[-1] == 0
            if base.mastery_case["stage"] == "base":
                np.testing.assert_array_equal(action[2:], 0)
            obs, *_ = base.step(action)
            assert base.control.last_action[-1] == 0
            assert obs["history"][-1, fields["proprio/previous_action"].stop - 1] == 0
            if base.mastery_case["stage"] == "base":
                np.testing.assert_array_equal(base.control.last_action[2:], 0)
    finally:
        env.close()


@pytest.mark.sim
def test_training_transfer_cannot_silently_drop_the_executor_mask(tmp_path):
    from dataclasses import asdict

    from emet.rl.manipulation.feedback_policy import FeedbackActorCriticPolicy
    from emet.rl.manipulation.initialization import _model, transfer_actor

    env = ExecutorCommandVecNormalize(DummyVecEnv([Requests]), config=config(), norm_reward=False)
    try:
        actor = model(env)
        _, fingerprint = _model("innate_mars", asdict(config()))
        export(actor, env, tmp_path, config(), fingerprint, steps=0)
        target = PPO(
            FeedbackActorCriticPolicy,
            env,
            device="cpu",
            n_steps=8,
            batch_size=8,
            policy_kwargs={
                "features_extractor_class": GoalFeedbackFeatures,
                "features_extractor_kwargs": {"robot": "innate_mars", "retention_context": True},
            },
        )
        with pytest.raises(ValueError, match="remove executor action masks"):
            transfer_actor(target, env, tmp_path, config(), fingerprint)
    finally:
        env.close()

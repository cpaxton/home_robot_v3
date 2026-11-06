"""Whole-body PPO must learn arm motion and export the same single actor."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from test_base_component import Toy, config, kwargs
from test_base_component import setup as setup

from emet.rl.manipulation.base_component import initialize, merge, verify_frozen
from emet.rl.manipulation.command_policy import ExecutorCommandVecNormalize
from emet.rl.manipulation.observation_layout import history_layout
from emet.rl.manipulation.profiles import MARS
from emet.rl.manipulation.whole_body import FrozenGripActions, WholeBodyPolicy


class ArmToy(Toy):
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        fields, _ = history_layout(MARS, retention_context=True)
        obs["history"][:, fields["hand_mask"]] = 1
        obs["preview"][:, 7] = 0.05
        obs["preview"][:, 10] = 1
        obs["preview"][:, 13] = 1
        obs["preview"][:, 16] = 1
        return obs, info

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        return obs, float(action[3]), done, truncated, info


@pytest.fixture
def whole(setup):
    source = setup[2]
    native = ArmToy()
    wrapper = FrozenGripActions(native, source)
    env = ExecutorCommandVecNormalize(DummyVecEnv([lambda: wrapper]), config=config(), norm_reward=False)
    try:
        model = PPO(
            WholeBodyPolicy,
            env,
            device="cpu",
            n_steps=8,
            batch_size=8,
            n_epochs=2,
            seed=19,
            use_sde=True,
            sde_sample_freq=2,
            policy_kwargs=kwargs(),
        )
        initialize(model, env, source, config(), Toy.fingerprint)
        yield model, env, source, native, wrapper
    finally:
        env.close()


def test_seven_motion_mean_transfer_update_export_and_reload(whole, tmp_path):
    model, env, source, _, _ = whole
    obs = env.reset()
    raw = {k: v[0] for k, v in env.get_original_obs().items()}
    np.testing.assert_allclose(model.predict(obs, deterministic=True)[0][0], source(raw)[:7], atol=1e-7, rtol=0)
    before = model.policy.action_net.gain.bias.detach().reshape(7, 35).clone()
    model.learn(32)
    after = model.policy.action_net.gain.bias.detach().reshape(7, 35)
    assert not torch.equal(before[2:], after[2:])
    verify_frozen(model, source)
    policy = merge(model, env, source, tmp_path / "merged", config(), Toy.fingerprint)
    expected = model.predict(obs, deterministic=True)[0][0]
    np.testing.assert_allclose(policy(raw)[:7], expected, atol=1e-7, rtol=0)
    assert policy(raw)[-1] == source(raw)[-1]
    assert policy.model.action_space.shape == (8,)
    assert policy.manifest["component_training"]["scope"] == "mars-whole-body-gsde-v1"
    model.save(tmp_path / "motion.zip")
    loaded = PPO.load(tmp_path / "motion.zip", env=env, device="cpu")
    assert loaded.policy.log_std.shape == (61, 7)
    assert loaded.policy.arm_exploration_coefficient_std == 0.03
    np.testing.assert_array_equal(loaded.predict(obs, deterministic=True)[0], model.predict(obs, deterministic=True)[0])
    loaded.learn(8)
    verify_frozen(loaded, source)


@pytest.mark.parametrize("base,hand", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_masks_control_sampling_mean_likelihood_entropy_and_gradients(whole, base, hand):
    model, env, _, _, _ = whole
    env.reset()
    raw = env.get_original_obs()
    fields, _ = history_layout(MARS, retention_context=True)
    raw["history"][..., fields["base_mask"]] = base
    raw["history"][..., fields["hand_mask"]] = hand
    tensor = model.policy.obs_to_tensor(env.normalize_obs(raw))[0]
    model.policy.reset_noise(1)
    dist = model.policy.get_distribution(tensor)
    allowed = torch.tensor([[bool(base or hand)] * 2 + [bool(hand)] * 5])
    for action in (dist.sample(), dist.mode()):
        assert torch.equal(action[~allowed], torch.zeros_like(action[~allowed]))
    sampled = dist.sample().detach()
    changed = torch.where(allowed, sampled, torch.ones_like(sampled) * 0.7)
    torch.testing.assert_close(dist.log_prob(sampled), dist.log_prob(changed), rtol=0, atol=0)
    expected = (dist.distribution.entropy() * allowed).sum(-1)
    torch.testing.assert_close(dist.entropy(), expected)
    gradient = torch.autograd.grad(dist.log_prob(sampled).sum(), dist.distribution.mean)[0]
    assert torch.equal(gradient[~allowed], torch.zeros_like(gradient[~allowed]))


def test_noise_coupling_respects_base_constraint_and_quiet_floor(whole):
    model, env, _, _, _ = whole
    env.reset()
    raw = env.get_original_obs()
    raw["preview"][..., 1:4] = 0
    raw["preview"][..., 7] = 0.5
    constrained = model.policy.get_distribution(model.policy.obs_to_tensor(env.normalize_obs(raw))[0])
    base_std = constrained.distribution.stddev[:, :2].detach().clone()
    assert base_std.max() < 0.002
    fields, _ = history_layout(MARS, retention_context=True)
    raw["history"][..., fields["base_mask"]] = 0
    free = model.policy.get_distribution(model.policy.obs_to_tensor(env.normalize_obs(raw))[0])
    assert torch.all(free.distribution.stddev[:, :2] > base_std * 10)
    raw["preview"][..., 7:16] = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0])
    quiet = model.policy.get_distribution(model.policy.obs_to_tensor(env.normalize_obs(raw))[0])
    assert quiet.distribution.stddev.max() < 0.002


@pytest.mark.parametrize("action", [[0] * 8, [0] * 6, [float("nan")] * 7, [2] * 7])
def test_grip_wrapper_rejects_invalid_motion(whole, action):
    wrapper = whole[-1]
    wrapper.reset()
    with pytest.raises(ValueError):
        wrapper.step(action)


def test_grip_wrapper_keeps_source_gripper_and_requires_reset(whole):
    _, _, source, native, wrapper = whole
    assert isinstance(wrapper.action_space, gym.spaces.Box)
    with pytest.raises(ValueError):
        wrapper.step(np.zeros(7))
    obs, _ = wrapper.reset()
    grip = source(obs)[-1]
    action = np.linspace(-0.1, 0.1, 7)
    wrapper.step(action)
    np.testing.assert_allclose(native.last, np.r_[action, grip], atol=1e-7, rtol=0)


@pytest.mark.sim
@pytest.mark.parametrize("stage,case", [("arm", "train:arm:translate_down"), ("coupled", "train:coupled:forward_turn")])
def test_whole_body_runs_native_physics_with_no_context_or_grip_change(tmp_path, stage, case):
    from pathlib import Path

    from emet.rl.manipulation.command_policy import ExecutorFeedbackPolicy
    from emet.rl.manipulation.mastery import MasteryEnv, TrainingCaseSampler, load_suite
    from emet.rl.manipulation.policy import ManipulationPolicy, export

    suite = load_suite(
        Path(__file__).resolve().parents[4] / "configs/benchmarks/mars_tracking_mastery_full_arm_v1.json"
    )
    native = MasteryEnv(suite, stages=[stage], split="train", population="training", training_stage_weights={stage: 1})
    native.cases = [c for c in native.cases if c["id"] == case]
    native.sampler = TrainingCaseSampler(native.cases, {stage: 1})
    params = kwargs()
    params["features_extractor_kwargs"]["orientation_scale"] = native.config.feedback_orientation_scale
    source_env = ExecutorCommandVecNormalize(DummyVecEnv([lambda: native]), config=native.config, norm_reward=False)
    env = None
    try:
        source_model = PPO(
            ExecutorFeedbackPolicy, source_env, device="cpu", n_steps=8, batch_size=8, policy_kwargs=params
        )
        export(source_model, source_env, tmp_path / "source", native.config, native.fingerprint, steps=0)
        source = ManipulationPolicy(tmp_path / "source")
        env = ExecutorCommandVecNormalize(
            DummyVecEnv([lambda: FrozenGripActions(native, source)]), config=native.config, norm_reward=False
        )
        model = PPO(
            WholeBodyPolicy,
            env,
            device="cpu",
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            use_sde=True,
            sde_sample_freq=2,
            policy_kwargs=params,
        )
        initialize(model, env, source, native.config, native.fingerprint)
        model.learn(16)
        assert model.num_timesteps == 16
        assert native.mastery_case["id"] == case
        assert native.mastery_admission["accepted"]
        verify_frozen(model, source)
        raw = {k: v[0] for k, v in env.get_original_obs().items()}
        expected = model.predict(env.normalize_obs({k: v[None] for k, v in raw.items()}), deterministic=True)[0][0]
        actor = merge(model, env, source, tmp_path / "actor", native.config, native.fingerprint)
        np.testing.assert_allclose(actor(raw)[:7], expected, atol=1e-7, rtol=0)
        assert actor(raw)[-1] == source(raw)[-1]
    finally:
        if env is not None:
            env.close()
        source_env.close()

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from emet.rl.manipulation.arm_adaptation import ArmAdaptationGaussian, ArmAdaptationPolicy
from emet.rl.manipulation.command_policy import ExecutorCommandVecNormalize, ExecutorFeedbackPolicy
from emet.rl.manipulation.feedback_policy import GoalFeedbackFeatures
from emet.rl.manipulation.observation_layout import history_layout, preview_layout
from emet.rl.manipulation.policy import ManipulationPolicy, export
from emet.rl.manipulation.profiles import MARS

from .test_command_policy import config
from .test_feedback_policy import Commands, equilibrium


class ArmRequests(Commands):
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        obs = equilibrium()
        fields, _ = history_layout(MARS, retention_context=True)
        preview, _ = preview_layout()
        obs["history"][:, fields["phase"]] = 0
        obs["history"][:, fields["phase"].start] = 1
        obs["history"][:, fields["base_mask"]] = 1
        obs["history"][:, fields["hand_mask"]] = 1
        obs["history"][:, fields["retain_object_command"]] = 0
        obs["preview"][:, preview["base_error"]] = [0.02, 0, 0.03]
        obs["preview"][:, preview["hand_error"].start] = 0.004
        obs["preview"][2, preview["hand_error"].start] = 0.006
        return obs, {}

    def step(self, action):
        return self.reset()[0], float(action[2]), False, False, {}


def make_model(env, cfg, policy=ArmAdaptationPolicy, *, shared=True, weight_decay=0):
    torch.set_num_threads(1)
    return PPO(
        policy,
        env,
        device="cpu",
        seed=21,
        n_steps=8,
        batch_size=8,
        n_epochs=2,
        learning_rate=0.001,
        policy_kwargs={
            "features_extractor_class": GoalFeedbackFeatures,
            "features_extractor_kwargs": GoalFeedbackFeatures.kwargs_for(cfg),
            "net_arch": {"pi": [16, 16], "vf": [16, 16]},
            "restoring_yaw_feedback": cfg.restoring_yaw_feedback,
            "turn_xy_tolerance": cfg.base_xy_tolerance,
            "share_features_extractor": shared,
            "optimizer_kwargs": {"weight_decay": weight_decay},
        },
    )


def probe_observations(env):
    rng = np.random.default_rng(2)
    fields, _ = history_layout(MARS, retention_context=True)
    rows = []
    for index in range(12):
        obs = ArmRequests().reset()[0]
        obs["history"][:, fields["proprio/positions"]] = rng.uniform(-0.5, 0.5, (20, MARS.arm_size))
        if index % 2:
            obs["history"][:, fields["phase"]] = 0
            obs["history"][:, fields["phase"].start + 3] = 1
            obs["history"][:, fields["retain_object_command"]] = 1
        rows.append(env.normalize_obs(obs))
    return {key: np.stack([obs[key] for obs in rows]) for key in rows[0]}


def test_arm_marginal_keeps_nonzero_frozen_means_without_noise_or_likelihood():
    means = torch.full((16, 8), 0.2, requires_grad=True)
    log_std = torch.full((8,), -0.5, requires_grad=True)
    mask = torch.ones_like(means)
    mask[0, 3] = 0
    distribution = ArmAdaptationGaussian(8, mask).proba_distribution(means, log_std)
    samples = torch.stack([distribution.sample() for _ in range(5)])
    torch.testing.assert_close(samples[:, :, [0, 1, 7]], means[None, :, [0, 1, 7]].expand(5, -1, -1))
    assert samples[:, :, 2:7].std() > 0.1
    assert samples[:, 0, 3].count_nonzero() == 0
    expected = distribution.log_prob(distribution.mode())
    changed = distribution.mode().detach().clone()
    changed[:, [0, 1, 7]] = 100
    torch.testing.assert_close(distribution.log_prob(changed), expected)
    (-expected.mean()).backward()
    torch.testing.assert_close(log_std.grad[[0, 1, 7]], torch.zeros(3))
    assert log_std.grad[2:7].abs().sum() > 0


@pytest.mark.parametrize("velocity", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_ppo_learns_arm_and_critic_with_bitwise_frozen_base_context_grip(tmp_path, velocity, shared):
    cfg = replace(config(), feedback_reference_velocity=velocity)
    env = ExecutorCommandVecNormalize(DummyVecEnv([ArmRequests]), config=cfg, norm_reward=False)
    try:
        env.training = False
        model = make_model(env, cfg, shared=shared)
        with torch.no_grad():
            gains = model.policy.action_net.gain.bias.reshape(7, -1)
            gains[0, 0], gains[1, 2] = 1.5, -0.7
            model.policy.action_net.grip.bias.fill_(0.2)
        frozen = model.policy.frozen_function_state()
        obs = probe_observations(env)
        before = model.predict(obs, deterministic=True)[0]
        arm_before = model.policy.action_net.gain.weight.detach().clone()
        critic_before = model.policy.value_net.weight.detach().clone()
        assert np.abs(before[:, :2]).max() > 0.01 and np.abs(before[:, -1]).max() > 0.1
        model.learn(32)
        model.policy.assert_frozen_function(frozen)
        after = model.predict(obs, deterministic=True)[0]
        np.testing.assert_array_equal(after[:, [0, 1, 7]], before[:, [0, 1, 7]])
        assert not torch.equal(arm_before, model.policy.action_net.gain.weight)
        assert not torch.equal(critic_before, model.policy.value_net.weight)
        for _ in range(5):
            sampled = model.predict(obs, deterministic=False)[0]
            np.testing.assert_array_equal(sampled[:, [0, 1, 7]], after[:, [0, 1, 7]])
        export(model, env, tmp_path, cfg, "arm-adaptation-test", steps=32)
        restored = ManipulationPolicy(tmp_path)
        raw = ArmRequests().reset()[0]
        np.testing.assert_allclose(
            restored(raw), model.predict(env.normalize_obs(raw), deterministic=True)[0], atol=1e-7
        )
        restored.model.set_env(env)
        loaded_frozen = restored.model.policy.frozen_function_state()
        restored.model.learn(16)
        restored.model.policy.assert_frozen_function(loaded_frozen)
    finally:
        env.close()


def test_protected_context_mutation_is_rejected():
    cfg = config()
    env = ExecutorCommandVecNormalize(DummyVecEnv([ArmRequests]), config=cfg, norm_reward=False)
    try:
        model = make_model(env, cfg)
        frozen = model.policy.frozen_function_state()
        with torch.no_grad():
            model.policy.mlp_extractor.policy_net[0].weight[0, 0] += 0.1
        with pytest.raises(RuntimeError, match="frozen base/context/grip function changed"):
            model.policy.assert_frozen_function(frozen)
        with pytest.raises(ValueError, match="zero weight decay"):
            make_model(env, cfg, weight_decay=0.01)
    finally:
        env.close()


@pytest.mark.parametrize("change", ["stage", "source", "normalization", "executor_mask", "arm_component"])
def test_training_requires_the_frozen_source_contract(monkeypatch, change):
    from emet.rl import regression_gate
    from emet.rl.manipulation import mastery

    monkeypatch.setattr(regression_gate, "verify_regressions", lambda path: None)
    args = SimpleNamespace(
        regression_receipt="unused",
        arm_adaptation_only=True,
        stages=["arm"],
        initialize_from="source",
        freeze_normalization=True,
        initialize_arm_from=None,
    )
    suite = {"training_policy": "executor_mask_v1"}
    if change == "stage":
        args.stages = ["base", "arm"]
    elif change == "source":
        args.initialize_from = None
    elif change == "normalization":
        args.freeze_normalization = False
    elif change == "executor_mask":
        suite["training_policy"] = "feedback"
    else:
        args.initialize_arm_from = "component"
    with pytest.raises(ValueError, match="arm adaptation requires"):
        mastery.train(args, suite)


@pytest.mark.sim
@pytest.mark.parametrize("velocity", [False, True])
def test_executor_checkpoint_transfer_preserves_all_initial_mean_commands(tmp_path, velocity):
    from emet.rl.manipulation.initialization import _model, transfer_actor

    cfg = config()
    env = ExecutorCommandVecNormalize(DummyVecEnv([ArmRequests]), config=cfg, norm_reward=False)
    new_cfg = replace(cfg, feedback_reference_velocity=velocity)
    target_env = ExecutorCommandVecNormalize(DummyVecEnv([ArmRequests]), config=new_cfg, norm_reward=False)
    try:
        source = make_model(env, cfg, policy=ExecutorFeedbackPolicy)
        with torch.no_grad():
            for parameter in source.policy.action_net.parameters():
                parameter.normal_(std=0.03)
        _, fingerprint = _model(cfg.robot, {})
        export(source, env, tmp_path, cfg, fingerprint, steps=0)
        target = make_model(target_env, new_cfg)
        transfer_actor(target, target_env, tmp_path, new_cfg, fingerprint)
        raw = ArmRequests().reset()[0]
        np.testing.assert_allclose(
            source.predict(env.normalize_obs(raw), deterministic=True)[0],
            target.predict(target_env.normalize_obs(raw), deterministic=True)[0],
            atol=1e-6,
        )
        frozen = target.policy.frozen_function_state()
        target_env.training = False
        target.learn(16)
        target.policy.assert_frozen_function(frozen)
    finally:
        env.close()
        target_env.close()

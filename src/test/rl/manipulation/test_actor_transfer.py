import json
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.feedback_policy import FeedbackActorCriticPolicy, GoalFeedbackFeatures
from emet.rl.manipulation.initialization import _model, transfer_actor
from emet.rl.manipulation.normalization import CommandVecNormalize
from emet.rl.manipulation.policy import ManipulationMLPFeatures, ManipulationPolicy, export


def environment(context):
    class Task(gym.Env):
        action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)
        observation_space = gym.spaces.Dict(
            {
                "history": gym.spaces.Box(-np.inf, np.inf, (20, 101 if context else 99), dtype=np.float32),
                "preview": gym.spaces.Box(-np.inf, np.inf, (7, 18), dtype=np.float32),
            }
        )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {k: np.zeros(s.shape, dtype=np.float32) for k, s in self.observation_space.spaces.items()}, {}

        def step(self, action):
            return self.reset()[0], -float(np.sum(action**2)), False, False, {}

    return DummyVecEnv([Task])


@pytest.fixture
def transfer(tmp_path):
    torch.set_num_threads(1)
    old_config = ManipulationConfig(robot="innate_mars", action_mode="direct_velocity", trajectory_preview=True)
    config = replace(
        old_config,
        retention_context=True,
        categorical_features_unscaled=True,
        contact_cone="elliptic",
        contact_impratio=3,
    )
    old_env = VecNormalize(environment(False), norm_reward=False)
    env = CommandVecNormalize(environment(True), config=config, norm_reward=False)
    kwargs = {
        "device": "cpu",
        "n_steps": 8,
        "batch_size": 4,
        "n_epochs": 1,
        "policy_kwargs": {"features_extractor_class": ManipulationMLPFeatures},
    }
    old, target = PPO("MultiInputPolicy", old_env, **kwargs), PPO("MultiInputPolicy", env, **kwargs)
    old.learn(8)
    _, source_fp = _model("innate_mars", {})
    _, target_fp = _model("innate_mars", {"contact_cone": "elliptic", "contact_impratio": 3})
    export(old, old_env, tmp_path, old_config, source_fp, steps=8)
    yield old, target, env, config, tmp_path, target_fp
    env.close()
    old_env.close()


def test_training_transfer_does_not_bypass_runtime_fingerprint_or_copy_optimizer(transfer):
    old, target, env, cfg, path, fingerprint = transfer
    with pytest.raises(ValueError, match="model mismatch"):
        ManipulationPolicy(path, robot="innate_mars", fingerprint=fingerprint)
    result = transfer_actor(target, env, path, cfg, fingerprint, prior_count=4)
    assert set(result["contact_solver_changes"]) == {"cone", "impratio"}
    assert result["optimizer_restored"] is False and result["runtime_admitted"] is False
    assert old.num_timesteps == 8 and target.num_timesteps == 0
    assert target.policy.optimizer.state_dict()["state"] == {}
    assert env.obs_rms["history"].count == 4
    layer = target.policy.mlp_extractor.policy_net[0].weight.detach()[:, : 4 * 101].reshape(-1, 4, 101)
    np.testing.assert_array_equal(layer[:, :, -2:].numpy(), 0)
    assert np.isfinite(env.obs_rms["history"].mean).all()


def test_transfer_rejects_unapproved_actuator_changes(transfer, monkeypatch):
    _, target, env, cfg, path, fingerprint = transfer

    def changed(robot, settings):
        model, fp = _model(robot, settings)
        if settings.get("contact_cone") == "elliptic":
            model.actuator_gear[0, 0] *= 2
        return model, fp

    monkeypatch.setattr("emet.rl.manipulation.initialization._model", changed)
    with pytest.raises(ValueError, match="robot or unsupported physics"):
        transfer_actor(target, env, path, cfg, fingerprint)


def test_legacy_feedback_contract_is_training_only_and_units_are_pinned(transfer):
    _, target, env, cfg, path, fingerprint = transfer
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["contract"]["version"] = 6
    manifest["contract"].pop("wheel_velocity_feedback")
    manifest_path.write_text(json.dumps(manifest))
    original = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="contract mismatch"):
        ManipulationPolicy(path)
    result = transfer_actor(target, env, path, cfg, fingerprint)
    assert result["control_contract_changes"]["version"] == {"from": 6, "to": 7}
    assert manifest_path.read_bytes() == original
    manifest["contract"]["base_speed"] *= 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported transfer control contract"):
        transfer_actor(target, env, path, cfg, fingerprint)


def test_tracking_context_transfer_reinitializes_feedback_mean_head(transfer):
    _, _, _, cfg, path, fingerprint = transfer
    cfg = replace(cfg, feedback_policy=True)
    env = CommandVecNormalize(environment(True), config=cfg, norm_reward=False)
    try:
        target = PPO(
            FeedbackActorCriticPolicy,
            env,
            device="cpu",
            n_steps=8,
            batch_size=4,
            policy_kwargs={
                "features_extractor_class": GoalFeedbackFeatures,
                "features_extractor_kwargs": {"robot": "innate_mars", "retention_context": True},
            },
        )
        result = transfer_actor(target, env, path, cfg, fingerprint)
        assert result["feedback_mean_head_reinitialized"]
        assert target.policy.action_net.gain.weight.detach().abs().sum() == 0
        assert target.policy.action_net.grip.bias.detach().abs().sum() == 0
        assert target.policy.optimizer.state_dict()["state"] == {}
    finally:
        env.close()

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from emet.rl.manipulation.__main__ import configure_observation_normalization
from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.normalization import CommandVecNormalize, categorical_columns
from emet.rl.manipulation.observation_layout import history_layout
from emet.rl.manipulation.policy import ManipulationMLPFeatures, ManipulationPolicy, export
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
        return {k: np.zeros(space.shape, dtype=np.float32) for k, space in self.observation_space.spaces.items()}, {}

    def step(self, action):
        return self.reset()[0], 0.0, False, False, {}


@pytest.mark.parametrize("frozen", [False, True])
def test_continuation_can_freeze_statistics_without_changing_the_observation_contract(frozen):
    cfg = ManipulationConfig(robot="innate_mars", retention_context=True, categorical_features_unscaled=True)
    env = CommandVecNormalize(DummyVecEnv([Commands]), config=cfg, norm_reward=False)
    try:
        env.reset()
        before = {k: (s.mean.copy(), s.var.copy(), s.count) for k, s in env.obs_rms.items()}
        configure_observation_normalization(env, frozen=frozen, initialized=True)
        env.reset()
        env.step(np.ones((1, 8)))
        for key, stats in env.obs_rms.items():
            if frozen:
                np.testing.assert_array_equal(stats.mean, before[key][0])
                np.testing.assert_array_equal(stats.var, before[key][1])
                assert stats.count == before[key][2]
            else:
                assert stats.count > before[key][2]
        obs = env.get_original_obs()
        for key, mask in categorical_columns(MARS, retention_context=True).items():
            np.testing.assert_array_equal(env.normalize_obs(obs)[key][..., mask], obs[key][..., mask])
    finally:
        env.close()


def test_freezing_requires_an_initialized_normalizer():
    with pytest.raises(ValueError, match="initialized"):
        configure_observation_normalization(None, frozen=True, initialized=False)


def test_unseen_phase_remains_binary_and_export_reproduces_normalization(tmp_path):
    torch.set_num_threads(1)
    cfg = ManipulationConfig(
        robot="innate_mars",
        action_mode="direct_velocity",
        trajectory_preview=True,
        retention_context=True,
        categorical_features_unscaled=True,
    )
    env = CommandVecNormalize(DummyVecEnv([Commands]), config=cfg, norm_reward=False)
    try:
        env.reset()
        for _ in range(8):
            env.step(np.zeros((1, 8)))
        obs = {k: np.zeros(space.shape, dtype=np.float32) for k, space in env.observation_space.spaces.items()}
        fields, _ = history_layout(MARS, retention_context=True)
        obs["history"][:, fields["phase"].start + 5] = 1
        obs["history"][:, fields["base_mask"]] = 1
        obs["preview"][:, 4:7] = 1
        normalized = env.normalize_obs(obs)
        for key, mask in categorical_columns(MARS, retention_context=True).items():
            np.testing.assert_array_equal(normalized[key][..., mask], obs[key][..., mask])
        assert np.max(normalized["history"][:, fields["phase"]]) == 1
        model = PPO(
            "MultiInputPolicy",
            env,
            device="cpu",
            n_steps=8,
            batch_size=4,
            policy_kwargs={"features_extractor_class": ManipulationMLPFeatures},
        )
        export(model, env, tmp_path, cfg, "normalization-test", steps=0)
        bundle = ManipulationPolicy(tmp_path, robot="innate_mars", fingerprint="normalization-test")
        np.testing.assert_allclose(bundle(obs), model.predict(normalized, deterministic=True)[0], atol=1e-7)
    finally:
        env.close()


def test_feedback_normalization_rejects_epsilon_that_would_store_negative_variance():
    cfg = ManipulationConfig(
        robot="innate_mars",
        feedback_policy=True,
        action_mode="direct_velocity",
        categorical_features_unscaled=True,
        trajectory_preview=True,
        retention_context=True,
    )
    with pytest.raises(ValueError, match="fixed physical variance"):
        CommandVecNormalize(DummyVecEnv([Commands]), config=cfg, epsilon=0.001, norm_reward=False)

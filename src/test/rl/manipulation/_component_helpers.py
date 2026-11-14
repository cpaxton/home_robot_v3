"""Shared lightweight motion-training fixtures; no tests import another test module."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from emet.rl.manipulation.base_component import BasePolicy, FrozenArmActions, initialize
from emet.rl.manipulation.command_policy import ExecutorCommandVecNormalize, ExecutorFeedbackPolicy
from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.feedback_policy import GoalFeedbackFeatures
from emet.rl.manipulation.observation_layout import history_layout
from emet.rl.manipulation.policy import ManipulationPolicy, export
from emet.rl.manipulation.profiles import MARS


class Toy(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)
    observation_space = gym.spaces.Dict(
        {
            "history": gym.spaces.Box(-np.inf, np.inf, (20, 101), dtype=np.float32),
            "preview": gym.spaces.Box(-np.inf, np.inf, (7, 18), dtype=np.float32),
        }
    )
    profile = MARS
    fingerprint = "base-component-test"

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = {k: np.zeros(v.shape, dtype=np.float32) for k, v in self.observation_space.spaces.items()}
        fields, _ = history_layout(MARS, retention_context=True)
        obs["history"][:, fields["phase"].start] = 1
        obs["history"][:, fields["base_mask"]] = 1
        obs["preview"][:, 1] = 0.02
        obs["preview"][:, 4:7] = 1
        self.last = None
        return obs, {}

    def step(self, action):
        obs = self.reset()[0]
        self.last = np.asarray(action).copy()
        return obs, float(action[0]), False, False, {}


def config():
    return ManipulationConfig(
        robot="innate_mars",
        action_mode="direct_velocity",
        feedback_policy=True,
        retention_context=True,
        trajectory_preview=True,
        categorical_features_unscaled=True,
        restoring_yaw_feedback=True,
        base_xy_tolerance=0.005,
    )


def kwargs():
    return {
        "features_extractor_class": GoalFeedbackFeatures,
        "features_extractor_kwargs": {"robot": "innate_mars", "retention_context": True, "orientation_scale": 1.0},
        "restoring_yaw_feedback": True,
        "turn_xy_tolerance": 0.005,
        "net_arch": {"pi": [16, 16], "vf": [16, 16]},
    }


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(1)
    source_env = ExecutorCommandVecNormalize(DummyVecEnv([Toy]), config=config(), norm_reward=False)
    try:
        source_model = PPO(
            ExecutorFeedbackPolicy, source_env, device="cpu", n_steps=8, batch_size=8, policy_kwargs=kwargs()
        )
        source_env.reset()
        with torch.no_grad():
            source_model.policy.action_net.gain.weight.reshape(7, 35, -1)[2:].fill_(0.01)
            source_model.policy.action_net.gain.bias.reshape(7, 35)[2:].fill_(0.03)
        export(source_model, source_env, tmp_path / "source", config(), Toy.fingerprint, steps=0)
        source = ManipulationPolicy(tmp_path / "source")
        base = Toy()
        wrapper = FrozenArmActions(base, source)
        env = ExecutorCommandVecNormalize(DummyVecEnv([lambda: wrapper]), config=config(), norm_reward=False)
        try:
            model = PPO(
                BasePolicy,
                env,
                device="cpu",
                n_steps=8,
                batch_size=8,
                n_epochs=1,
                seed=19,
                use_sde=True,
                sde_sample_freq=2,
                policy_kwargs=kwargs(),
            )
            initialize(model, env, source, config(), Toy.fingerprint)
            yield model, env, source, base, wrapper
        finally:
            env.close()
    finally:
        source_env.close()



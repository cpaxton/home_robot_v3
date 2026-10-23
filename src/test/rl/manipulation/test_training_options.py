from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from emet.rl.manipulation.__main__ import FixedResetSeed, PhaseLearningRate, resume_training, training_options
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.policy import ManipulationMLPFeatures, ManipulationPolicy, export


def test_learning_rate_decay_covers_additional_steps_instead_of_lifetime_steps():
    schedule = PhaseLearningRate(2e-5, 2e-6, 262144, 65536)
    assert schedule(0.2) == pytest.approx(2e-5)
    assert schedule(0.1) == pytest.approx(1.1e-5)
    assert schedule(0) == pytest.approx(2e-6)
    assert schedule(-0.1) == pytest.approx(2e-6)
    assert schedule(1) == pytest.approx(2e-5)
    assert PhaseLearningRate(1e-4, 1e-4, 0, 10)(0.5) == 1e-4
    for start, end in [(0, 0), (1e-4, 0), (1e-4, float("nan")), (1e-5, 1e-4)]:
        with pytest.raises(ValueError):
            PhaseLearningRate(start, end, 0, 10)


@pytest.mark.sim
def test_fixed_case_training_wrapper_repeats_physics_and_reference_after_motion():
    env = FixedResetSeed(
        ManipulationEnv(ManipulationConfig(robot="innate_mars", scenario="reach", trajectory_preview=True)),
        fixed_seed=250000,
    )
    try:
        first, _ = env.reset(seed=1)
        qpos = env.unwrapped.data.qpos.copy()
        reference = env.unwrapped.control.hand_goal.copy()
        env.step(np.ones(env.action_space.shape, dtype=np.float32) * 0.1)
        again, _ = env.reset(seed=2)
        np.testing.assert_array_equal(env.unwrapped.data.qpos, qpos)
        np.testing.assert_array_equal(env.unwrapped.control.hand_goal, reference)
        for key in first:
            np.testing.assert_array_equal(first[key], again[key])
    finally:
        env.close()


def test_scratch_and_dagger_are_independent_options_with_appropriate_exploration():
    args = SimpleNamespace(
        dagger=False,
        action_mode="direct_velocity",
        initialization_steps=0,
        envs=4,
        exploration_std=None,
        architecture="mlp",
    )
    assert training_options(args) == {"initialization": "none", "architecture": "mlp", "exploration_std": 0.3}
    args.dagger = True
    with pytest.raises(ValueError):
        training_options(args)
    args.initialization_steps = 20
    assert training_options(args)["initialization"] == "dagger"
    assert training_options(args)["exploration_std"] == 0.03
    args.exploration_std = 0.2
    assert training_options(args)["exploration_std"] == 0.2
    args.action_mode = "residual"
    with pytest.raises(ValueError):
        training_options(args)
    args.dagger = False
    assert training_options(args)["initialization"] == "critic_warmup"
    args.exploration_std = float("nan")
    with pytest.raises(ValueError):
        training_options(args)


@pytest.mark.parametrize("preview", [False, True])
def test_mlp_ppo_update_and_bundle_roundtrip_without_demonstrations(tmp_path, preview):
    class Task(gym.Env):
        observation_space = gym.spaces.Dict(
            {
                "history": gym.spaces.Box(-10, 10, (20, 99), dtype=np.float32),
                **({"preview": gym.spaces.Box(-10, 10, (7, 18), dtype=np.float32)} if preview else {}),
            }
        )
        action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {
                key: np.zeros(space.shape, dtype=np.float32) for key, space in self.observation_space.spaces.items()
            }, {}

        def step(self, action):
            return self.reset()[0], -float(np.sum(action**2)), False, False, {}

    torch.set_num_threads(1)
    env = VecNormalize(DummyVecEnv([Task]), norm_reward=False)
    try:
        model = PPO(
            "MultiInputPolicy",
            env,
            device="cpu",
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            seed=4,
            policy_kwargs={"features_extractor_class": ManipulationMLPFeatures, "log_std_init": np.log(0.3)},
        )
        model.learn(16)
        export(
            model,
            env,
            tmp_path,
            ManipulationConfig(robot="innate_mars", action_mode="direct_velocity", trajectory_preview=preview),
            "test-only-fingerprint",
            steps=16,
        )
        loaded = ManipulationPolicy(tmp_path, robot="innate_mars", fingerprint="test-only-fingerprint")
        obs = {key: np.zeros(space.shape, dtype=np.float32) for key, space in env.observation_space.spaces.items()}
        expected = model.predict(
            env.normalize_obs({key: value[None] for key, value in obs.items()}), deterministic=True
        )[0][0]
        np.testing.assert_allclose(loaded(obs), expected, atol=1e-7)
        assert loaded.manifest["feature_extractor"] == "ManipulationMLPFeatures"
        assert loaded.action_mode == "direct_velocity"
        before_steps = model.num_timesteps
        optimizer = model.policy.optimizer.state_dict()
        for stats in env.obs_rms.values():
            stats.mean[:] = 777
            stats.var[:] = 999
            stats.count = 1
        resumed = resume_training(
            tmp_path,
            env,
            ManipulationConfig(robot="innate_mars", action_mode="direct_velocity", trajectory_preview=preview),
            "test-only-fingerprint",
            "mlp",
        )
        assert resumed.num_timesteps == before_steps
        for key, (mean, var) in loaded.normalization.items():
            np.testing.assert_array_equal(env.obs_rms[key].mean, mean)
            np.testing.assert_array_equal(env.obs_rms[key].var, var)
            assert env.obs_rms[key].count == loaded.manifest["normalization"][key]["count"]
        restored = resumed.policy.optimizer.state_dict()
        for key, state in optimizer["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(restored["state"][key][name], value)
        np.testing.assert_allclose(
            resumed.predict(env.normalize_obs({k: v[None] for k, v in obs.items()}), deterministic=True)[0][0],
            expected,
            atol=1e-7,
        )
        resumed.learning_rate = resumed.lr_schedule = PhaseLearningRate(1e-4, 1e-5, before_steps, 8)
        resumed.learn(8, reset_num_timesteps=False)
        assert resumed.num_timesteps > before_steps
        assert resumed.policy.optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
        with pytest.raises(ValueError, match="semantics"):
            resume_training(
                tmp_path,
                env,
                ManipulationConfig(robot="innate_mars", trajectory_preview=preview),
                "test-only-fingerprint",
                "mlp",
            )
        if preview:
            with pytest.raises(ValueError):
                loaded({"history": obs["history"]})
            obs["preview"][0, 0] = np.nan
            with pytest.raises(ValueError):
                loaded(obs)
    finally:
        env.close()

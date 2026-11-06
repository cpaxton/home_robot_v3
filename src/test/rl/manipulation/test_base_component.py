"""Base optimization must preserve the arm function and export one actor."""

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from emet.rl.manipulation.base_component import (
    BasePolicy,
    FrozenArmActions,
    initialize,
    merge,
    verify_frozen,
    verify_source_fidelity,
)
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


def test_sde_noise_is_coherent_until_resampled_and_quiet_at_zero_error(setup):
    model, env, _, _, _ = setup
    obs = env.reset()
    model.policy.reset_noise(1)
    first = model.predict(obs, deterministic=False)[0]
    np.testing.assert_array_equal(model.predict(obs, deterministic=False)[0], first)
    model.policy.reset_noise(1)
    assert not np.array_equal(model.predict(obs, deterministic=False)[0], first)
    raw = env.get_original_obs()
    raw["preview"][..., 1:4] = 0
    quiet = env.normalize_obs(raw)
    distribution = model.policy.get_distribution(model.policy.obs_to_tensor(quiet)[0])
    assert distribution.distribution.stddev.max() < 0.002
    model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0])
    actions = model.predict(obs, deterministic=False)[0]
    values, logp, entropy = model.policy.evaluate_actions(model.policy.obs_to_tensor(obs)[0], torch.as_tensor(actions))
    assert torch.isfinite(logp).all() and torch.isfinite(entropy).all() and torch.isfinite(values).all()


def test_actual_updates_preserve_arm_and_context_and_merge_means(setup, tmp_path):
    model, env, source, _, _ = setup
    before = model.policy.action_net.gain.weight.detach().clone()
    model.learn(16)
    assert not torch.equal(before, model.policy.action_net.gain.weight)
    verify_frozen(model, source)
    obs = env.reset()
    raw = {k: v[0] for k, v in env.get_original_obs().items()}
    expected_base = model.predict(obs, deterministic=True)[0][0]
    expected_arm = source(raw)[2:]
    original = {k: v.clone() for k, v in source.model.policy.state_dict().items()}
    policy = merge(model, env, source, tmp_path / "merged", config(), Toy.fingerprint)
    np.testing.assert_allclose(policy(raw)[:2], expected_base, rtol=0, atol=1e-7)
    np.testing.assert_allclose(policy(raw)[2:], expected_arm, rtol=0, atol=1e-7)
    fields, _ = history_layout(MARS, retention_context=True)
    raw["history"][:, fields["hand_mask"]] = 1
    raw["preview"][:, 16] = 1
    raw["preview"][:, 7] = 0.005
    assert np.max(abs(source(raw)[2:7])) > 1e-6
    np.testing.assert_allclose(policy(raw)[2:7], source(raw)[2:7], rtol=0, atol=1e-7)
    assert policy.model.action_space.shape == (8,)
    assert policy.manifest["component_training"]["one_runtime_actor"]
    for key, value in original.items():
        torch.testing.assert_close(source.model.policy.state_dict()[key], value, rtol=0, atol=0)
    model.save(tmp_path / "component.zip")
    loaded = PPO.load(tmp_path / "component.zip", env=env, device="cpu")
    np.testing.assert_allclose(
        loaded.predict(obs, deterministic=True)[0], model.predict(obs, deterministic=True)[0], atol=1e-7
    )
    loaded.learn(8)
    verify_frozen(loaded, source)


@pytest.mark.parametrize("action", [[0] * 8, [float("nan")] * 2, [1.1, 0]])
def test_wrapper_rejects_malformed_base_requests(setup, action):
    wrapper = setup[-1]
    wrapper.reset()
    with pytest.raises(ValueError):
        wrapper.step(action)


def test_wrapper_uses_frozen_learned_arm_and_rejects_pre_reset(setup):
    _, _, source, base, wrapper = setup
    with pytest.raises(ValueError):
        wrapper.step([0, 0])
    obs, _ = wrapper.reset()
    expected = source(obs).copy()
    expected[:2] = [0.1, -0.2]
    wrapper.step([0.1, -0.2])
    np.testing.assert_allclose(base.last, expected, rtol=0, atol=1e-7)


def test_frozen_guard_detects_arm_mutation(setup):
    model, _, source, _, _ = setup
    with torch.no_grad():
        model.policy.action_net.gain.bias.reshape(7, 35)[2, 0] += 1
    with pytest.raises(RuntimeError, match="frozen arm/context"):
        verify_frozen(model, source)


def test_initialization_rejects_environment_drift(setup):
    model, env, source, _, _ = setup
    cfg = replace(config(), horizon=config().horizon + 1)
    with pytest.raises(ValueError, match="exact source"):
        initialize(model, env, source, cfg, Toy.fingerprint)


def test_quieter_exploration_is_serialized_without_changing_mean(setup, tmp_path):
    original, env, source, _, _ = setup
    quiet = PPO(
        BasePolicy,
        env,
        device="cpu",
        n_steps=8,
        batch_size=8,
        use_sde=True,
        policy_kwargs={**kwargs(), "exploration_coefficient_std": 0.1},
    )
    initialize(quiet, env, source, config(), Toy.fingerprint)
    obs = env.reset()
    np.testing.assert_array_equal(
        original.predict(obs, deterministic=True)[0], quiet.predict(obs, deterministic=True)[0]
    )
    assert quiet.policy.log_std.exp().max().item() == pytest.approx(0.1)
    quiet.save(tmp_path / "quiet.zip")
    loaded = PPO.load(tmp_path / "quiet.zip", device="cpu")
    assert loaded.policy.exploration_coefficient_std == 0.1
    torch.testing.assert_close(loaded.policy.log_std, quiet.policy.log_std)


def test_source_fidelity_records_known_failure_without_qualifying_it():
    from copy import deepcopy

    case = {
        "id": "a",
        "admitted": True,
        "code": "ok",
        "mastered": False,
        "metrics": {"base_linear_jerk": 0.5},
        "failure_reasons": ["smoothness/base_linear_jerk"],
        "admission": {"initial_state_sha256": "initial", "reference_sha256": "reference"},
        "trace": [{"action": [0] * 8, "reward": 1, "hand": [0] * 9, "base": [0] * 3}],
    }
    report = {"complete": True, "suite_sha256": "suite", "source_hashes": {"source": "hash"}, "cases": [case]}
    result = verify_source_fidelity(report, deepcopy(report), expected_ids={"a"})
    assert result == {"exact": True, "cases": 1, "source_failures": {"a": case["failure_reasons"]}}
    for change in ("metrics", "trace", "identity", "missing", "source"):
        changed = deepcopy(report)
        if change == "metrics":
            changed["cases"][0]["metrics"]["base_linear_jerk"] += 0.01
        elif change == "trace":
            changed["cases"][0]["trace"][0]["reward"] += 0.01
        elif change == "identity":
            changed["cases"][0]["admission"]["reference_sha256"] = "other"
        elif change == "missing":
            changed["cases"] = []
        else:
            changed["source_hashes"] = {"source": "other"}
        with pytest.raises(RuntimeError, match="source"):
            verify_source_fidelity(changed, report, expected_ids={"a"})


@pytest.mark.sim
@pytest.mark.parametrize("stages", [["base"], ["base", "arm"]])
def test_base_component_runs_native_physics_and_preserves_arm_rows(tmp_path, stages):
    from pathlib import Path

    from emet.rl.manipulation.mastery import MasteryEnv, load_suite

    spec = load_suite(
        Path(__file__).resolve().parents[4] / "configs/benchmarks/mars_tracking_mastery_command_history_v1.json"
    )
    base = MasteryEnv(spec, stages=stages, split="train", population="training")
    source_env = ExecutorCommandVecNormalize(DummyVecEnv([lambda: base]), config=base.config, norm_reward=False)
    params = kwargs()
    params["features_extractor_kwargs"]["orientation_scale"] = base.config.feedback_orientation_scale
    try:
        full = PPO(ExecutorFeedbackPolicy, source_env, device="cpu", n_steps=8, batch_size=8, policy_kwargs=params)
        export(full, source_env, tmp_path / "source", base.config, base.fingerprint, steps=0)
        source = ManipulationPolicy(tmp_path / "source")
        reward_env = base
        if stages == ["base"]:
            from copy import deepcopy

            from emet.rl.manipulation.command_runtime import task_pose_teacher_action
            from emet.rl.manipulation.mastery import TrainingCaseSampler
            from emet.rl.manipulation.mastery_objective import TeacherContractReward, mastery_metrics

            case = next(c for c in base.cases if c["id"] == "train:base:hold")
            base.cases = [case]
            base.sampler = TrainingCaseSampler(base.cases, {"base": 1}, sampling="control_steps")
            base.reset(options={"case_id": case["id"]})
            admission = deepcopy(base.mastery_admission)
            while True:
                _, _, done, truncated, info = base.step(task_pose_teacher_action(base.control))
                if done or truncated:
                    break
            row = {"id": case["id"], "mastered": True, "admission": admission, "metrics": mastery_metrics(base, info)}
            reward_env = TeacherContractReward(base, {"cases": [row]})
        wrapper = FrozenArmActions(reward_env, source)
        component_env = ExecutorCommandVecNormalize(
            DummyVecEnv([lambda: wrapper]), config=base.config, norm_reward=False
        )
        model = PPO(
            BasePolicy,
            component_env,
            device="cpu",
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            use_sde=True,
            sde_sample_freq=2,
            policy_kwargs=params,
        )
        initialize(model, component_env, source, base.config, base.fingerprint)
        if stages == ["base"]:
            component_env.env_method("set_reward_fraction", 0)
            assert reward_env.next_fraction == 0
        model.learn(16)
        verify_frozen(model, source)
        if stages == ["base"]:
            columns = [*range(3, 12), *range(15, 24)]
            for key in ("weight", "bias"):
                current = getattr(model.policy.action_net.gain, key).reshape(7, 35, -1)
                original = getattr(source.model.policy.action_net.gain, key).reshape(7, 35, -1)
                torch.testing.assert_close(current[:2, columns], original[:2, columns], rtol=0, atol=0)
        assert model.num_timesteps == 16
        merged = merge(model, component_env, source, tmp_path / "actor", base.config, base.fingerprint)
        assert merged.model.action_space.shape == (8,) and merged.manifest["steps"] == 16
        assert merged.model.num_timesteps == 16
    finally:
        source_env.close()

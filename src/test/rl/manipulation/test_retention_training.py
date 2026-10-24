from dataclasses import replace
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from emet.rl.manipulation.__main__ import initialize_retention_context, retention_options
from emet.rl.manipulation.env import MIXTURE, SCENARIOS, ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.held_object import HeldObjectEstimator
from emet.rl.manipulation.observation_layout import proprioceptive_layout
from emet.rl.manipulation.policy import ManipulationHistoryFeatures, ManipulationMLPFeatures, ManipulationPolicy, export
from emet.rl.manipulation.profiles import PROFILES
from emet.rl.manipulation.retention_training import TRACKING_RETENTION_MIX, HeldObjectLabel, validate_mixture


def options(**overrides):
    return SimpleNamespace(
        **{
            "task_mix": "tracking-retention",
            "scenario": "mixed",
            "scenario_weights": None,
            "retention_context": False,
            "held_object_predictor": None,
            "empty_grasp_probability": None,
            "carry_hold_probability": None,
            "payload_mass_range_kg": None,
            "payload_mass_kg": None,
            "payload_com_randomization_m": None,
            **overrides,
        }
    )


def test_tracking_majority_preset_is_explicit_and_legacy_is_preserved():
    config = ManipulationConfig(**retention_options(options()))
    assert config.scenario_weights == TRACKING_RETENTION_MIX
    assert config.scenario_weights[0] == 0.7
    draws = np.random.default_rng(9).choice(SCENARIOS, 100000, p=config.scenario_weights)
    assert np.mean(draws == "reach") == pytest.approx(0.7, abs=0.005)
    assert config.empty_grasp_probability == 0.2
    assert config.carry_hold_probability == 0.25
    legacy = ManipulationConfig(**retention_options(options(task_mix="legacy")))
    assert legacy.scenario_weights is None and not legacy.retention_context
    assert legacy.payload_mass_range_kg is None and legacy.carry_hold_probability == 0
    assert MIXTURE == (0.25, 0.20, 0.30, 0.15, 0.10)
    with pytest.raises(ValueError, match="50%"):
        retention_options(options(scenario_weights=(0.1, 0.1, 0.6, 0.1, 0.1)))
    with pytest.raises(ValueError, match="mixed"):
        retention_options(options(scenario="carry"))


@pytest.mark.parametrize("weights", [(1, 0), (0.8, -0.1, 0.1, 0.1, 0.1), (1, 1, 1, 1, 1), (np.nan, 0, 0, 0, 0)])
def test_invalid_task_distributions_rejected(weights):
    with pytest.raises(ValueError):
        validate_mixture(weights)


def test_held_supervision_abstains_on_transitions_and_invalid_physics():
    label = HeldObjectLabel()
    label.update(bilateral=True, clearly_empty=False, dt=0.05)
    assert not label.valid
    label.update(bilateral=True, clearly_empty=False, dt=0.05)
    assert label.valid and label.target == 1
    label.update(bilateral=False, clearly_empty=False, dt=0.05)
    assert not label.valid  # Single-finger contact is ambiguous, not empty.
    for _ in range(2):
        label.update(bilateral=False, clearly_empty=True, dt=0.05)
    assert label.valid and label.target == 0
    label.update(bilateral=True, clearly_empty=False, dt=1, physics_valid=False)
    assert not label.valid


def test_estimator_learns_closed_empty_counterexamples_and_exports_without_pickle(tmp_path):
    torch.set_num_threads(1)
    model = HeldObjectEstimator("innate_mars", (20, 101), seed=7)
    fields, _ = proprioceptive_layout(PROFILES["innate_mars"])
    samples = []
    for target in (0, 1):
        history = np.zeros((20, 101), dtype=np.float32)
        history[:, -2] = 1  # Both request retention; command alone is insufficient.
        history[:, fields["grip_position"]] = target * 0.25
        history[:, fields["grip_error"]] = -target * 0.25
        samples.append(history)
        for _ in range(64):
            model.add(history, {"target": target, "valid": True})
        model.add(history, {"target": target, "valid": True}, audit=True)
    before = [p.detach().clone() for p in model.head.parameters()]
    for _ in range(30):
        model.fit(batches=8)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.head.parameters(), strict=True))
    assert model.estimate(samples[0])["probability"] < 0.1
    assert model.estimate(samples[1])["probability"] > 0.9
    assert model.audit_metrics()["audit_empty_false_positive_rate"] == 0
    assert model.audit_metrics()["audit_held_recall"] == 1
    assert len(model.training) == 128 and len(model.audit) == 2
    model.save(tmp_path)
    loaded = HeldObjectEstimator.load(tmp_path)
    assert loaded.estimate(samples[1]) == model.estimate(samples[1])
    assert not loaded.estimate(samples[1])["calibrated"]
    perturbed = samples[1].copy()
    for name in ("velocities", "target_error", "previous_action"):
        perturbed[:, fields[name]] = 100
    assert loaded.estimate(perturbed) == loaded.estimate(samples[1])
    legacy = HeldObjectEstimator("innate_mars", (20, 101), feature_set="legacy_proprioception")
    (tmp_path / "legacy").mkdir()
    legacy.save(tmp_path / "legacy")
    assert HeldObjectEstimator.load(tmp_path / "legacy").feature_set == "legacy_proprioception"
    (tmp_path / "held_object.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        HeldObjectEstimator.load(tmp_path)


@pytest.mark.parametrize("architecture", [ManipulationMLPFeatures, ManipulationHistoryFeatures])
@pytest.mark.parametrize("preview", [False, True])
def test_warm_start_and_auxiliary_export_preserve_actor_actions(tmp_path, architecture, preview):
    class Task(gym.Env):
        action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)

        def __init__(self, width):
            self.observation_space = gym.spaces.Dict(
                {
                    "history": gym.spaces.Box(-100, 100, (20, width), dtype=np.float32),
                    **({"preview": gym.spaces.Box(-100, 100, (7, 18), dtype=np.float32)} if preview else {}),
                }
            )

        def reset(self, *, seed=None, options=None):
            return {k: np.zeros(v.shape, dtype=np.float32) for k, v in self.observation_space.spaces.items()}, {}

    torch.set_num_threads(1)
    old_env, new_env = [VecNormalize(DummyVecEnv([lambda w=w: Task(w)]), norm_reward=False) for w in (99, 101)]
    try:
        kwargs = {
            "device": "cpu",
            "n_steps": 4,
            "batch_size": 4,
            "policy_kwargs": {"features_extractor_class": architecture},
        }
        old_model, model = (
            PPO("MultiInputPolicy", old_env, seed=3, **kwargs),
            PPO("MultiInputPolicy", new_env, seed=4, **kwargs),
        )
        config = ManipulationConfig(robot="innate_mars", action_mode="direct_velocity", trajectory_preview=preview)
        old_path = tmp_path / "old"
        export(old_model, old_env, old_path, config, "test-model", steps=0)
        config = replace(config, retention_context=True)
        result = initialize_retention_context(model, new_env, old_path, config, "test-model")
        assert not result["optimizer_restored"]
        obs = old_env.reset()
        obs["history"][:] = np.random.default_rng(1).normal(0, 0.2, obs["history"].shape)
        extended = {
            **obs,
            "history": np.concatenate([obs["history"], np.full((1, 20, 2), 3.0)], axis=-1).astype(np.float32),
        }
        expected = old_model.predict(obs, deterministic=True)[0]
        np.testing.assert_allclose(model.predict(extended, deterministic=True)[0], expected, atol=1e-6)
        estimator = HeldObjectEstimator("innate_mars", (20, 101))
        before = {k: v.clone() for k, v in model.policy.state_dict().items()}
        for target in (0, 1):
            for _ in range(32):
                estimator.add(extended["history"][0], {"valid": True, "target": target})
        estimator.fit()
        for k, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, before[k], rtol=0, atol=0)
        export(model, new_env, tmp_path / "new", config, "test-model", steps=0, held_estimator=estimator)
        loaded = ManipulationPolicy(tmp_path / "new")
        assert loaded.estimate_held_object({"history": extended["history"][0]})["available"]
        np.testing.assert_allclose(loaded.model.predict(extended, deterministic=True)[0], expected, atol=1e-6)
    finally:
        old_env.close()
        new_env.close()


@pytest.mark.sim
@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_retention_context_preserves_tracking_rewards_and_has_no_ground_truth(robot):
    config = ManipulationConfig(robot=robot, scenario="reach", randomization_scale=0, horizon=1)
    legacy = ManipulationEnv(config)
    extended = ManipulationEnv(replace(config, retention_context=True))
    try:
        old, _ = legacy.reset(seed=42)
        new, _ = extended.reset(seed=42)
        assert new["history"].shape[-1] == old["history"].shape[-1] + 2
        np.testing.assert_array_equal(new["history"][:, :-2], old["history"])
        for _ in range(3):
            action = legacy.teacher_action()
            old, reward_old, done_old, truncated_old, _ = legacy.step(action)
            new, reward_new, done_new, truncated_new, info = extended.step(action)
            assert reward_new == pytest.approx(reward_old, abs=1e-10)
            assert (done_old, truncated_old) == (done_new, truncated_new)
            np.testing.assert_array_equal(new["history"][:, :-2], old["history"])
            terms = info["reward_terms"]
            assert reward_new == pytest.approx(
                terms["tracking_and_progress"]
                - terms["physical_cost"]
                + terms["success_bonus"]
                - terms["failure_penalty"]
            )
        before = extended.control.observe()["history"]
        extended.held_object_label.target = 1
        extended.held_object_label.valid = True
        np.testing.assert_array_equal(extended.control.observe()["history"], before)
    finally:
        legacy.close()
        extended.close()


@pytest.mark.sim
def test_empty_and_loaded_hands_share_retention_intent_but_different_labels():
    for scenario, expected in [("reach", 0), ("carry", 1)]:
        env = ManipulationEnv(
            ManipulationConfig(
                robot="innate_mars",
                scenario=scenario,
                randomization_scale=0,
                horizon=1,
                retention_context=True,
                empty_grasp_probability=1 if scenario == "reach" else 0,
                carry_hold_probability=1 if scenario == "carry" else 0,
            )
        )
        try:
            obs, info = env.reset(seed=42)
            assert not info["initialization_failed"]
            for _ in range(3):
                obs, _, done, truncated, info = env.step(env.teacher_action())
                assert obs["history"][-1, -2] == 1
                if done or truncated:
                    break
            assert info["held_object_supervision"]["valid"]
            assert info["held_object_supervision"]["target"] == expected
            if scenario == "carry":
                assert info["carry_hold_trial"]
                assert not env._released
                env.control.set_reference(env.control.positions(), phase="release", release=True)
                assert not env.control.retain_object
        finally:
            env.close()


@pytest.mark.sim
def test_payload_mass_com_randomization_is_episode_fixed_and_reset_reproducible():
    env = ManipulationEnv(
        ManipulationConfig(
            robot="innate_mars",
            scenario="reach",
            randomization_scale=1,
            payload_mass_range_kg=(0.02, 0.1),
            payload_com_randomization_m=0.002,
        )
    )
    try:
        _, info = env.reset(seed=42)
        mass = env.model.body_mass[env.payload]
        com = env.model.body_ipos[env.payload].copy()
        assert 0.02 <= mass <= 0.1
        assert np.max(abs(com - env.body_ipos[env.payload])) <= 0.002
        np.testing.assert_allclose(
            env.model.body_inertia[env.payload], env.inertia[env.payload] * mass / env.mass[env.payload]
        )
        env.step(env.teacher_action())
        assert env.model.body_mass[env.payload] == mass
        np.testing.assert_array_equal(env.model.body_ipos[env.payload], com)
        env.reset(seed=42)
        assert env.model.body_mass[env.payload] == mass
        np.testing.assert_array_equal(env.model.body_ipos[env.payload], com)
        assert info["randomization"]["payload_mass_kg"] == mass
    finally:
        env.close()

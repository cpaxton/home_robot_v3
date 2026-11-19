"""Physical jaw feedback stays observable through a source-preserving warmup."""

import numpy as np
import pytest
from stable_baselines3.common.logger import configure

from emet.rl.manipulation.base_component import merge
from emet.rl.manipulation.normalization_schedule import GripNormalizationSchedule, validate_grip_floor
from emet.rl.manipulation.observation_layout import history_layout
from emet.rl.manipulation.profiles import MARS

from ._component_helpers import Toy, config


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), 1])
def test_invalid_physical_grip_floor_is_rejected(value):
    with pytest.raises(ValueError, match="physical jaw travel"):
        validate_grip_floor(value, MARS)


def test_grip_schedule_keeps_warmup_exact_and_only_widens_selected_variances(setup):
    _, env, _, _, _ = setup
    stats = env.obs_rms["history"]
    before, mean, count = stats.var.copy(), stats.mean.copy(), stats.count
    schedule = GripNormalizationSchedule(env, minimum_std=0.08, warmup_steps=8, ramp_steps=16)
    for steps in (0, 8):
        assert not schedule.apply(steps)
        np.testing.assert_array_equal(stats.var, before)
    schedule.apply(16)
    columns = schedule.columns
    initial_sigma = np.sqrt(before[:, columns] + env.epsilon)
    expected_midpoint = np.sqrt(initial_sigma * np.maximum(initial_sigma, 0.08))
    np.testing.assert_allclose(np.sqrt(stats.var[:, columns] + env.epsilon), expected_midpoint)
    schedule.apply(24)
    np.testing.assert_allclose(np.sqrt(stats.var[:, columns] + env.epsilon), np.maximum(initial_sigma, 0.08))
    rest = [i for i in range(before.shape[1]) if i not in columns]
    np.testing.assert_array_equal(stats.var[:, rest], before[:, rest])
    np.testing.assert_array_equal(stats.mean, mean)
    assert stats.count == count and not env.training
    assert schedule.record()["fraction"] == 1.0


@pytest.mark.parametrize("drift", ["mean", "count", "running_updates"])
def test_running_statistic_drift_fails_closed(setup, drift):
    _, env, _, _, _ = setup
    schedule = GripNormalizationSchedule(env, minimum_std=0.08, warmup_steps=0, ramp_steps=16)
    if drift == "mean":
        env.obs_rms["history"].mean[0, 0] += 1
    elif drift == "count":
        env.obs_rms["history"].count += 1
    else:
        env.training = True
    with pytest.raises(RuntimeError, match="frozen means and running counts"):
        schedule.apply(16)


def test_rollout_boundary_refreshes_ppo_input_without_reset_or_measurement_change(setup, monkeypatch):
    model, env, _, _, _ = setup
    env.reset()
    raw = env.get_original_obs()
    fields, _ = history_layout(MARS, retention_context=True)
    raw["history"][:, :, fields["proprio/grip_position"]] = 0.35
    unchanged = {k: v.copy() for k, v in raw.items()}
    monkeypatch.setattr(env, "get_original_obs", lambda: raw)
    model._last_obs = env.normalize_obs(raw)
    old = {k: v.copy() for k, v in model._last_obs.items()}
    schedule = GripNormalizationSchedule(env, minimum_std=0.08, warmup_steps=0, ramp_steps=16)
    model.set_logger(configure(format_strings=[]))
    schedule.init_callback(model)
    model.num_timesteps = 16
    schedule.on_rollout_start()
    assert not np.array_equal(model._last_obs["history"], old["history"])
    for key in raw:
        np.testing.assert_array_equal(raw[key], unchanged[key])
        np.testing.assert_array_equal(model._last_obs[key], env.normalize_obs(raw)[key])


def test_exported_bundle_uses_the_same_final_grip_units_as_training(setup, tmp_path):
    model, env, source, _, _ = setup
    schedule = GripNormalizationSchedule(env, minimum_std=0.08, warmup_steps=0, ramp_steps=16)
    schedule.apply(16)
    actor = merge(model, env, source, tmp_path / "actor", config(), Toy.fingerprint)
    raw = env.reset()
    expected = model.predict(raw, deterministic=True)[0][0]
    physical = {k: v[0] for k, v in env.get_original_obs().items()}
    np.testing.assert_allclose(actor(physical)[:2], expected, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(actor.var, env.obs_rms["history"].var)


def test_short_real_ppo_update_applies_scaling_only_after_warmup(setup):
    model, env, _, _, _ = setup
    schedule = GripNormalizationSchedule(env, minimum_std=0.08, warmup_steps=8, ramp_steps=8)
    model.learn(24, callback=schedule)
    assert schedule.fraction == 1
    assert np.isfinite(env.obs_rms["history"].var).all()

"""Dense fine tracking, quiet completion and final motor smoothing contracts."""

from pathlib import Path

import numpy as np
import pytest

from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.mastery import MasteryEnv, load_suite
from emet.rl.manipulation.tracking import (
    base_progress_potential,
    normalized_jerk_cost,
    precision_base_score,
    whole_body_quiet_score,
)
from emet.rl.manipulation.velocity_filter import smooth_velocity

V2 = Path(__file__).resolve().parents[4] / "configs/benchmarks/mars_tracking_mastery_v2.json"


def test_v2_changes_objective_and_command_filter_without_relaxing_frozen_problem():
    a, b = load_suite(), load_suite(V2)
    for name in ["cases", "reference", "gate", "stages", "robot"]:
        assert a[name] == b[name]
    cfg = ManipulationConfig(**b["config"])
    assert cfg.terminal_hold_seconds == b["gate"]["terminal_hold_s"] == 2
    assert cfg.precision_tracking_reward and cfg.normalized_jerk_weight > 0
    assert cfg.wheel_jerk_limits == (0.24, 1.2)


@pytest.mark.parametrize(
    "updates",
    [
        {"precision_tracking_reward": True},
        {"terminal_hold_seconds": True},
        {"terminal_hold_seconds": np.nan},
        {"terminal_hold_seconds": 0},
        {"normalized_jerk_weight": 0.1},
        {"wheel_jerk_limits": (0.24, 1.2)},
        {"physical_wheel_slew": True, "wheel_jerk_limits": (0.24, -1)},
        {"physical_wheel_slew": True, "wheel_jerk_limits": (np.nan, 1)},
    ],
)
def test_invalid_precision_or_actuator_settings_rejected(updates):
    with pytest.raises(ValueError):
        ManipulationConfig(**updates)


def test_fine_yaw_credit_and_progress_are_continuous_and_direction_symmetric():
    errors = np.array([0, 0.005, 0.01, 0.02, 0.04])
    scores = [precision_base_score(0, e, 0.005, 0.01) for e in errors]
    assert scores[0] == 1 and np.all(np.diff(scores) < 0)
    assert scores[1] - scores[2] > 0.15
    for error in errors:
        assert precision_base_score(0, error, 0.005, 0.01) == precision_base_score(0, -error, 0.005, 0.01)
    assert base_progress_potential(0, 0.01) > base_progress_potential(0, 0.02)
    assert base_progress_potential(0.005, 0) > base_progress_potential(0.01, 0)


def test_quiet_reward_has_no_binary_speed_cliff_and_covers_all_base_dofs():
    values = []
    for speed in [0, 0.0025, 0.005, 0.0075]:
        values.append(whole_body_quiet_score(np.zeros(5), [speed, 0, 0, 0, 0, 0]))
    np.testing.assert_allclose(values, np.exp(-(np.array([0, 0.5, 1, 1.5]) ** 2)))
    for axis in range(6):
        v = np.zeros(6)
        v[axis] = 0.005 if axis < 3 else 0.02
        assert whole_body_quiet_score(np.zeros(5), v) == pytest.approx(np.exp(-1))
    assert whole_body_quiet_score([0.03, 0, 0, 0, 0], np.zeros(6)) == pytest.approx(np.exp(-1))


def test_jerk_cost_uses_each_physical_scale_and_remains_finite_for_large_spikes():
    assert normalized_jerk_cost(np.zeros(5), np.zeros(2), np.ones(5), np.ones(2)) == 0
    a = normalized_jerk_cost(np.ones(5), [0.24, 1.2], np.ones(5), [0.24, 1.2])
    b = normalized_jerk_cost(np.ones(5) * 2, [0.48, 2.4], np.ones(5) * 2, [0.48, 2.4])
    assert a == b == pytest.approx(7 * (np.sqrt(2) - 1))
    assert np.isfinite(normalized_jerk_cost([1e6] * 5, [1e6] * 2, np.ones(5), np.ones(2)))


def test_motor_filter_bounds_derivatives_rejects_tick_jitter_and_tracks_signed_steps():
    speed = np.array([0.12, 0.5])
    accel = np.array([0.12, 0.6])
    jerk = np.array([0.24, 1.2])
    dt = 0.002
    for sign in [-1, 1]:
        v = np.zeros(2)
        a = np.zeros(2)
        previous_a = a.copy()
        target = sign * np.array([0.0036, 0.015])
        for _ in range(1500):
            previous_v = v.copy()
            v, a = smooth_velocity(v, a, target, dt, speed, accel, jerk)
            assert np.all(abs(v) <= speed) and np.all(abs(a) <= accel + 1e-10)
            assert np.all(abs(a - previous_a) <= jerk * dt + 1e-10)
            np.testing.assert_allclose(v - previous_v, a * dt, atol=1e-12)
            previous_a = a.copy()
        np.testing.assert_allclose(v, target, atol=1e-7)
        unchanged = smooth_velocity(v, a, -target, 0, speed, accel, jerk)
        np.testing.assert_array_equal(unchanged[0], v)
        np.testing.assert_array_equal(unchanged[1], a)
    v = np.zeros(2)
    a = np.zeros(2)
    samples = []
    for tick in range(2000):
        target = np.array([0.0036, 0.015]) * (1 if (tick // 25) % 2 else -1)
        v, a = smooth_velocity(v, a, target, dt, speed, accel, jerk)
        samples.append(v.copy())
    assert np.max(np.abs(samples), axis=0)[0] < 0.0036 / 2


@pytest.mark.sim
def test_v2_reward_is_dense_at_20hz_and_bonus_waits_for_two_seconds():
    env = MasteryEnv(load_suite(V2), stages=["base"], split="train")
    try:
        _, info = env.reset(options={"case_id": "train:base:hold"})
        assert not info["initialization_failed"]
        reference_end = env.mastery_reference.end
        bonuses = []
        quiet = []
        for _ in range(150):
            _, reward, done, truncated, info = env.step(np.zeros(8))
            assert not done and not truncated
            assert np.isfinite(reward) and info["reward_terms"]["base_tracking_reward"] > 0
            if info["reward_terms"]["success_bonus"]:
                bonuses.append(info["time"])
            if info["reward_terms"]["terminal_quiet_reward"] > 0:
                quiet.append(info["time"])
        assert len(bonuses) == 1
        assert reference_end + 1.9 <= bonuses[0] <= reference_end + 2.2
        assert quiet and min(quiet) < bonuses[0] - 1.5
        env.control.apply()  # First apply at the new physics instant is a real update.
        before = env.control.applied_base_velocity.copy()
        accel = env.control.applied_base_acceleration.copy()
        env.control.apply()
        np.testing.assert_array_equal(before, env.control.applied_base_velocity)
        np.testing.assert_array_equal(accel, env.control.applied_base_acceleration)
        env.control.stop()
        np.testing.assert_array_equal(env.control.applied_base_velocity, 0)
        np.testing.assert_array_equal(env.control.applied_base_acceleration, 0)
    finally:
        env.close()

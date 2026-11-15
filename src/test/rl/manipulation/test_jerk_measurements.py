# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Measurement definitions and nonintrusive native observer behavior."""

import numpy as np
import pytest

from emet.rl.manipulation.jerk_measurements import (
    DerivativeChannels,
    NativeJerkObserver,
    current_feedback_velocity,
    linear_magnitude,
)


@pytest.mark.sim
def test_current_com_probe_matches_fresh_cache_but_responds_to_new_qvel_without_forward():
    from types import SimpleNamespace

    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="base"><freejoint/><geom type="box" size=".1 .1 .1" '
        'pos=".03 0 .07"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    control = SimpleNamespace(model=model, data=data, base=1, axle_offset=np.array([0.01, 0, -0.05]))
    data.qvel[:] = [0.1, 0, 0, 0, 0.2, 0.3]
    mujoco.mj_forward(model, data)
    spatial = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, spatial, 0)
    expected = spatial[[3, 2]]
    np.testing.assert_allclose(current_feedback_velocity(control, "com"), expected, atol=1e-15)
    # Integration changes qvel without refreshing cvel. Current-qvel diagnosis
    # isolates this timing difference without advancing or re-solving physics.
    data.qvel[:] *= 2
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, spatial, 0)
    np.testing.assert_allclose(spatial[[3, 2]], expected, atol=1e-15)
    np.testing.assert_allclose(current_feedback_velocity(control, "com"), 2 * expected, atol=1e-15)
    np.testing.assert_allclose(current_feedback_velocity(control, "axle"), [0.18, 0.6], atol=1e-15)
    with pytest.raises(ValueError, match="point"):
        current_feedback_velocity(control, "origin")


@pytest.mark.parametrize("dt,peak", [(0.1, 1.0), (0.05, 2.0), (0.025, 4.0)])
def test_reversal_distinguishes_unsigned_speed_from_vector_jerk(dt, peak):
    times = np.arange(-1.0, 1.0 + dt / 2, dt)
    meter = DerivativeChannels([0.05, 0, 0], [-0.1, 0, 0], dt)
    rows = [meter.sample([0.05 * t**2, 0, 0], [0.1 * t, 0, 0]) for t in times[1:]]
    assert max(linear_magnitude(row["gate"], "gate") for row in rows[2:]) == pytest.approx(peak)
    for name in ("displacement_vector", "jacobian", "jacobian_trapezoid"):
        assert max(linear_magnitude(row[name], name) for row in rows[2:]) < 1e-9


def test_nonzero_reset_velocity_creates_only_the_zero_history_transient():
    meter = DerivativeChannels([0, 0, 0], [0.001, 0, 0], 0.05)
    rows = [meter.sample([0.001 * t, 0, 0], [0.001, 0, 0]) for t in (0.05, 0.1, 0.15, 0.2)]
    assert max(linear_magnitude(row["gate"], "gate") for row in rows) == pytest.approx(0.4)
    assert max(linear_magnitude(row["jacobian_zero_reset"], "jacobian_zero_reset") for row in rows) == pytest.approx(
        0.4
    )
    assert max(linear_magnitude(row["jacobian"], "jacobian") for row in rows) == 0
    assert max(linear_magnitude(row["gate"], "gate") for row in rows[2:]) < 1e-12


def test_trapezoidal_instantaneous_velocity_matches_constant_acceleration_interval():
    meter = DerivativeChannels([0, 0, 0], [0, 0, 0], 0.05)
    for t in np.arange(0.05, 0.51, 0.05):
        row = meter.sample([0.05 * t**2, 0, 0], [0.1 * t, 0, 0])
        np.testing.assert_allclose(
            row["jacobian_trapezoid"]["velocity"], row["displacement_vector"]["velocity"], atol=1e-15
        )


def test_heading_wrap_and_copy_ownership():
    pose = np.array([0.0, 0, np.pi - 0.001])
    meter = DerivativeChannels(pose, [0, 0, 0.04], 0.05)
    pose[:] = 100
    row = meter.sample([0, 0, -np.pi + 0.001], [0, 0, 0.04])
    assert row["gate"]["velocity"][1] == pytest.approx(0.04)
    row["gate"]["velocity"][:] = 100
    row = meter.sample([0, 0, -np.pi + 0.003], [0, 0, 0.04])
    assert row["gate"]["acceleration"][1] == pytest.approx(0, abs=1e-11)


@pytest.mark.parametrize(
    "pose,velocity,dt", [([0, 0], [0, 0, 0], 0.05), ([0, 0, 0], [np.nan, 0, 0], 0.05), ([0] * 3, [0] * 3, 0)]
)
def test_invalid_measurement_state_is_rejected(pose, velocity, dt):
    with pytest.raises(ValueError):
        DerivativeChannels(pose, velocity, dt)


@pytest.mark.parametrize("count", [1, 12])
def test_replay_validator_rejects_changed_actions_and_duplicate_case_records(count):
    import copy
    import runpy
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    compare = runpy.run_path(str(root / "scripts/replay_mars_jerk_measurements.py"))["compare_replay"]
    trace = {
        "time": 0.05,
        "base": [0] * 3,
        "hand": [0] * 9,
        "action": [0] * 8,
        "reference_base": [0] * 3,
        "reference_hand": [0] * 9,
    }
    expected = {
        "cases": [
            {
                "id": str(i),
                "admission": {"initial_state_sha256": str(i), "reference_sha256": str(i)},
                "code": "ok",
                "mastered": True,
                "tracking_and_settling_pass": True,
                "smoothness_pass": True,
                "failure_reasons": [],
                "metrics": {"base_jerk": 0},
                "trace": [copy.deepcopy(trace)],
            }
            for i in range(count)
        ]
    }
    actual = copy.deepcopy(expected)
    assert compare(expected, actual, cases=count)["cases"] == count
    actual["cases"][0]["trace"][0]["action"][0] = 1
    with pytest.raises(AssertionError):
        compare(expected, actual, cases=count)
    duplicate = copy.deepcopy(expected)
    duplicate["cases"].append(copy.deepcopy(duplicate["cases"][0]))
    with pytest.raises(ValueError, match="unique"):
        compare(duplicate, expected, cases=count)


@pytest.mark.sim
@pytest.mark.parametrize("record_physics", [False, True])
@pytest.mark.parametrize("case_id", ["train:base:hold", "train:arm:supported_A_up@amplitude:0.25"])
def test_observer_preserves_native_state_observations_and_exact_gate_channels(case_id, record_physics):
    from pathlib import Path

    from emet.rl.manipulation.basic_learning import load_basic_learning
    from emet.rl.manipulation.mastery import MasteryEnv
    from emet.rl.manipulation.whole_body_training import load_development

    _, parent = load_development()
    _, _, suite = load_basic_learning(Path("configs/benchmarks/mars_whole_body_basic_learning_v1.json"), parent)
    plain, observed = (MasteryEnv(suite, stages=("base", "arm"), split="all", population="canonical") for _ in range(2))
    observer = NativeJerkObserver(record_physics=record_physics)
    try:
        a, _ = plain.reset(options={"case_id": case_id})
        b, _ = observed.reset(options={"case_id": case_id})
        case = next(c for c in suite["cases"] if c["id"] == case_id)
        observer.reset(observed, case)
        for _ in range(5):
            action = np.zeros(observed.profile.action_size)
            observer.before_step(observed, action)
            a, *_ = plain.step(action)
            b, _, _, _, info = observed.step(action)
            observer.after_step(observed, action, info)
            assert observed.physics_observer is None
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            for key in ("qpos", "qvel", "ctrl", "act"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(plain.data, key)).copy(), np.asarray(getattr(observed.data, key)).copy()
                )
            np.testing.assert_array_equal(plain.peak_base_axis_jerk, observed.peak_base_axis_jerk)
        result = observer.summarize()[0]
        assert result["samples"] == 5 and result["max_dt_error_seconds"] < 1e-10
        assert result["physics_samples"] == (5 * observed.substeps if record_physics else 0)
        if record_physics:
            records = observer.current["physics"]
            assert all(np.isfinite(row["qvel"]).all() for row in records)
            np.testing.assert_array_equal(records[-1]["qvel"], observed.data.qvel.copy())
            observed.reset(options={"case_id": case_id})
            assert observed.physics_observer is None
            assert len(records) == 5 * observed.substeps  # Reset/conditioning was not observed.
    finally:
        plain.close()
        observed.close()

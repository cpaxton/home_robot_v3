"""Raw-command penalties train smoothness without inserting an action filter."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.mastery import MasteryEnv, evaluate_suite, load_suite, save_report, train
from emet.rl.manipulation.tracking import command_motion_cost

SUITE = Path(__file__).resolve().parents[4] / "configs/benchmarks/mars_tracking_mastery_unfiltered_v1.json"


def test_unfiltered_experiment_preserves_cases_gates_and_native_limits():
    old, new = load_suite(), load_suite(SUITE)
    for name in ["cases", "reference", "gate", "robot", "stages"]:
        assert new[name] == old[name]
    cfg = ManipulationConfig(**new["config"])
    assert cfg.wheel_jerk_limits is None and cfg.physical_wheel_slew
    assert cfg.command_smoothness_weight == 0.1 and cfg.terminal_hold_seconds == 2


@pytest.mark.parametrize("weight", [np.nan, -0.1, 2.1])
def test_invalid_command_cost_rejected(weight):
    with pytest.raises(ValueError):
        ManipulationConfig(command_smoothness_weight=weight)


def test_command_cost_is_zero_at_constant_command_and_uses_physical_derivatives():
    current = np.array([0.1, 0])
    zero = np.zeros(2)
    speed = np.array([0.12, 0.5])
    accel = np.array([0.12, 0.6])
    mask = np.ones(2)
    c, a = command_motion_cost(current, zero, zero, 0.05, speed, accel, 2 * accel, mask)
    assert c > 0
    np.testing.assert_allclose(a, [0.24, 0])
    # Changing normalization while preserving physical commands preserves cost.
    other, b = command_motion_cost(current / 2, zero, zero, 0.05, 2 * speed, accel, 2 * accel, mask)
    assert c == other
    np.testing.assert_array_equal(a, b)
    steady, a = command_motion_cost(current, current, zero, 0.05, speed, accel, 2 * accel, mask)
    assert steady == 0
    np.testing.assert_array_equal(a, zero)
    ignored, a = command_motion_cost(current, zero, [10, 10], 0.05, speed, accel, 2 * accel, np.zeros(2))
    assert ignored == 0
    np.testing.assert_array_equal(a, zero)


@pytest.mark.sim
def test_command_penalty_changes_reward_without_changing_actuation(monkeypatch):
    from emet.rl.manipulation import velocity_filter

    monkeypatch.setattr(
        velocity_filter, "smooth_velocity", lambda *a: pytest.fail("unfiltered path invoked low-pass filter")
    )
    suite = load_suite(SUITE)
    a, b = MasteryEnv(suite, stages=["base"]), MasteryEnv(suite, stages=["base"])
    try:
        for env in [a, b]:
            _, info = env.reset(options={"case_id": "train:base:forward"})
            assert not info["initialization_failed"]
            assert env.control.physical_wheel_jerk_limits is None
        a.set_command_smoothness_fraction(0)
        b.set_command_smoothness_fraction(1)
        penalties = []
        for index in range(8):
            action = np.zeros(8)
            action[0] = 0.01 if index % 2 else -0.01
            _, ra, da, ta, ia = a.step(action)
            _, rb, db, tb, ib = b.step(action)
            np.testing.assert_array_equal(a.data.qpos.copy(), b.data.qpos.copy())
            np.testing.assert_array_equal(a.data.ctrl.copy(), b.data.ctrl.copy())
            np.testing.assert_array_equal(a.control.base_command, b.control.base_command)
            assert (da, ta) == (db, tb)
            assert ra - rb == pytest.approx(ib["reward_terms"]["command_motion_cost"])
            assert ia["reward_terms"]["command_motion_cost"] == 0
            penalties.append(ib["reward_terms"]["command_motion_cost"])
        assert max(penalties) > 0
        for bad in [-1, 2, np.nan]:
            with pytest.raises(ValueError):
                a.set_command_smoothness_fraction(bad)
    finally:
        a.close()
        b.close()


@pytest.mark.sim
def test_real_ppo_command_penalty_ramps_only_during_training(tmp_path, monkeypatch):
    import torch

    from emet.rl import regression_gate

    torch.set_num_threads(1)
    suite = deepcopy(load_suite(SUITE))
    suite["cases"] = [c for c in suite["cases"] if c["stage"] == "base" and c["id"].endswith(":hold")]
    teacher = evaluate_suite(suite, stages=["base"])
    assert all(c["mastered"] for c in teacher["cases"])
    preflight = tmp_path / "teacher.json.gz"
    save_report(preflight, teacher)
    monkeypatch.setattr(regression_gate, "verify_regressions", lambda path: None)
    args = SimpleNamespace(
        preflight=preflight,
        regression_receipt=tmp_path / "unused",
        output=tmp_path / "run",
        stages=["base"],
        seed=90,
        steps=512,
        learning_rate=5e-5,
        discount=0.9975,
        base_std=0.005,
        arm_std=0.01,
        initialize_from=None,
        wandb_mode="disabled",
        eval_every=8192,
        command_smoothness_warmup=512,
    )
    train(args, suite)
    rows = [json.loads(line) for line in (args.output / "metrics.jsonl").read_text().splitlines()]
    fractions = []
    for row in rows:
        count = sum(c["steps"] for c in row["rollout/cases"].values())
        fraction = (
            sum(c["reward_terms_sum"]["command_smoothness_fraction"] for c in row["rollout/cases"].values()) / count
        )
        fractions.append(fraction)
    assert fractions == [0, 0.5]
    assert json.loads((args.output / "run.json").read_text())["command_smoothness_warmup_steps"] == 512

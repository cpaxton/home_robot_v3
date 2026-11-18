"""Nominal handoffs preserve history, native guards and controller-independent targets."""

import copy
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.goal_change_tracking import GoalChangeTrackingEnv, nominal_successor, validate_change


def case_and_change():
    case = {
        "id": "innate_mars:goal-change:test:0",
        "seed": 410400000,
        "split": "train",
        "mode": "hand_free_base",
        "travel_m": 0.0,
        "turn_rad": 0.0,
        "arm_delta": [0.0, 0.01, -0.01, 0.0, 0.0],
        "base_mask": [False] * 3,
        "purpose": "tracking",
    }
    change = {"travel_m": 0.0, "turn_rad": 0.0, "arm_delta": [0.0, -0.01, 0.01, 0.0, 0.0], "dwell_s": 0.25}
    return case, change


@pytest.mark.parametrize(
    "patch",
    [
        {"dwell_s": -1},
        {"dwell_s": 2},
        {"dwell_s": True},
        {"turn_rad": float("nan")},
        {"travel_m": 0.21},
        {"turn_rad": 0.51},
        {"arm_delta": [0] * 4},
        {"arm_delta": None},
        {"arm_delta": [False] * 5},
        {"travel_m": "0.1"},
        {"extra": 1},
    ],
)
def test_invalid_change_rejected_without_simulator(patch):
    _, spec = case_and_change()
    with pytest.raises(ValueError):
        validate_change(spec | patch, 5)


@pytest.mark.sim
def test_nominal_successor_ignores_live_tracking_error_and_preserves_physics():
    case, change = case_and_change()
    env = GoalChangeTrackingEnv("innate_mars", [case], changes={case["id"]: change})
    try:
        _, info = env.reset()
        e, c = env.unwrapped, env.unwrapped.control
        before = {name: getattr(e.data, name).copy() for name in ("qpos", "qvel", "ctrl")}
        first = c.task_trajectory
        altered = copy.copy(c)
        altered.data = mujoco.MjData(e.model)
        altered.data.qpos[:] = e.data.qpos.copy()
        altered.data.qpos[c.q[0]] += 0.02
        mujoco.mj_forward(e.model, altered.data)
        candidate = nominal_successor(altered, first, case, change)
        assert digest(candidate.record()) == info["goal_change"]["second_reference_sha256"]
        for name, values in before.items():
            np.testing.assert_array_equal(getattr(e.data, name).copy(), values)
        assert not c.stopped and e.steps == 0
    finally:
        env.close()


@pytest.mark.sim
def test_switch_is_observed_before_next_action_without_history_or_clock_reset():
    case, change = case_and_change()
    env = GoalChangeTrackingEnv("innate_mars", [case], changes={case["id"]: change})
    try:
        _, info = env.reset()
        e, c = env.unwrapped, env.unwrapped.control
        origin, meter, original = e.start_time, env.meter, c.task_trajectory
        for _ in range(250):
            history, steps, time = c.history.copy(), e.steps, float(e.data.time)
            observation, _, done, truncated, info = env.step(np.zeros(e.profile.action_size))
            assert not done and not truncated
            if info["goal_change_delivered"]:
                assert c.task_trajectory is env.successor and c.task_trajectory is not original
                assert e.start_time == origin and e.steps == steps + 1
                assert float(e.data.time) == pytest.approx(time + e.profile.dt)
                assert env.meter is meter and not c.stopped
                np.testing.assert_array_equal(observation["history"][:-1], history[1:])
                np.testing.assert_array_equal(observation["preview"], c.observe()["preview"])
                break
        else:
            pytest.fail("announced goal change was not delivered")
    finally:
        env.close()


@pytest.mark.sim
def test_native_failure_never_installs_successor_or_rearms_stop():
    case, change = case_and_change()
    env = GoalChangeTrackingEnv("innate_mars", [case], changes={case["id"]: change})
    try:
        env.reset()
        e, c = env.unwrapped, env.unwrapped.control
        original = c.task_trajectory
        # A real scored deadline stops control; a handoff must not bypass it.
        e.config = replace(e.config, horizon=e.profile.dt)
        _, _, done, truncated, info = env.step(np.zeros(e.profile.action_size))
        assert done and not truncated and c.stopped
        assert not info["goal_change_delivered"] and not info["primitive_success"]
        assert c.task_trajectory is original
    finally:
        env.close()


@pytest.mark.sim
def test_short_horizon_rejected_and_control_stopped():
    case, change = case_and_change()
    env = GoalChangeTrackingEnv("innate_mars", [case], changes={case["id"]: change})
    try:
        env.unwrapped.config = replace(env.unwrapped.config, horizon=1.0)
        with pytest.raises(ValueError, match="exceeds_episode_horizon"):
            env.reset()
        assert env.unwrapped.control.stopped
        with pytest.raises(RuntimeError, match="reset an active"):
            env.step(np.zeros(8))
    finally:
        env.close()


@pytest.mark.sim
def test_external_stop_cannot_be_cleared_or_advance_physics():
    case, change = case_and_change()
    env = GoalChangeTrackingEnv("innate_mars", [case], changes={case["id"]: change})
    try:
        env.reset()
        e = env.unwrapped
        e.control.stop()
        frozen, time = e.control.target.copy(), float(e.data.time)
        with pytest.raises(RuntimeError, match="reset an active"):
            env.step(np.zeros(8))
        np.testing.assert_array_equal(e.control.target, frozen)
        assert e.control.stopped and float(e.data.time) == time
        assert not env.change_delivered
    finally:
        env.close()


def test_curriculum_splits_and_tick_share_are_explicit_and_disjoint():
    import json
    from pathlib import Path

    from emet.rl.manipulation.goal_change_tracking import load_curriculum

    curriculum = load_curriculum("configs/benchmarks/mars_goal_changes_v1.json")
    cases = curriculum["robots"]["innate_mars"]
    assert len(cases) == 16 and sum(c["split"] == "train" for c in cases) == 8
    prior = json.loads(Path("paper/data/whole_body_control/fresh_tracking_confirmation_v1/cases.json").read_text())
    assert {c["id"] for c in cases}.isdisjoint(c["id"] for c in prior)
    assert {c["seed"] for c in cases}.isdisjoint(c["seed"] for c in prior)
    assert curriculum["training_mix"]["general"] * curriculum["training_mix"]["within_general"]["goal_changes"] == 0.1


def test_curriculum_does_not_allow_silent_sampling_or_reference_drift(tmp_path):
    import json
    from pathlib import Path

    from emet.rl.manipulation.goal_change_tracking import load_curriculum

    original = json.loads(Path("configs/benchmarks/mars_goal_changes_v1.json").read_text())
    for kind in ("missing_change", "wrong_share", "nonfinite"):
        changed = copy.deepcopy(original)
        cid = changed["robots"]["innate_mars"][0]["id"]
        if kind == "missing_change":
            del changed["goal_changes"][cid]
        elif kind == "wrong_share":
            changed["training_mix"]["within_general"]["goal_changes"] = 0.8
        else:
            changed["goal_changes"][cid]["turn_rad"] = float("nan")
        path = tmp_path / (kind + ".json")
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            load_curriculum(path)


@pytest.mark.sim
def test_nominal_pose_join_wraps_yaw_without_moving_live_robot():
    from emet.rl.manipulation.coupled_tracking import CoupledReference

    case, change = case_and_change()
    env = GoalChangeTrackingEnv("innate_mars", [case], changes={case["id"]: change})
    try:
        env.reset()
        e = env.unwrapped
        before = e.data.qpos.copy()
        view = copy.copy(e.control)
        view.data = mujoco.MjData(e.model)
        view.data.qpos[:] = before
        base_joint = e.model.body_jntadr[view.base]
        address = e.model.jnt_qposadr[base_joint]
        view.data.qpos[address + 3 : address + 7] = [np.cos(3.05 / 2), 0, 0, np.sin(3.05 / 2)]
        mujoco.mj_forward(e.model, view.data)
        first_case = {**case, "mode": "coupled", "travel_m": 0.12, "turn_rad": 0.3, "base_mask": [True] * 3}
        first = CoupledReference(view, first_case)
        second = nominal_successor(view, first, first_case, {**change, "travel_m": -0.12, "turn_rad": -0.3})
        assert first.start_base[2] > 3
        assert first.sample(first.end)[0][2] < -2.9
        assert second.sample(0)[0][2] < 0
        np.testing.assert_allclose(first.sample(first.end)[1], second.sample(0)[1], atol=1e-9, rtol=0)
        np.testing.assert_array_equal(e.data.qpos.copy(), before)
    finally:
        env.close()

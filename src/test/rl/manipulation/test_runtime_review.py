"""Regression tests for all-DOF stops, explicit units and independent deadlines."""

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from emet.controller.task.tamp.mars_sim_robot import MarsSimRobot
from emet.core.command_client import command_receipt
from emet.eval.mars_tamp_control import CANONICAL_SUITE, final_predicates, fixture
from emet.rl.manipulation import command_runtime as module
from emet.rl.manipulation.command_runtime import (
    TaskPoseCommandRuntime,
    TaskPoseRuntimeTiming,
    TaskPoseTeacherConfig,
    send_task_pose_trajectory,
    task_pose_teacher_action,
)
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.safety import whole_body_stopped
from emet.rl.manipulation.tracking import TaskPoseTrajectory


def fake_control():
    data = SimpleNamespace(qvel=np.zeros(6), time=0.0)
    return SimpleNamespace(
        data=data, base_dof=0, profile=SimpleNamespace(arm_size=5), velocities=lambda: np.zeros(5), stop=lambda: None
    )


@pytest.mark.parametrize("axis", range(6))
def test_stop_requires_all_six_base_dofs_and_has_sim_time_bound(axis, monkeypatch):
    control = fake_control()
    assert whole_body_stopped(control)
    control.data.qvel[axis] = 0.006 if axis < 3 else 0.021
    assert not whole_body_stopped(control)
    env = SimpleNamespace(
        control=control, data=control.data, _physics_invalid=False, action_space=SimpleNamespace(shape=(8,))
    )

    def tick(action, **kwargs):
        env.data.time += 0.05

    env._tick = tick
    runtime = object.__new__(TaskPoseCommandRuntime)
    runtime.env = env
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 0.0))
    assert not runtime.cancel_reactive_command()
    assert 0.75 <= env.data.time <= 0.8


@pytest.mark.parametrize(
    "velocity",
    [
        np.full(6, np.nan),
        np.full(6, np.inf),
        np.zeros(5),
        np.array([0.004, 0.004, 0, 0, 0, 0]),
        np.array([0, 0, 0, 0.015, 0.015, 0]),
    ],
)
def test_nonfinite_malformed_or_combined_motion_cannot_confirm_stop(velocity):
    control = fake_control()
    control.data.qvel = velocity
    assert not whole_body_stopped(control)
    control.data.qvel = np.zeros(6)
    control.velocities = lambda: np.full(5, np.nan)
    assert not whole_body_stopped(control)


@pytest.mark.parametrize("axis", [2, 3, 4])
def test_terminal_quiet_hold_does_not_ignore_vertical_roll_or_pitch(axis):
    robot = object.__new__(MarsSimRobot)
    robot.servo = fake_control()
    robot.data = robot.servo.data
    robot.env = SimpleNamespace(profile=SimpleNamespace(dt=0.05))
    robot.failure = None
    robot.step = lambda dt: None
    robot.data.qvel[axis] = 0.1
    assert not robot._verify_quiet_hold(0.1, 0.2)
    robot.data.qvel[:] = 0
    assert robot._verify_quiet_hold(0.1, 0.2)


def test_spatial_feedback_has_separate_gains_and_explicit_task_metric():
    config = TaskPoseTeacherConfig(position_gain_per_s=2, orientation_gain_per_s=3, rotation_length_m_per_rad=0.1)
    np.testing.assert_allclose(config.feedback_twist([0.01, 0, 0], [0, 0, 0.1]), [0.02, 0, 0, 0, 0, 0.3])
    defaults = TaskPoseTeacherConfig()
    np.testing.assert_array_equal(defaults.feedback_twist([1, 2, 3], [4, 5, 6]), 4 * np.arange(1, 7))
    assert defaults.rotation_length_m_per_rad == 1


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "0.2"])
def test_gain_and_timing_limits_fail_closed(bad):
    for name in ("position_gain_per_s", "orientation_gain_per_s", "rotation_length_m_per_rad"):
        with pytest.raises(ValueError):
            TaskPoseTeacherConfig(**{name: bad})
    for name in ("inference_budget_s", "cycle_budget_s", "max_state_age_s"):
        with pytest.raises(ValueError):
            TaskPoseRuntimeTiming(**{name: bad})


def test_timing_defaults_preserve_limits_and_invalid_budget_combinations_reject():
    assert TaskPoseRuntimeTiming().resolve(0.05) == {
        "inference_budget_s": 0.1,
        "cycle_budget_s": 0.1,
        "max_state_age_s": 0.25,
    }
    with pytest.raises(ValueError):
        TaskPoseRuntimeTiming(inference_budget_s=0.2).resolve(0.05)
    with pytest.raises(ValueError):
        TaskPoseRuntimeTiming(max_state_age_s=None)


@pytest.fixture
def environment():
    env = ManipulationEnv(
        ManipulationConfig(
            robot="innate_mars",
            scenario="reach",
            action_mode="direct_velocity",
            trajectory_preview=True,
            randomization_scale=0,
            horizon=15,
        )
    )
    env.reset(seed=1900000)
    yield env
    env.close()


def start_runtime(env, monkeypatch, timing, *, command_timeout=15):
    clock = SimpleNamespace(now=0.0)
    clock.monotonic = lambda: clock.now
    monkeypatch.setattr(module, "time", clock)
    runtime = TaskPoseCommandRuntime(env, timing=timing)
    runtime.command_tracker._clock = clock.monotonic
    client = SimpleNamespace(_act_lock=threading.Lock(), _iter=0, _state=runtime.command_message({}))

    def send(action):
        runtime.dispatch_command(action)
        client._state = runtime.command_message({})

    client.send_message = send
    control = env.control
    path = TaskPoseTrajectory(
        [0, 1], [control.base_pose()] * 2, [control.ee_pose()] * 2, [control.grip_reference] * 2, base_mask=[True] * 3
    )
    action = send_task_pose_trajectory(client, path, motion_timeout_s=command_timeout)
    return runtime, client, action, clock


@pytest.mark.sim
@pytest.mark.parametrize(
    ("inference", "cycle", "age", "expected"),
    [
        (0.2, 0.4, 0.3, None),
        (0.12, 0.4, 0.3, "policy inference deadline exceeded"),
        (0.2, 0.25, 0.3, "control step deadline exceeded"),
        (0.2, 0.4, 0.1, "stale physics state"),
    ],
)
def test_explicit_budgets_do_not_waive_independent_freshness(environment, monkeypatch, inference, cycle, age, expected):
    runtime, client, action, clock = start_runtime(
        environment,
        monkeypatch,
        TaskPoseRuntimeTiming(inference_budget_s=inference, cycle_budget_s=cycle, max_state_age_s=age),
    )
    steps = []
    original = environment.step

    def infer(obs):
        clock.now += 0.15
        return np.zeros(environment.action_space.shape)

    def step(value):
        steps.append(value.copy())
        result = original(value)
        clock.now += 0.15
        return result

    runtime.policy = infer
    monkeypatch.setattr(environment, "step", step)
    runtime.advance()
    client._state = runtime.command_message({})
    receipt = command_receipt(client, action)
    assert client._state["task_pose_runtime_timing"] == runtime.timing_limits
    if expected is None:
        assert receipt["status"] == "running"
        assert len(steps) == 1
        assert runtime.last_timing["cycle_s"] == pytest.approx(0.3)
    else:
        assert receipt["status"] == "failed" and receipt["result"]["failure"] == expected
        assert environment.control.stopped
        assert len(steps) == int(expected == "control step deadline exceeded")


@pytest.mark.sim
def test_command_expiry_during_inference_prevents_another_action(environment, monkeypatch):
    runtime, client, action, clock = start_runtime(
        environment,
        monkeypatch,
        TaskPoseRuntimeTiming(inference_budget_s=0.5, cycle_budget_s=1, max_state_age_s=1),
        command_timeout=0.1,
    )
    steps = []

    def infer(obs):
        clock.now += 0.2
        return np.ones(environment.action_space.shape)

    runtime.policy = infer
    monkeypatch.setattr(environment, "step", lambda action: steps.append(action))
    runtime.advance()
    client._state = runtime.command_message({})
    receipt = command_receipt(client, action)
    assert receipt["status"] == "failed" and "motion deadline exceeded" in receipt["reason"]
    assert not steps and environment.control.stopped
    assert not environment.control.last_action.any()


@pytest.mark.sim
def test_default_teacher_config_preserves_action_and_metric_changes_are_bounded(environment):
    c = environment.control
    hand = c.ee_pose().copy()
    hand[:3] += [0.002, 0.001, 0]
    c.set_pose_targets(c.base_pose(), [True] * 3, hand, True)
    np.testing.assert_array_equal(
        task_pose_teacher_action(c), task_pose_teacher_action(c, config=TaskPoseTeacherConfig())
    )
    for length in (0.1, 1, 2):
        action = task_pose_teacher_action(c, config=TaskPoseTeacherConfig(rotation_length_m_per_rad=length))
        assert np.isfinite(action).all() and np.max(abs(action)) <= 1


@pytest.mark.sim
@pytest.mark.parametrize("axis", [2, 3, 4])
def test_public_task_predicates_require_all_base_dofs_stopped(axis):
    case = json.loads(CANONICAL_SUITE.read_text())["cases"][0]
    env, admission = fixture(case)
    try:
        assert admission["accepted"]
        robot = MarsSimRobot(env)
        env.data.qvel[:] = 0
        assert final_predicates(robot, admission)["robot_still"]
        env.data.qvel[robot.servo.base_dof + axis] = 0.1
        assert not final_predicates(robot, admission)["robot_still"]
    finally:
        env.close()

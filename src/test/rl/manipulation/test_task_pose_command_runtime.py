"""Exercise task-space schedules through actual ownership and receipt logic."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from emet.core.command_client import cancel_reactive_goal, command_receipt, send_command, update_reactive_goal
from emet.rl.manipulation.command_runtime import (
    TaskPoseCommandRuntime,
    parse_trajectory_goal,
    send_task_pose_trajectory,
)
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.tracking import TaskFrameFK, TaskPoseTrajectory


@pytest.fixture
def runtime():
    env = ManipulationEnv(ManipulationConfig(
        robot='innate_mars', scenario='reach', action_mode='direct_velocity',
        trajectory_preview=True, randomization_scale=0, horizon=15,
    ))
    env.reset(seed=1900000)
    server = TaskPoseCommandRuntime(env)
    yield server
    env.close()


def client(server):
    c = SimpleNamespace(_act_lock=threading.Lock(), _iter=0, _state=server.command_message({}))
    def send(action):
        server.dispatch_command(action)
        c._state = server.command_message({})
    c.send_message = send
    return c


def path(server, offset=0):
    c = server.env.control
    goal = c.positions().copy()
    goal[2] += offset
    fk = TaskFrameFK(server.env.model, server.env.data, c)
    return TaskPoseTrajectory(
        [0, 1], np.repeat(c.base_pose()[None], 2, axis=0),
        [c.ee_pose(), fk(c.base_pose(), goal)], [c.grip_reference] * 2,
        base_mask=[True] * 3,
    )


@pytest.mark.sim
def test_reachable_pose_schedule_completes_through_command_receipts(runtime):
    c = client(runtime)
    action = send_task_pose_trajectory(c, path(runtime, 0.01))
    for _ in range(300):
        runtime.advance()
        c._state = runtime.command_message({})
        receipt = command_receipt(c, action)
        if receipt['status'] in {'succeeded', 'failed'}:
            break
    assert receipt['status'] == 'succeeded', receipt
    assert receipt['result']['stop_confirmed']
    assert receipt['result']['measurements']['ee_position_error'] < 0.005
    assert not runtime.env.control.release_authorized


@pytest.mark.sim
def test_replace_cancel_ownership_and_faults_use_shared_protocol(runtime):
    c = client(runtime)
    action = send_task_pose_trajectory(c, path(runtime, 0.02))
    runtime.advance()
    update_reactive_goal(c, action, {'trajectory': path(runtime, -0.01).record()})
    with pytest.raises(RuntimeError, match='another client'):
        send_task_pose_trajectory(client(runtime), path(runtime))
    assert cancel_reactive_goal(c, action)
    assert command_receipt(c, action)['result']['stop_confirmed']
    with pytest.raises(RuntimeError):
        send_task_pose_trajectory(c, path(runtime))  # cancellation cannot re-arm the stage


@pytest.mark.sim
def test_stale_physics_stops_before_another_command(runtime):
    c = client(runtime)
    action = send_task_pose_trajectory(c, path(runtime))
    runtime.last_advance -= 1
    qpos = runtime.env.data.qpos.copy()
    runtime.advance()
    c._state = runtime.command_message({})
    receipt = command_receipt(c, action)
    assert receipt['status'] == 'failed'
    assert receipt['result']['failure'] == 'stale physics state'
    assert runtime.env.control.stopped
    # Stop confirmation may step physics; it must never update the requested target.
    assert runtime.env.control.task_trajectory is not None
    assert np.isfinite(qpos).all()


@pytest.mark.sim
def test_unadvertised_or_malformed_schedule_never_starts(runtime):
    c = client(runtime)
    c._state['task_pose_control'] = False
    with pytest.raises(RuntimeError, match='advertise'):
        send_task_pose_trajectory(c, path(runtime))
    record = path(runtime).record()
    record['base_mask'] = [1, 1, 1]
    with pytest.raises(ValueError, match='booleans'):
        parse_trajectory_goal({'trajectory': record})
    with pytest.raises(RuntimeError):
        send_command(client(runtime), {'reactive_goal': {'trajectory': record}})
    assert runtime._reactive_command is None


@pytest.mark.sim
def test_tamp_planner_pose_schedule_runs_through_protocol_without_relocation(runtime):
    from emet.rl.manipulation.tamp_cases import execute_pose_case, plan_case

    env = runtime.env
    before = env.data.qpos.copy()
    goal = env.control.positions() + [0, 0, .01, 0, 0]
    case = plan_case(env, goal)
    assert case['planning_error'] is None, case
    np.testing.assert_array_equal(env.data.qpos, before)
    result = execute_pose_case(env, case)
    assert result['execution_verified'], result
    assert result['execution_interface'] == 'task_pose_command_protocol'
    assert result['command_receipt']['result']['stop_confirmed']
    assert not env.control.release_authorized


@pytest.mark.sim
def test_unconfirmed_stop_retains_command_ownership(runtime, monkeypatch):
    c = client(runtime)
    send_task_pose_trajectory(c, path(runtime))
    monkeypatch.setattr(runtime, 'cancel_reactive_command', lambda: False)
    from emet.core.command_client import close_command_session

    close_command_session(c)
    assert runtime._reactive_fault
    assert runtime._reactive_command is not None
    with pytest.raises(RuntimeError, match='another client'):
        send_task_pose_trajectory(client(runtime), path(runtime))


@pytest.mark.sim
def test_mid_motion_retarget_keeps_deadline_and_reaches_replacement(runtime):
    c = client(runtime)
    action = send_task_pose_trajectory(c, path(runtime, .02))
    original_deadline = command_receipt(c, action)['deadline']
    for _ in range(4):
        runtime.advance()
    update_reactive_goal(c, action, {'trajectory': path(runtime, -.01).record()})
    assert command_receipt(c, action)['deadline'] == original_deadline
    for _ in range(300):
        runtime.advance()
        c._state = runtime.command_message({})
        receipt = command_receipt(c, action)
        if receipt['status'] in {'succeeded', 'failed'}:
            break
    assert receipt['status'] == 'succeeded', receipt
    assert receipt['result']['stop_confirmed']


@pytest.mark.sim
def test_late_policy_inference_stops_before_accepting_action(runtime):
    import time

    c = client(runtime)
    action = send_task_pose_trajectory(c, path(runtime))
    def slow(obs):
        time.sleep(.11)
        return np.ones(runtime.env.action_space.shape)
    runtime.policy = slow
    runtime.advance()
    c._state = runtime.command_message({})
    receipt = command_receipt(c, action)
    assert receipt['status'] == 'failed'
    assert receipt['result']['failure'] == 'policy inference deadline exceeded'
    assert runtime.env.control.stopped
    assert not runtime.env.control.last_action.any()

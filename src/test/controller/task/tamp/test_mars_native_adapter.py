"""Public CHAT tools over measured Mars simulation; no hardware transport."""

import json

import numpy as np
import pytest

from emet.agent.tools import get_tools
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult
from emet.controller.task.tamp.mars_native_adapter import MarsNativeTableAdapter
from emet.controller.task.tamp.mars_sim_robot import MarsSimRobot
from emet.controller.task.tamp.native_adapter import NativePlan, adapter_for
from emet.eval.mars_tamp_control import fixture, run_case, suite

pytestmark = pytest.mark.sim


@pytest.fixture
def mars():
    env, admission = fixture(suite()["cases"][0])
    assert admission["accepted"], admission
    robot = MarsSimRobot(env)
    context = {"robot": robot, "manip_mode": "physical", "tamp_inputs": "privileged", "_native_tamp_model": env.model}
    tools = {t.name: t for t in get_tools(context)}

    def call(name, **kwargs):
        robot.refresh()
        return json.loads(tools[name].func(**kwargs))

    yield env, robot, context, call
    env.close()


def test_public_discovery_routes_mars_and_preserves_live_state(mars):
    env, robot, context, call = mars
    before = env.data.qpos.copy(), env.data.qvel.copy(), float(env.data.time)
    result = call("scene_tasks", object_filter="blue", robot="innate_mars")
    assert result["status"] == "ok", result
    assert {x["receptacle_name"] for x in result["data"]["tasks"]} == {"source support", "destination support"}
    assert isinstance(adapter_for(context, robot), MarsNativeTableAdapter)
    assert not robot.commands
    np.testing.assert_array_equal(before[0], env.data.qpos)
    np.testing.assert_array_equal(before[1], env.data.qvel)
    assert before[2] == env.data.time
    assert call("scene_tasks", robot="stretch")["code"] == "native_robot_unsupported"


def test_policy_backend_receives_only_actor_observations_without_teacher_fallback(mars, monkeypatch):
    env, _, _, _ = mars
    observations = []

    def actor(observation):
        observations.append(observation)
        return np.zeros(env.action_space.shape)

    def forbidden(*args):
        raise AssertionError("teacher must not run during learned execution")

    robot = MarsSimRobot(env, policy=actor, record_observations=True)
    monkeypatch.setattr(robot, "_joint_teacher_action", forbidden)
    monkeypatch.setattr("emet.controller.task.tamp.mars_sim_robot.task_pose_teacher_action", forbidden)
    before = env.data.time
    robot.command_arm_reference(robot.servo.positions(), blocking=False)
    robot.step(env.profile.dt)
    assert robot.policy_calls == 1 and env.data.time > before
    assert set(observations[0]) == {"history", "preview"}
    assert not robot.failure and robot.trace[-1]["controller"] == "checkpoint_policy"
    assert robot.commands[-1]["target_source"] == "certified_joint_witness_fk"
    recorded = robot.trace[-1]["actor_observation"]
    np.testing.assert_array_equal(recorded["history"], observations[0]["history"])
    np.testing.assert_array_equal(recorded["preview"], observations[0]["preview"])


@pytest.mark.parametrize("bad", [np.full(8, np.nan), np.zeros(7), np.full(8, 2.0)])
def test_invalid_policy_action_latches_stop_and_never_falls_back(mars, bad):
    env, _, _, _ = mars
    robot = MarsSimRobot(env, policy=lambda obs: bad)
    robot.command_arm_reference(robot.servo.positions(), blocking=False)
    robot.step(env.profile.dt)
    assert robot.failure == "policy_inference_failed"
    assert robot.servo.stopped and robot.policy_error
    assert robot.policy_calls == 1
    np.testing.assert_array_equal(robot.trace[-1]["normalized_action"], np.zeros(8))
    robot.step(env.profile.dt)
    assert robot.policy_calls == 1


def test_policy_deadline_discards_the_action_and_latches_stop(mars, monkeypatch):
    from types import SimpleNamespace

    env, _, _, _ = mars
    clock = [0.0]

    def slow_actor(observation):
        clock[0] += 0.2
        return np.ones(env.action_space.shape)

    robot = MarsSimRobot(env, policy=slow_actor)
    robot.command_arm_reference(robot.servo.positions(), blocking=False)
    monkeypatch.setattr(
        "emet.controller.task.tamp.mars_sim_robot.time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 0.0),
    )
    robot.step(env.profile.dt)
    assert robot.failure == "policy_inference_deadline" and robot.servo.stopped
    np.testing.assert_array_equal(robot.trace[-1]["normalized_action"], np.zeros(8))


def test_public_tamp_execution_calls_selected_actor_and_keeps_partial_failure(mars, monkeypatch):
    env, robot, _, call = mars
    robot.policy = lambda observation: np.zeros(env.action_space.shape)

    def forbidden(*args):
        raise AssertionError("no teacher fallback")

    monkeypatch.setattr(robot, "_joint_teacher_action", forbidden)
    tasks = call("scene_tasks", object_filter="blue")["data"]["tasks"]
    selected = next(x for x in tasks if x["receptacle_name"] == "destination support")
    planned = call("plan_pick_place", task_ref=selected["task_ref"])
    assert planned["status"] == "ok", planned
    assert planned["data"]["controller"] == "mars_checkpoint_policy"
    with robot.stepped_clock():
        result = call("execute_pick_place_plan", plan_ref=planned["data"]["plan_ref"])
    assert result["status"] == "partial" and robot.policy_calls > 0
    assert result["data"]["policy_calls"] == robot.policy_calls
    assert robot.failure and robot.stop_confirmed
    first_arm = next(c for c in robot.commands if "arm" in c)
    first_actor = next(t for t in robot.trace if t["action_source"] == "policy")
    assert first_actor["time"] > first_arm["sim_time"]
    assert all(t["policy_calls"] == 0 for t in robot.trace if t["time"] <= first_arm["sim_time"])


def test_checkpoint_stays_in_actuator_hold_until_an_explicit_motion_command(mars):
    env, _, _, _ = mars
    calls = []

    def actor(observation):
        calls.append(observation)
        return np.zeros(env.action_space.shape)

    robot = MarsSimRobot(env, policy=actor, record_observations=True)
    robot.step(0.5)
    assert not calls and not robot.failure
    assert {t["action_source"] for t in robot.trace} == {"executor_idle_hold"}
    assert robot.servo.history is not None
    assert robot.command_arm_reference(robot.servo.positions(), blocking=False)
    robot.step(env.profile.dt)
    assert len(calls) == 1 and robot.trace[-1]["action_source"] == "policy"
    robot.servo.hold_position()
    robot.step(env.profile.dt)
    assert len(calls) == 1 and robot.trace[-1]["action_source"] == "executor_terminal_hold"


def test_explicit_trajectory_activates_actor_and_terminal_hold_stops_inference(mars):
    from emet.rl.manipulation.tracking import TaskPoseTrajectory

    env, _, _, _ = mars
    robot = MarsSimRobot(env, policy=lambda observation: np.zeros(env.action_space.shape))
    path = TaskPoseTrajectory(
        [0, 0.1],
        [robot.servo.base_pose()] * 2,
        [robot.servo.ee_pose()] * 2,
        [env.profile.gripper_open] * 2,
        base_mask=np.ones(3, dtype=bool),
    )
    result = robot.follow_trajectory(path, loaded=False)
    assert result.success, result
    assert robot.policy_calls > 0
    assert any(t["action_source"] == "policy" for t in robot.trace)
    assert robot.trace[-1]["action_source"] == "executor_terminal_hold"


def test_explicit_model_required_and_observed_mode_cannot_read_gt(mars):
    env, robot, context, call = mars
    with pytest.raises(ObservationUnavailable, match="native_mars_model_required"):
        adapter_for({}, robot)
    context["tamp_inputs"] = "observed"
    result = call("scene_tasks", object_filter="blue")
    assert result["code"] == "observed_scene_unavailable"
    assert not robot.commands


def test_scene_geometry_change_invalidates_cached_handles(mars):
    env, robot, context, call = mars
    tasks = call("scene_tasks", object_filter="blue")["data"]["tasks"]
    old = adapter_for(context, robot)
    env.model.body_pos[env.model.body("destination").id, 0] += 0.01
    robot.refresh()
    new = adapter_for(context, robot)
    assert new is not old
    assert new.namespace != old.namespace
    assert call("plan_pick_place", task_ref=tasks[0]["task_ref"])["code"] == "unknown_task_ref"
    assert not robot.commands


def test_soft_stop_opening_certifies_both_jaws_without_clamping_live_state(mars):
    env, robot, context, call = mars
    adapter = adapter_for(context, robot)
    executor = adapter._executor("object1")
    before = env.data.qpos.copy()
    path, error = executor.plan_gripper_open()
    assert error is None, error
    assert len(path[-1]) == 2
    for name, target in executor.gripper_open_configuration.items():
        lo, hi = env.model.joint(name).range
        assert lo <= target <= hi
    np.testing.assert_array_equal(before, env.data.qpos)
    address = env.model.joint("joint6").qposadr[0]
    executor.data.qpos[address] = env.model.joint("joint6").range[1] + 0.004
    assert executor.plan_gripper_open()[1] == "gripper_open_measured_bounds"


def test_task_selector_and_partial_execution_do_not_reuse_plan(mars, monkeypatch):
    env, robot, context, call = mars
    tasks = call("scene_tasks", object_filter="blue")["data"]["tasks"]
    assert (
        call("plan_pick_place", task_ref=tasks[0]["task_ref"], object_name="red cylinder")["code"]
        == "task_selector_mismatch"
    )
    adapter = adapter_for(context, robot)
    adapter.plans["frozen"] = NativePlan(
        "object1",
        "destination",
        env.data.qpos.copy(),
        np.zeros(3),
        np.eye(3),
        None,
        adapter.boot,
        object_name="blue cube",
        receptacle_name="destination support",
    )
    monkeypatch.setattr(adapter, "_approach", lambda plan: PhysicalMotionResult(False, "diagnostic_stop", "approach"))
    result = call("execute_pick_place_plan", plan_ref="frozen")
    assert result["status"] == "partial" and result["data"]["failed_stage"] == "approach"
    assert call("execute_pick_place_plan", plan_ref="frozen")["code"] == "unknown_plan"
    assert not robot.commands


def test_fabricated_driver_success_cannot_pass_physical_score():
    row = run_case(suite()["cases"][0], tool_driver=lambda call, admission, tools: {"status": "ok"})
    assert not row["success"]
    assert not row["final_predicates"]["destination_support"]
    assert row["calls"] == []


def test_frozen_fixture_change_rejected_before_commands():
    case = dict(suite()["cases"][0])
    env, admission = fixture(case)
    try:
        case["frozen_fixture"] = admission
        case["model_signature"] = "changed"
        case["initial_qpos"] = env.data.qpos.tolist()
    finally:
        env.close()
    row = run_case(case)
    assert row["fixture_admission"]["reason"] == "frozen_fixture_changed"
    assert not row["success"]
    assert "commands" not in row


def test_joint_commands_do_not_acquire_world_pose_counter_rotation(mars):
    env, robot, context, call = mars
    robot.servo.base_goal[2] += 0.1
    robot.servo.reference[:] = robot.servo.target
    action = robot._joint_teacher_action()
    np.testing.assert_array_equal(action[2:-1], np.zeros(5))
    assert action[1] > 0


def test_clock_context_excludes_concurrent_sessions_and_restores_modules(mars):
    from emet.controller.manipulation import physical_pick_place
    from emet.controller.task.tamp import native_adapter

    env, robot, context, call = mars
    old = physical_pick_place.time, native_adapter.time
    with robot.stepped_clock():
        with pytest.raises(RuntimeError, match="isolated processes"):
            with robot.stepped_clock():
                pytest.fail("nested clock entered")
    assert (physical_pick_place.time, native_adapter.time) == old


def test_transport_turns_about_axle_without_lateral_wheel_motion(mars):
    from types import SimpleNamespace

    import mujoco

    from emet.controller.task.tamp.mars_transport import plan_transport
    from emet.rl.manipulation.profiles import MARS

    env, robot, context, call = mars
    scratch = mujoco.MjData(env.model)
    mujoco.mj_copyData(scratch, env.model, env.data)
    # Test path kinematics independently of obstacle screening, which is tested
    # by the physical planner. Never edit live data or claim collision admission.
    checker = SimpleNamespace(configuration_collides=lambda model, data: False)
    goal = robot.servo.base_pose() + [0.05, 0.08, 0.4]
    before = env.data.qpos.copy()
    result, error = plan_transport(env.model, scratch, checker, None, None, base_target=goal)
    assert error is None, error
    path = result["trajectory"].record()
    base = np.asarray(path["base"])
    yaw = base[:, 2]
    offset = robot.servo.axle_offset[:2]
    rotated = np.column_stack(
        [np.cos(yaw) * offset[0] - np.sin(yaw) * offset[1], np.sin(yaw) * offset[0] + np.cos(yaw) * offset[1]]
    )
    axle = base[:, :2] + rotated
    delta = np.diff(axle, axis=0)
    heading = (yaw[1:] + yaw[:-1]) / 2
    lateral = -np.sin(heading) * delta[:, 0] + np.cos(heading) * delta[:, 1]
    assert np.max(abs(lateral)) < 1e-9
    dt = np.diff(path["times"])
    assert np.max(np.linalg.norm(delta, axis=1) / dt) <= MARS.base_speed
    assert np.max(abs(np.diff(yaw) / dt)) <= MARS.base_yaw_speed
    np.testing.assert_allclose(base[-1], goal, atol=1e-10)
    np.testing.assert_array_equal(before, env.data.qpos)


def test_public_tools_complete_frozen_20g_transfer_with_supported_release():
    from emet.eval.mars_tamp_control import CANONICAL_SUITE

    row = run_case(json.loads(CANONICAL_SUITE.read_text())["cases"][0])
    assert row["frozen_fixture_verified"]
    assert row["success"], {k: row[k] for k in ["robot_failure", "final_predicates", "events"]}
    assert row["peak_base_displacement_m"] > 0.02
    assert all(row["final_predicates"].values()) and row["post_execution_stable_s"] == 2
    assert row["max_payload_rotation_rad"] <= 0.1
    assert not row["contact_metrics"]["wheel_lift"]
    confirmations = [i for i, command in enumerate(row["commands"]) if "support_confirmed_N" in command]
    release = max(i for i, command in enumerate(row["commands"]) if command.get("gripper") == "open")
    assert confirmations and confirmations[-1] < release


def test_loaded_gripper_cannot_open_or_probe_outside_place(mars):
    env, robot, context, call = mars
    robot.retention_reference = np.eye(4)
    robot.placement_support = "destination"
    before = env.data.qpos.copy()
    assert not robot.confirm_support_contact()
    assert not robot.open_gripper(blocking=True)
    assert robot.failure == "release_support_unverified"
    assert not robot.commands and robot.servo.stopped
    np.testing.assert_array_equal(before, env.data.qpos)


def test_joint_reference_preview_uses_measured_world_frame_after_base_tilt(mars):
    import mujoco

    env, robot, context, call = mars
    q = robot.servo.positions().copy()
    original = robot.servo.ee_pose().copy()
    joint = env.model.body(env.profile.base_body).jntadr[0]
    address = env.model.jnt_qposadr[joint]
    tilt = np.array([np.cos(0.01 / 2), 0, np.sin(0.01 / 2), 0])
    quat = env.data.qpos[address + 3 : address + 7].copy()
    mujoco.mju_mulQuat(env.data.qpos[address + 3 : address + 7], tilt, quat)
    mujoco.mj_forward(env.model, env.data)
    expected = robot.servo.ee_pose().copy()
    assert np.linalg.norm(expected[:3] - original[:3]) > 0.001
    assert robot.command_arm_reference(q, blocking=False)
    np.testing.assert_allclose(robot.servo.hand_goal, expected, atol=1e-10)


@pytest.mark.parametrize("sign", [-1, 1])
def test_transport_final_turn_crosses_pi_by_the_short_path(mars, sign):
    from types import SimpleNamespace

    import mujoco

    from emet.controller.task.tamp.mars_transport import plan_transport

    env, robot, _, _ = mars
    scratch = mujoco.MjData(env.model)
    mujoco.mj_copyData(scratch, env.model, env.data)
    adr = int(env.model.joint("manipulation_base").qposadr[0])
    yaw = sign * (np.pi - 0.02)
    scratch.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(env.model, scratch)
    base = scratch.body("base_link").xpos[:2].copy()
    end = -yaw

    def rot(a):
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    axle = base + rot(yaw) @ robot.servo.axle_offset[:2]
    goal = np.r_[axle - rot(end) @ robot.servo.axle_offset[:2], end]
    result, error = plan_transport(
        env.model, scratch, SimpleNamespace(configuration_collides=lambda *args: False), None, None, base_target=goal
    )
    assert error is None, error
    path = result["trajectory"]
    assert path.end < 2
    turns = np.unwrap(path.base[:, 2])
    assert abs(turns[-1] - turns[0]) < 0.05
    assert abs(np.arctan2(np.sin(turns[-1] - end), np.cos(turns[-1] - end))) < 1e-8

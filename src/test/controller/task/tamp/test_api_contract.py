"""Agent contract and guards: real tool registry, no robot actuation required."""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from emet.agent.tools import get_tools
from emet.controller.task.tamp import agent_bridge as bridge
from emet.controller.task.tamp.api import TAMP_TOOLS, failure_code, json_tool, plan_data, refusal_from_teleport_collision, response
from emet.controller.task.tamp.task_search import TaskPlan, TaskPlanStep


class Robot:
    def __init__(self):
        self._state = {"command_protocol": {"version": 2, "server_boot_id": "boot"}, "sim_base_pose_query": True}
        self.placements = {
            "object_private": {"cat": "bowl", "pos": [0.0, 0.0, 0.8], "quat": [1.0, 0.0, 0.0, 0.0]},
            "receptacle_private": {"cat": "table", "pos": [1.0, 0.0, 0.8], "quat": [1.0, 0.0, 0.0, 0.0]},
        }
        self.caps = {"kinematic_manip": True}
        self.clear = True

    def get_emet_session(self):
        return {"is_simulation": True, "capabilities": self.caps, "sim_object_placements": self.placements}

    def check_base_poses(self, poses):
        return {"poses": poses, "clear": [self.clear] * len(poses)}


def setup_plan():
    robot = Robot()
    task = bridge.AgentTaskRef("task:1", "bowl", "table", "object_private", "receptacle_private")
    plan = TaskPlan(
        [TaskPlanStep("approach", {"xyt": [0.0, 0.5, 0.0]}), TaskPlanStep("grasp"), TaskPlanStep("place")],
        task.object_body,
        task.receptacle_body,
        success=True,
    )
    snapshot = bridge.PlanningSnapshot(
        "boot", bridge.robot_session_key(robot), json.dumps(robot.caps, sort_keys=True), json.dumps(robot.placements)
    )
    build = bridge.AgentPlanBuild(task, plan, "kinematic", True, snapshot=snapshot)
    context = {"robot": robot}
    return robot, plan, build, context


def decoded(value):
    result = json.loads(value, parse_constant=lambda s: pytest.fail(s))
    assert set(result) == {"schema_version", "tool", "status", "code", "message", "data", "recovery"}
    assert result["schema_version"] == 1
    assert result["status"] in {"ok", "partial", "error"}
    assert "object_private" not in value and "receptacle_private" not in value
    return result


def test_all_tools_return_json_on_missing_state_and_bad_arguments():
    tools = {t.name: t for t in get_tools({})}
    args = {
        "scene_tasks": {},
        "plan_pick_place": {},
        "execute_pick_place_plan": {"plan_ref": "missing"},
        "pick_place": {"object_name": "bowl", "receptacle_name": "table"},
    }
    for name in TAMP_TOOLS:
        assert decoded(tools[name].func(**args[name]))["tool"] == name
        assert decoded(tools[name].func(unexpected=True))["status"] == "error"


@pytest.mark.parametrize(
    "change,code",
    [
        ("translation", "scene_changed_replan"),
        ("rotation", "scene_changed_replan"),
        ("nan", "invalid_plan"),
        ("boot", "scene_changed_replan"),
        ("capability", "kinematic_capability_missing"),
        ("blocked", "approach_changed_replan"),
        ("query_error", "approach_validation_failed"),
        ("malformed", "approach_validation_failed"),
    ],
)
def test_stale_or_unsafe_plan_is_consumed_without_motion(monkeypatch, change, code):
    robot, plan, build, context = setup_plan()
    ref = bridge.store_agent_plan(context, robot, build)
    if change == "translation":
        robot.placements["object_private"]["pos"][0] += 0.011
    if change == "rotation":
        robot.placements["receptacle_private"]["quat"] = [np.cos(0.1), 0, 0, np.sin(0.1)]
    if change == "nan":
        robot.placements["object_private"]["pos"][0] = float("nan")
    if change == "boot":
        robot._state["command_protocol"]["server_boot_id"] = "new"
    if change == "capability":
        robot.caps.clear()
    if change == "blocked":
        robot.clear = False
    if change == "query_error":
        robot.check_base_poses = Mock(side_effect=TimeoutError("private detail"))
    if change == "malformed":
        robot.check_base_poses = Mock(return_value={"poses": [[0.0, 0.5, 0.0]], "clear": [1]})
    execute = Mock()
    monkeypatch.setattr(bridge, "execute_agent_plan", execute)
    result = bridge.execute_stored_agent_plan_result(robot, context, ref)
    assert result["code"] == code
    assert not execute.called
    assert bridge.execute_stored_agent_plan_result(robot, context, ref)["code"] == "unknown_plan"


@pytest.mark.parametrize("success", [True, False])
def test_public_plan_execute_and_partial_evidence(monkeypatch, success):
    robot, plan, build, context = setup_plan()
    monkeypatch.setattr(bridge, "build_agent_pick_place_plan", lambda *a, **k: build)

    def execute(*_):
        plan.completed_ops = ["approach", "grasp", "place"] if success else ["approach"]
        plan.success = success
        plan.failed_op = None if success else "grasp"
        plan.message = "ok" if success else "grasp_failed:pregrasp_tracking_failed"
        plan.measurements = [{"stage": "grasp", "error_m": 0.086}]
        return plan

    monkeypatch.setattr(bridge, "execute_agent_plan", execute)
    tools = {t.name: t for t in get_tools(context)}
    result = decoded(tools["plan_pick_place"].func(object_name="bowl", receptacle_name="table"))
    assert result["status"] == "ok"
    assert result["data"]["completed_ops"] == []
    result = decoded(tools["execute_pick_place_plan"].func(plan_ref=result["data"]["plan_ref"]))
    assert result["status"] == ("ok" if success else "partial")
    assert result["code"] == ("ok" if success else "tracking_timeout")
    assert result["data"]["measurements"][0]["error_m"] == 0.086
    assert result["data"]["failed_stage"] == (None if success else "grasp")


def test_nonfinite_measurement_is_null_and_exception_is_private():
    @json_tool
    def scene_tasks():
        return response("scene_tasks", data={"error_m": float("inf")})

    assert decoded(scene_tasks())["data"]["error_m"] is None

    @json_tool
    def pick_place():
        raise RuntimeError("object_private")

    assert decoded(pick_place())["code"] == "internal_error"


def test_missing_pose_or_server_identity_cannot_be_stored():
    robot, _, build, context = setup_plan()
    robot._state.clear()
    with pytest.raises(ValueError, match="scene_changed_replan"):
        bridge.store_agent_plan(context, robot, build)
    assert "_tamp_plans" not in context


def test_kinematic_bridge_tries_collision_and_ik_alternatives(monkeypatch):
    from types import SimpleNamespace

    from emet.controller.manipulation import kinematic_pick_place
    from emet.controller.task.tamp import task_search

    robot = Robot()
    robot.check_base_poses = lambda poses: {"poses": np.asarray(poses).tolist(), "clear": [False] + [True] * 15}
    seen = []
    executor = SimpleNamespace(_ensure_model=lambda: True, _model=None, _data=None, ee_body="tool", joint_names=["arm"])
    monkeypatch.setattr(kinematic_pick_place, "KinematicPickPlaceExecutor", lambda *a, **k: executor)
    monkeypatch.setattr(task_search, "_sync_executor_base_to_xyt", lambda ex, pose: seen.append(pose.copy()))
    monkeypatch.setattr(
        task_search, "rank_grasps_by_ik", lambda *a, **k: [(0, 0.001 if len(seen) > 1 else 1.0, len(seen) > 1)]
    )
    result = bridge.build_agent_pick_place_plan(robot, "bowl", "table", seed=0)
    assert result.mode == "kinematic"
    assert result.plan.success, result.reason
    np.testing.assert_allclose(
        result.plan.steps[0].args["xyt"], task_search.approach_candidates_for_object_xy([0, 0])[2]
    )


def test_stale_joint_observation_cannot_pass_arrival():
    import time

    import mujoco

    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

    ex = object.__new__(KinematicPickPlaceExecutor)
    ex._model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="tool"/></worldbody></mujoco>')
    ex._data = mujoco.MjData(ex._model)
    ex.robot = Robot()
    ex.robot._state_received_monotonic = time.monotonic() - 3
    assert not ex._sync_qpos_from_robot()
    assert ex._last_motion_failure == "stale_observation"


@pytest.mark.parametrize("failure", ["approach", "retract", "release", "missing_receptacle"])
def test_place_does_not_hide_failed_submotions(monkeypatch, failure):
    from emet.controller.manipulation import kinematic_pick_place as module

    ex = object.__new__(module.KinematicPickPlaceExecutor)
    ex.robot = Robot()
    ex.ee_body = "tool"
    ex.place_z_offset_m = 0.0
    ex._ensure_model = lambda: True
    ex._placements = lambda: ex.robot.placements
    ex._approach_xy = lambda *a, **k: failure != "approach"
    ex._sleep = lambda *a: None
    ex._verify_place_xy = lambda *a: (True, 0.001)
    motions = iter([(True, 0.001), (True, 0.001), (failure != "retract", 0.08)])
    ex._plan_and_execute_ee = lambda *a: next(motions)
    ex._last_motion_failure = "tracking_failed"
    ex._set_gripper = Mock(side_effect=RuntimeError() if failure == "release" else None)
    attach = Mock()
    monkeypatch.setattr(module, "robot_zmq_attach_body", attach)
    monkeypatch.setattr(module, "robot_zmq_detach_body", Mock())
    monkeypatch.setattr(module, "robot_zmq_set_body_pose", Mock())
    result = ex.place_only(
        "table",
        object_gt_body="object_private",
        receptacle_gt_body="missing" if failure == "missing_receptacle" else "receptacle_private",
    )
    assert not result.success
    if failure in {"approach", "missing_receptacle"}:
        assert not attach.called


@pytest.mark.parametrize("change", ["position", "rotation", "boot", "capability", "nan"])
def test_change_between_planning_and_storage_refuses_handle(change):
    robot, _, build, context = setup_plan()
    if change == "position":
        robot.placements["object_private"]["pos"][0] += 0.5
    elif change == "rotation":
        robot.placements["receptacle_private"]["quat"] = [0.0, 0.0, 0.0, 1.0]
    elif change == "boot":
        robot._state["command_protocol"]["server_boot_id"] = "new"
    elif change == "capability":
        robot.caps["sim_set_body_pose"] = True
    else:
        robot.placements["object_private"]["pos"][0] = float("nan")
    with pytest.raises(ValueError, match="scene_changed_replan|invalid_plan"):
        bridge.store_agent_plan(context, robot, build)
    assert "_tamp_plans" not in context


def test_planner_uses_captured_inputs_and_rejects_change_during_grounding(monkeypatch):
    robot, plan, _, _ = setup_plan()
    robot.caps = {"sim_set_body_pose": True}
    seen = []

    def changing_planner(*args, **kwargs):
        robot.placements["object_private"]["pos"][0] += 0.5
        seen.append(kwargs["placements"]["object_private"]["pos"][0])
        return plan

    monkeypatch.setattr(bridge, "plan_pick_place_mcts", changing_planner)
    result = bridge.build_agent_pick_place_plan(robot, "bowl", "table")
    assert seen == [0.0]
    assert result.reason == "scene_changed_replan"
    assert result.plan is None


def test_snapshot_copies_are_independent_and_quaternion_sign_is_equivalent():
    robot, _, build, context = setup_plan()
    decoded = build.snapshot.placements()
    decoded["object_private"]["pos"][0] = 50
    robot.placements["object_private"]["quat"] = [-1.0, 0.0, 0.0, 0.0]
    assert bridge.store_agent_plan(context, robot, build)
    assert build.snapshot.placements()["object_private"]["pos"][0] == 0.0


def test_handles_never_recycle_across_sessions_or_contexts():
    from types import SimpleNamespace

    robot, _, build, context = setup_plan()
    task = SimpleNamespace(object="bowl", goal_recep="table", start_recep="counter", object_gt_body="object_private")
    old = bridge.stable_scene_task_refs(context, [task], robot.placements, session_key=("old",))[0].ref
    new = bridge.stable_scene_task_refs(context, [task], robot.placements, session_key=("new",))[0].ref
    assert old != new
    assert old not in context["_tamp_task_refs"]
    assert bridge.store_agent_plan({}, robot, build) != bridge.store_agent_plan({}, robot, build)


def test_place_exception_cannot_inherit_grasp_evidence(monkeypatch):
    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor
    from emet.controller.task.tamp import task_search
    from emet.controller.task.tamp.task_search import execute_task_plan
    messages = []
    monkeypatch.setattr(task_search.logger, 'warning', messages.append)

    class Executor:
        begin_operation = KinematicPickPlaceExecutor.begin_operation

        def grasp_only(self, *_args, **_kwargs):
            from types import SimpleNamespace

            self.last_ee_verification = {"error_m": 0.02, "accepted": True}
            self.last_grasp_verification = {"target_error_m": 0.003, "accepted": True}
            return SimpleNamespace(success=True)

        def place_only(self, *_args, **_kwargs):
            raise RuntimeError("base approach rejected")

    ex = Executor()
    robot, plan, _, _ = setup_plan()
    robot.move_base_to = lambda *a, **k: True
    plan.steps[1] = TaskPlanStep("grasp", {"grasp_index": 0, "object_query": "bowl"})
    plan.steps[2] = TaskPlanStep("place", {"receptacle_query": "table"})
    result = execute_task_plan(robot, plan, executor=ex, grasp_poses=[np.eye(4)], manip_mode="kinematic")
    assert result.completed_ops == ["approach", "grasp"]
    assert result.failed_op == "place"
    assert any("base approach rejected" in message for message in messages)
    assert "base approach rejected" not in result.message
    assert [m["stage"] for m in result.measurements] == ["grasp"]
    saved = result.measurements[0]["last_ee_verification"]["error_m"]
    ex.begin_operation("next-task")
    assert saved == 0.02 and result.measurements[0]["last_ee_verification"]["error_m"] == 0.02


def test_mcts_and_approach_grounding_share_supplied_placements(monkeypatch):
    from emet.controller.task.tamp import task_search
    from emet.controller.task.tamp.task_search import plan_pick_place_mcts

    seen = []

    def synthetic_grasps(body, placements, **kwargs):
        seen.append(placements[body]["pos"][0])
        return []

    monkeypatch.setattr(task_search, "resolve_scene_grasps", synthetic_grasps)

    robot = Robot()
    captured = json.loads(json.dumps(robot.placements))
    robot.placements["object_private"]["pos"][0] = 5.0
    plan = plan_pick_place_mcts(
        robot,
        candidates=[
            {
                "object_query": "bowl",
                "receptacle_query": "table",
                "object_gt_body": "object_private",
                "receptacle_gt_body": "receptacle_private",
            }
        ],
        placements=captured,
        seed=0,
    )
    assert plan.success
    assert abs(plan.steps[0].args["xyt"][0]) < 1e-6
    pose = getattr(plan.grasp_poses[0], "T_world", plan.grasp_poses[0])
    assert abs(pose[0, 3]) < 1e-6
    assert seen == [0.0]


def test_robocasa_pick_place_task_parses_and_resolves():
    from emet.controller.task.tamp.agent_bridge import robocasa_pick_place_task

    environment = {"kind": "robocasa", "task": "PickPlaceCounterToCabinet", "style": 1, "layout": 1}
    # Realistic RoboCasa placements: fixtures (cab_main/counter_main), a distractor
    # (distr_counter_main), and the task object (obj_main) all end in '_main'.
    placements = {
        "cab_main": {"cat": "cab", "pos": [1.0, 0.0, 0.8]},
        "counter_main": {"cat": "counter", "pos": [0.0, 1.0, 0.8]},
        "distr_counter_main": {"cat": "beer", "pos": [0.1, 0.1, 0.8]},
        "obj_main": {"cat": "kiwi", "pos": [0.0, 0.0, 0.8]},
    }
    task = robocasa_pick_place_task(environment, placements)
    assert task is not None
    assert task.object == "kiwi"            # task object category, NOT 'cab'
    assert task.goal_recep == "cab"         # task "Cabinet" -> fixture key "cab_main" (cat 'cab')
    assert task.start_recep == "counter"
    assert task.object_gt_body == "obj_main"


def test_robocasa_pick_place_task_requires_obj_main():
    from emet.controller.task.tamp.agent_bridge import robocasa_pick_place_task

    environment = {"kind": "robocasa", "task": "PickPlaceCounterToCabinet"}
    placements = {
        "cab_main": {"cat": "cab", "pos": [1.0, 0.0, 0.8]},
        "counter_main": {"cat": "counter", "pos": [0.0, 1.0, 0.8]},
        "distr_counter_main": {"cat": "beer", "pos": [0.1, 0.1, 0.8]},
    }
    # No obj_main (the task object) -> cannot ground the task.
    assert robocasa_pick_place_task(environment, placements) is None


def test_robocasa_pick_place_task_none_for_non_pickplace_or_non_robocasa():
    from emet.controller.task.tamp.agent_bridge import robocasa_pick_place_task

    assert robocasa_pick_place_task(None, {}) is None
    assert robocasa_pick_place_task({"kind": "molmospaces"}, {}) is None
    # Non-PickPlace RoboCasa task (open/close) is not a pick-place task.
    assert robocasa_pick_place_task({"kind": "robocasa", "task": "OpenCabinet"}, {}) is None
    # Missing goal receptacle -> None.
    assert robocasa_pick_place_task({"kind": "robocasa", "task": "PickPlaceCounterToCabinet"}, {"obj_main": {"cat": "mug"}}) is None


def test_failure_code_maps_ik_failed_to_unreachable():
    assert failure_code("grasp_failed:pregrasp_ik_failed") == "unreachable"
    assert failure_code("grasp_failed:pregrasp_tracking_failed") == "tracking_timeout"
    assert failure_code("grasp_failed:grasp_joint_bounds") == "joint_bounds"


def test_record_manip_attempt_writes_jsonl(tmp_path, monkeypatch):
    from emet.controller.manipulation.failure_record import record_manip_attempt

    monkeypatch.setenv("EMET_MANIP_FAILURE_DIR", str(tmp_path))
    path = record_manip_attempt(
        {"failure_mode": "tracking_failed", "target_xyz": [0.0, 0.0, 1.0], "trace": []},
        context={"op": "grasp", "object_body": "obj_main"},
    )
    assert path is not None
    lines = (tmp_path / "manip_failures.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["attempt"]["failure_mode"] == "tracking_failed"
    assert rec["context"]["op"] == "grasp"


def test_record_manip_attempt_noop_for_empty(tmp_path, monkeypatch):
    from emet.controller.manipulation.failure_record import record_manip_attempt

    monkeypatch.setenv("EMET_MANIP_FAILURE_DIR", str(tmp_path))
    assert record_manip_attempt({}, context={}) is None
    assert not (tmp_path / "manip_failures.jsonl").exists()


def test_refusal_from_teleport_collision_parses_deepest_margin():
    msg = (
        "Command 47 failed: navigation start failed: teleport_endpoint_collision: "
        "[{'bodies': ['dishwasher_x', 'base_link'], 'distance_m': -0.0736}, "
        "{'bodies': ['cabinet_y', 'base_link'], 'distance_m': -0.1162}]"
    )
    refusal = refusal_from_teleport_collision(msg)
    assert refusal is not None
    assert refusal["code"] == "teleport_collision"
    assert refusal["refused"] is True
    assert abs(refusal["margin_m"] - (-0.1162)) < 1e-9
    assert refusal["threshold_m"] is None
    assert refusal["relax_allowed"] is False
    # Body identities are never exposed in the semantic refusal.
    assert "dishwasher_x" not in json.dumps(refusal)
    assert "cabinet_y" not in json.dumps(refusal)


def test_refusal_from_teleport_collision_none_for_other_errors():
    assert refusal_from_teleport_collision("base approach rejected") is None
    assert refusal_from_teleport_collision("grasp_failed:pregrasp_tracking_failed") is None
    assert refusal_from_teleport_collision("") is None


def test_teleport_collision_surfaces_refusal_through_plan_and_data():
    from emet.controller.task.tamp.task_search import execute_task_plan

    class Executor:
        def begin_operation(self, *_):
            pass

        def grasp_only(self, *_args, **_kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(success=True)

        def place_only(self, *_args, **_kwargs):
            raise RuntimeError(
                "Command 99 failed: navigation start failed: teleport_endpoint_collision: "
                "[{'bodies': ['dishwasher_x', 'base_link'], 'distance_m': -0.0736}, "
                "{'bodies': ['cabinet_y', 'base_link'], 'distance_m': -0.1162}]"
            )

    ex = Executor()
    robot, plan, _, _ = setup_plan()
    robot.move_base_to = lambda *a, **k: True
    plan.steps[1] = TaskPlanStep("grasp", {"grasp_index": 0, "object_query": "bowl"})
    plan.steps[2] = TaskPlanStep("place", {"receptacle_query": "table"})
    result = execute_task_plan(robot, plan, executor=ex, grasp_poses=[np.eye(4)], manip_mode="kinematic")
    assert result.failed_op == "place"
    assert result.refusal is not None
    assert result.refusal["code"] == "teleport_collision"
    assert abs(result.refusal["margin_m"] - (-0.1162)) < 1e-9
    data = plan_data(result, "kinematic", "plan:1")
    assert data["refusal"] == result.refusal
    assert "dishwasher_x" not in json.dumps(data)
    assert "cabinet_y" not in json.dumps(data)

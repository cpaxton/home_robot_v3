"""Agent contract and guards: real tool registry, no robot actuation required."""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from emet.agent.tools import get_tools
from emet.controller.task.tamp import agent_bridge as bridge
from emet.controller.task.tamp.api import TAMP_TOOLS, json_tool, response
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
    ex._approach_xy = lambda *a: failure != "approach"
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


@pytest.mark.parametrize("provider", ["absent", "false", "nonboolean", "no_client"])
def test_unavailable_base_validation_rejects_build_and_consumes_plan(monkeypatch, provider):
    robot, _, build, context = setup_plan()
    ref = bridge.store_agent_plan(context, robot, build)
    query = Mock()
    robot.check_base_poses = query
    if provider == "absent":
        robot._state.pop("sim_base_pose_query")
    elif provider == "false":
        robot._state["sim_base_pose_query"] = False
    elif provider == "nonboolean":
        robot._state["sim_base_pose_query"] = 1
    else:
        robot.check_base_poses = None
    ground = Mock(side_effect=AssertionError("must reject before grounding"))
    execute = Mock(side_effect=AssertionError("must reject before motion"))
    monkeypatch.setattr(bridge, "plan_pick_place_mcts", ground)
    monkeypatch.setattr(bridge, "execute_agent_plan", execute)
    tools = {t.name: t for t in get_tools(context)}
    planned = decoded(tools["plan_pick_place"].func(object_name="bowl", receptacle_name="table"))
    executed = decoded(tools["execute_pick_place_plan"].func(plan_ref=ref))
    for result in (planned, executed):
        assert result["code"] == "base_pose_validation_unavailable"
        assert result["status"] == "error"
        assert result["recovery"] == "rediscover"
    assert not ground.called and not execute.called and not query.called
    assert bridge.execute_stored_agent_plan_result(robot, context, ref)["code"] == "unknown_plan"

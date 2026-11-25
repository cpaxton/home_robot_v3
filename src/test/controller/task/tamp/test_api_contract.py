"""Agent contract and guards: real tool registry, no robot actuation required."""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from emet.agent.tools import get_tools
from emet.controller.task.tamp import agent_bridge as bridge
from emet.controller.task.tamp.api import (
    TAMP_TOOLS,
    failure_code,
    json_tool,
    plan_data,
    refusal_from_teleport_collision,
    response,
)
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
        "plan_task": {"instruction": "move bowl to table"},
        "execute_task_plan": {"plan_ref": "missing"},
        "set_receptacle_state": {"task_ref": "missing", "state": "open"},
        "plan_pick_place": {},
        "execute_pick_place_plan": {"plan_ref": "missing"},
        "pick_place": {"object_name": "bowl", "receptacle_name": "table"},
    }
    assert set(args) == TAMP_TOOLS
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
    robot.check_base_poses = lambda poses: {"poses": np.asarray(poses).tolist(), "clear": [False] + [True] * (len(poses) - 1)}
    seen = []
    executor = SimpleNamespace(_ensure_model=lambda: True, _model=None, _data=None, ee_body="tool",
                               joint_names=["arm"], pregrasp_standoff_m=.18, lift_m=.22,
                               ik_tol_m=.024, ik_max_iters=123)
    monkeypatch.setattr(kinematic_pick_place, "KinematicPickPlaceExecutor", lambda *a, **k: executor)
    monkeypatch.setattr(task_search, "_sync_executor_base_to_xyt", lambda ex, pose: seen.append(pose.copy()))
    def rank(*args, **kwargs):
        assert kwargs["pregrasp_standoff_m"] == .18
        assert kwargs["lift_m"] == .22
        assert kwargs["ik_tol_m"] == .006
        assert kwargs["ik_max_iters"] == 123
        return [(0, 0.001 if len(seen) > 1 else 1.0, len(seen) > 1)]

    monkeypatch.setattr(task_search, "rank_grasps_by_ik", rank)
    result = bridge.build_agent_pick_place_plan(robot, "bowl", "table", seed=0)
    assert result.mode == "kinematic"
    assert result.plan.success, result.reason
    np.testing.assert_allclose(
        result.plan.steps[0].args["xyt"], task_search.approach_candidates_for_object_xy([0, 0], count=8)[2]
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


@pytest.mark.parametrize("failure", ["approach", "retract", "release", "missing_receptacle",
                                     "no_path", "invalidated", "preplace", "place", "runtime", "timeout",
                                     "move_exception", "attachment_changed", "state_search", "state_refresh", "success", "missing_body_pose"])
def test_place_does_not_hide_failed_submotions(monkeypatch, failure):
    from emet.controller.manipulation import kinematic_pick_place as module

    ex = object.__new__(module.KinematicPickPlaceExecutor)
    ex.robot = Robot()
    ex.ee_body = "tool"
    ex.place_z_offset_m = 0.0
    ex._ensure_model = lambda: True
    ex._placements = lambda: ex.robot.placements
    from types import SimpleNamespace

    from emet.motion.placement_geometry import HeldObject

    selected = SimpleNamespace(base_xyt=np.zeros(3), object_center=np.ones(3),
        ee_targets=(np.ones(3), np.ones(3)), ee_rotation=np.eye(3), segments=([np.zeros(3)], [np.zeros(3)]))
    checker = SimpleNamespace(payload=HeldObject(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        body_pose_ee=None if failure == "missing_body_pose" else np.eye(4)),
                              margin=0., released_bounds=None)
    calls = []

    def search(*args, **kwargs):
        calls.append(kwargs)
        if failure == "state_search":
            raise ValueError("placement_stale_observation")
        if failure in {"runtime", "timeout"}:
            raise (RuntimeError if failure == "runtime" else TimeoutError)("query failed")
        if failure == "attachment_changed" and len(calls) > 1:
            checker.payload = HeldObject(checker.payload.vertices_ee + .1)
        return SimpleNamespace(paths=[] if failure == "no_path" or (failure == "invalidated" and len(calls) > 1)
                               else [selected]), checker

    ex._search_placement = search
    ex._refresh_placement_checker = Mock(return_value=checker, side_effect=ValueError("placement_missing_joint_state")
                                        if failure == "state_refresh" else None)
    ex.robot.move_base_to = Mock(return_value=failure != "approach",
                                 side_effect=RuntimeError("transport failed") if failure == "move_exception" else None)
    ex._collision = None
    ex._sleep = lambda *a: None
    ex._verify_place_xy = lambda *a: (True, 0.001)
    motions = iter([(failure != "preplace", 0.001), (failure != "place", 0.001)])
    ex._execute_placement_segment = lambda *a: next(motions)
    ex._plan_and_execute_ee = lambda *a: (failure != "retract", 0.08)
    ex._last_motion_failure = "tracking_failed"
    ex._set_gripper = Mock(side_effect=RuntimeError() if failure == "release" else None)
    detach = Mock()
    monkeypatch.setattr(module, "robot_zmq_detach_body", detach)
    monkeypatch.setattr(module, "robot_zmq_set_body_pose", Mock())
    result = ex.place_only(
        "table",
        object_gt_body="object_private",
        receptacle_gt_body="missing" if failure == "missing_receptacle" else "receptacle_private",
    )
    assert result.success == (failure == "success")
    if failure == "success":
        assert calls == [{"approach_base": True}, {"approach_base": False}]
        assert ex._refresh_placement_checker.call_count == 2
        detach.assert_called_once()
    if failure == "missing_body_pose":
        detach.assert_not_called()
        assert result.message == "placement_geometry_unavailable"
    if failure in {"state_search", "state_refresh"}:
        assert result.message == ("placement_stale_observation" if failure == "state_search" else "placement_missing_joint_state")
    if failure in {"approach", "missing_receptacle", "no_path", "invalidated", "preplace", "place", "runtime",
                   "timeout", "move_exception", "attachment_changed", "state_search", "state_refresh"}:
        assert not detach.called
    if failure in {"no_path", "missing_receptacle", "runtime", "timeout"}:
        ex.robot.move_base_to.assert_not_called()


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
    assert [m["stage"] for m in result.measurements] == ["grasp", "place"]
    assert result.measurements[-1] == {"stage": "place", "release": {
        "detach_command": "not_attempted", "placement_verified": None, "held_state": "unknown"}}
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


@pytest.mark.parametrize("reason, expected", [
    ("no_plan_within_budget:last_grounding=no_reachable_grasp", "no_reachable_grasp"),
    ("no_plan_within_budget:last_grounding=private_body_id", "no_plan_within_budget"),
    ("no_plan_within_budget:candidates=3", "no_plan_within_budget"),
])
def test_mcts_grounding_failure_preserves_only_known_public_code(reason, expected):
    from emet.controller.task.tamp.api import failure_code
    assert failure_code(reason) == expected


@pytest.mark.parametrize("reason", [
    "grasp_verification_failed",
    "place_verification_failed",
    "private_body_id:grasp_verification_failed:/private/path",
])
def test_native_verification_failure_remains_actionable_without_private_details(reason):
    from emet.controller.task.tamp.api import failure_code
    from emet.controller.task.tamp.task_session import TaskExecutionResult

    code = failure_code(reason)
    assert code == "verification_failed"
    assert TaskExecutionResult(False, True, code).failure_code == code


def test_unsupported_adapter_does_not_ignore_placement_region():
    tools = {t.name: t for t in get_tools({})}
    result = decoded(tools['plan_pick_place'].func(object_name='bowl', receptacle_name='table',
                                                placement_region=[[0, 0], [1, 1]]))
    assert result['code'] == 'placement_region_unsupported'


@pytest.mark.parametrize('value', [
    {'substage': '/private/body', 'reason': 'payload_not_retained'},
    {'substage': 'post_lift_hold', 'reason': '/private/body'},
    {'substage': 'post_lift_hold', 'reason': ['payload_not_retained']},
    {'substage': 'post_lift_hold'},
    {'substage': 'post_lift_hold', 'reason': 'payload_not_retained', 'body': '/private/body'},
])
def test_verification_evidence_rejects_private_or_malformed_values(value):
    from emet.controller.task.tamp.api import VerificationFailure
    assert VerificationFailure.from_data(value) is None


def test_robocasa_pick_place_task_is_semantic():
    from emet.controller.task.tamp.agent_bridge import robocasa_pick_place_task

    task = robocasa_pick_place_task(
        {
            "kind": "robocasa",
            "task": "PickPlaceCounterToCabinet",
            "object_category": "sugar_cube",
        }
    )
    assert task is not None
    assert (task.object, task.start_recep, task.goal_recep) == ("sugar_cube", "counter", "cab")
    assert not hasattr(task, "object_gt_body")
    assert not hasattr(task, "goal_gt_body")


def test_robocasa_task_tokens_are_vocabulary_only():
    from emet.controller.task.tamp.agent_bridge import _robocasa_task_tokens_semantic

    assert _robocasa_task_tokens_semantic("PickPlaceCounterToStove") == ("Counter", "Stove")
    assert _robocasa_task_tokens_semantic("PickPlaceStoveToCounter") == ("Stove", "Counter")
    assert _robocasa_task_tokens_semantic("PickPlaceToasterToCounter") == ("Toaster", "Counter")
    assert _robocasa_task_tokens_semantic("PickPlaceCounterToToasterOven") == ("Counter", "ToasterOven")
    assert _robocasa_task_tokens_semantic("PickPlaceToasterOvenToCounter") == ("ToasterOven", "Counter")


def test_robocasa_pick_place_task_requires_semantic_object_category():
    from emet.controller.task.tamp.agent_bridge import robocasa_pick_place_task

    assert robocasa_pick_place_task({"kind": "robocasa", "task": "PickPlaceCounterToCabinet"}) is None
    assert robocasa_pick_place_task(
        {"kind": "robocasa", "task": "PickPlaceCounterToCabinet", "object_category": ""}
    ) is None


def test_robocasa_pick_place_task_none_for_non_robocasa_or_non_pickplace():
    from emet.controller.task.tamp.agent_bridge import robocasa_pick_place_task

    assert robocasa_pick_place_task(None) is None
    assert robocasa_pick_place_task({"kind": "molmospaces"}) is None
    assert robocasa_pick_place_task(
        {"kind": "robocasa", "task": "OpenCabinet", "object_category": "mug"}
    ) is None


def test_build_semantic_task_refs_has_no_bodies():
    from emet.controller.task.tamp.agent_bridge import (
        build_semantic_task_refs,
        robocasa_pick_place_task,
    )

    task = robocasa_pick_place_task(
        {"kind": "robocasa", "task": "PickPlaceCounterToCabinet", "object_category": "mug"}
    )
    refs = build_semantic_task_refs([task])
    assert len(refs) == 1
    ref = refs[0]
    assert ref.object_body is None and ref.receptacle_body is None
    assert ref.semantic_only is True
    assert ref.task_name == "PickPlaceCounterToCabinet"


def test_privileged_task_from_semantic_resolves_gt_for_teacher():
    from emet.controller.task.tamp.agent_bridge import (
        build_semantic_task_refs,
        privileged_task_from_semantic,
        robocasa_pick_place_task,
    )

    task = robocasa_pick_place_task(
        {"kind": "robocasa", "task": "PickPlaceCounterToCabinet", "object_category": "sugar_cube"}
    )
    ref = build_semantic_task_refs([task])[0]
    privileged = privileged_task_from_semantic(
        ref,
        {
            "obj_main": {"cat": "sugar_cube"},
            "counter_main": {"cat": "counter"},
            "cab_main": {"cat": "cab"},
        },
    )
    assert privileged is not None
    assert privileged.object_body == "obj_main"
    assert privileged.receptacle_body == "cab_main"
    assert privileged.privileged is True


class _RobocasaRobot:
    def __init__(self, task: str, object_category: str, placements: dict):
        self._task = task
        self._object_category = object_category
        self._placements = placements
        self._state = {"command_protocol": {"version": 2, "server_boot_id": "boot"}}

    def get_emet_session(self):
        return {
            "is_simulation": True,
            "environment": {
                "kind": "robocasa",
                "task": self._task,
                "object_category": self._object_category,
                "style": 1,
                "layout": 1,
            },
            "sim_object_placements": self._placements,
        }


def test_scene_tasks_robocasa_is_semantic_and_never_reads_gt(monkeypatch):
    import emet.memory.graph_eqa.sim_ground_truth_graph as gt
    from emet.agent.chat_tamp import build_tamp_handlers

    def _fail(*args, **kwargs):
        raise AssertionError("RoboCasa assessment must not read GT placements")

    monkeypatch.setattr(gt, "read_sim_object_placements", _fail)
    placements = {
        "obj_main": {"cat": "sugar_cube", "pos": [0.0, 0.0, 0.8]},
        "counter_main": {"cat": "counter", "pos": [0.0, 1.0, 0.8]},
        "cab_main": {"cat": "cab", "pos": [1.0, 0.0, 0.8]},
    }
    handlers = build_tamp_handlers(
        {"robot": _RobocasaRobot("PickPlaceCounterToCabinet", "sugar_cube", placements)}
    )
    data = decoded(handlers["scene_tasks"]())["data"]
    assert data["pickable_categories"] == ["sugar_cube"]
    assert data["receptacle_categories"] == ["cab"]
    assert len(data["tasks"]) == 1
    task = data["tasks"][0]
    assert task["object_name"] == "sugar_cube"
    assert task["receptacle_name"] == "cab"
    assert task["start_receptacle"] == "counter"
    assert task["task_ref"].startswith("task:")
    assert "obj_main" not in json.dumps(data)

    filtered = decoded(handlers["scene_tasks"](object_filter="banana"))["data"]
    assert filtered["tasks"] == []


def test_semantic_plan_returns_grounding_unavailable_without_teacher(monkeypatch):
    from emet.agent.chat_tamp import build_tamp_handlers

    monkeypatch.delenv("EMET_GT_TEACHER", raising=False)
    placements = {"obj_main": {"cat": "sugar_cube", "pos": [0.0, 0.0, 0.8]}}
    handlers = build_tamp_handlers(
        {"robot": _RobocasaRobot("PickPlaceCounterToCabinet", "sugar_cube", placements)}
    )
    task_ref = decoded(handlers["scene_tasks"]())["data"]["tasks"][0]["task_ref"]
    result = decoded(handlers["plan_pick_place"](task_ref=task_ref))
    assert result["status"] == "error"
    assert result["code"] == "grounding_unavailable"


def test_failure_code_maps_ik_failed_to_unreachable():
    assert failure_code("grasp_failed:pregrasp_ik_failed") == "unreachable"
    assert failure_code("grasp_failed:pregrasp_tracking_failed") == "tracking_timeout"
    assert failure_code("grasp_failed:grasp_joint_bounds") == "joint_bounds"


def test_scenario_from_attempt_is_semantic_and_marks_privileged():
    from emet.controller.manipulation.failure_record import scenario_from_attempt

    attempt = {
        "target_xyz": [0.0, 0.0, 1.0],
        "trace": [{"qpos": [0.0, 0.0]}],
        "joint_names": ["j0", "j1"],
        "failure_mode": "tracking_failed",
    }
    assessment = scenario_from_attempt(
        attempt, context={"object_query": "mug", "receptacle_query": "cab"}
    )
    assert assessment.goal["object"] == "mug"
    assert assessment.goal["receptacle"] == "cab"
    assert assessment.meta["privileged"] is False
    assert "object_body" not in assessment.meta

    teacher = scenario_from_attempt(
        attempt,
        context={
            "object_query": "mug",
            "receptacle_query": "cab",
            "privileged": True,
            "object_body": "obj_main",
            "receptacle_body": "cab_main",
        },
    )
    assert teacher.meta["privileged"] is True
    assert teacher.meta["object_body"] == "obj_main"
    assert teacher.meta["receptacle_body"] == "cab_main"


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



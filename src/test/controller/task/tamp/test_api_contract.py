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
    build = bridge.AgentPlanBuild(task, plan, "kinematic", True)
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
    with pytest.raises(ValueError, match="server_identity_missing"):
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

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Agent-facing TAMP grounding and stale-plan regressions (no sim)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from emet.controller.task.tamp import agent_bridge
from emet.controller.task.tamp.task_search import TaskPlan, TaskPlanStep, execute_task_plan


class _Robot:
    def __init__(self, placements: dict, *, capabilities: dict | None = None, is_simulation: bool = True):
        self.placements = placements
        self.capabilities = capabilities or {"sim_set_body_pose": True}
        self.is_simulation = is_simulation

    def get_emet_session(self):
        return {
            "is_simulation": self.is_simulation,
            "capabilities": self.capabilities,
            "sim_object_placements": self.placements,
        }


def _placements() -> dict:
    return {
        "bowl_hash_1_0_0": {"cat": "bowl", "pos": [0.1, 0.2, 0.8]},
        "table_main": {"cat": "table", "pos": [1.0, 0.2, 0.9]},
    }


def test_scene_task_refs_resolve_mesh_child_to_live_parent():
    task = SimpleNamespace(
        object="bowl",
        goal_recep="table",
        object_gt_body="bowl_hash_1_1_0",
        start_recep="cabinet",
    )

    refs = agent_bridge.build_scene_task_refs([task], _placements())

    assert len(refs) == 1
    assert refs[0].ref == "task:1"
    assert refs[0].object_query == "bowl"
    assert refs[0].object_body == "bowl_hash_1_0_0"
    assert refs[0].receptacle_body == "table_main"
    assert "bowl_hash" not in refs[0].ref


def test_scene_task_refs_are_stable_until_session_changes():
    task = SimpleNamespace(
        object="bowl",
        goal_recep="table",
        object_gt_body="bowl_hash_1_1_0",
        start_recep="cabinet",
    )
    context: dict = {}

    first = agent_bridge.stable_scene_task_refs(
        context,
        [task],
        _placements(),
        session_key=("molmospaces", "ithor", "0", "FloorPlan1.xml"),
    )
    repeated = agent_bridge.stable_scene_task_refs(
        context,
        [task],
        _placements(),
        session_key=("molmospaces", "ithor", "0", "FloorPlan1.xml"),
    )
    changed = agent_bridge.stable_scene_task_refs(
        context,
        [task],
        _placements(),
        session_key=("molmospaces", "ithor", "1", "FloorPlan2.xml"),
    )

    assert first[0].ref == repeated[0].ref == "task:1"
    assert changed[0].ref == "task:1"


def test_semantic_plan_build_keeps_grounding_inside_adapter(monkeypatch):
    robot = _Robot(_placements())
    seen: dict = {}
    planned = TaskPlan(
        steps=[TaskPlanStep("approach"), TaskPlanStep("grasp"), TaskPlanStep("place")],
        object_body="bowl_hash_1_0_0",
        receptacle_body="table_main",
        chosen_grasp_index=0,
        grasp_poses=[object()],
        success=True,
        message="planned",
    )

    def fake_plan(_robot, *, candidates, **_kwargs):
        seen["candidates"] = candidates
        return planned

    monkeypatch.setattr(agent_bridge, "plan_pick_place_mcts", fake_plan)
    build = agent_bridge.build_agent_pick_place_plan(robot, "bowl", "table")

    assert build.mode == "teleport"
    assert build.task is not None
    assert build.task.object_query == "bowl"
    assert build.task.receptacle_query == "table"
    assert seen["candidates"][0]["object_gt_body"] == "bowl_hash_1_0_0"
    assert seen["candidates"][0]["receptacle_gt_body"] == "table_main"


def test_explicit_stale_grounding_is_not_remapped_to_another_object():
    robot = _Robot(_placements())

    task, reason, live_sim = agent_bridge.resolve_agent_task(
        robot,
        "bowl",
        "table",
        object_body="old_bowl_1_0_0",
        receptacle_body="table_main",
    )

    assert task is None
    assert reason == "object_not_in_live_scene"
    assert live_sim is True


def test_hardware_session_does_not_enter_sim_tamp_path():
    robot = _Robot(_placements(), is_simulation=False)

    task, reason, live_sim = agent_bridge.resolve_agent_task(robot, "bowl", "table")

    assert task is None
    assert reason == "no_live_scene"
    assert live_sim is False


def test_stored_plan_revalidates_pose_and_is_one_shot():
    placements = _placements()
    robot = _Robot(placements, capabilities={"kinematic_manip": True})
    task = agent_bridge.AgentTaskRef(
        ref="task:1",
        object_query="bowl",
        receptacle_query="table",
        object_body="bowl_hash_1_0_0",
        receptacle_body="table_main",
    )
    plan = TaskPlan(
        steps=[TaskPlanStep("approach")],
        object_body=task.object_body,
        receptacle_body=task.receptacle_body,
        success=True,
        message="planned",
    )
    build = agent_bridge.AgentPlanBuild(task=task, plan=plan, mode="kinematic", live_sim=True)
    context: dict = {}
    plan_ref = agent_bridge.store_agent_plan(context, robot, build)

    placements["bowl_hash_1_0_0"]["pos"] = [0.5, 0.2, 0.8]
    with patch.object(agent_bridge, "execute_agent_plan") as execute:
        ok, message = agent_bridge.execute_stored_agent_plan(robot, context, plan_ref)

    assert ok is False
    assert message == "scene_changed_replan"
    assert execute.called is False
    assert plan_ref not in context["_tamp_plans"]


def test_task_plan_reports_partial_execution_failure():
    class _MotionRobot:
        def move_base_to(self, *_args, **_kwargs):
            return None

    class _FailingExecutor:
        def grasp_only(self, *_args, **_kwargs):
            return SimpleNamespace(success=False, message="attach_verify_failed")

    plan = TaskPlan(
        steps=[
            TaskPlanStep("approach", {"xyt": [0.0, 0.0, 0.0]}),
            TaskPlanStep(
                "grasp",
                {"object_query": "bowl", "object_gt_body": "bowl", "grasp_index": 0},
            ),
        ],
        object_body="bowl",
        receptacle_body="table",
        success=True,
    )

    result = execute_task_plan(
        _MotionRobot(),
        plan,
        executor=_FailingExecutor(),
        grasp_poses=[object()],
        manip_mode="kinematic",
    )

    assert result.success is False
    assert result.completed_ops == ["approach"]
    assert result.failed_op == "grasp"
    assert result.message == "grasp_failed:attach_verify_failed"

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Agent-facing bridge for semantic TAMP planning and guarded execution.

The planner needs concrete simulator bodies to ground IK and teleport actions, but
those body names are implementation details and often expose simulator ground truth.
This module keeps them behind session-scoped semantic task references used by CHAT
tools.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from emet.controller.task.tamp.task_search import TaskPlan, execute_task_plan, plan_pick_place_mcts
from emet.utils.logger import Logger

logger = Logger(__name__)


@dataclass(frozen=True)
class AgentTaskRef:
    """Semantic task handle with private body IDs for the execution adapter."""

    ref: str
    object_query: str
    receptacle_query: str
    object_body: str
    receptacle_body: str
    start_receptacle: str = ""


@dataclass(frozen=True)
class AgentPlanBuild:
    """Result of resolving and grounding an agent task."""

    task: AgentTaskRef | None
    plan: TaskPlan | None
    mode: str | None
    live_sim: bool
    reason: str = ""


def _robot_session_key(robot: Any) -> tuple[str, ...]:
    if robot is None or not hasattr(robot, "get_emet_session"):
        return ()
    session = robot.get_emet_session()
    if not isinstance(session, dict) or not session.get("is_simulation"):
        return ()
    environment = session.get("environment") or {}
    if not isinstance(environment, dict):
        environment = {}
    return (
        str(environment.get("kind") or ""),
        str(environment.get("scene") or ""),
        str(environment.get("index") if environment.get("index") is not None else ""),
        str(session.get("scene_source_basename") or ""),
    )


def _read_placements(robot: Any) -> dict[str, dict[str, Any]]:
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    if robot is None or not hasattr(robot, "get_emet_session"):
        return {}
    session = robot.get_emet_session()
    if not isinstance(session, dict) or not session.get("is_simulation"):
        return {}
    return read_sim_object_placements(session) or {}


def _body_aliases(body: str | None) -> tuple[str, ...]:
    raw = str(body or "").strip()
    if not raw:
        return ()
    aliases = [raw]
    if raw.endswith("_1_1_0"):
        aliases.append(raw[: -len("_1_1_0")] + "_1_0_0")
    elif raw.endswith("_1_0_0"):
        aliases.append(raw[: -len("_1_0_0")] + "_1_1_0")
    return tuple(dict.fromkeys(aliases))


def resolve_body_alias(body: str | None, placements: dict[str, dict[str, Any]]) -> str | None:
    """Resolve a metadata mesh/parent name to the live placement key."""

    return next((candidate for candidate in _body_aliases(body) if candidate in placements), None)


def build_scene_task_refs(
    tasks: list[Any],
    placements: dict[str, dict[str, Any]] | None,
    *,
    max_tasks: int = 24,
) -> list[AgentTaskRef]:
    """Build opaque semantic handles for metadata tasks present in the live scene."""

    from emet.eval.ovmm_find_phase import bodies_matching_category

    live = placements or {}
    refs: list[AgentTaskRef] = []
    for task in tasks:
        object_body = resolve_body_alias(getattr(task, "object_gt_body", None), live)
        if object_body is None:
            object_matches = bodies_matching_category(live, str(getattr(task, "object", "")))
            if len(object_matches) == 1:
                object_body = object_matches[0]
        if object_body is None:
            continue
        receptacle_bodies = bodies_matching_category(live, str(getattr(task, "goal_recep", "")))
        for receptacle_body in receptacle_bodies:
            if len(refs) >= int(max_tasks):
                return refs
            refs.append(
                AgentTaskRef(
                    ref=f"task:{len(refs) + 1}",
                    object_query=str(getattr(task, "object", "")),
                    receptacle_query=str(getattr(task, "goal_recep", "")),
                    object_body=object_body,
                    receptacle_body=receptacle_body,
                    start_receptacle=str(getattr(task, "start_recep", "")),
                )
            )
    return refs


def stable_scene_task_refs(
    context: dict[str, Any],
    tasks: list[Any],
    placements: dict[str, dict[str, Any]] | None,
    *,
    session_key: tuple[str, ...] = (),
) -> list[AgentTaskRef]:
    """Register semantic task handles for one live-scene session.

    Handles are stable across repeated or filtered ``scene_tasks`` calls, but the
    registry is discarded when the session identity changes.
    """

    if context.get("_tamp_scene_key") != session_key:
        context["_tamp_scene_key"] = session_key
        context["_tamp_task_refs"] = {}
        context["_tamp_task_ref_keys"] = {}
        context["_tamp_task_counter"] = 0
    by_key = context.setdefault("_tamp_task_ref_keys", {})
    refs_by_name = context.setdefault("_tamp_task_refs", {})
    out: list[AgentTaskRef] = []
    for ref in build_scene_task_refs(tasks, placements):
        key = (
            ref.object_query,
            ref.receptacle_query,
            ref.start_receptacle,
            ref.object_body,
            ref.receptacle_body,
        )
        handle = by_key.get(key)
        if not isinstance(handle, str) or not handle:
            counter = int(context.get("_tamp_task_counter", 0)) + 1
            context["_tamp_task_counter"] = counter
            handle = f"task:{counter}"
            by_key[key] = handle
        registered = replace(ref, ref=handle)
        refs_by_name[handle] = registered
        out.append(registered)
    return out


def resolve_agent_task(
    robot: Any,
    object_query: str,
    receptacle_query: str,
    *,
    object_body: str | None = None,
    receptacle_body: str | None = None,
) -> tuple[AgentTaskRef | None, str, bool]:
    """Resolve semantic queries against the live sim without exposing body IDs."""

    from emet.eval.ovmm_find_phase import bodies_matching_category

    placements = _read_placements(robot)
    if not placements:
        return None, "no_live_scene", False

    if object_body:
        resolved_object = resolve_body_alias(object_body, placements)
        if resolved_object is None:
            return None, "object_not_in_live_scene", True
    else:
        objects = bodies_matching_category(placements, object_query)
        if not objects:
            return None, "object_not_found", True
        if len(objects) > 1:
            return None, "ambiguous_object_use_scene_tasks", True
        resolved_object = objects[0]

    if receptacle_body:
        resolved_receptacle = resolve_body_alias(receptacle_body, placements)
        if resolved_receptacle is None:
            return None, "receptacle_not_in_live_scene", True
    else:
        receptacles = bodies_matching_category(placements, receptacle_query)
        if not receptacles:
            return None, "receptacle_not_found", True
        if len(receptacles) > 1:
            return None, "ambiguous_receptacle_use_scene_tasks", True
        resolved_receptacle = receptacles[0]

    return (
        AgentTaskRef(
            ref="query",
            object_query=str(object_query).strip(),
            receptacle_query=str(receptacle_query).strip(),
            object_body=resolved_object,
            receptacle_body=resolved_receptacle,
        ),
        "",
        True,
    )


def build_agent_pick_place_plan(
    robot: Any,
    object_query: str,
    receptacle_query: str,
    *,
    object_body: str | None = None,
    receptacle_body: str | None = None,
    manip_mode: str = "auto",
    seed: int | None = None,
) -> AgentPlanBuild:
    """Resolve a semantic task and ground it with the available sim manipulator."""

    task, reason, live_sim = resolve_agent_task(
        robot,
        object_query,
        receptacle_query,
        object_body=object_body,
        receptacle_body=receptacle_body,
    )
    if task is None:
        return AgentPlanBuild(task=None, plan=None, mode=None, live_sim=live_sim, reason=reason)

    from emet.motion.arm_manip_profile import resolve_manip_mode_for_robot

    try:
        mode = resolve_manip_mode_for_robot(robot, manip_mode=manip_mode)
    except (RuntimeError, ValueError):
        return AgentPlanBuild(task=task, plan=None, mode=None, live_sim=True, reason="manipulation_unavailable")

    grounding_executor = None
    if mode == "kinematic":
        from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

        grounding_executor = KinematicPickPlaceExecutor(robot, manip_collision="none", traj_dt=0.05)

    try:
        plan = plan_pick_place_mcts(
            robot,
            candidates=[
                {
                    "object_query": task.object_query,
                    "receptacle_query": task.receptacle_query,
                    "object_gt_body": task.object_body,
                    "receptacle_gt_body": task.receptacle_body,
                }
            ],
            executor=grounding_executor,
            mcts_iterations=64,
            seed=seed,
        )
    except Exception:
        return AgentPlanBuild(task=task, plan=None, mode=mode, live_sim=True, reason="planner_error")
    return AgentPlanBuild(task=task, plan=plan, mode=mode, live_sim=True, reason=plan.message)


def store_agent_plan(context: dict[str, Any], robot: Any, build: AgentPlanBuild) -> str:
    """Store a grounded plan under an opaque, one-shot session handle."""

    if build.task is None or build.plan is None or build.mode is None:
        raise ValueError("cannot store an incomplete TAMP plan")
    placements = _read_placements(robot)
    object_pos = np.asarray(placements[build.task.object_body]["pos"], dtype=np.float64).reshape(3).copy()
    receptacle_pos = np.asarray(placements[build.task.receptacle_body]["pos"], dtype=np.float64).reshape(3).copy()
    plans = context.setdefault("_tamp_plans", {})
    counter = int(context.get("_tamp_plan_counter", 0)) + 1
    context["_tamp_plan_counter"] = counter
    plan_ref = f"plan:{counter}"
    plans[plan_ref] = {
        "task": build.task,
        "plan": build.plan,
        "mode": build.mode,
        "session_key": _robot_session_key(robot),
        "object_pos": object_pos,
        "receptacle_pos": receptacle_pos,
    }
    return plan_ref


def _validate_stored_plan(robot: Any, record: dict[str, Any]) -> str | None:
    session = robot.get_emet_session() if robot is not None and hasattr(robot, "get_emet_session") else None
    if not isinstance(session, dict) or not session.get("is_simulation"):
        return "not_simulation"
    mode = str(record.get("mode") or "")
    caps = session.get("capabilities") or {}
    if mode == "kinematic" and not caps.get("kinematic_manip"):
        return "kinematic_capability_missing"
    if mode == "teleport" and not caps.get("sim_set_body_pose"):
        return "teleport_capability_missing"
    stored_session_key = record.get("session_key")
    if stored_session_key and tuple(stored_session_key) != _robot_session_key(robot):
        return "scene_changed_replan"
    task = record.get("task")
    placements = _read_placements(robot)
    if not isinstance(task, AgentTaskRef):
        return "invalid_plan"
    if task.object_body not in placements or task.receptacle_body not in placements:
        return "scene_changed_replan"
    for body, key in (
        (task.object_body, "object_pos"),
        (task.receptacle_body, "receptacle_pos"),
    ):
        if key not in record or placements[body].get("pos") is None:
            return "invalid_plan"
        try:
            expected = np.asarray(record[key], dtype=np.float64).reshape(3)
            current = np.asarray(placements[body]["pos"], dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            return "invalid_plan"
        if float(np.linalg.norm(current - expected)) > 0.20:
            return "scene_changed_replan"
    return None


def execute_agent_plan(robot: Any, plan: TaskPlan, mode: str) -> TaskPlan:
    """Execute a grounded plan using the selected capability path."""

    if mode == "kinematic":
        from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

        executor = KinematicPickPlaceExecutor(robot, manip_collision="none", traj_dt=0.05)
    elif mode == "teleport":
        executor = None
    else:
        plan.success = False
        plan.message = "unsupported_manipulation_mode"
        return plan
    return execute_task_plan(
        robot,
        plan,
        executor=executor,
        grasp_poses=plan.grasp_poses,
        manip_mode=mode,
    )


def execute_stored_agent_plan(robot: Any, context: dict[str, Any], plan_ref: str) -> tuple[bool, str]:
    """Validate and execute one stored plan, returning agent-safe text."""

    plans = context.get("_tamp_plans") or {}
    record = plans.pop(str(plan_ref), None)
    if not isinstance(record, dict):
        return False, "unknown_plan"
    reason = _validate_stored_plan(robot, record)
    if reason:
        return False, reason
    plan = record["plan"]
    try:
        result = execute_agent_plan(robot, plan, str(record["mode"]))
    except Exception as exc:
        logger.warning(f"TAMP plan execution failed: {type(exc).__name__}")
        return False, "execution_error"
    return bool(result.success), str(result.message or ("ok" if result.success else "failed"))

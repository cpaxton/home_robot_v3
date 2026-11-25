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

import copy
import json
import re
import uuid
from dataclasses import dataclass, replace
from types import SimpleNamespace
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
class PlanningSnapshot:
    """Immutable grounding inputs; decoding always returns a private copy."""

    server_boot_id: str
    session_key: tuple[str, ...]
    capabilities_json: str
    placements_json: str
    articulation_revision: int = 0

    def placements(self) -> dict:
        return json.loads(self.placements_json)


@dataclass(frozen=True)
class AgentPlanBuild:
    """Result of resolving and grounding an agent task."""

    task: AgentTaskRef | None
    plan: TaskPlan | None
    mode: str | None
    live_sim: bool
    reason: str = ""
    snapshot: PlanningSnapshot | None = None


def robot_session_key(robot: Any) -> tuple[str, ...]:
    if robot is None or not hasattr(robot, "get_emet_session"):
        return ()
    session = robot.get_emet_session()
    if not isinstance(session, dict) or not session.get("is_simulation"):
        return ()
    return _session_key(session, _server_boot(robot))


def _session_key(session: dict, boot: str | None) -> tuple[str, ...]:
    environment = session.get("environment") or {}
    if not isinstance(environment, dict):
        environment = {}
    return (
        str(environment.get("kind") or ""),
        str(environment.get("scene") or ""),
        str(environment.get("index") if environment.get("index") is not None else ""),
        str(session.get("scene_source_basename") or ""),
        str(boot or ""),
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


# RoboCasa task-name → fixture-key aliases (task names use long words, the wizard
# fixture scan uses short group keys like ``cab`` / ``counter`` / ``sink``).
_ROBOCASA_FIXTURE_ALIASES: dict[str, str] = {
    "cabinet": "cab",
    "countertop": "counter",
    "toaster": "toaster",
    "toasteroven": "toasteroven",
    "standmixer": "standmixer",
    "fridgedrawer": "fridgedrawer",
    "fridgeshelf": "fridgeshelf",
    "electrickettle": "electrickettle",
    "coffeemachine": "coffeemachine",
}
_ROBOCASA_KNOWN_FIXTURES = frozenset(_ROBOCASA_FIXTURE_ALIASES) | frozenset(_ROBOCASA_FIXTURE_ALIASES.values())


def _robocasa_fixture_token(name: str) -> str:
    token = str(name).strip().lower().replace(" ", "")
    return _ROBOCASA_FIXTURE_ALIASES.get(token, token)


def _match_placement_by_category(
    placements: dict[str, dict[str, Any]],
    token: str,
    *,
    reference_body: str | None = None,
) -> str | None:
    """Resolve a task-name receptacle token to one placement body key (internal).

    Exact matches (body key or GT ``cat``) win. Prefix matches cover the wizard
    short keys (``cab`` ↔ ``cab_main``). When several fixtures match (layouts
    have multiple cabinets), prefer the one nearest ``reference_body`` — the task
    object sits on the start counter beside its goal cabinet. Ties fall back to
    shortest key then lexical order so dict order cannot change the choice.
    """
    want = _robocasa_fixture_token(token)
    if not want:
        return None

    def rank(key: str) -> tuple[float, int, str]:
        if reference_body:
            ref = placements.get(reference_body) or {}
            info = placements.get(key) or {}
            try:
                distance = float(
                    np.linalg.norm(
                        np.asarray(info["pos"], dtype=np.float64)[:2]
                        - np.asarray(ref["pos"], dtype=np.float64)[:2]
                    )
                )
            except (KeyError, TypeError, ValueError):
                distance = float("inf")
            return (distance, len(key), key)
        return (0.0, len(key), key)

    exact: list[str] = []
    prefix: list[str] = []
    for key, info in placements.items():
        k = str(key).lower()
        cat = str(info.get("cat") or "").lower()
        if k == want or cat == want:
            exact.append(str(key))
        elif k and len(k) >= 3 and (k.startswith(want) or want.startswith(k)):
            prefix.append(str(key))
    for group in (exact, prefix):
        if group:
            return sorted(group, key=rank)[0]
    return None


def _robocasa_task_tokens(task_name: str, placements: dict[str, dict[str, Any]]) -> tuple[str, str] | None:
    """Split ``PickPlace<Start>To<Goal>`` on the CamelCase delimiter.

    RoboCasa task names are CamelCase, so the start/goal delimiter is a capital
    ``To``. Fixture names can contain it too (``Toaster``, ``ToasterOven``), so
    prefer the split where both sides resolve against the live placements; a
    case-insensitive greedy match would read ``PickPlaceCounterToStove`` as
    ``countertos`` → ``ve``.
    """
    match = re.match(r"PickPlace(.+)", str(task_name or "").strip())
    if not match:
        return None
    rest = match.group(1)
    splits = [index for index in range(len(rest) - 1) if rest.startswith("To", index)]
    if not splits:
        return None

    def known(token: str) -> bool:
        return (
            _robocasa_fixture_token(token) in _ROBOCASA_KNOWN_FIXTURES
            or _match_placement_by_category(placements, token) is not None
        )

    for index in splits:
        start, goal = rest[:index], rest[index + 2 :]
        if start and goal and known(start) and known(goal):
            return start, goal
    if len(splits) == 1:
        start, goal = rest[: splits[0]], rest[splits[0] + 2 :]
        if start and goal:
            return start, goal
    return None


def robocasa_pick_place_task(environment: Any, placements: dict[str, dict[str, Any]]) -> Any | None:
    """Build a semantic pick-place task from the active RoboCasa task name.

    RoboCasa has no iTHOR-style task catalog, but the active task is in the session
    (``environment.task``, e.g. ``PickPlaceCounterToCabinet``). Returns a task object
    with semantic ``object`` / ``goal_recep`` / ``start_recep`` plus the internal
    ``object_gt_body``; body IDs stay private and are never surfaced to the agent.
    """
    if not isinstance(environment, dict):
        return None
    if str(environment.get("kind") or "").lower() != "robocasa":
        return None
    live = placements or {}
    tokens = _robocasa_task_tokens(str(environment.get("task") or ""), live)
    if tokens is None:
        return None
    start_token, goal_token = tokens
    # The task object is the wizard object named 'obj' (body 'obj_main'); fixtures
    # ('cab_main', 'counter_main', …) and distractors ('distr_*') also end in '_main'.
    object_body = "obj_main" if "obj_main" in live else None
    if object_body is None:
        return None
    object_cat = str((live.get(object_body) or {}).get("cat") or "obj")
    goal_body = _match_placement_by_category(live, goal_token, reference_body=object_body)
    if goal_body is None:
        return None
    start_body = _match_placement_by_category(live, start_token, reference_body=object_body)
    return SimpleNamespace(
        object=object_cat,
        goal_recep=str((live.get(goal_body) or {}).get("cat") or _robocasa_fixture_token(goal_token)),
        start_recep=(
            str((live.get(start_body) or {}).get("cat") or _robocasa_fixture_token(start_token))
            if start_body
            else _robocasa_fixture_token(start_token)
        ),
        object_gt_body=object_body,
        goal_gt_body=goal_body,
        start_gt_body=start_body,
    )


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
        goal_hint = getattr(task, "goal_gt_body", None)
        receptacle_bodies: list[str] = []
        if goal_hint:
            resolved_goal = resolve_body_alias(str(goal_hint), live)
            if resolved_goal is not None:
                receptacle_bodies = [resolved_goal]
        if not receptacle_bodies:
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
        context["_tamp_task_namespace"] = uuid.uuid4().hex
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
            namespace = context.setdefault("_tamp_task_namespace", uuid.uuid4().hex)
            handle = f"task:{namespace}:{counter}"
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
    placements: dict | None = None,
) -> tuple[AgentTaskRef | None, str, bool]:
    """Resolve semantic queries against the live sim without exposing body IDs."""

    from emet.eval.ovmm_find_phase import bodies_matching_category

    placements = _read_placements(robot) if placements is None else placements
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


def _base_pose_validation_available(robot: Any) -> bool:
    """Require both the advertised server endpoint and its client implementation."""
    state = getattr(robot, "_state", None)
    return (
        isinstance(state, dict)
        and state.get("sim_base_pose_query") is True
        and callable(getattr(robot, "check_base_poses", None))
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

    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    session = copy.deepcopy(robot.get_emet_session()) if robot is not None and hasattr(robot, 'get_emet_session') else None
    if not isinstance(session, dict) or not session.get('is_simulation'):
        return AgentPlanBuild(None, None, None, False, 'no_live_scene')
    # Capture a fresh session between boot checks, not after grounding.
    boot = _server_boot(robot)
    session = copy.deepcopy(robot.get_emet_session())
    placements = read_sim_object_placements(session) or {}
    if not placements:
        return AgentPlanBuild(None, None, None, False, 'no_live_scene')
    if not boot:
        return AgentPlanBuild(None, None, None, True, 'server_identity_missing')
    if not _base_pose_validation_available(robot):
        return AgentPlanBuild(None, None, None, True, 'base_pose_validation_unavailable')
    snapshot = PlanningSnapshot(boot, _session_key(session, boot),
                                json.dumps(session.get('capabilities') or {}, sort_keys=True),
                                json.dumps(placements, default=lambda v: v.tolist()),
                                int(session.get('articulation_revision', 0)))
    if snapshot.session_key != robot_session_key(robot):
        return AgentPlanBuild(None, None, None, True, 'scene_changed_replan')
    task, reason, live_sim = resolve_agent_task(
        robot,
        object_query,
        receptacle_query,
        object_body=object_body,
        receptacle_body=receptacle_body,
        placements=snapshot.placements(),
    )
    if task is None:
        return AgentPlanBuild(task=None, plan=None, mode=None, live_sim=live_sim, reason=reason)

    from emet.motion.arm_manip_profile import resolve_manip_mode_for_robot

    try:
        mode = resolve_manip_mode_for_robot(robot, manip_mode=manip_mode)
    except (RuntimeError, ValueError):
        return AgentPlanBuild(task=task, plan=None, mode=None, live_sim=True, reason="manipulation_unavailable")

    reason = _validate_snapshot(robot, snapshot, task)
    if reason:
        return AgentPlanBuild(task, None, mode, True, reason)
    from emet.simulation.articulation import access_precondition
    access_reason = access_precondition(session, task.receptacle_body)
    if access_reason:
        return AgentPlanBuild(task, None, mode, True, access_reason)
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
            placements=snapshot.placements(),
        )
    except Exception:
        return AgentPlanBuild(task=task, plan=None, mode=mode, live_sim=True, reason="planner_error")
    reason = _validate_snapshot(robot, snapshot, task)
    if reason:
        return AgentPlanBuild(task, None, mode, True, reason)
    return AgentPlanBuild(task=task, plan=plan, mode=mode, live_sim=True, reason=plan.message, snapshot=snapshot)


def _server_boot(robot: Any) -> str | None:
    from emet.core.command_client import _check_peer
    try:
        return _check_peer(robot)
    except (RuntimeError, TypeError, ValueError):
        return None


def _pose(info: dict) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(info['pos'], dtype=float).reshape(3)
    quat = np.asarray(info['quat'], dtype=float).reshape(4)
    if not np.isfinite(pos).all() or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-12:
        raise ValueError('invalid pose')
    return pos.copy(), quat / np.linalg.norm(quat)


def _validate_snapshot(robot: Any, snapshot: PlanningSnapshot, task: AgentTaskRef) -> str | None:
    if snapshot.session_key != robot_session_key(robot):
        return 'scene_changed_replan'
    session = robot.get_emet_session()
    if json.dumps(session.get('capabilities') or {}, sort_keys=True) != snapshot.capabilities_json:
        return 'scene_changed_replan'
    from emet.simulation.articulation import access_precondition
    if int(session.get("articulation_revision", 0)) != snapshot.articulation_revision:
        return 'scene_changed_replan'
    access_reason = access_precondition(session, task.receptacle_body)
    if access_reason:
        return access_reason
    expected = snapshot.placements()
    current = _read_placements(robot)
    for body in (task.object_body, task.receptacle_body):
        try:
            p0, q0 = _pose(expected[body])
            p1, q1 = _pose(current[body])
        except (KeyError, TypeError, ValueError):
            return 'invalid_plan'
        angle = 2 * np.arccos(np.clip(abs(float(np.dot(q0, q1))), 0, 1))
        if np.linalg.norm(p1 - p0) > .01 or angle > np.deg2rad(5):
            return 'scene_changed_replan'
    if snapshot.session_key != robot_session_key(robot):
        return 'scene_changed_replan'
    return None


def store_agent_plan(context: dict[str, Any], robot: Any, build: AgentPlanBuild) -> str:
    """Store a grounded plan under an opaque, one-shot session handle."""

    if build.task is None or build.plan is None or not build.plan.success or build.mode is None:
        raise ValueError("cannot store an incomplete TAMP plan")
    if build.snapshot is None:
        raise ValueError('invalid_plan')
    reason = _validate_snapshot(robot, build.snapshot, build.task)
    if reason:
        raise ValueError(reason)
    placements = build.snapshot.placements()
    boot = build.snapshot.server_boot_id
    object_pos, object_quat = _pose(placements[build.task.object_body])
    receptacle_pos, receptacle_quat = _pose(placements[build.task.receptacle_body])
    plans = context.setdefault("_tamp_plans", {})
    counter = int(context.get("_tamp_plan_counter", 0)) + 1
    context["_tamp_plan_counter"] = counter
    namespace = context.setdefault("_tamp_plan_namespace", uuid.uuid4().hex)
    plan_ref = f"plan:{namespace}:{counter}"
    plans[plan_ref] = {
        "task": build.task,
        "plan": build.plan,
        "mode": build.mode,
        "session_key": build.snapshot.session_key,
        "snapshot": build.snapshot,
        "server_boot_id": boot,
        "object_quat": object_quat,
        "receptacle_quat": receptacle_quat,
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
    if _server_boot(robot) is None:
        return "server_identity_missing"
    if record.get('server_boot_id') != _server_boot(robot):
        return "scene_changed_replan"
    stored_session_key = record.get("session_key")
    if stored_session_key and tuple(stored_session_key) != robot_session_key(robot):
        return "scene_changed_replan"
    task = record.get("task")
    placements = _read_placements(robot)
    if not isinstance(task, AgentTaskRef):
        return "invalid_plan"
    if task.object_body not in placements or task.receptacle_body not in placements:
        return "scene_changed_replan"
    snapshot = record.get('snapshot')
    if not isinstance(snapshot, PlanningSnapshot):
        return 'invalid_plan'
    reason = _validate_snapshot(robot, snapshot, task)
    if reason:
        return reason
    if not _base_pose_validation_available(robot):
        return 'base_pose_validation_unavailable'
    try:
        poses = [step.args['xyt'] for step in record['plan'].steps if step.op == 'approach']
        if not poses:
            return "invalid_plan"
        evidence = robot.check_base_poses(poses)
        if evidence.get('poses') != [np.asarray(p).tolist() for p in poses]:
            return "approach_validation_failed"
        clear = evidence.get('clear')
        if not isinstance(clear, list) or len(clear) != len(poses) or any(type(c) is not bool for c in clear):
            return "approach_validation_failed"
        if not all(clear):
            return "approach_changed_replan"
    except Exception:
        return "approach_validation_failed"
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


def execute_stored_agent_plan_result(robot: Any, context: dict[str, Any], plan_ref: str) -> dict:
    """Consume once, validate, and return the shared semantic result envelope."""
    from emet.controller.task.tamp.api import failure_code, plan_data, response

    plans = context.get("_tamp_plans") or {}
    record = plans.pop(str(plan_ref), None)
    if not isinstance(record, dict):
        return response('execute_pick_place_plan', code='unknown_plan')
    plan = record['plan']
    mode = str(record['mode'])
    try:
        reason = _validate_stored_plan(robot, record)
        if reason:
            return response('execute_pick_place_plan', code=reason, data=plan_data(plan, mode, plan_ref))
        result = execute_agent_plan(robot, plan, mode)
        code = 'ok' if result.success else failure_code(result.message)
    except Exception as exc:
        logger.warning(f"TAMP plan execution failed: {type(exc).__name__}")
        result, code = plan, 'execution_error'
    return response('execute_pick_place_plan', code=code,
                    status='ok' if code == 'ok' else 'partial' if result.completed_ops else 'error',
                    data=plan_data(result, mode, plan_ref))


def execute_stored_agent_plan(robot: Any, context: dict[str, Any], plan_ref: str) -> tuple[bool, str]:
    """Compatibility for Python callers; agent tools use the structured result."""
    result = execute_stored_agent_plan_result(robot, context, plan_ref)
    return result['status'] == 'ok', result['code']

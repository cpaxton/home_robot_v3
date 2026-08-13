# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Deterministic beam-style search for sim pick-place (no LLM / VLM).

Symbolic operators reuse existing primitives:
  approach(body) → grasp(body, grasp_i) → place(receptacle)

Grasp branches are ranked by offline position-IK feasibility before execution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from emet.utils.logger import Logger

logger = Logger(__name__)


@dataclass(frozen=True)
class TaskPlanStep:
    op: str
    args: dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass
class TaskPlan:
    steps: list[TaskPlanStep]
    object_body: str
    receptacle_body: str | None
    chosen_grasp_index: int | None = None
    grasp_scores: list[tuple[int, float, bool]] = field(default_factory=list)
    expanded_nodes: list[str] = field(default_factory=list)
    success: bool = False
    message: str = ""
    # Grasp candidates used to ground this plan (executor needs the exact list,
    # e.g. synthesized teleport grasps, to map chosen_grasp_index -> pose).
    grasp_poses: list[Any] = field(default_factory=list)


def approach_pose_for_object_xy(obj_xy: np.ndarray, *, standoff: float = 0.55) -> np.ndarray:
    """Face −Y (table / iTHOR convention used in default_table smoke).

    Default standoff 0.55 m clears the Galaxea base footprint (0.35 embedded the chassis).
    """
    xy = np.asarray(obj_xy, dtype=np.float64).reshape(-1)[:2]
    return np.array([float(xy[0]), float(xy[1]) + float(standoff), -np.pi / 2], dtype=np.float64)


def rank_grasps_by_ik(
    model: Any,
    data: Any,
    *,
    ee_body: str,
    joint_names: Sequence[str],
    grasp_poses: Sequence[Any],
    pregrasp_standoff_m: float = 0.12,
    ik_tol_m: float = 0.05,
    ik_max_iters: int = 80,
    top_k: int = 8,
) -> list[tuple[int, float, bool]]:
    """Return ``(index, pos_error_m, reachable)`` sorted best-first for each grasp candidate."""
    from emet.controller.manipulation.kinematic_pick_place import _targets_from_grasp_T
    from emet.motion.mujoco_arm_ik import solve_position_ik_multiseed

    scored: list[tuple[int, float, bool]] = []
    for i, g in enumerate(grasp_poses[: max(1, int(top_k) * 2)]):
        T = getattr(g, "T_world", g)
        pregrasp, grasp, _lift = _targets_from_grasp_T(T, pregrasp_standoff_m=pregrasp_standoff_m, lift_m=0.12)
        # Probe grasp XYZ (harder than pregrasp).
        res = solve_position_ik_multiseed(
            model,
            data,
            ee_body=ee_body,
            joint_names=list(joint_names),
            target_pos=grasp,
            seeds=None,
            try_midrange=True,
            tol_m=float(ik_tol_m),
            max_iters=int(ik_max_iters),
        )
        scored.append((i, float(res.pos_error_m), bool(res.success)))
    scored.sort(key=lambda t: (not t[2], t[1]))
    return scored[: int(top_k)]


def _sync_executor_base_to_xyt(executor: Any, xyt: np.ndarray) -> None:
    """Write approach base XYT into the executor's offline MJCF freejoint for IK ranking."""
    import mujoco

    model = getattr(executor, "_model", None)
    data = getattr(executor, "_data", None)
    profile = getattr(executor, "profile", None)
    if model is None or data is None or profile is None:
        return
    name = getattr(profile, "base_freejoint_name", None)
    if not name:
        return
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
    if jid < 0:
        return
    qadr = int(model.jnt_qposadr[jid])
    x, y, th = float(xyt[0]), float(xyt[1]), float(xyt[2])
    z = float(data.qpos[qadr + 2])
    half = 0.5 * th
    data.qpos[qadr : qadr + 7] = [x, y, z, float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]
    # Seed arm near home so ranking matches post-approach posture.
    home = getattr(profile, "home_cmd", None)
    joint_names = list(getattr(executor, "joint_names", ()) or ())
    act_names = list(getattr(profile, "actuator_names", ()) or ())
    if home is not None and joint_names and act_names:
        from emet.motion.mujoco_arm_ik import joint_qpos_addrs

        qadr_list = joint_qpos_addrs(model, joint_names)
        # Map profile home actuators → arm joints when names align via pack helpers.
        for i, jn in enumerate(joint_names):
            # Prefer matching ``left_arm_jointK`` ↔ ``left_armK`` style.
            short = jn.replace("_joint", "") if "_joint" in jn else jn
            if short in act_names:
                ai = act_names.index(short)
                if ai < len(home) and i < len(qadr_list):
                    data.qpos[int(qadr_list[i])] = float(home[ai])
    mujoco.mj_forward(model, data)


def plan_pick_place(
    robot: Any,
    *,
    object_query: str,
    receptacle_query: str,
    grasp_poses: Sequence[Any],
    object_gt_body: str,
    receptacle_gt_body: str | None = None,
    approach_standoff_m: float = 0.55,
    top_k_grasps: int = 8,
    executor: Any | None = None,
) -> TaskPlan:
    """Build a grounded approach → grasp → place plan with IK-ranked grasps.

    Does not stream motion; optional ``executor`` supplies an offline MuJoCo model for ranking.
    Ranking uses the **approach** base pose (synced into the offline model), not spawn.
    """
    expanded: list[str] = [f"goal:on({object_gt_body},{receptacle_query})"]
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    pl = read_sim_object_placements(robot.get_emet_session()) or {}
    if object_gt_body not in pl:
        return TaskPlan(
            steps=[],
            object_body=object_gt_body,
            receptacle_body=receptacle_gt_body,
            success=False,
            message="object_not_in_gt",
            expanded_nodes=expanded,
        )
    obj_xy = np.asarray(pl[object_gt_body]["pos"], dtype=np.float64).reshape(3)[:2]
    approach = approach_pose_for_object_xy(obj_xy, standoff=approach_standoff_m)
    expanded.append(f"approach@{approach.tolist()}")

    scores: list[tuple[int, float, bool]] = []
    chosen: int | None = None
    if executor is not None and getattr(executor, "_ensure_model", None) and executor._ensure_model():
        _sync_executor_base_to_xyt(executor, approach)
        scores = rank_grasps_by_ik(
            executor._model,
            executor._data,
            ee_body=executor.ee_body,
            joint_names=executor.joint_names,
            grasp_poses=grasp_poses,
            top_k=top_k_grasps,
        )
        for idx, err, ok in scores:
            expanded.append(f"grasp[{idx}] err={err:.3f} reachable={ok}")
        for idx, _err, ok in scores:
            if ok:
                chosen = idx
                break
        if chosen is None:
            return TaskPlan(
                steps=[],
                object_body=object_gt_body,
                receptacle_body=receptacle_gt_body,
                grasp_scores=scores,
                chosen_grasp_index=None,
                expanded_nodes=expanded,
                success=False,
                message="no_reachable_grasp",
            )
    elif grasp_poses:
        chosen = 0
        scores = [(i, float("inf"), True) for i in range(min(len(grasp_poses), top_k_grasps))]
        expanded.append("ik_rank_skipped")

    if chosen is None:
        return TaskPlan(
            steps=[],
            object_body=object_gt_body,
            receptacle_body=receptacle_gt_body,
            grasp_scores=scores,
            success=False,
            message="no_grasp_candidates",
            expanded_nodes=expanded,
        )

    steps = [
        TaskPlanStep("approach", {"xyt": approach.tolist(), "world_frame": True}, note="base standoff"),
        TaskPlanStep(
            "grasp",
            {
                "object_query": object_query,
                "object_gt_body": object_gt_body,
                "grasp_index": int(chosen),
            },
            note=f"oracle grasp[{chosen}]",
        ),
        TaskPlanStep(
            "place",
            {
                "receptacle_query": receptacle_query,
                "object_gt_body": object_gt_body,
                "receptacle_gt_body": receptacle_gt_body,
            },
            note="place then detach",
        ),
    ]
    expanded.append(f"chosen_grasp={chosen}")
    return TaskPlan(
        steps=steps,
        object_body=object_gt_body,
        receptacle_body=receptacle_gt_body,
        chosen_grasp_index=int(chosen),
        grasp_scores=scores,
        expanded_nodes=expanded,
        success=True,
        message="planned",
    )


def execute_task_plan(
    robot: Any,
    plan: TaskPlan,
    *,
    executor: Any,
    grasp_poses: Sequence[Any],
    manip_mode: str = "kinematic",
    video_recorder: Any | None = None,
) -> TaskPlan:
    """Execute a :class:`TaskPlan` in order; updates ``plan.success`` / ``message``.

    Optional *video_recorder* (``ManipVideoRecorder``) gets status updates per step.
    """
    if not plan.steps:
        plan.success = False
        plan.message = plan.message or "empty_plan"
        return plan

    def _status(action: str, *, detail: str = "") -> None:
        if video_recorder is None:
            return
        goal = f"{plan.object_body}"
        if plan.receptacle_body:
            goal = f"{plan.object_body} → {plan.receptacle_body}"
        video_recorder.set_status(action, goal=goal, detail=detail)

    for step in plan.steps:
        op = step.op
        args = step.args
        logger.info(f"TAMP execute: {op} {args}")
        if op == "approach":
            _status("approach", detail=f"xyt={args.get('xyt')}")
            xyt = np.asarray(args["xyt"], dtype=np.float64)
            robot.move_base_to(xyt, blocking=True, world_frame=bool(args.get("world_frame", True)))
        elif op == "grasp":
            gi = int(args["grasp_index"])
            _status("grasp", detail=f"grasp_index={gi} object={args.get('object_query')!r}")
            if gi < 0 or gi >= len(grasp_poses):
                plan.success = False
                plan.message = f"bad_grasp_index_{gi}"
                return plan
            g = grasp_poses[gi]
            T = getattr(g, "T_world", g)
            if str(manip_mode).lower() == "kinematic":
                result = executor.grasp_only(
                    args["object_query"],
                    object_gt_body=args.get("object_gt_body"),
                    grasp_T_world=T,
                )
                if not result.success:
                    plan.success = False
                    plan.message = f"grasp_failed:{result.message}"
                    return plan
            else:
                from emet.simulation.sim_manipulation import sim_teleport_to_grasp_pose

                pos = np.asarray(getattr(g, "position", T[:3, 3]), dtype=np.float64).reshape(3)
                ok = sim_teleport_to_grasp_pose(robot, args["object_gt_body"], pos, lift_m=0.12)
                if not ok:
                    plan.success = False
                    plan.message = "teleport_grasp_failed"
                    return plan
        elif op == "place":
            _status("place", detail=f"receptacle={args.get('receptacle_query')!r}")
            if str(manip_mode).lower() == "kinematic":
                result = executor.place_only(
                    args["receptacle_query"],
                    object_gt_body=args.get("object_gt_body"),
                )
                if not result.success:
                    plan.success = False
                    plan.message = f"place_failed:{result.message}"
                    return plan
            else:
                from emet.simulation.sim_manipulation import sim_teleport_place

                ok = sim_teleport_place(
                    robot,
                    args["receptacle_query"],
                    object_gt_body=args.get("object_gt_body"),
                )
                if not ok:
                    plan.success = False
                    plan.message = "teleport_place_failed"
                    return plan
        else:
            plan.success = False
            plan.message = f"unknown_op:{op}"
            return plan
        if video_recorder is not None:
            video_recorder.capture_once()

    _status("done", detail=plan.message or "ok")
    plan.success = True
    plan.message = "ok"
    return plan


# ---------------------------------------------------------------------------
# MCTS pick-place: search task assignment, then ground via plan_pick_place
# ---------------------------------------------------------------------------


def plan_pick_place_mcts(
    robot: Any,
    *,
    candidates: Sequence[dict[str, Any]],
    grasp_poses: Sequence[Any],
    executor: Any | None = None,
    approach_standoff_m: float = 0.55,
    top_k_grasps: int = 8,
    mcts_iterations: int = 120,
    mcts_breadth: int = 4,
    mcts_depth: int = 5,
    mcts_uct_c: float = 1.3,
    seed: int | None = None,
) -> TaskPlan:
    """MCTS over candidate (object, receptacle) task assignments.

    Each *candidate* is a dict with ``object_query``, ``receptacle_query``,
    ``object_gt_body``, ``receptacle_gt_body``. The search uses
    :class:`PickPlaceDistancePolicy` over the live scene geometry (base at the
    scene origin), then grounds the winning assignment with
    :func:`plan_pick_place` (approach standoff + IK-ranked reachable grasps).

    This is the "agent-call-wrapping" TAMP seam: the distance heuristic policy
    stands in for an LLM proposer, and the executor/MuJoCo grounding is the
    simulator. Returns the best reachable plan (or an empty failed TaskPlan).
    """
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements
    from emet.motion.agent_mcts import AgentMCTSPlanner, MCTSConfig, PickPlaceDistancePolicy

    pl = read_sim_object_placements(robot.get_emet_session()) or {}
    cands = [c for c in candidates if c.get("object_gt_body") in pl]
    if not cands:
        return TaskPlan(
            steps=[],
            object_body="",
            receptacle_body=None,
            success=False,
            message="no_gt_candidates",
            expanded_nodes=[f"candidates={len(candidates)}"],
        )

    def _pos(body: str) -> np.ndarray:
        return np.asarray(pl[body]["pos"], dtype=np.float64).reshape(3)[:2]

    # Build MCTS over assignments: state = base-at-origin + current (object, receptacle).
    # Each candidate is an independent 1-deep task decision; simulate grounds it.

    policy = PickPlaceDistancePolicy(seed=seed)
    cfg = MCTSConfig(
        n_iterations=int(mcts_iterations),
        expansion_breadth=int(mcts_breadth),
        depth_limit=int(mcts_depth),
        uct_c=float(mcts_uct_c),
        seed=seed,
    )

    # State schema the distance policy expects: robot / object / carrying / receptacle.
    def make_state(obj_body: str, recep_body: str) -> dict:
        return {
            "robot": np.zeros(2, dtype=np.float64),
            "object": _pos(obj_body),
            "carrying": False,
            "receptacle": _pos(recep_body),
        }

    best: TaskPlan | None = None
    for cand in cands:
        obj_body = str(cand["object_gt_body"])
        recep_body = str(cand.get("receptacle_gt_body") or "")
        state = make_state(obj_body, recep_body)
        goal = _pos(recep_body) if recep_body else state["object"]
        planner = AgentMCTSPlanner(policy=policy, simulate=policy_rollout, config=cfg)
        seq = planner.search(state, goal)
        if not seq:
            continue
        # Ground the best assignment through the deterministic TAMP planner. When no
        # grasp candidates were supplied (teleport path), synthesize a top-down grasp
        # at the object COM so the plan still grounds and executes via sim teleport.
        grounding_grasps = list(grasp_poses)
        if not grounding_grasps:
            from emet.controller.task.tamp.grasp_frames import top_down_grasp_T

            grounding_grasps = [top_down_grasp_T(np.asarray(pl[obj_body]["pos"], dtype=np.float64).reshape(3))]
        plan = plan_pick_place(
            robot,
            object_query=str(cand["object_query"]),
            receptacle_query=str(cand["receptacle_query"]),
            grasp_poses=grounding_grasps,
            object_gt_body=obj_body,
            receptacle_gt_body=recep_body or None,
            approach_standoff_m=approach_standoff_m,
            top_k_grasps=top_k_grasps,
            executor=executor,
        )
        plan.grasp_poses = list(grounding_grasps)
        plan.expanded_nodes = [a.name for a in seq] + list(plan.expanded_nodes or ())
        if plan.success and (best is None or len(best.steps) <= len(plan.steps)):
            best = plan
    if best is not None:
        return best
    return TaskPlan(
        steps=[],
        object_body="",
        receptacle_body=None,
        success=False,
        message="no_reachable_task",
        expanded_nodes=[c.get("object_query", "") for c in cands],
    )


def policy_rollout(state: dict, action: Any) -> tuple[dict, float, bool]:
    """Deterministic geometry rollout used by :func:`plan_pick_place_mcts`.

    Mirrors the ``PickPlaceDistancePolicy`` test sim: moving reduces distance,
    pickup/place apply when in range. Purely geometric — no physics, no server.
    """

    next_state = {k: np.asarray(v, dtype=float) if isinstance(v, np.ndarray) else v for k, v in state.items()}
    next_state["carrying"] = bool(state["carrying"])
    obj = np.asarray(next_state["object"], dtype=float)
    rec = np.asarray(next_state["receptacle"], dtype=float)
    robot = np.asarray(next_state["robot"], dtype=float)
    cost_raw: Any = getattr(action, "cost", None)
    cost = float(cost_raw if cost_raw is not None else 1.0)

    if action.name == "move_to":
        target = np.asarray(action.args["xy"], dtype=float).reshape(2)
        next_state["robot"] = target.copy()
        progress = max(0.0, float(np.linalg.norm(obj - rec)) - float(np.linalg.norm(obj - target)))
        return next_state, progress - cost, False
    if action.name == "pickup":
        if float(np.linalg.norm(robot - obj)) <= 0.25 and not bool(state["carrying"]):
            next_state["carrying"] = True
            return next_state, -0.1, False
        return state, -1.0, True
    if action.name == "place":
        if bool(state["carrying"]) and float(np.linalg.norm(robot - rec)) <= 0.30:
            next_state["carrying"] = False
            next_state["object"] = rec.copy()
            return next_state, 10.0, True
        return state, -1.0, True
    return state, -float(cost), True

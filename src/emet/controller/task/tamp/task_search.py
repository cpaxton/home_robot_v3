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
    completed_ops: list[str] = field(default_factory=list)
    failed_op: str | None = None


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
    """Write approach base XYT into the executor's offline MJCF for IK ranking.

    Handles both freejoint bases (rby1/nori/galaxea: ``base_freejoint`` 7-DoF quat)
    and planar slide bases (sourccey/innate_mars/xlerobot: ``base_x/base_y/base_yaw``).
    """
    import mujoco

    model = getattr(executor, "_model", None)
    data = getattr(executor, "_data", None)
    profile = getattr(executor, "profile", None)
    if model is None or data is None or profile is None:
        return
    x, y, th = float(xyt[0]), float(xyt[1]), float(xyt[2])

    def _set_planar_base() -> bool:
        planar = ("base_x", "base_y", "base_yaw")
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in planar]
        if any(jid < 0 for jid in ids):
            return False
        for jn, val in zip(planar, (x, y, th), strict=True):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            data.qpos[int(model.jnt_qposadr[jid])] = float(val)
        return True

    name = getattr(profile, "base_freejoint_name", None)
    if not name or not _set_planar_base():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name)) if name else -1
        if jid < 0:
            return
        qadr = int(model.jnt_qposadr[jid])
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

    plan.completed_ops.clear()
    plan.failed_op = None

    def _status(action: str, *, detail: str = "") -> None:
        if video_recorder is None:
            return
        goal = f"{plan.object_body}"
        if plan.receptacle_body:
            goal = f"{plan.object_body} → {plan.receptacle_body}"
        video_recorder.set_status(action, goal=goal, detail=detail)

    def _fail(op: str, code: str) -> TaskPlan:
        plan.success = False
        plan.failed_op = op
        plan.message = code
        logger.warning(f"TAMP execute failed op={op}: {code}")
        return plan

    for step in plan.steps:
        op = step.op
        args = step.args
        logger.info(f"TAMP execute: {op} {args}")
        if op == "approach":
            _status("approach", detail=f"xyt={args.get('xyt')}")
            xyt = np.asarray(args["xyt"], dtype=np.float64)
            try:
                robot.move_base_to(xyt, blocking=True, world_frame=bool(args.get("world_frame", True)))
            except Exception as exc:
                return _fail(op, f"approach_failed:{type(exc).__name__}")
        elif op == "grasp":
            gi = int(args["grasp_index"])
            _status("grasp", detail=f"grasp_index={gi} object={args.get('object_query')!r}")
            if gi < 0 or gi >= len(grasp_poses):
                return _fail(op, f"bad_grasp_index_{gi}")
            g = grasp_poses[gi]
            T = getattr(g, "T_world", g)
            if str(manip_mode).lower() == "kinematic":
                try:
                    result = executor.grasp_only(
                        args["object_query"],
                        object_gt_body=args.get("object_gt_body"),
                        grasp_T_world=T,
                    )
                except Exception as exc:
                    return _fail(op, f"grasp_execution_error:{type(exc).__name__}")
                if not result.success:
                    return _fail(op, f"grasp_failed:{result.message}")
            else:
                from emet.simulation.sim_manipulation import sim_teleport_to_grasp_pose

                pos = np.asarray(getattr(g, "position", T[:3, 3]), dtype=np.float64).reshape(3)
                try:
                    ok = sim_teleport_to_grasp_pose(robot, args["object_gt_body"], pos, lift_m=0.12)
                except Exception as exc:
                    return _fail(op, f"grasp_execution_error:{type(exc).__name__}")
                if not ok:
                    return _fail(op, "teleport_grasp_failed")
        elif op == "place":
            _status("place", detail=f"receptacle={args.get('receptacle_query')!r}")
            if str(manip_mode).lower() == "kinematic":
                try:
                    result = executor.place_only(
                        args["receptacle_query"],
                        object_gt_body=args.get("object_gt_body"),
                        receptacle_gt_body=args.get("receptacle_gt_body"),
                    )
                except Exception as exc:
                    return _fail(op, f"place_execution_error:{type(exc).__name__}")
                if not result.success:
                    return _fail(op, f"place_failed:{result.message}")
            else:
                from emet.simulation.sim_manipulation import sim_teleport_place

                try:
                    ok = sim_teleport_place(
                        robot,
                        args["receptacle_query"],
                        object_gt_body=args.get("object_gt_body"),
                        receptacle_gt_body=args.get("receptacle_gt_body"),
                    )
                except Exception as exc:
                    return _fail(op, f"place_execution_error:{type(exc).__name__}")
                if not ok:
                    return _fail(op, "teleport_place_failed")
        else:
            return _fail(op, f"unknown_op:{op}")
        plan.completed_ops.append(op)
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
    grasp_poses: Sequence[Any] = (),
    grasp_poses_by_body: dict[str, Sequence[Any]] | None = None,
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

    Grasps are per-object: pass ``grasp_poses_by_body`` keyed by ``object_gt_body``
    (preferred), or a flat ``grasp_poses`` list used for every candidate. When
    neither is available (teleport path) a top-down grasp is synthesized at the
    object COM.

    This is the "agent-call-wrapping" TAMP seam: the distance heuristic policy
    stands in for an LLM proposer, and the executor/MuJoCo grounding is the
    simulator. Returns the best reachable plan (or an empty failed TaskPlan).
    """
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements
    from emet.motion.agent_mcts import AgentMCTSPlanner, MCTSConfig, PickPlaceDistancePolicy

    pl = read_sim_object_placements(robot.get_emet_session()) or {}
    cands = [
        c
        for c in candidates
        if c.get("object_gt_body") in pl and (not c.get("receptacle_gt_body") or c.get("receptacle_gt_body") in pl)
    ]
    if not cands:
        return TaskPlan(
            steps=[],
            object_body="",
            receptacle_body=None,
            success=False,
            message="no_gt_candidates",
            expanded_nodes=[f"candidates={len(candidates)}"],
        )

    by_body = {str(c.get("object_gt_body")): c for c in cands}
    flat_by_body: dict[str, list[Any]] = {}
    for g in grasp_poses:
        body = str(getattr(g, "object_body", "") or "")
        flat_by_body.setdefault(body, []).append(g)

    def _grasps_for(body: str) -> list[Any]:
        if grasp_poses_by_body is not None and body in grasp_poses_by_body:
            return list(grasp_poses_by_body[body])
        if body in flat_by_body:
            return list(flat_by_body[body])
        if body in by_body and by_body[body].get("grasp_poses"):
            return list(by_body[body]["grasp_poses"])
        if grasp_poses:
            return list(grasp_poses)
        # No caller-supplied grasps: resolve real DROID grasps from the scene asset +
        # live placement pose (in-process oracle). Falls back to an empty list so the
        # caller synthesizes a top-down grasp (teleport path).
        try:
            return resolve_scene_grasps(body, pl, category=by_body.get(body, {}).get("object_query"))
        except Exception:
            return []

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
            "receptacle": _pos(recep_body) if recep_body else _pos(obj_body),
        }

    best: TaskPlan | None = None
    last_fail: str | None = None
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
        grounding_grasps = list(_grasps_for(obj_body))
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
        if not plan.success:
            # Keep the most informative failure for diagnostics.
            last_fail = str(plan.message or "")
        if plan.success and (best is None or len(best.steps) <= len(plan.steps)):
            best = plan
    if best is not None:
        return best
    detail = f"last_grounding={last_fail}" if last_fail else f"candidates={len(cands)}"
    return TaskPlan(
        steps=[],
        object_body="",
        receptacle_body=None,
        success=False,
        message=f"no_reachable_task:{detail}",
        expanded_nodes=[c.get("object_query", "") for c in cands],
    )


def replace_step_op(steps: Sequence[TaskPlanStep], op: str, receptacle_gt_body: str) -> list[TaskPlanStep]:
    """Return *steps* with the ``op`` step's ``receptacle_gt_body`` argument replaced."""
    out: list[TaskPlanStep] = []
    for s in steps:
        if s.op == op:
            args = dict(s.args)
            args["receptacle_gt_body"] = str(receptacle_gt_body)
            out.append(TaskPlanStep(s.op, args, s.note))
        else:
            out.append(s)
    return out


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


def resolve_scene_grasps(body: str, placements: dict[str, Any], *, category: str | None = None) -> list[Any]:
    """Resolve real DROID grasp poses for ``body`` from its live placement pose.

    Uses the in-process :class:`MolmoGraspOracle` over on-disk DROID grasp assets
    (``~/.cache/molmospaces/assets/grasps/droid/...``). Returns an empty list when
    no asset exists for the body (callers fall back to a synthetic top-down grasp).
    """
    from emet.perception.grasps.molmo_grasp_library import pose_matrix_from_pos_quat
    from emet.perception.grasps.oracle import MolmoGraspOracle

    info = placements.get(body)
    if not info:
        return []
    pos = info.get("pos")
    if pos is None:
        return []
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    quat = info.get("quat")
    T_obj = pose_matrix_from_pos_quat(pos, quat if quat is not None else [1.0, 0.0, 0.0, 0.0])
    oracle = MolmoGraspOracle()
    asset = str(info.get("asset_id") or "")
    if asset and oracle.has_asset(asset):
        return oracle.predict_from_asset(asset, T_obj, top_k=8)
    try:
        return oracle.predict_for_body(body, T_obj, category=category, top_k=8)
    except Exception:
        return []

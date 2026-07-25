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


def approach_pose_for_object_xy(obj_xy: np.ndarray, *, standoff: float = 0.35) -> np.ndarray:
    """Face −Y (table / iTHOR convention used in default_table smoke)."""
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


def plan_pick_place(
    robot: Any,
    *,
    object_query: str,
    receptacle_query: str,
    grasp_poses: Sequence[Any],
    object_gt_body: str,
    receptacle_gt_body: str | None = None,
    approach_standoff_m: float = 0.35,
    top_k_grasps: int = 8,
    executor: Any | None = None,
) -> TaskPlan:
    """Build a grounded approach → grasp → place plan with IK-ranked grasps.

    Does not stream motion; optional ``executor`` supplies an offline MuJoCo model for ranking.
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
        if chosen is None and scores:
            chosen = scores[0][0]
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
) -> TaskPlan:
    """Execute a :class:`TaskPlan` in order; updates ``plan.success`` / ``message``."""
    if not plan.steps:
        plan.success = False
        plan.message = plan.message or "empty_plan"
        return plan

    for step in plan.steps:
        op = step.op
        args = step.args
        logger.info(f"TAMP execute: {op} {args}")
        if op == "approach":
            xyt = np.asarray(args["xyt"], dtype=np.float64)
            robot.move_base_to(xyt, blocking=True, world_frame=bool(args.get("world_frame", True)))
        elif op == "grasp":
            gi = int(args["grasp_index"])
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

    plan.success = True
    plan.message = "ok"
    return plan

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Multi-object TAMP chain for the clutter-clearance benchmark.

``plan_clear_clutter`` clears a set of scattered floor objects by repeatedly grounding
and executing a pick-and-place plan per object (to a shared drop receptacle / bin),
re-reading live ``sim_object_placements`` after each relocation. For ``nav_goal`` mode
it then snaps the base to the goal landmark only if interpolated chord samples
are footprint-clear of leftover clutter and furniture.

Each grasp follows the benchmark's latch contract: the end-effector reaches the object's
grasp frame, the gripper closes, and the sim ``attach`` welds the object to the gripper
(no physics-grasp requirement). Failures are per-object so a single flaky grasp does not
abort the whole episode.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from emet.utils.logger import Logger

logger = Logger(__name__)


def _clamp_xy(xy: Any) -> np.ndarray:
    return np.asarray(xy, dtype=np.float64).reshape(-1)[:2]


def _dist_xy(a: Any, b: Any) -> float:
    return float(np.linalg.norm(_clamp_xy(a) - _clamp_xy(b)))


def _world_base_xy(robot: Any) -> np.ndarray | None:
    """Best-effort world-frame base XY (session navigation origin composed)."""
    from emet.utils.geometry import nav_xyt_to_world_xyt

    try:
        pose = np.asarray(robot.get_base_pose(timeout=2.0), dtype=np.float64).reshape(-1)
        if pose.size < 3:
            return None
        sess = robot.get_emet_session() if callable(getattr(robot, "get_emet_session", None)) else None
        world = nav_xyt_to_world_xyt(pose[:3], sess)
        state = getattr(robot, "_state", None)
        if isinstance(state, dict) and state.get("base_xyz") is not None:
            xyz = np.asarray(state["base_xyz"], dtype=np.float64).reshape(-1)
            if xyz.size >= 2:
                return np.array([float(xyz[0]), float(xyz[1])], dtype=np.float64)
        return _clamp_xy(world)
    except Exception:  # noqa: BLE001
        return None


# Drop-receptacle aliases are shared (emet.eval.tamp_clutter.BIN_FALLBACKS) so the runner's
# scatter exclusion and the chain's resolution use the same trash-only set.
from emet.eval.tamp_clutter import BIN_FALLBACKS  # noqa: E402


def _resolve_bin_body(placements: Mapping[str, Any], bin_query: str) -> tuple[str | None, str | None]:
    """Pick a GT receptacle body for the drop receptacle (bin), with aliases.

    Returns ``(body, matched_query)`` so callers can flag when an alias
    (e.g. ashcan) matched rather than the requested ``bin_query``.
    """
    from emet.eval.ovmm_find_phase import bodies_matching_category

    queries = [bin_query] if bin_query else []
    for q in BIN_FALLBACKS:
        if q not in queries:
            queries.append(q)
    for q in queries:
        bodies = bodies_matching_category(dict(placements), q)
        if bodies:
            return sorted(bodies)[0], q
    return None, None


def plan_clear_clutter(
    robot: Any,
    *,
    objects: Sequence[Mapping[str, Any]],
    mode: str,
    bin_query: str,
    goal_xy: np.ndarray | list | None = None,
    goal_radius_m: float = 0.5,
    drop_radius_m: float = 0.5,
    manip_mode: str = "latch",
    executor: Any | None = None,
    grasp_poses_by_body: Mapping[str, Sequence[Any]] | None = None,
    approach_standoff_m: float = 0.55,
    mcts_iterations: int = 120,
    seed: int | None = None,
    robot_move_goal: Any = None,
    clearance_m: float = 0.22,
) -> dict[str, Any]:
    """Clear scattered floor objects as part of a plan; return a metrics dict.

    ``objects`` is a list of ``{object_query, object_gt_body}``. Every object is
    relocated to the drop receptacle (``bin_query``). ``mode`` ``cleanup`` succeeds when
    all objects are relocated; ``nav_goal`` additionally requires the base to reach
    ``goal_xy`` within ``goal_radius_m``.

    ``manip_mode``: ``latch`` (default; kinematic IK + sim attach) or ``sim``
    (teleport oracle). ``executor`` is a :class:`KinematicPickPlaceExecutor` for
    ``latch``; when omitted one is built for the robot.

    Nav-to-landmark does **not** snap through leftover clutter. After relocating,
    the same GT occupancy probe as ``episode_valid`` must report a free route;
    only then is a single ``move_base_to`` issued (MolmoSpaces already teleports).
    Interpolating that snap along a straight line still cuts the ring; stepping
    via repeated ZMQ teleports only adds ``at_goal`` waits.
    """
    from emet.controller.task.tamp.task_search import execute_task_plan, plan_pick_place_mcts
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    t0 = time.monotonic()
    pl = read_sim_object_placements(robot.get_emet_session()) or {}
    bin_body, bin_matched = _resolve_bin_body(pl, bin_query)
    if bin_body is None:
        return {
            "mode": mode,
            "n_objects": int(len(objects)),
            "n_cleared": 0,
            "n_relocated": 0,
            "goal_reached": False,
            "nav_success": False,
            "task_success": False,
            "manip_success_rate": 0.0,
            "motion_failures": 0,
            "planning_wall_s": float(time.monotonic() - t0),
            "manip_wall_s": 0.0,
            "error": f"missing_bin_body:{bin_query}",
            "nav_path_open": False,
        }

    exec_manip = "kinematic" if str(manip_mode).lower() in ("latch", "attempt") else "teleport"
    if executor is None and exec_manip == "kinematic":
        from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

        executor = KinematicPickPlaceExecutor(robot, manip_collision="none", traj_dt=0.05)

    # Filter objects present in GT placements (all are scattered away from the bin).
    pending: list[Mapping[str, Any]] = []
    for obj in objects:
        body = str(obj.get("object_gt_body") or "")
        if body in pl:
            pending.append(obj)
    if not pending:
        return {
            "mode": mode,
            "n_objects": int(len(objects)),
            "n_cleared": 0,
            "n_relocated": 0,
            "goal_reached": False,
            "nav_success": False,
            "task_success": False,
            "manip_success_rate": 0.0,
            "motion_failures": 0,
            "planning_wall_s": float(time.monotonic() - t0),
            "manip_wall_s": 0.0,
            "error": "no_gt_objects",
            "nav_path_open": False,
        }

    relocated: list[str] = []
    failed: list[str] = []
    motion_failures = 0
    plan_wall = 0.0
    manip_wall = 0.0

    for obj in pending:
        body = str(obj["object_gt_body"])
        obj_query = str(obj.get("object_query") or body)
        candidates = [
            {
                "object_query": obj_query,
                "receptacle_query": bin_query,
                "object_gt_body": body,
                "receptacle_gt_body": bin_body,
            }
        ]
        t_p = time.monotonic()
        plan = plan_pick_place_mcts(
            robot,
            candidates=candidates,
            grasp_poses_by_body=dict(grasp_poses_by_body) if grasp_poses_by_body else None,
            executor=executor,
            approach_standoff_m=approach_standoff_m,
            mcts_iterations=mcts_iterations,
            seed=seed,
        )
        plan_wall += time.monotonic() - t_p
        if not plan.success or plan.receptacle_body != bin_body:
            failed.append(body)
            logger.warning(f"clutter plan failed body={body}: {plan.message}")
            continue
        t_m = time.monotonic()
        plan = execute_task_plan(
            robot,
            plan,
            executor=executor,
            grasp_poses=plan.grasp_poses,
            manip_mode=exec_manip,
        )
        manip_wall += time.monotonic() - t_m
        if not plan.success:
            failed.append(body)
            if plan.failed_op in ("grasp", "place"):
                motion_failures += 1
            logger.warning(f"clutter execute failed body={body}: {plan.failed_op}:{plan.message}")
            continue
        # Re-read live placements and confirm the object reached the bin.
        after = read_sim_object_placements(robot.get_emet_session()) or {}
        bin_xy = after.get(bin_body, {}).get("pos")
        obj_after = after.get(body, {}).get("pos")
        if bin_xy is not None and obj_after is not None and _dist_xy(obj_after, bin_xy) <= drop_radius_m:
            relocated.append(body)
        else:
            failed.append(body)
            logger.warning(f"clutter relocate verify failed body={body}")

    goal_reached = False
    nav_success = False
    nav_path_open = True
    nav_probe_after: dict[str, Any] | None = None
    if mode == "nav_goal" and goal_xy is not None:
        goal_reached, nav_success, nav_path_open, nav_probe_after = nav_to_landmark_if_clear(
            robot,
            goal_xy=goal_xy,
            objects=objects,
            goal_radius_m=goal_radius_m,
            clearance_m=clearance_m,
            robot_move_goal=robot_move_goal,
        )

    n_total = int(len(objects))
    n_relocated = int(len(relocated))
    n_cleared = n_relocated
    if mode == "nav_goal":
        task_success = bool(goal_reached) and bool(nav_path_open)
    else:
        task_success = bool(n_total > 0 and n_relocated >= n_total)

    out: dict[str, Any] = {
        "mode": mode,
        "n_objects": n_total,
        "n_cleared": n_cleared,
        "n_relocated": n_relocated,
        "goal_reached": bool(goal_reached),
        "nav_success": bool(nav_success),
        "nav_path_open": bool(nav_path_open),
        "task_success": bool(task_success),
        "manip_success_rate": float(n_relocated / n_total) if n_total else 0.0,
        "motion_failures": int(motion_failures),
        "planning_wall_s": float(plan_wall),
        "manip_wall_s": float(manip_wall),
        "bin_body": bin_body,
        "bin_query": bin_query,
        "bin_fallback": bool(bin_matched is not None and bin_matched != bin_query),
        "bin_matched_query": bin_matched,
        "failed_bodies": failed,
        "relocated_bodies": relocated,
    }
    if nav_probe_after is not None:
        out["nav_probe_after"] = nav_probe_after
    return out


def nav_to_landmark_if_clear(
    robot: Any,
    *,
    goal_xy: np.ndarray | list,
    objects: Sequence[Mapping[str, Any]],
    goal_radius_m: float,
    clearance_m: float,
    robot_move_goal: Any = None,
) -> tuple[bool, bool, bool, dict[str, Any] | None]:
    """Snap to the landmark only if an 8-connected route around disks exists.

    Scoring matches the GT validity probe: a landmark behind furniture is
    reachable if a path exists around leftover clutter + furniture disks.
    ``move_base_to`` is still a teleport snap; the probe is the reachability
    gate, not a curved follower.
    """
    from emet.eval.tamp_clutter import (
        bodies_near_xy,
        nav_path_open_around_disks,
        placement_obstacle_disks,
    )
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    target = _clamp_xy(goal_xy)
    here = _world_base_xy(robot)
    live = read_sim_object_placements(robot.get_emet_session()) or {}
    if here is None:
        logger.warning("clutter nav refused: missing base pose")
        return False, False, False, None
    skip = bodies_near_xy(live, target, keepout_m=0.75)
    disks = placement_obstacle_disks(live, skip_bodies=skip)
    known = {name for _xy, _r, name in disks}
    for obj in objects:
        body = str(obj.get("object_gt_body") or "")
        if not body or body in skip or body in known:
            continue
        pos = (live.get(body) or {}).get("pos")
        if pos is not None:
            disks.append((_clamp_xy(pos), 0.08, body))
            known.add(body)
    # Post-clear route: 8-connected path around furniture + leftover clutter (the
    # GT-planner analogue of the validity probe), not the straight-line teleport
    # chord — a landmark behind furniture is reachable by planning around it.
    path_open, probe = nav_path_open_around_disks(here, target, disks, clearance_m=float(clearance_m))
    if not path_open:
        logger.warning(
            "clutter nav refused: no 8-connected route to landmark "
            f"around {len(disks)} obstacle disks (blocked={not path_open})"
        )
        return False, False, False, probe
    try:
        ok = robot.move_base_to(
            np.array([float(target[0]), float(target[1]), 0.0], dtype=np.float64),
            blocking=True,
            world_frame=True,
        )
        if not ok:
            # Client refused (e.g. >12 m from navigation_origin) — not an exception.
            logger.warning("clutter nav refused by client (move_base_to returned False)")
            return False, False, True, probe
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"clutter nav to goal failed: {exc}")
        return False, False, True, probe
    if robot_move_goal is not None:
        achieved = _clamp_xy(robot_move_goal)
    else:
        achieved = _world_base_xy(robot)
    goal_reached = achieved is not None and _dist_xy(achieved, target) <= goal_radius_m
    return bool(goal_reached), bool(goal_reached), True, probe

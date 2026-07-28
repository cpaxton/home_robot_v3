# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Habitat navmesh navigation helpers for EQA (optional vs voxel A*)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class NavAttemptResult:
    """Outcome of one navigate_to_target_pose call."""

    success: bool
    finished: bool
    dist_m: float
    method: str
    note: str
    target_obs_id: int | None = None
    goal_xy: tuple[float, float] | None = None
    effective_goal_xy: tuple[float, float] | None = None
    path_xy: list[list[float]] | None = None


@dataclass(frozen=True)
class ResolvedNavGoal:
    """Planar nav target after navmesh snap / path-end projection."""

    request_xy: tuple[float, float]
    effective_xy: tuple[float, float]
    mode: str  # raw | snapped | path_end


def habitat_explore_frontiers_enabled(parameters: Any) -> bool:
    if isinstance(parameters, dict):
        eqa = parameters.get("eqa", {}) or {}
    else:
        eqa = parameters.get("eqa", {}) if hasattr(parameters, "get") else {}
    if not isinstance(eqa, dict):
        return True
    return bool(eqa.get("habitat_explore_frontiers", True))


def habitat_nav_would_be_noop(
    robot: Any,
    goal_xy: np.ndarray | tuple[float, float],
    *,
    finish_radius_m: float = 0.28,
) -> bool:
    """True when the robot is already within finish radius of a resolved nav goal."""
    resolved = apply_habitat_nav_resolution(robot, goal_xy)
    if resolved is None:
        return False
    travel = habitat_goal_travel_m(robot, resolved[:2])
    return travel <= finish_radius_m


def goal_key_xy(xy: np.ndarray | tuple[float, float]) -> tuple[float, float]:
    return (round(float(xy[0]), 2), round(float(xy[1]), 2))


def habitat_goal_travel_m(robot: Any, goal_xy: np.ndarray | tuple[float, float]) -> float:
    """Planar distance from robot base to ``goal_xy``."""
    robot_xy = robot_planar_xy(robot)
    arr = np.asarray(goal_xy, dtype=np.float64).reshape(-1)
    return _planar_dist(robot_xy, (float(arr[0]), float(arr[1])))


def robot_planar_xy(robot: Any) -> tuple[float, float]:
    """Current Habitat nav pose ``(x, z)``."""
    pose = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)
    return float(pose[0]), float(pose[1])


def _planar_dist(a: tuple[float, float] | np.ndarray, b: tuple[float, float] | np.ndarray) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _recent_goal_penalty(
    xy: tuple[float, float],
    recent: list[tuple[float, float]] | None,
    *,
    radius_m: float = 1.25,
) -> float:
    """Discourage picking a frontier on top of a goal we just visited."""
    if not recent:
        return 0.0
    near = min(_planar_dist(xy, r) for r in recent)
    if near >= radius_m:
        return 0.0
    return (radius_m - near) * 4.0


def _frontier_explore_sort_key(
    node: Any,
    robot_xy: tuple[float, float],
    *,
    recent: list[tuple[float, float]] | None = None,
    grid_resolution_m: float = 0.1,
    min_travel_m: float = 0.0,
) -> tuple[float, float, int]:
    """Rank a frontier node by region utility (area gain per unit travel).

    Nearest-first creep left the robot circling its spawn area while whole rooms
    stayed unexplored (holdout q104/q105). Nodes without cluster metadata score on
    proximity alone, so non-Habitat backends keep their previous ordering.
    """
    from emet.memory.graph_eqa.frontier_regions import frontier_region_utility, region_from_node

    utility = frontier_region_utility(
        region_from_node(node),
        robot_xy,
        grid_resolution_m=grid_resolution_m,
        recent=recent,
        min_travel_m=min_travel_m,
    )
    nx, nz = float(node.xyz[0]), float(node.xyz[1])
    return (
        -utility,
        _planar_dist(robot_xy, (nx, nz)),
        int(getattr(node, "obs_id", 0)),
    )


def explore_grid_resolution_m(agent: Any, default: float = 0.1) -> float:
    voxel_map = getattr(agent, "voxel_map", None)
    return float(getattr(voxel_map, "grid_resolution", default) or default)


def explore_min_travel_m(agent: Any) -> float:
    """Escape floor set by the EQA loop after repeated 'target not visible' views."""
    return float(getattr(agent, "_explore_min_travel_m", 0.0) or 0.0)


def apply_habitat_nav_resolution(
    robot: Any,
    target_xy: np.ndarray | tuple[float, float],
) -> np.ndarray | None:
    """Map a voxel/frontier XY to a reachable navmesh goal, or ``None`` when no path exists."""
    arr = np.asarray(target_xy, dtype=np.float64).reshape(-1)
    gx, gz = float(arr[0]), float(arr[1])
    sim = getattr(robot, "_sim", None)
    if sim is None or not hasattr(sim, "find_path_to_xy"):
        return np.array([gx, gz, 1.0], dtype=float)
    resolved = resolve_habitat_nav_goal(sim, gx, gz)
    if resolved is None:
        return None
    return np.array([resolved.effective_xy[0], resolved.effective_xy[1], 1.0], dtype=float)


def pick_habitat_exploration_target(
    agent: Any,
    *,
    question: str | None = None,
    blocked: set[tuple[float, float]] | None = None,
    recent_goals: list[tuple[float, float]] | None = None,
) -> np.ndarray | None:
    """Choose a frontier nav goal for Habitat (graph node, heuristic, or voxel sample)."""
    blocked = blocked or set()
    robot = getattr(agent, "robot", None)
    recent = list(recent_goals or getattr(agent, "_habitat_recent_goals", None) or [])

    def _accept(raw: np.ndarray | None) -> np.ndarray | None:
        if raw is None:
            return None
        key = goal_key_xy(raw)
        if key in blocked:
            return None
        if robot is None:
            return raw
        resolved = apply_habitat_nav_resolution(robot, raw)
        if resolved is None:
            blocked.add(key)
            return None
        eff = (float(resolved[0]), float(resolved[1]))
        if goal_key_xy(eff) in blocked:
            blocked.add(key)
            return None
        if habitat_nav_would_be_noop(robot, resolved):
            blocked.add(goal_key_xy(eff))
            blocked.add(key)
            return None
        # Recent is a soft skip only — permanently blocking emptied the frontier
        # set on HM-EQA q104 (every explore_frontier logged frontier_xyz=null).
        if _recent_goal_penalty(eff, recent, radius_m=1.25) > 0.0:
            return None
        return resolved

    gm = getattr(agent, "graph_memory", None)
    if gm is not None:
        nodes = [n for n in gm.get_nodes() if getattr(n, "is_frontier", False)]
        if nodes:
            robot_xy = robot_planar_xy(robot) if robot is not None else (0.0, 0.0)
            nodes.sort(
                key=lambda n: _frontier_explore_sort_key(
                    n,
                    robot_xy,
                    recent=recent,
                    grid_resolution_m=explore_grid_resolution_m(agent),
                    min_travel_m=explore_min_travel_m(agent),
                )
            )
            for node in nodes:
                raw = np.array([float(node.xyz[0]), float(node.xyz[1]), 1.0], dtype=float)
                pt = _accept(raw)
                if pt is not None:
                    return pt
    if hasattr(agent, "_best_frontier_point_from_graph"):
        pt = _accept(agent._best_frontier_point_from_graph(question))
        if pt is not None:
            return pt
    if hasattr(agent, "space") and hasattr(agent.space, "sample_frontier"):
        for _ in range(4):
            fr = agent.space.sample_frontier(
                agent.planner,
                agent._planning_base_xyt(agent.robot.get_base_pose()),
                text=question,
            )
            if fr is None:
                break
            raw = np.array([float(fr[0]), float(fr[1]), 1.0], dtype=float)
            pt = _accept(raw)
            if pt is not None:
                return pt
    return None


def _mujoco_accept_explore_xy(
    raw: np.ndarray | None,
    *,
    blocked: set[tuple[float, float]],
    recent: list[tuple[float, float]],
) -> np.ndarray | None:
    """Accept a planar explore target for non-Habitat (Molmo/Robocasa) stacks."""
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    key = goal_key_xy(arr)
    if key in blocked:
        return None
    if _recent_goal_penalty(key, recent, radius_m=1.25) > 0.0:
        return None
    return np.array([float(arr[0]), float(arr[1]), 1.0], dtype=float)


def pick_uncovered_explore_target(
    agent: Any,
    *,
    question: str | None = None,
    candidates: list[np.ndarray | None] | None = None,
    blocked: set[tuple[float, float]] | None = None,
    recent_goals: list[tuple[float, float]] | None = None,
    min_travel_m: float = 0.0,
) -> np.ndarray | None:
    """Blocked/recent-aware explore target for Habitat **and** MuJoCo Dynagraph EQA.

    Habitat uses navmesh resolution + noop rejection; MuJoCo uses planar blocked/recent
    filtering then ``space.sample_frontier``. ``min_travel_m`` rejects candidates that do
    not leave the current area (set once the verifier keeps reporting the target absent).
    """
    blocked_set = blocked if blocked is not None else getattr(agent, "_habitat_blocked_goals", None)
    if blocked_set is None:
        blocked_set = set()
    recent = list(recent_goals or getattr(agent, "_habitat_recent_goals", None) or [])
    robot = getattr(agent, "robot", None)
    habitat = robot is not None and is_habitat_robot_client(robot)
    if min_travel_m <= 0.0:
        min_travel_m = explore_min_travel_m(agent)
    robot_xy = robot_planar_xy(robot) if robot is not None else (0.0, 0.0)

    for cand in candidates or []:
        if cand is None:
            continue
        if min_travel_m > 0.0 and _planar_dist((float(cand[0]), float(cand[1])), robot_xy) < min_travel_m:
            continue
        if habitat:
            key = goal_key_xy(cand)
            if key in blocked_set:
                continue
            resolved = apply_habitat_nav_resolution(robot, cand)
            if resolved is None:
                blocked_set.add(key)
                continue
            eff_key = goal_key_xy(resolved)
            if eff_key in blocked_set or habitat_nav_would_be_noop(robot, resolved):
                blocked_set.add(key)
                blocked_set.add(eff_key)
                continue
            if _recent_goal_penalty(eff_key, recent, radius_m=1.25) > 0.0:
                continue
            return resolved
        accepted = _mujoco_accept_explore_xy(cand, blocked=blocked_set, recent=recent)
        if accepted is not None:
            return accepted

    if habitat:
        return pick_habitat_exploration_target(
            agent,
            question=question,
            blocked=blocked_set,
            recent_goals=recent,
        )

    # MuJoCo / Molmo: graph frontiers then voxel sample_frontier with retries.
    gm = getattr(agent, "graph_memory", None)
    if gm is not None:
        nodes = [n for n in gm.get_nodes() if getattr(n, "is_frontier", False)]
        robot_xy = robot_planar_xy(robot) if robot is not None else (0.0, 0.0)
        nodes.sort(
            key=lambda n: _frontier_explore_sort_key(
                n,
                robot_xy,
                recent=recent,
                grid_resolution_m=explore_grid_resolution_m(agent),
                min_travel_m=explore_min_travel_m(agent),
            )
        )
        for node in nodes:
            raw = np.array([float(node.xyz[0]), float(node.xyz[1]), 1.0], dtype=float)
            accepted = _mujoco_accept_explore_xy(raw, blocked=blocked_set, recent=recent)
            if accepted is not None:
                return accepted
    if hasattr(agent, "_best_frontier_point_from_graph"):
        accepted = _mujoco_accept_explore_xy(
            agent._best_frontier_point_from_graph(question),
            blocked=blocked_set,
            recent=recent,
        )
        if accepted is not None:
            return accepted
    if hasattr(agent, "space") and hasattr(agent.space, "sample_frontier") and robot is not None:
        start = agent._planning_base_xyt(robot.get_base_pose())
        for _ in range(6):
            fr = agent.space.sample_frontier(agent.planner, start, text=question)
            if fr is None:
                break
            accepted = _mujoco_accept_explore_xy(
                np.array([float(fr[0]), float(fr[1]), 1.0], dtype=float),
                blocked=blocked_set,
                recent=recent,
            )
            if accepted is not None:
                return accepted
    return None


def habitat_body_scan(robot: Any, *, turns: int = 6, on_step: Any | None = None) -> None:
    """Rotate in place on Habitat (head stubs are no-ops; body turns build the map)."""
    sim = getattr(robot, "_sim", None)
    if sim is None or not hasattr(sim, "step"):
        return
    for _ in range(max(1, int(turns))):
        sim.step("turn_left")
        if hasattr(robot, "_sync_pose_from_sim"):
            robot._sync_pose_from_sim()
        if on_step is not None:
            on_step()


def habitat_perfect_nav_enabled(parameters: Any) -> bool:
    """True when Habitat EQA should use navmesh pathing instead of voxel A* only."""
    if isinstance(parameters, dict):
        eqa = parameters.get("eqa", {}) or {}
    else:
        eqa = parameters.get("eqa", {}) if hasattr(parameters, "get") else {}
    if not isinstance(eqa, dict):
        return False
    return bool(eqa.get("habitat_perfect_nav", eqa.get("habitat_navmesh_nav", False)))


def is_habitat_robot_client(robot: Any) -> bool:
    return type(robot).__name__ == "HabitatRobotClient" or (
        hasattr(robot, "_sim") and hasattr(getattr(robot, "_sim", None), "find_path_to_xy")
    )


def resolve_habitat_nav_goal(sim: Any, goal_x: float, goal_z: float) -> ResolvedNavGoal | None:
    """Map a voxel/frontier ``(x,z)`` to a reachable navmesh goal.

    Frontier cluster centers often lie off the navmesh. We snap to the mesh, plan a
    path, and when the path endpoint still differs from the request we navigate to
    the **path end** (coverage) instead of failing 0.6 m short forever.
    """
    if sim is None or not hasattr(sim, "find_path_to_xy"):
        return None
    req = (float(goal_x), float(goal_z))
    sx, sz = req
    mode = "raw"
    if hasattr(sim, "snap_navmesh_xz"):
        sx, sz, ok = sim.snap_navmesh_xz(req[0], req[1])
        if ok:
            mode = "snapped" if math.hypot(sx - req[0], sz - req[1]) > 0.05 else "raw"
    path_pts = sim.find_path_to_xy(sx, sz)
    if path_pts is None or len(path_pts) < 2:
        return None
    end_x, end_z = float(path_pts[-1][0]), float(path_pts[-1][2])
    eff = (end_x, end_z)
    if math.hypot(end_x - req[0], end_z - req[1]) > 0.2:
        mode = "path_end"
    elif math.hypot(end_x - sx, end_z - sz) > 0.05 and mode != "path_end":
        mode = "snapped"
    return ResolvedNavGoal(request_xy=req, effective_xy=eff, mode=mode)


def sample_habitat_navmesh_approach_xy(
    sim: Any,
    *,
    anchor_xy: tuple[float, float],
    robot_xy: tuple[float, float] | None,
    approach_index: int = 0,
    radius_inner_m: float = 0.45,
    radius_outer_m: float = 1.60,
    n_draws: int = 24,
    avoid_xy: list[tuple[float, float]] | None = None,
    avoid_m: float = 0.55,
) -> tuple[float, float] | None:
    """Sample a navmesh-reachable approach near ``anchor``.

    Used by agentic investigate under Habitat perfect-nav so goals can sit on the
    navmesh (including through doorways) even when the voxel free ring is sparse.
    """
    if sim is None:
        return None
    ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
    seed = (int(round(ax * 100)) * 1009 + int(round(ay * 100)) * 9176 + int(approach_index) * 131) & 0xFFFFFFFF
    rng = random.Random(seed)
    r_in = max(0.2, float(radius_inner_m))
    r_out = max(r_in + 0.15, float(radius_outer_m))

    def _too_close(x: float, y: float) -> bool:
        if not avoid_xy:
            return False
        return any(math.hypot(x - ox, y - oy) < float(avoid_m) for ox, oy in avoid_xy)

    best: tuple[float, float, float] | None = None  # score, x, y
    for _ in range(max(8, int(n_draws))):
        ang = rng.uniform(0.0, 2.0 * math.pi)
        rad = math.sqrt(rng.uniform(r_in * r_in, r_out * r_out))
        x = ax + rad * math.cos(ang)
        y = ay + rad * math.sin(ang)
        if _too_close(x, y):
            continue
        resolved = resolve_habitat_nav_goal(sim, x, y)
        if resolved is None:
            continue
        ex, ey = resolved.effective_xy
        if _too_close(ex, ey):
            continue
        # Prefer snaps that stay close to the requested approach (not path_end far away).
        snap_err = math.hypot(ex - x, ey - y)
        score = -snap_err - 0.35 * math.hypot(ex - ax, ey - ay) + rng.random() * 0.05
        if best is None or score > best[0]:
            best = (score, ex, ey)
    if best is None:
        # Last resort: snap the anchor itself / classic standoff.
        if robot_xy is not None:
            rx, ry = float(robot_xy[0]), float(robot_xy[1])
            dx, dy = ax - rx, ay - ry
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                travel = dist if dist <= r_in else max(r_in, dist - r_in)
                gx, gy = rx + (dx / dist) * travel, ry + (dy / dist) * travel
                resolved = resolve_habitat_nav_goal(sim, gx, gy)
                if resolved is not None:
                    return resolved.effective_xy
        resolved = resolve_habitat_nav_goal(sim, ax, ay)
        if resolved is not None:
            return resolved.effective_xy
        return None
    return (best[1], best[2])


def navmesh_waypoints_to_xyt(path_pts: np.ndarray, *, max_waypoints: int = 24) -> list[np.ndarray]:
    """Convert Habitat navmesh ``(N,3)`` XYZ points to planar ``(x,z,yaw)`` trajectory."""
    pts = np.asarray(path_pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 3:
        return []
    if pts.shape[0] > max_waypoints:
        idx = np.linspace(0, pts.shape[0] - 1, num=max_waypoints, dtype=int)
        pts = pts[idx]
    out: list[np.ndarray] = []
    for i in range(pts.shape[0]):
        x, z = float(pts[i, 0]), float(pts[i, 2])
        if i + 1 < pts.shape[0]:
            nxt = pts[i + 1]
            yaw = float(math.atan2(float(nxt[2]) - z, float(nxt[0]) - x))
        elif out:
            yaw = float(out[-1][2])
        else:
            yaw = 0.0
        out.append(np.array([x, z, yaw], dtype=np.float64))
    return out


def habitat_random_walk_step(robot: Any, *, rng: random.Random | None = None) -> str:
    """One discrete exploration step when frontier nav is blocked (no VLM needed)."""
    r = rng or random
    sim = getattr(robot, "_sim", None)
    if sim is None or not hasattr(sim, "step"):
        return "noop"
    if r.random() < 0.35:
        action = "turn_left" if r.random() < 0.5 else "turn_right"
    else:
        action = "move_forward"
    sim.step(action)
    if hasattr(robot, "_sync_pose_from_sim"):
        robot._sync_pose_from_sim()
    return action


def habitat_navmesh_navigate(
    robot: Any,
    target_xy: np.ndarray | tuple[float, float],
    *,
    start_xyt: np.ndarray | None = None,
    target_theta: float | None = None,
    max_waypoints: int = 24,
    min_success_dist_m: float = 0.08,
    finish_radius_m: float = 0.28,
) -> NavAttemptResult:
    """Follow navmesh to ``target_xy`` (Habitat X/Z). Returns movement outcome."""
    sim = getattr(robot, "_sim", None)
    if sim is None or not hasattr(sim, "find_path_to_xy"):
        return NavAttemptResult(
            success=False,
            finished=False,
            dist_m=0.0,
            method="habitat_navmesh",
            note="no_sim",
        )
    goal_x, goal_z = float(target_xy[0]), float(target_xy[1])
    resolved = resolve_habitat_nav_goal(sim, goal_x, goal_z)
    if resolved is None:
        return NavAttemptResult(
            success=False,
            finished=False,
            dist_m=0.0,
            method="habitat_navmesh",
            note="navmesh_no_path",
            goal_xy=(goal_x, goal_z),
        )
    eff_x, eff_z = resolved.effective_xy
    before = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)[:3].copy()
    start_goal_m = float(np.hypot(eff_x - before[0], eff_z - before[1]))
    finish_tol = max(min_success_dist_m, finish_radius_m)
    if start_goal_m <= finish_tol:
        return NavAttemptResult(
            success=False,
            finished=False,
            dist_m=0.0,
            method="habitat_navmesh",
            note=f"already_at_goal_{start_goal_m:.2f}m",
            goal_xy=(goal_x, goal_z),
            effective_goal_xy=(eff_x, eff_z),
        )
    yaw = float(target_theta if target_theta is not None else before[2])
    path_pts = sim.find_path_to_xy(eff_x, eff_z)
    path_xy: list[list[float]] | None = None
    if path_pts is not None and len(path_pts) >= 2:
        path_xy = [[float(p[0]), float(p[2])] for p in np.asarray(path_pts)]
        waypoints = navmesh_waypoints_to_xyt(path_pts, max_waypoints=max_waypoints)
        if len(waypoints) >= 2 and hasattr(robot, "execute_trajectory"):
            robot.execute_trajectory(waypoints[1:], blocking=True)
        else:
            robot.move_base_to(np.array([eff_x, eff_z, yaw], dtype=np.float64), blocking=True)
    else:
        robot.move_base_to(np.array([eff_x, eff_z, yaw], dtype=np.float64), blocking=True)
    after = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)[:3]
    dist_m = float(np.hypot(after[0] - before[0], after[1] - before[1]))
    goal_dist = float(np.hypot(after[0] - eff_x, after[1] - eff_z))
    req_dist = float(np.hypot(after[0] - goal_x, after[1] - goal_z))
    at_goal = goal_dist <= finish_tol
    moved_enough = dist_m >= min_success_dist_m or (at_goal and start_goal_m > min_success_dist_m)
    finished = moved_enough and at_goal
    success = finished
    if finished:
        note = f"ok_{resolved.mode}"
    elif at_goal and not moved_enough:
        note = f"already_at_goal_{start_goal_m:.2f}m"
    else:
        note = f"moved_{dist_m:.2f}m_eff_{goal_dist:.2f}m_req_{req_dist:.2f}m_{resolved.mode}"
    return NavAttemptResult(
        success=success,
        finished=finished,
        dist_m=dist_m,
        method="habitat_navmesh",
        note=note,
        goal_xy=(goal_x, goal_z),
        effective_goal_xy=(eff_x, eff_z),
        path_xy=path_xy,
    )

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


def goal_key_xy(xy: np.ndarray | tuple[float, float]) -> tuple[float, float]:
    return (round(float(xy[0]), 2), round(float(xy[1]), 2))


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
) -> tuple[float, float, float, int]:
    nx, nz = float(node.xyz[0]), float(node.xyz[1])
    dist = _planar_dist(robot_xy, (nx, nz))
    penalty = _recent_goal_penalty((nx, nz), recent)
    return (
        float(int(getattr(node, "nav_failures", 0))),
        dist + penalty,
        -float(int(getattr(node, "last_seen", 0))),
        int(getattr(node, "obs_id", 0)),
    )


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
    robot_xy = robot_planar_xy(robot) if robot is not None else (0.0, 0.0)

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
        if _recent_goal_penalty(eff, recent, radius_m=0.85) > 0.0:
            blocked.add(key)
            return None
        return resolved

    gm = getattr(agent, "graph_memory", None)
    if gm is not None:
        nodes = [n for n in gm.get_nodes() if getattr(n, "is_frontier", False)]
        if nodes:
            nodes.sort(key=lambda n: _frontier_explore_sort_key(n, robot_xy, recent=recent))
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
    min_success_dist_m: float = 0.12,
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
    finished = goal_dist <= max(min_success_dist_m, finish_radius_m)
    success = dist_m >= min_success_dist_m or finished
    if finished:
        note = f"ok_{resolved.mode}"
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

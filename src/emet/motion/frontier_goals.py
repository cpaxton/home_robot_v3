# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Collect top-K explore frontier XYs for multi-goal base A*."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from emet.controller.habitat_nav import (
    _frontier_explore_sort_key,
    _mujoco_accept_explore_xy,
    _planar_dist,
    explore_grid_resolution_m,
    explore_min_travel_m,
    goal_key_xy,
    is_habitat_robot_client,
    robot_planar_xy,
)


def _dedup_key(xy: np.ndarray | tuple[float, float], *, grid_m: float = 0.35) -> tuple[int, int]:
    return (int(round(float(xy[0]) / grid_m)), int(round(float(xy[1]) / grid_m)))


def _as_xyz1(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size < 2 or not np.isfinite(arr[:2]).all():
        return None
    z = float(arr[2]) if arr.size > 2 and np.isfinite(arr[2]) and abs(float(arr[2])) > 1e-9 else 1.0
    return np.array([float(arr[0]), float(arr[1]), z], dtype=np.float64)


def collect_explore_frontier_candidates(
    agent: Any,
    *,
    question: str | None = None,
    k: int = 8,
    blocked: set[tuple[float, float]] | None = None,
    recent_goals: list[tuple[float, float]] | None = None,
    seeds: list[np.ndarray | None] | None = None,
    dedup_m: float = 0.35,
    min_travel_m: float = 0.0,
) -> list[np.ndarray]:
    """Return up to ``k`` planar explore targets for multi-goal A*.

    MuJoCo / Dynamem: graph frontiers (region-ranked) then top heuristic cells from
    ``sample_exploration``, then ``sample_frontier`` retries. Habitat navmesh clients
    return an empty list — they keep using ``pick_uncovered_explore_target``.
    """
    k = max(1, int(k))
    blocked_set = blocked if blocked is not None else getattr(agent, "_habitat_blocked_goals", None)
    if blocked_set is None:
        blocked_set = set()
    recent = list(recent_goals or getattr(agent, "_habitat_recent_goals", None) or [])
    robot = getattr(agent, "robot", None)
    if robot is not None and is_habitat_robot_client(robot):
        return []

    if min_travel_m <= 0.0:
        min_travel_m = explore_min_travel_m(agent)
    robot_xy = robot_planar_xy(robot) if robot is not None else (0.0, 0.0)

    out: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()

    def _try_add(raw: Any) -> bool:
        if len(out) >= k:
            return False
        xyz = _as_xyz1(raw)
        if xyz is None:
            return False
        if min_travel_m > 0.0 and _planar_dist((float(xyz[0]), float(xyz[1])), robot_xy) < min_travel_m:
            return False
        accepted = _mujoco_accept_explore_xy(xyz, blocked=blocked_set, recent=recent)
        if accepted is None:
            return False
        key = _dedup_key(accepted, grid_m=dedup_m)
        if key in seen:
            return False
        # Also skip exact blocked keys already rounded by goal_key_xy
        if goal_key_xy(accepted) in blocked_set:
            return False
        seen.add(key)
        out.append(accepted)
        return True

    for seed in seeds or []:
        _try_add(seed)

    gm = getattr(agent, "graph_memory", None)
    if gm is not None:
        nodes = [n for n in gm.get_nodes() if getattr(n, "is_frontier", False)]
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
            if len(out) >= k:
                break
            _try_add(np.array([float(node.xyz[0]), float(node.xyz[1]), 1.0], dtype=float))

    if hasattr(agent, "_best_frontier_point_from_graph") and len(out) < k:
        _try_add(agent._best_frontier_point_from_graph(question))

    space = getattr(agent, "space", None)
    planner = getattr(agent, "planner", None)
    if space is not None and planner is not None and robot is not None and len(out) < k:
        start = agent._planning_base_xyt(robot.get_base_pose())
        if hasattr(space, "sample_exploration") and hasattr(space, "voxel_map"):
            try:
                _index, _th, _ah, total_heuristics = space.sample_exploration(
                    start, planner, text=question, debug=False
                )
                th = np.asarray(total_heuristics, dtype=np.float64)
                if th.ndim == 2 and th.size > 0:
                    order = np.argsort(th.ravel())[::-1]
                    vm = space.voxel_map
                    for fi in order:
                        if len(out) >= k:
                            break
                        if th.ravel()[fi] <= 0.0:
                            break
                        ii, jj = np.unravel_index(int(fi), th.shape)
                        xyt = vm.grid_coords_to_xyt(torch.tensor([float(ii), float(jj)]))
                        _try_add(xyt)
            except Exception:
                pass
        if hasattr(space, "sample_frontier"):
            for _ in range(max(0, k - len(out) + 2)):
                if len(out) >= k:
                    break
                fr = space.sample_frontier(planner, start, text=question)
                if fr is None:
                    break
                _try_add(fr)

    return out

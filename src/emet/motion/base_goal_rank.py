# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Multi-goal base XY planning on a 2D navigable grid (one shared search).

Prefer this over K independent RRT plans for frontier / multi-option base nav:
expand once from the start until the nearest reachable goal is hit.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from emet.motion.voxel_arm_collision import GridConvention, world_xy_to_grid


def _neighbors8(pt: tuple[int, int]) -> list[tuple[int, int]]:
    i, j = pt
    return [(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if not (di == 0 and dj == 0)]


def _euclid(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


@dataclass(frozen=True)
class MultiGoalPlanResult:
    """Result of a single multi-goal grid search."""

    success: bool
    goal_index: int | None = None
    path_ij: list[tuple[int, int]] = field(default_factory=list)
    path_xy: list[np.ndarray] = field(default_factory=list)
    cost: float | None = None
    reason: str | None = None
    # Per-goal reachability from the same search (index, reachable, path_cost or None).
    goal_scores: list[tuple[int, bool, float | None]] = field(default_factory=list)


def plan_grid_multi_goal(
    start_ij: tuple[int, int],
    goal_ijs: Sequence[tuple[int, int] | None],
    *,
    navigable: np.ndarray,
    stop_at_first: bool = True,
) -> MultiGoalPlanResult:
    """A* from ``start_ij`` toward a set of goal cells on a boolean navigable grid.

    ``navigable[i, j]`` is True for free cells. Invalid / None goals are scored unreachable.
    If ``stop_at_first``, returns as soon as the nearest (by path cost) goal is reached;
    otherwise continues until the open set is empty so every goal is classified.
    """
    nav = np.asarray(navigable, dtype=bool)
    h, w = nav.shape[:2]

    def in_bounds(p: tuple[int, int]) -> bool:
        return 0 <= p[0] < h and 0 <= p[1] < w

    def free(p: tuple[int, int]) -> bool:
        return in_bounds(p) and bool(nav[p[0], p[1]])

    scores: list[tuple[int, bool, float | None]] = []
    goal_to_indices: dict[tuple[int, int], list[int]] = {}
    for i, g in enumerate(goal_ijs):
        if g is None or not free(g):
            scores.append((i, False, None))
            continue
        goal_to_indices.setdefault(g, []).append(i)
        scores.append((i, False, None))  # filled in when reached

    if not free(start_ij):
        return MultiGoalPlanResult(False, reason="invalid_start", goal_scores=scores)
    if not goal_to_indices:
        return MultiGoalPlanResult(False, reason="no_valid_goals", goal_scores=scores)

    pending = set(goal_to_indices.keys())

    def h_multi(p: tuple[int, int]) -> float:
        # Nearest-goal heuristic when seeking any goal; Dijkstra when classifying all.
        if not stop_at_first or not pending:
            return 0.0
        return min(_euclid(p, g) for g in pending)

    # (priority, counter, cell) — counter breaks ties for heapq.
    counter = 0
    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    heapq.heappush(open_heap, (h_multi(start_ij), counter, start_ij))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_ij: None}
    cost_so_far: dict[tuple[int, int], float] = {start_ij: 0.0}
    reached_goal_pt: tuple[int, int] | None = None

    while open_heap and pending:
        _prio, _, current = heapq.heappop(open_heap)
        if current in pending:
            c = cost_so_far[current]
            for idx in goal_to_indices[current]:
                scores[idx] = (idx, True, c)
            pending.discard(current)
            if reached_goal_pt is None:
                reached_goal_pt = current
            if stop_at_first:
                break
            # Fall through: keep expanding so other goals can be classified.

        for nxt in _neighbors8(current):
            if not free(nxt):
                continue
            step = _euclid(current, nxt)
            new_cost = cost_so_far[current] + step
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + h_multi(nxt)
                counter += 1
                heapq.heappush(open_heap, (priority, counter, nxt))
                came_from[nxt] = current

    if reached_goal_pt is None:
        return MultiGoalPlanResult(False, reason="no_reachable_goal", goal_scores=scores)

    # Reconstruct path to the first (nearest) reached goal.
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = reached_goal_pt
    while cur is not None:
        path.append(cur)
        cur = came_from.get(cur)
    path.reverse()
    goal_index = goal_to_indices[reached_goal_pt][0]
    return MultiGoalPlanResult(
        True,
        goal_index=goal_index,
        path_ij=path,
        cost=float(cost_so_far[reached_goal_pt]),
        goal_scores=scores,
    )


def plan_xy_multi_goal(
    start_xy: np.ndarray | Sequence[float],
    goals_xy: Sequence[np.ndarray | Sequence[float]],
    *,
    navigable: np.ndarray,
    grid_origin: np.ndarray | Sequence[float],
    resolution: float,
    convention: GridConvention = "grid_params",
    stop_at_first: bool = True,
    ij_to_xy: Callable[[tuple[int, int]], np.ndarray] | None = None,
) -> MultiGoalPlanResult:
    """World-frame multi-goal plan using ``world_xy_to_grid`` indexing."""
    go = np.asarray(grid_origin, dtype=np.float64).reshape(-1)[:2]
    res = float(resolution)
    start = np.asarray(start_xy, dtype=np.float64).reshape(2)
    start_ij = world_xy_to_grid(float(start[0]), float(start[1]), grid_origin=go, resolution=res, convention=convention)

    goal_ijs: list[tuple[int, int] | None] = []
    for g in goals_xy:
        gg = np.asarray(g, dtype=np.float64).reshape(2)
        goal_ijs.append(
            world_xy_to_grid(float(gg[0]), float(gg[1]), grid_origin=go, resolution=res, convention=convention)
        )

    result = plan_grid_multi_goal(start_ij, goal_ijs, navigable=navigable, stop_at_first=stop_at_first)
    if not result.success:
        return result

    def _default_ij_to_xy(ij: tuple[int, int]) -> np.ndarray:
        # Inverse of grid_params: world = (ij - origin) * res
        if convention == "grid_params":
            return np.array([(ij[0] - go[0]) * res, (ij[1] - go[1]) * res], dtype=np.float64)
        return np.array([ij[0] * res + go[0], ij[1] * res + go[1]], dtype=np.float64)

    to_xy = ij_to_xy or _default_ij_to_xy
    path_xy = [to_xy(ij) for ij in result.path_ij]
    # Snap endpoints to the exact requested start / chosen goal XY.
    if path_xy:
        path_xy[0] = start.copy()
        gi = int(result.goal_index) if result.goal_index is not None else -1
        if 0 <= gi < len(goals_xy):
            path_xy[-1] = np.asarray(goals_xy[gi], dtype=np.float64).reshape(2).copy()
    return MultiGoalPlanResult(
        True,
        goal_index=result.goal_index,
        path_ij=result.path_ij,
        path_xy=path_xy,
        cost=result.cost,
        goal_scores=result.goal_scores,
    )


def choose_first_reachable(
    scores: Sequence[tuple[int, bool, str]] | Sequence[tuple[int, bool, float | None]],
) -> int | None:
    """Return the first reachable goal index from score tuples, or None."""
    for row in scores:
        idx, ok = int(row[0]), bool(row[1])
        if ok:
            return idx
    return None


def rank_xy_goals_by_plan(
    start_xy: np.ndarray | Sequence[float],
    goals_xy: Sequence[np.ndarray | Sequence[float]],
    *,
    navigable: np.ndarray,
    grid_origin: np.ndarray | Sequence[float],
    resolution: float,
    convention: GridConvention = "grid_params",
    **_unused,
) -> list[tuple[int, bool, str]]:
    """Classify each goal via one multi-goal grid search (reachable-first sort).

    ``**_unused`` absorbs legacy RRT kwargs (planner, max_iter, …) so old call sites
    fail soft into the grid path.
    """
    result = plan_xy_multi_goal(
        start_xy,
        goals_xy,
        navigable=navigable,
        grid_origin=grid_origin,
        resolution=resolution,
        convention=convention,
        stop_at_first=False,
    )
    scored: list[tuple[int, bool, str]] = []
    for idx, ok, cost in result.goal_scores:
        if ok:
            scored.append((idx, True, f"cost={cost:.3f}" if cost is not None else "planned"))
        else:
            scored.append((idx, False, "unreachable"))
    # Prefer the multi-goal winner first, then other reachables by index.
    winner = result.goal_index
    scored.sort(key=lambda t: (0 if (winner is not None and t[0] == winner and t[1]) else (0 if t[1] else 1), t[0]))
    return scored

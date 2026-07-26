# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Rank base XY(T) goals by motion-plan feasibility (frontier multi-option smoke)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from emet.motion.algo import get_planner
from emet.motion.base import ConfigurationSpace


def rank_xy_goals_by_plan(
    start_xy: np.ndarray | Sequence[float],
    goals_xy: Sequence[np.ndarray | Sequence[float]],
    *,
    is_valid: Callable[[np.ndarray], bool],
    planner: str = "rrt_connect",
    step_size: float = 0.12,
    max_iter: int = 800,
    goal_tolerance: float = 0.15,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    seed: int | None = 0,
) -> list[tuple[int, bool, str]]:
    """Score each goal: ``(index, reachable, reason)`` sorted reachable-first.

    *is_valid* must accept a length-2 XY state (same contract as base RRT validate_fn).
    """
    start = np.asarray(start_xy, dtype=np.float64).reshape(2)
    if not is_valid(start):
        raise ValueError("start_xy is not valid under is_valid")

    if bounds is None:
        pts = [start] + [np.asarray(g, dtype=np.float64).reshape(2) for g in goals_xy]
        stack = np.vstack(pts)
        pad = 1.0
        mins = stack.min(axis=0) - pad
        maxs = stack.max(axis=0) + pad
    else:
        mins, maxs = bounds

    space = ConfigurationSpace(
        2, mins=np.asarray(mins, dtype=np.float64), maxs=np.asarray(maxs, dtype=np.float64), step_size=float(step_size)
    )
    if seed is not None:
        np.random.seed(int(seed))
    planner_obj: Any = get_planner(
        str(planner),
        space,
        is_valid,
        max_iter=int(max_iter),
        goal_tolerance=float(goal_tolerance),
    )

    scored: list[tuple[int, bool, str]] = []
    for i, g in enumerate(goals_xy):
        goal = np.asarray(g, dtype=np.float64).reshape(2)
        if not is_valid(goal):
            scored.append((i, False, "goal_invalid"))
            continue
        result = planner_obj.plan(start, goal, verbose=False)
        if bool(getattr(result, "success", False)):
            scored.append((i, True, "planned"))
        else:
            scored.append((i, False, str(getattr(result, "reason", None) or "plan_failed")))
    scored.sort(key=lambda t: (not t[1], t[0]))
    return scored


def choose_first_reachable(scores: Sequence[tuple[int, bool, str]]) -> int | None:
    """Return the first reachable goal index, or None."""
    for idx, ok, _reason in scores:
        if ok:
            return int(idx)
    return None

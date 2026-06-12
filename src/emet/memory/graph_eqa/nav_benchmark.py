# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""GT navigation and frontier-exploration scoring for Dynagraph sim benchmarks."""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def dist_xy(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.hypot(float(aa[0]) - float(bb[0]), float(aa[1]) - float(bb[1])))


def find_gt_target_xy(
    placements: dict[str, dict[str, Any]],
    query: str,
    *,
    body_key: str | None = None,
) -> tuple[str, np.ndarray] | None:
    """Pick a GT body / category best matching a natural-language nav query."""
    if body_key and body_key in placements:
        pos = np.asarray(placements[body_key]["pos"], dtype=np.float64).reshape(3)
        return str(body_key), pos

    tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2]
    if not tokens:
        return None

    best: tuple[str, np.ndarray] | None = None
    best_score = 0
    for body, info in placements.items():
        cat = str(info.get("cat") or body).lower()
        blob = f"{body.lower()} {cat}"
        score = sum(1 for tok in tokens if tok in blob)
        if score > best_score:
            best_score = score
            best = (str(body), np.asarray(info["pos"], dtype=np.float64).reshape(3))
    return best if best_score > 0 else None


def score_nav_toward_target(
    start_xy: np.ndarray | list[float],
    end_xy: np.ndarray | list[float],
    target_xy: np.ndarray | list[float],
    *,
    min_improvement_m: float = 0.08,
) -> dict[str, Any]:
    """Return whether base XY moved closer to a GT target in the planning frame."""
    d0 = dist_xy(start_xy, target_xy)
    d1 = dist_xy(end_xy, target_xy)
    delta = d0 - d1
    return {
        "dist_start_m": d0,
        "dist_end_m": d1,
        "delta_m": delta,
        "improved": bool(delta >= min_improvement_m),
        "reached": bool(d1 <= max(min_improvement_m * 2.0, 0.35)),
    }


def score_explore_metrics(
    *,
    n_success: int,
    n_iters: int,
    explored_area_m2: float | None = None,
    frontier_nodes: int | None = None,
) -> dict[str, Any]:
    return {
        "explore_successes": int(n_success),
        "explore_iterations": int(n_iters),
        "explored_area_m2": explored_area_m2,
        "frontier_nodes": frontier_nodes,
        "pass": bool(n_success >= 1 and (explored_area_m2 is None or explored_area_m2 > 0.0)),
    }

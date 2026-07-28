# Copyright (c) Chris Paxton 2026

"""Place-card approach sampling + local frontier completeness for agentic investigate.

Approach goals are random navigable floor samples in an annulus around the object
(probabilistically covering viewpoints). Completeness uses the object's XY footprint
vs the unexplored frontier mask: if the dilated footprint no longer touches frontier,
local geometry is ``coverage=closed`` and further orbits are low value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

DEFAULT_FOOTPRINT_HALF_M = 0.60
DEFAULT_ANNULUS_INNER_M = 0.45
DEFAULT_ANNULUS_OUTER_M = 1.60
DEFAULT_FOOTPRINT_DILATE_M = 0.75
DEFAULT_AVOID_XY_M = 0.55
DEFAULT_CANDIDATE_DRAWS = 48


@dataclass(frozen=True)
class PlaceFootprint:
    """Axis-aligned planar footprint for a place card (meters, world XY)."""

    cx: float
    cy: float
    half_x: float
    half_y: float

    @property
    def min_xy(self) -> tuple[float, float]:
        return (self.cx - self.half_x, self.cy - self.half_y)

    @property
    def max_xy(self) -> tuple[float, float]:
        return (self.cx + self.half_x, self.cy + self.half_y)


@dataclass(frozen=True)
class PlaceCoverage:
    """Local observation completeness relative to the unexplored frontier."""

    status: str  # open | closed | unknown
    local_frontier_cells: int = 0

    @property
    def complete(self) -> bool:
        return self.status == "closed"


def footprint_from_node(
    node: Any,
    *,
    default_half_m: float = DEFAULT_FOOTPRINT_HALF_M,
) -> PlaceFootprint | None:
    """Build a planar footprint from graph node bounds / extent / xyz."""
    if node is None:
        return None
    xyz = getattr(node, "xyz", None)
    if xyz is None:
        return None
    arr = np.asarray(xyz, dtype=float).reshape(-1)
    if arr.size < 2:
        return None
    cx, cy = float(arr[0]), float(arr[1])
    bounds = getattr(node, "bounds_3d", None)
    if isinstance(bounds, dict) and "min" in bounds and "max" in bounds:
        lo = np.asarray(bounds["min"], dtype=float).reshape(-1)
        hi = np.asarray(bounds["max"], dtype=float).reshape(-1)
        if lo.size >= 2 and hi.size >= 2:
            hx = max(0.15, 0.5 * abs(float(hi[0]) - float(lo[0])))
            hy = max(0.15, 0.5 * abs(float(hi[1]) - float(lo[1])))
            return PlaceFootprint(
                cx=0.5 * (float(lo[0]) + float(hi[0])),
                cy=0.5 * (float(lo[1]) + float(hi[1])),
                half_x=hx,
                half_y=hy,
            )
    extent = getattr(node, "extent_half", None)
    if extent is not None:
        ext = np.asarray(extent, dtype=float).reshape(-1)
        if ext.size >= 2:
            return PlaceFootprint(
                cx=cx,
                cy=cy,
                half_x=max(0.15, float(ext[0])),
                half_y=max(0.15, float(ext[1])),
            )
    half = max(0.15, float(default_half_m))
    return PlaceFootprint(cx=cx, cy=cy, half_x=half, half_y=half)


def footprint_from_xyz(
    xyz: Any,
    *,
    default_half_m: float = DEFAULT_FOOTPRINT_HALF_M,
) -> PlaceFootprint | None:
    arr = np.asarray(xyz, dtype=float).reshape(-1)
    if arr.size < 2:
        return None
    half = max(0.15, float(default_half_m))
    return PlaceFootprint(cx=float(arr[0]), cy=float(arr[1]), half_x=half, half_y=half)


def _as_bool(mask: Any) -> np.ndarray:
    if hasattr(mask, "detach"):
        return np.asarray(mask.detach().cpu().numpy(), dtype=bool)
    return np.asarray(mask, dtype=bool)


def count_frontier_in_footprint(
    footprint: PlaceFootprint,
    frontier: np.ndarray,
    *,
    xy_to_ij: Callable[[float, float], tuple[int, int] | None],
    dilate_m: float = DEFAULT_FOOTPRINT_DILATE_M,
    resolution_m: float = 0.1,
) -> int:
    """Count unexplored-frontier cells inside the dilated footprint AABB."""
    fr = _as_bool(frontier)
    if fr.size == 0:
        return 0
    h, w = fr.shape[:2]
    pad = max(0.0, float(dilate_m))
    x0, y0 = footprint.min_xy
    x1, y1 = footprint.max_xy
    x0 -= pad
    y0 -= pad
    x1 += pad
    y1 += pad
    step = max(0.05, float(resolution_m))
    cells: set[tuple[int, int]] = set()
    x = x0
    while x <= x1 + 1e-9:
        y = y0
        while y <= y1 + 1e-9:
            ij = xy_to_ij(float(x), float(y))
            if ij is not None:
                i, j = int(ij[0]), int(ij[1])
                if 0 <= i < h and 0 <= j < w:
                    cells.add((i, j))
            y += step
        x += step
    return sum(1 for i, j in cells if fr[i, j])


def coverage_from_frontier_count(n_frontier: int | None) -> PlaceCoverage:
    if n_frontier is None:
        return PlaceCoverage(status="unknown", local_frontier_cells=0)
    n = int(n_frontier)
    if n > 0:
        return PlaceCoverage(status="open", local_frontier_cells=n)
    return PlaceCoverage(status="closed", local_frontier_cells=0)


def _min_dist_xy(
    xy: tuple[float, float],
    others: Sequence[tuple[float, float]] | None,
) -> float:
    if not others:
        return float("inf")
    return min(math.hypot(xy[0] - o[0], xy[1] - o[1]) for o in others)


def sample_annulus_approach_xy(
    *,
    anchor_xy: tuple[float, float],
    robot_xy: tuple[float, float] | None,
    obstacles: np.ndarray | None,
    reachable: np.ndarray | None,
    frontier: np.ndarray | None,
    footprint: PlaceFootprint | None,
    xy_to_ij: Callable[[float, float], tuple[int, int] | None],
    ij_to_xy: Callable[[int, int], tuple[float, float] | None],
    avoid_xy: Sequence[tuple[float, float]] | None = None,
    radius_inner_m: float = DEFAULT_ANNULUS_INNER_M,
    radius_outer_m: float = DEFAULT_ANNULUS_OUTER_M,
    avoid_m: float = DEFAULT_AVOID_XY_M,
    n_draws: int = DEFAULT_CANDIDATE_DRAWS,
    rng: np.random.Generator | None = None,
    approach_index: int = 0,
) -> tuple[float, float] | None:
    """Sample a navigable floor XY in an annulus; prefer frontier-adjacent samples.

    Deterministic when ``rng`` is omitted (seeded from anchor + approach_index).
    Falls back to classic standoff when no free cell is found.
    """
    ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
    if rng is None:
        seed = (int(round(ax * 100)) * 1009 + int(round(ay * 100)) * 9176 + int(approach_index) * 131) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

    r_in = max(0.2, float(radius_inner_m))
    r_out = max(r_in + 0.15, float(radius_outer_m))
    obs = _as_bool(obstacles) if obstacles is not None else None
    reach = _as_bool(reachable) if reachable is not None else None
    fr = _as_bool(frontier) if frontier is not None else None
    h = w = 0
    if obs is not None:
        h, w = obs.shape[:2]
    elif reach is not None:
        h, w = reach.shape[:2]
    elif fr is not None:
        h, w = fr.shape[:2]

    candidates: list[tuple[float, float, float]] = []  # score, x, y
    for _ in range(max(8, int(n_draws))):
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        rad = float(math.sqrt(rng.uniform(r_in * r_in, r_out * r_out)))
        x = ax + rad * math.cos(ang)
        y = ay + rad * math.sin(ang)
        if _min_dist_xy((x, y), avoid_xy) < float(avoid_m):
            continue
        if h > 0 and w > 0:
            ij = xy_to_ij(x, y)
            if ij is None:
                continue
            i, j = int(ij[0]), int(ij[1])
            if not (0 <= i < h and 0 <= j < w):
                continue
            if obs is not None and obs[i, j]:
                continue
            if reach is not None and not reach[i, j]:
                continue
            # Snap to cell center when we have a grid.
            snapped = ij_to_xy(i, j)
            if snapped is not None:
                x, y = float(snapped[0]), float(snapped[1])
            if _min_dist_xy((x, y), avoid_xy) < float(avoid_m):
                continue
        # Score: prefer near local frontier / footprint ring; slight preference for novelty.
        score = float(rng.random()) * 0.1
        if fr is not None and h > 0:
            ij = xy_to_ij(x, y)
            if ij is not None:
                i, j = int(ij[0]), int(ij[1])
                near_fr = 0
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and fr[ni, nj]:
                            near_fr += 1
                score += 2.0 * float(near_fr)
        if footprint is not None:
            # Prefer standing outside the object box looking in.
            if (
                footprint.min_xy[0] - 0.05 <= x <= footprint.max_xy[0] + 0.05
                and footprint.min_xy[1] - 0.05 <= y <= footprint.max_xy[1] + 0.05
            ):
                score -= 1.5
        candidates.append((score, x, y))

    if candidates:
        candidates.sort(key=lambda t: -t[0])
        return (candidates[0][1], candidates[0][2])

    # Fallback: classic standoff from current robot pose, else east of anchor.
    if robot_xy is not None:
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        dx, dy = ax - rx, ay - ry
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return (ax + r_in, ay)
        travel = dist if dist <= r_in else max(r_in, dist - r_in)
        return (rx + (dx / dist) * travel, ry + (dy / dist) * travel)
    return (ax + r_in, ay)


def make_grid_converters(
    voxel_map: Any,
) -> tuple[
    Callable[[float, float], tuple[int, int] | None],
    Callable[[int, int], tuple[float, float] | None],
    float,
] | None:
    """Build xy↔ij helpers from a voxel map (returns None if unavailable)."""
    if voxel_map is None or not hasattr(voxel_map, "xy_to_grid_coords"):
        return None
    res = float(getattr(voxel_map, "grid_resolution", 0.1) or 0.1)

    def xy_to_ij(x: float, y: float) -> tuple[int, int] | None:
        try:
            gc = voxel_map.xy_to_grid_coords(np.array([x, y], dtype=float))
        except Exception:
            return None
        if gc is None:
            return None
        if hasattr(gc, "detach"):
            gc = gc.detach().cpu().numpy()
        arr = np.asarray(gc, dtype=float).reshape(-1)
        if arr.size < 2 or not np.isfinite(arr[:2]).all():
            return None
        return (int(round(float(arr[0]))), int(round(float(arr[1]))))

    def ij_to_xy(i: int, j: int) -> tuple[float, float] | None:
        try:
            xy = voxel_map.grid_coords_to_xy(np.array([i, j], dtype=float))
        except Exception:
            return None
        if hasattr(xy, "detach"):
            xy = xy.detach().cpu().numpy()
        arr = np.asarray(xy, dtype=float).reshape(-1)
        if arr.size < 2:
            return None
        return (float(arr[0]), float(arr[1]))

    return xy_to_ij, ij_to_xy, res

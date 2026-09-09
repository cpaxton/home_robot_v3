# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""TAMP clutter-clearance benchmark: episode schema, scatter geometry, GT validity probe,
and metric aggregation.

A "clutter" episode starts the robot surrounded by small floor objects that must be
moved out of the way as part of a plan. Two modes:

* ``cleanup``  — relocate all scattered objects to a bin (e.g. "clean up the room").
* ``nav_goal`` — clear a path of objects to reach a sampled scene landmark (e.g.
  "get to the sofa" / "reach the fridge").

The offline/GT pieces here are deliberately dependency-light and unit-testable:
episode loading, ring-scatter target geometry, a clearance-aware A* validity probe
(whether the scattered clutter blocks the direct route), and aggregate-metric helpers.
Live execution (server launch, scatter via ``sim_set_body_pose``, the multi-object TAMP
chain, scoring) lives in ``scripts/eval_tamp_clutter.py`` and
``emet.controller.task.tamp.clutter_chain``.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from emet.utils.logger import Logger

logger = Logger(__name__)

# New manip-mode seam added by the clutter benchmark on top of OVMM's modes.
CLUTTER_MANIP_MODES = ("latch", "sim", "attempt")

CLUTTER_MODES = ("cleanup", "nav_goal")

# Default manipulation mode per robot. rby1 (Galaxea) has a verified kinematic latch
# path; innate_mars / nori default to the **sim (teleport) oracle** because their floor
# reach is model-limited (Nori's arm bottoms out at z≈0.29 m — it cannot latch-grasp
# true-floor objects at z≈0.02 m). Use ``--manip-mode latch`` for a latch experiment.
ROBOT_DEFAULT_MANIP_MODE = {
    "rby1": "latch",
    "galaxea_r1": "latch",
    "innate_mars": "sim",
    "nori": "sim",
    "nori_a3": "sim",
    "stretch": "sim",
}

# Drop-receptacle aliases only (iTHOR names the garbage can body "ashcan ..."). Deliberately
# NOT falling back to furniture/appliances — a stove/cabinet drop would not be "cleanup".
# Shared by the runner and the TAMP chain so bin resolution + scatter exclusion cannot drift.
BIN_FALLBACKS = (
    "ashcan",
    "garbagecan",
    "trashcan",
    "basket",
)

# Static furniture landmarks used to rotate nav-goal goals across the episode set.
NAV_GOAL_LANDMARKS = ("Sofa", "Fridge", "Table", "Bed", "Counter", "Desk")

# Teleport-chord collision samples. 2 cm is well below the 0.22 m nav footprint so a
# thin object cannot sit between two checks; numpy cost is negligible vs ZMQ.
NAV_CHORD_STEP_M = 0.02


@dataclass(frozen=True)
class ClutterEpisode:
    """One TAMP clutter-clearance episode.

    ``goal_landmark`` is a static-furniture category to navigate to (``nav_goal``
    mode); ``None``/``auto`` samples a landmark from the scene. ``bin_query`` is the
    receptacle category for relocated objects (``cleanup`` mode).

    ``clutter`` and ``episode_valid`` are populated by the GT offline generator at
    problem-set build time (fixed deterministic episodes) or at eval time on the live
    scene; they are optional in the YAML.
    """

    id: str
    tier: str
    sim: str  # scene config yaml (e.g. configs/sim/molmospaces_ithor_train_0.yaml)
    robot: str  # rby1 | stretch | innate_mars | …
    mode: str  # cleanup | nav_goal
    n_objects: int
    success_radius_m: float = 0.5
    scatter_radius_m: float = 0.8
    floor_z_m: float = 0.02
    goal_landmark: str | None = None
    bin_query: str | None = None
    manip_mode: str | None = None  # None → ROBOT_DEFAULT_MANIP_MODE[robot]
    backend: str = "dynagraph"
    seed: int | None = None
    # Closed (deterministic blocking) ring: minimal scatter angle/radius jitter.
    tight_ring: bool = False
    # Scene index override (MolmoSpaces iTHOR FloorPlan index; replaces sim_cfg.index).
    scene_index: int | None = None
    clutter: tuple[dict[str, Any], ...] = ()
    robot_start_xy: tuple[float, float] | None = None
    goal_xy: tuple[float, float] | None = None
    episode_valid: bool | None = None

    def __post_init__(self) -> None:
        if self.mode not in CLUTTER_MODES:
            raise ValueError(f"invalid mode={self.mode!r} in {self.id} (expected {CLUTTER_MODES})")
        if int(self.n_objects) < 0:
            raise ValueError(f"n_objects must be >= 0 in {self.id} (0 = no clutter / pure nav)")
        mode = self.manip_mode or ROBOT_DEFAULT_MANIP_MODE.get(self.robot.lower(), "sim")
        if mode not in CLUTTER_MANIP_MODES:
            raise ValueError(f"invalid manip_mode={mode!r} in {self.id}")

    def resolved_manip_mode(self) -> str:
        return self.manip_mode or ROBOT_DEFAULT_MANIP_MODE.get(self.robot.lower(), "sim")

    def phrase(self) -> str:
        """Natural-language command phrasing for this episode (LLM-agent mode)."""
        if self.mode == "cleanup":
            return f"clean up the room by putting the scattered objects into {self.bin_query or 'a bin'}"
        landmark = self.goal_landmark or "the landmark"
        return f"get to {landmark}"


def load_clutter_episodes(path: str | Path) -> list[ClutterEpisode]:
    """Load clutter episodes from a YAML registry (``episodes:`` list)."""
    full = Path(path).expanduser().resolve()
    with full.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    rows = raw.get("episodes") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError(f"expected list under 'episodes' in {full}")
    out: list[ClutterEpisode] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        clutter = tuple(dict(c) for c in (row.get("clutter") or []))
        out.append(
            ClutterEpisode(
                id=str(row["id"]),
                tier=str(row.get("tier", "")),
                sim=str(row["sim"]),
                robot=str(row.get("robot", "rby1")),
                mode=str(row["mode"]),
                n_objects=int(row["n_objects"]),
                success_radius_m=float(row.get("success_radius_m", 0.5)),
                scatter_radius_m=float(row.get("scatter_radius_m", 0.8)),
                floor_z_m=float(row.get("floor_z_m", 0.02)),
                goal_landmark=(str(row["goal_landmark"]) if row.get("goal_landmark") else None),
                bin_query=(str(row["bin_query"]) if row.get("bin_query") else None),
                manip_mode=(str(row["manip_mode"]) if row.get("manip_mode") else None),
                backend=str(row.get("backend", "dynagraph")),
                seed=(int(row["seed"]) if row.get("seed") is not None else None),
                tight_ring=bool(row.get("tight_ring", False)),
                scene_index=(int(row["scene_index"]) if row.get("scene_index") is not None else None),
                clutter=clutter,
                robot_start_xy=tuple(float(x) for x in row["robot_start_xy"]) if row.get("robot_start_xy") else None,
                goal_xy=tuple(float(x) for x in row["goal_xy"]) if row.get("goal_xy") else None,
                episode_valid=(bool(row["episode_valid"]) if row.get("episode_valid") is not None else None),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Scatter geometry (pure / testable)
# ---------------------------------------------------------------------------


def _clamp_xy(xy: np.ndarray) -> np.ndarray:
    return np.asarray(xy, dtype=np.float64).reshape(-1)[:2]


def ring_angle_offsets(
    n: int,
    *,
    rng: np.random.Generator,
    jitter_rad: float = 0.35,
) -> np.ndarray:
    """Evenly spaced ring angles with jitter, offset so one points at the goal."""
    base = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    return base + rng.uniform(-jitter_rad, jitter_rad, size=int(n))


def scatter_ring_targets(
    robot_xy: np.ndarray | list,
    goal_xy: np.ndarray | list | None,
    n: int,
    *,
    radius_m: float,
    rng: np.random.Generator,
    radius_jitter: float = 0.15,
    angle_jitter_rad: float = 0.35,
) -> list[np.ndarray]:
    """Return ``n`` floor target positions in a ring around the robot.

    The ring is biased toward the goal direction (nav_goal) so objects tend to sit
    between the robot and its goal; for cleanup (no goal) the ring is uniform.
    Radius is ``radius_m`` plus small jitter. ``angle_jitter_rad`` opens/closes gaps
    between ring slots — set near 0 for a deterministically closed (blocking) ring.
    Pure geometry — no sim required.
    """
    robot = _clamp_xy(robot_xy)
    n = max(1, int(n))
    angles = ring_angle_offsets(n, rng=rng, jitter_rad=angle_jitter_rad)
    if goal_xy is not None:
        goal = _clamp_xy(goal_xy)
        delta = goal - robot
        if float(np.linalg.norm(delta)) > 1e-6:
            # Rotate the first ring slot onto the goal bearing.
            goal_bearing = float(np.arctan2(delta[1], delta[0]))
            angles = angles - angles[0] + goal_bearing
    radii = np.ones(n) * float(radius_m) + rng.uniform(-float(radius_jitter), float(radius_jitter), size=n)
    radii = np.clip(radii, 0.2, None)
    out: list[np.ndarray] = []
    for a, r in zip(angles, radii, strict=True):
        out.append(robot + np.array([float(r) * np.cos(a), float(r) * np.sin(a)], dtype=np.float64))
    return out


# ---------------------------------------------------------------------------
# GT validity probe: is the direct route blocked by the scattered clutter?
# ---------------------------------------------------------------------------


def _shortest_path(occ: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """8-connected Dijkstra on a boolean occupancy grid (True = free)."""
    h, w = occ.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w and 0 <= goal[0] < h and 0 <= goal[1] < w):
        return None
    if not occ[start[0], start[1]] or not occ[goal[0], goal[1]]:
        return None
    dist = np.full((h, w), -1, dtype=np.int32)
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    dist[start[0], start[1]] = 0
    pq = [(0, start)]
    while pq:
        _, (r, c) = heapq.heappop(pq)
        if (r, c) == goal:
            path = [(r, c)]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return path
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and occ[nr, nc] and dist[nr, nc] < 0:
                    dist[nr, nc] = dist[r, c] + 1
                    parent[(nr, nc)] = (r, c)
                    heapq.heappush(pq, (dist[nr, nc], (nr, nc)))
    return None


def _path_exists(occ: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    return _shortest_path(occ, start, goal) is not None


def densify_polyline(points: Sequence[np.ndarray] | Sequence[list], step_m: float) -> list[np.ndarray]:
    """Insert samples so consecutive points are at most ``step_m`` apart."""
    pts = [_clamp_xy(p) for p in points]
    if not pts:
        return []
    step = max(1e-3, float(step_m))
    out: list[np.ndarray] = [pts[0]]
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        delta = b - a
        dist = float(np.linalg.norm(delta))
        n = max(1, int(math.ceil(dist / step)))
        for j in range(1, n + 1):
            out.append(a + delta * (j / n))
    return out


def first_footprint_hit(
    waypoints: Sequence[np.ndarray] | Sequence[list],
    disks: Sequence[tuple[np.ndarray | list, float, str]],
    *,
    footprint_r_m: float,
    skip_first: bool = True,
) -> dict[str, Any] | None:
    """Return the first waypoint (after start) whose footprint overlaps an obstacle disk."""
    fr = float(footprint_r_m)
    for i, raw in enumerate(waypoints):
        if skip_first and i == 0:
            continue
        xy = _clamp_xy(raw)
        for disk_xy, radius, name in disks:
            d = float(np.linalg.norm(xy - _clamp_xy(disk_xy)))
            if d < fr + float(radius):
                return {
                    "step": int(i),
                    "xy": xy.tolist(),
                    "hit": str(name),
                    "dist_m": d,
                    "clearance_m": fr + float(radius),
                }
    return None


def placement_obstacle_disks(
    placements: Mapping[str, Any],
    *,
    skip_bodies: Sequence[str] = (),
    default_radius_m: float = 0.08,
    max_center_z_m: float = 1.15,
    min_radius_m: float = 0.05,
    max_radius_m: float = 1.25,
) -> list[tuple[np.ndarray, float, str]]:
    """Planar obstacle disks from GT ``sim_object_placements`` (skip high/hanging bodies)."""
    skip = {str(b) for b in skip_bodies}
    out: list[tuple[np.ndarray, float, str]] = []
    for body, info in placements.items():
        if str(body) in skip or not isinstance(info, dict):
            continue
        pos = info.get("pos")
        if pos is None:
            continue
        xyz = np.asarray(pos, dtype=np.float64).reshape(-1)
        if xyz.size < 2:
            continue
        if xyz.size >= 3 and float(xyz[2]) > float(max_center_z_m):
            continue
        radius = float(default_radius_m)
        bounds = info.get("bounds")
        if bounds is not None:
            b = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
            half = 0.5 * (b[1, :2] - b[0, :2])
            radius = float(np.hypot(float(half[0]), float(half[1])))
        radius = min(float(max_radius_m), max(float(min_radius_m), radius))
        out.append((_clamp_xy(xyz), radius, str(body)))
    return out


def bodies_near_xy(
    placements: Mapping[str, Any],
    xy: np.ndarray | list,
    *,
    keepout_m: float = 0.75,
) -> list[str]:
    """Bodies whose XY center is within ``keepout_m`` of ``xy`` (landmark near the approach)."""
    target = _clamp_xy(xy)
    lim = float(keepout_m)
    names: list[str] = []
    for body, info in placements.items():
        if not isinstance(info, dict) or info.get("pos") is None:
            continue
        if float(np.linalg.norm(_clamp_xy(info["pos"]) - target)) <= lim:
            names.append(str(body))
    return names


def clutter_blocks_path(
    robot_xy: np.ndarray | list,
    goal_xy: np.ndarray | list | None,
    obj_xy: list[np.ndarray] | list[list],
    *,
    grid_res_m: float = 0.05,
    clearance_m: float = 0.22,
    obj_radius_m: float = 0.08,
    pad_cells: int = 4,
) -> tuple[bool, dict[str, Any]]:
    """GT validity probe: do the scattered objects block a path from robot to goal?

    Builds a synthetic occupancy grid around the robot/goal, marks the scattered
    objects as obstacle disks dilated by ``clearance_m + obj_radius_m`` (mirroring the
    nav ``min_clearance_m`` hard gate), then 8-connected A*/BFS. Returns
    ``(blocked, info)`` where ``blocked=True`` means **no** free path exists
    (the episode requires manipulation to proceed).

    ``obj_xy`` is the set of scatter target positions; ``goal_xy`` may be ``None``
    (cleanup: validity instead requires at least one object near the robot).
    """
    robot = _clamp_xy(robot_xy)
    objs = [np.asarray(o, dtype=np.float64).reshape(2) for o in obj_xy]
    if goal_xy is None:
        # Cleanup: no nav goal; validity means the robot is actually surrounded.
        near_max_m = 1.5
        near = sum(1 for o in objs if float(np.linalg.norm(o - robot)) <= near_max_m)
        info: dict[str, Any] = {"probe": "cleanup_near", "n_near": int(near), "blocked": bool(near >= 1)}
        return bool(near >= 1), info

    goal = _clamp_xy(goal_xy)

    all_pts = [robot, goal, *objs]
    xs = [float(p[0]) for p in all_pts]
    ys = [float(p[1]) for p in all_pts]
    margin = max(1.0, float(clearance_m) + float(obj_radius_m) + 2 * float(grid_res_m) * pad_cells)
    x0, x1 = float(min(xs)) - margin, float(max(xs)) + margin
    y0, y1 = float(min(ys)) - margin, float(max(ys)) + margin
    res = float(grid_res_m)
    w = max(3, int(math.ceil((x1 - x0) / res)))
    h = max(3, int(math.ceil((y1 - y0) / res)))

    occ = np.ones((h, w), dtype=bool)
    inflate = int(math.ceil((float(clearance_m) + float(obj_radius_m)) / res))

    def to_cell(p: np.ndarray) -> tuple[int, int]:
        # Return (row, col) = (y index, x index) to match occ[r, c] indexing.
        col = int(round((float(p[0]) - x0) / res))
        row = int(round((float(p[1]) - y0) / res))
        return row, col

    for o in objs:
        cr, cc = to_cell(o)
        for dr in range(-inflate, inflate + 1):
            for dc in range(-inflate, inflate + 1):
                r, c = cr + dr, cc + dc
                if 0 <= r < h and 0 <= c < w and (dr * dr + dc * dc) <= inflate * inflate:
                    occ[r, c] = False

    # Objects too close to the goal/robot would mark start/goal occupied; relax the
    # immediate neighbors only if the target cell itself is free.
    start_cell = to_cell(robot)
    goal_cell = to_cell(goal)
    for cell in (start_cell, goal_cell):
        r, c = cell
        if 0 <= r < h and 0 <= c < w and not occ[r, c]:
            occ[r, c] = True

    reachable = _path_exists(occ, start_cell, goal_cell)
    blocked = not reachable
    info = {
        "probe": "gt_nav",
        "grid_res_m": res,
        "clearance_m": float(clearance_m),
        "obj_radius_m": float(obj_radius_m),
        "n_objects": int(len(objs)),
        "start_xy": robot.tolist(),
        "goal_xy": goal.tolist(),
        "blocked": bool(blocked),
    }
    return blocked, info


def nav_path_open_around_disks(
    robot_xy: np.ndarray | list,
    goal_xy: np.ndarray | list | None,
    disks: Sequence[tuple[np.ndarray | list, float, str]],
    *,
    grid_res_m: float = 0.05,
    clearance_m: float = 0.22,
    pad_cells: int = 4,
) -> tuple[bool, dict[str, Any]]:
    """8-connected path exists from robot to goal around obstacle disks.

    This is the **post-clear** analogue of :func:`clutter_blocks_path`: it routes
    around *all* obstacle disks (static furniture + leftover clutter) instead of the
    straight-line teleport chord. A teleport nav cannot follow a curved path, so a
    benchmark that wants ``nav_goal`` to mean ``cleared a path, then reach the
    landmark`` must score the route the GT planner would take, not the chord.
    Returns ``(path_open, info)``.
    """
    if goal_xy is None:
        return True, {"probe": "no_goal", "blocked": False}
    robot = _clamp_xy(robot_xy)
    goal = _clamp_xy(goal_xy)
    parsed: list[tuple[np.ndarray, float, str]] = [
        (_clamp_xy(xy), float(radius), str(name)) for xy, radius, name in disks
    ]
    if not parsed:
        return True, {"probe": "nav_path_8conn", "n_disks": 0, "blocked": False}

    xs = [float(robot[0]), float(goal[0])] + [float(xy[0]) for xy, _r, _n in parsed]
    ys = [float(robot[1]), float(goal[1])] + [float(xy[1]) for xy, _r, _n in parsed]
    max_r = max((float(r) for _xy, r, _n in parsed), default=0.0)
    margin = max(1.0, float(clearance_m) + float(max_r) + 2 * float(grid_res_m) * pad_cells)
    x0, x1 = float(min(xs)) - margin, float(max(xs)) + margin
    y0, y1 = float(min(ys)) - margin, float(max(ys)) + margin
    res = float(grid_res_m)
    w = max(3, int(math.ceil((x1 - x0) / res)))
    h = max(3, int(math.ceil((y1 - y0) / res)))
    occ = np.ones((h, w), dtype=bool)

    def to_cell(p: np.ndarray) -> tuple[int, int]:
        col = int(round((float(p[0]) - x0) / res))
        row = int(round((float(p[1]) - y0) / res))
        return row, col

    for xy, radius, _name in parsed:
        cr, cc = to_cell(xy)
        inflate = int(math.ceil((float(clearance_m) + float(radius)) / res))
        for dr in range(-inflate, inflate + 1):
            for dc in range(-inflate, inflate + 1):
                r, c = cr + dr, cc + dc
                if 0 <= r < h and 0 <= c < w and (dr * dr + dc * dc) <= inflate * inflate:
                    occ[r, c] = False

    start_cell = to_cell(robot)
    goal_cell = to_cell(goal)
    for cell in (start_cell, goal_cell):
        r, c = cell
        if 0 <= r < h and 0 <= c < w and not occ[r, c]:
            occ[r, c] = True

    path_open = _path_exists(occ, start_cell, goal_cell)
    info = {
        "probe": "nav_path_8conn",
        "grid_res_m": res,
        "clearance_m": float(clearance_m),
        "n_disks": int(len(parsed)),
        "start_xy": robot.tolist(),
        "goal_xy": goal.tolist(),
        "blocked": not bool(path_open),
    }
    return bool(path_open), info


def nav_interpolated_route(
    robot_xy: np.ndarray | list,
    goal_xy: np.ndarray | list | None,
    disks: Sequence[tuple[np.ndarray | list, float, str]],
    *,
    clearance_m: float = 0.22,
    step_m: float = NAV_CHORD_STEP_M,
    **_unused: Any,
) -> tuple[bool, dict[str, Any]]:
    """Interpolate the teleport chord and fail if any step's footprint hits a disk.

    MolmoSpaces ``move_base_to`` is a single snap, so the physical path is the
    straight line from the current base to the landmark — not an A* detour.
    Checking densified chord samples (no extra ZMQ) invalidates a snap that
    would pass through leftover clutter or furniture.
    """
    if goal_xy is None:
        return True, {"probe": "no_goal", "blocked": False}
    robot = _clamp_xy(robot_xy)
    goal = _clamp_xy(goal_xy)
    parsed: list[tuple[np.ndarray, float, str]] = [
        (_clamp_xy(xy), float(radius), str(name)) for xy, radius, name in disks
    ]
    waypoints = densify_polyline([robot, goal], step_m)
    info: dict[str, Any] = {
        "probe": "interpolated_nav",
        "clearance_m": float(clearance_m),
        "step_m": float(step_m),
        "n_disks": int(len(parsed)),
        "n_steps": int(len(waypoints)),
        "start_xy": robot.tolist(),
        "goal_xy": goal.tolist(),
        "blocked": False,
    }
    hit = first_footprint_hit(waypoints, parsed, footprint_r_m=float(clearance_m), skip_first=True)
    if hit is not None:
        info["blocked"] = True
        info["hit"] = hit
        return False, info
    return True, info


def nav_route_open(
    robot_xy: np.ndarray | list,
    goal_xy: np.ndarray | list | None,
    obj_xy: list[np.ndarray] | list[list],
    **kwargs: Any,
) -> tuple[bool, dict[str, Any]]:
    """True when an interpolated, footprint-checked path to ``goal_xy`` exists.

    Scatter-only convenience wrapper (equal-radius disks). Post-clear nav should
    pass full placement disks into :func:`nav_interpolated_route`.
    """
    if goal_xy is None:
        return True, {"probe": "no_goal", "blocked": False}
    obj_radius = float(kwargs.pop("obj_radius_m", 0.08))
    kwargs.pop("grid_res_m", None)
    kwargs.pop("pad_cells", None)
    disks = [(xy, obj_radius, f"obj_{i}") for i, xy in enumerate(obj_xy)]
    return nav_interpolated_route(robot_xy, goal_xy, disks, **kwargs)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def clutter_success_flags(metrics: dict[str, Any]) -> dict[str, bool | float]:
    """Derive the standard aggregate columns from a per-run metrics dict.

    ``task_success``: cleanup → all scattered objects relocated; nav_goal → goal
    reached, route open after clearance, and not ``skipped_invalid``.
    Cluttered nav_goal whose *initial* GT probe did not show a blocked route is
    excluded from the scored success rate (``skipped_invalid``).
    """
    skipped = bool(metrics.get("skipped_invalid", False))
    mode = str(metrics.get("mode", ""))
    n = int(metrics.get("n_objects", 0))
    path_open = metrics.get("nav_path_open")
    if path_open is None and mode == "nav_goal" and n > 0:
        # Older rows without a post-clear probe: do not count a blocked episode as
        # success unless something was actually relocated.
        path_open = int(metrics.get("n_relocated", 0)) >= 1 or not bool(metrics.get("episode_valid", False))
    elif path_open is None:
        path_open = True
    if skipped:
        task_success = False
    elif mode == "cleanup":
        cleared = int(metrics.get("n_relocated", 0))
        task_success = bool(n > 0 and cleared >= n)
    else:
        task_success = bool(metrics.get("goal_reached", False)) and bool(path_open)
    return {
        "task_success": bool(task_success),
        "skipped_invalid": skipped,
        "goal_reached": bool(metrics.get("goal_reached", False)),
        "nav_success": bool(metrics.get("nav_success", False)),
        "nav_path_open": bool(path_open),
        "n_cleared": int(metrics.get("n_cleared", 0)),
        "n_relocated": int(metrics.get("n_relocated", 0)),
        "manip_success_rate": float(metrics.get("manip_success_rate", 0.0)),
        "motion_failures": int(metrics.get("motion_failures", 0)),
        "planning_wall_s": float(metrics.get("planning_wall_s", 0.0)),
        "manip_wall_s": float(metrics.get("manip_wall_s", 0.0)),
    }

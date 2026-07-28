# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Clearance-aware A* + yaw unwrap unit tests (no robot / GPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from emet.motion.algo.a_star import AStar, default_min_clearance_m, unwrap_yaw


class _FakeVoxelMap:
    def __init__(self, obs: np.ndarray, exp: np.ndarray, *, resolution: float = 0.1):
        self._obs = obs
        self._exp = exp
        self.grid_resolution = resolution
        self.grid_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def get_2d_map(self):
        return self._obs.copy(), self._exp.copy()


class _FakeSpace:
    def __init__(self, vm: _FakeVoxelMap):
        self.voxel_map = vm

    def to_pt(self, xy: tuple[float, float]) -> tuple[int, int]:
        res = float(self.voxel_map.grid_resolution)
        # Match Dynamem convention: grid index = floor(xy / res) with origin at 0.
        return int(xy[0] / res), int(xy[1] / res)

    def to_xy(self, pt: tuple[int, int]) -> tuple[float, float]:
        res = float(self.voxel_map.grid_resolution)
        return (pt[0] + 0.5) * res, (pt[1] + 0.5) * res

    def is_valid(self, *_args, **_kwargs) -> bool:
        return True


def _corridor_map(nx: int = 80, ny: int = 41, *, wall_clearance_cells: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Free strip along +x with thick walls in y (grid index 0=x, 1=y)."""
    obs = np.ones((nx, ny), dtype=bool)
    exp = np.ones((nx, ny), dtype=bool)
    y0 = wall_clearance_cells
    y1 = ny - wall_clearance_cells
    obs[:, y0:y1] = False
    return obs, exp


def test_default_min_clearance_stretch_footprint():
    assert default_min_clearance_m(0.34) == pytest.approx(0.22)


def test_unwrap_yaw_shortest_turn():
    assert abs(unwrap_yaw(0.0, math.pi / 2) - math.pi / 2) < 1e-6
    # 5.50 wraps near -0.78; from -1.89 the short delta is not a +2π jump.
    prev = -1.89
    raw = 5.50
    unwrapped = unwrap_yaw(prev, raw)
    delta = unwrapped - prev
    assert abs(delta) <= math.pi + 1e-6
    assert abs(math.atan2(math.sin(raw - prev), math.cos(raw - prev)) - delta) < 1e-6


def test_clean_path_for_xy_yaw_consecutive_delta_le_pi():
    obs, exp = _corridor_map()
    space = _FakeSpace(_FakeVoxelMap(obs, exp, resolution=0.1))
    planner = AStar(space, min_clearance_m=0.0, clearance_cost_weight=0.0, grid_resolution_m=0.1)
    waypoints = [
        [0.5, 2.0, 0.0],
        [1.5, 2.0, 0.0],
        [1.5, 3.0, 0.0],
        [0.5, 3.0, 5.50],  # discontinuous absolute goal yaw
    ]
    traj = planner.clean_path_for_xy(waypoints, start_yaw=0.0)
    assert len(traj) >= 2
    for i in range(1, len(traj)):
        d = traj[i][2] - traj[i - 1][2]
        assert abs(d) <= math.pi + 1e-5, f"yaw jump at {i}: {traj[i - 1][2]} -> {traj[i][2]}"


def test_clearance_astar_prefers_corridor_center():
    """With soft clearance cost, path midpoints stay farther from walls than weight=0."""
    obs, exp = _corridor_map(nx=80, ny=41, wall_clearance_cells=5)
    space = _FakeSpace(_FakeVoxelMap(obs, exp, resolution=0.1))
    soft = AStar(space, min_clearance_m=0.0, clearance_cost_weight=4.0, grid_resolution_m=0.1)
    binary = AStar(space, min_clearance_m=0.0, clearance_cost_weight=0.0, grid_resolution_m=0.1)

    # Start near the lower free edge (y≈0.55) so binary A* may hug the wall.
    start = (0.5, 0.55, 0.0)
    goal = (7.5, 0.55, 0.0)
    res_soft = soft.plan(start, goal, verbose=False)
    res_bin = binary.plan(start, goal, verbose=False)
    assert res_soft.success and res_bin.success

    def mean_clearance(planner: AStar, result) -> float:
        planner.reset()
        vals = [planner.clearance_at_xy(np.asarray(node.state).reshape(-1)[:2]) for node in result.trajectory]
        return float(np.mean(vals))

    soft_c = mean_clearance(soft, res_soft)
    bin_c = mean_clearance(binary, res_bin)
    assert soft_c + 1e-6 >= bin_c * 0.95
    assert soft_c >= 0.15


def test_hard_min_clearance_rejects_wall_hug_cells():
    obs, exp = _corridor_map(nx=40, ny=30, wall_clearance_cells=2)
    space = _FakeSpace(_FakeVoxelMap(obs, exp, resolution=0.1))
    planner = AStar(space, min_clearance_m=0.25, clearance_cost_weight=1.0, grid_resolution_m=0.1)
    planner.reset()
    # First free y-cell (index 2) is < 0.25 m from the wall → occupied under hard gate.
    assert planner.point_is_occupied(10, 2)
    mid_y = 15
    assert not planner.point_is_occupied(10, mid_y)


def _l_obstacle_map(nx: int = 40, ny: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Open room with an L-shaped obstacle that invites corner-cutting chords."""
    obs = np.zeros((nx, ny), dtype=bool)
    exp = np.ones((nx, ny), dtype=bool)
    # Vertical bar + horizontal bar meeting near (20, 20).
    obs[18:23, 5:21] = True
    obs[10:23, 18:23] = True
    return obs, exp


def test_los_rejects_chord_through_obstacle_corner():
    """Direct chord across an L-corner must fail even if endpoints are free."""
    obs, exp = _l_obstacle_map()
    space = _FakeSpace(_FakeVoxelMap(obs, exp, resolution=0.1))
    planner = AStar(space, min_clearance_m=0.15, clearance_cost_weight=0.0, grid_resolution_m=0.1)
    planner.reset()
    # Endpoints sit in free space on either side of the L.
    a, b = (12, 12), (28, 12)
    assert not planner.point_is_occupied(*a)
    assert not planner.point_is_occupied(*b)
    assert not planner.is_in_line_of_sight(a, b)
    assert not planner.is_clearance_line_of_sight(a, b)


def test_clean_path_keeps_detour_instead_of_cutting_corner():
    """clean_path must not collapse a detour into a chord through the L obstacle."""
    obs, exp = _l_obstacle_map()
    space = _FakeSpace(_FakeVoxelMap(obs, exp, resolution=0.1))
    planner = AStar(space, min_clearance_m=0.15, clearance_cost_weight=1.0, grid_resolution_m=0.1)
    planner.reset()
    # Detour south of the vertical bar (y < 5) then up — every consecutive
    # pair must be free; coarse jumps that skip the wall are invalid inputs.
    detour = [
        (12, 12),
        (12, 3),
        (16, 3),
        (20, 3),
        (24, 3),
        (28, 3),
        (28, 12),
    ]
    for pt in detour:
        assert not planner.point_is_occupied(*pt), pt
    for i in range(len(detour) - 1):
        assert planner.is_in_line_of_sight(detour[i], detour[i + 1]), (detour[i], detour[i + 1])
    cleaned = planner.clean_path(detour)
    assert cleaned[0][:2] == detour[0]
    assert cleaned[-1][:2] == detour[-1]
    for i in range(len(cleaned) - 1):
        assert planner.is_in_line_of_sight(cleaned[i][:2], cleaned[i + 1][:2])
    # Collapsing start→goal in one hop would cut the corner — must not happen.
    assert not planner.is_in_line_of_sight(detour[0], detour[-1])
    assert len(cleaned) >= 3


def test_clean_path_for_xy_clearance_holds_along_segments():
    obs, exp = _l_obstacle_map()
    space = _FakeSpace(_FakeVoxelMap(obs, exp, resolution=0.1))
    planner = AStar(space, min_clearance_m=0.15, clearance_cost_weight=1.0, grid_resolution_m=0.1)
    start = (1.25, 1.25, 0.0)
    goal = (2.85, 1.25, 0.0)
    res = planner.plan(start, goal, verbose=False)
    assert res.success, getattr(res, "reason", None)
    traj = planner.clean_path_for_xy(
        [np.asarray(n.state).reshape(-1).tolist() for n in res.trajectory],
        start_yaw=0.0,
    )
    min_c = float(planner.min_clearance_m)
    for p in traj:
        xy = (float(p[0]), float(p[1]))
        assert planner.clearance_at_xy(xy) + 1e-6 >= min_c
    for i in range(len(traj) - 1):
        a = planner.to_pt((traj[i][0], traj[i][1]))
        b = planner.to_pt((traj[i + 1][0], traj[i + 1][1]))
        assert planner.is_in_line_of_sight(a, b)

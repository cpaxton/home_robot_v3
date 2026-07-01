# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""2D grid helpers for top-down map export (connected components, island pruning)."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from emet.visualization.map_snapshot import world_xy_to_grid_ij


def explored_connected_components(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Return explored connected components sorted by size (largest first)."""
    arr = np.asarray(mask, dtype=bool)
    h, w = arr.shape
    seen = np.zeros_like(arr, dtype=bool)
    components: list[tuple[int, np.ndarray]] = []
    for i in range(h):
        for j in range(w):
            if not arr[i, j] or seen[i, j]:
                continue
            q: deque[tuple[int, int]] = deque([(i, j)])
            seen[i, j] = True
            cells: list[tuple[int, int]] = []
            while q:
                ci, cj = q.popleft()
                cells.append((ci, cj))
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < h and 0 <= nj < w and arr[ni, nj] and not seen[ni, nj]:
                        seen[ni, nj] = True
                        q.append((ni, nj))
            components.append((len(cells), np.asarray(cells, dtype=np.int32)))
    components.sort(key=lambda x: x[0], reverse=True)
    return components


def build_trajectory_anchor_mask(
    shape_hw: tuple[int, int],
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    *,
    robot_xy: np.ndarray | tuple[float, float] | None = None,
    trajectory_xyt: list[tuple[float, float, float] | list[float]] | None = None,
    radius_cells: int = 8,
) -> np.ndarray:
    """Disk union along trajectory + robot pose (cells marked True)."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    mask = np.zeros((h, w), dtype=bool)
    rr = max(1, int(radius_cells))
    points: list[tuple[float, float]] = []
    if robot_xy is not None:
        xy = np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2]
        points.append((float(xy[0]), float(xy[1])))
    for raw in trajectory_xyt or []:
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.size >= 2:
            points.append((float(arr[0]), float(arr[1])))
    for x, y in points:
        ri, rj = world_xy_to_grid_ij((x, y), grid_origin_xy, grid_resolution, (h, w))
        i0, i1 = max(0, ri - rr), min(h, ri + rr + 1)
        j0, j1 = max(0, rj - rr), min(w, rj + rr + 1)
        for ii in range(i0, i1):
            for jj in range(j0, j1):
                if (ii - ri) ** 2 + (jj - rj) ** 2 <= rr * rr:
                    mask[ii, jj] = True
    return mask


def prune_explored_islands(
    explored: Any,
    *,
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    robot_xy: np.ndarray | tuple[float, float] | None = None,
    trajectory_xyt: list[tuple[float, float, float] | list[float]] | None = None,
    min_component_cells: int = 6,
    anchor_radius_cells: int = 8,
) -> np.ndarray:
    """Drop explored blobs disconnected from the robot path (remote depth islands)."""
    exp = np.asarray(explored, dtype=bool)
    if not exp.any():
        return exp
    anchor = build_trajectory_anchor_mask(
        exp.shape,
        grid_origin_xy,
        grid_resolution,
        robot_xy=robot_xy,
        trajectory_xyt=trajectory_xyt,
        radius_cells=anchor_radius_cells,
    )
    components = explored_connected_components(exp)
    keep = np.zeros_like(exp, dtype=bool)
    for count, cells in components:
        if count < min_component_cells:
            continue
        comp_mask = np.zeros_like(exp, dtype=bool)
        comp_mask[cells[:, 0], cells[:, 1]] = True
        if anchor.any() and not np.any(comp_mask & anchor):
            continue
        keep |= comp_mask
    if not keep.any() and components:
        largest = components[0][1]
        keep[largest[:, 0], largest[:, 1]] = True
    return exp & keep

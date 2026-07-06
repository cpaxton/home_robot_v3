# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Rasterize Habitat navmesh navigability onto emet voxel-map grids for GT overlays."""

from __future__ import annotations

from typing import Any

import numpy as np

from emet.visualization.map_snapshot import (
    finalize_export_topdown_rgb,
    world_xy_to_grid_ij,
)


def _pathfinder_xz_bounds(pathfinder: Any) -> tuple[np.ndarray, np.ndarray]:
    bounds = pathfinder.get_bounds()
    min_b = np.array([float(bounds[0][0]), float(bounds[0][2])], dtype=np.float64)
    max_b = np.array([float(bounds[1][0]), float(bounds[1][2])], dtype=np.float64)
    return min_b, max_b


def rasterize_habitat_navmesh_grid(
    pathfinder: Any,
    shape_hw: tuple[int, int],
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    *,
    floor_y: float,
) -> np.ndarray:
    """Map Habitat navmesh navigability onto the emet voxel 2D grid (planar X/Z)."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if pathfinder is None or not getattr(pathfinder, "is_loaded", False):
        return np.zeros((h, w), dtype=bool)
    mpp = float(grid_resolution)
    if mpp <= 0:
        mpp = 0.1
    go = np.asarray(grid_origin_xy, dtype=np.float64).reshape(-1)[:2]
    try:
        td = np.asarray(pathfinder.get_topdown_view(mpp, float(floor_y)), dtype=bool)
    except Exception:
        td = np.zeros((0, 0), dtype=bool)
    if td.size == 0:
        return _rasterize_navmesh_sampled(pathfinder, (h, w), go, mpp, floor_y=float(floor_y))
    min_xz, _max_xz = _pathfinder_xz_bounds(pathfinder)
    ii, jj = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    wx = (ii - go[0]) * mpp
    wz = (jj - go[1]) * mpp
    cols = np.floor((wx - min_xz[0]) / mpp).astype(np.int32)
    rows = np.floor((wz - min_xz[1]) / mpp).astype(np.int32)
    out = np.zeros((h, w), dtype=bool)
    valid = (rows >= 0) & (rows < td.shape[0]) & (cols >= 0) & (cols < td.shape[1])
    out[valid] = td[rows[valid], cols[valid]]
    return out


def _rasterize_navmesh_sampled(
    pathfinder: Any,
    shape_hw: tuple[int, int],
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    *,
    floor_y: float,
) -> np.ndarray:
    """Fallback per-cell ``is_navigable`` sampling when topdown view is unavailable."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    res = float(grid_resolution)
    go = np.asarray(grid_origin_xy, dtype=np.float64).reshape(-1)[:2]
    y = float(floor_y)
    out = np.zeros((h, w), dtype=bool)
    for i in range(h):
        wx = (float(i) - go[0]) * res
        for j in range(w):
            wz = (float(j) - go[1]) * res
            pt = np.array([wx, y, wz], dtype=np.float32)
            try:
                out[i, j] = bool(pathfinder.is_navigable(pt))
            except Exception:
                out[i, j] = False
    return out


def habitat_gt_topdown_rgb(
    navigable: np.ndarray,
    *,
    crop_slice: tuple[int, int, int, int] | None = None,
    max_side: int | None = 1280,
    min_map_side: int = 1024,
) -> np.ndarray:
    """GT navmesh layer: navigable = light slate, non-nav = white."""
    nav = np.asarray(navigable, dtype=bool)
    if crop_slice is not None:
        i0, i1, j0, j1 = crop_slice
        nav = nav[i0:i1, j0:j1]
    rgb = np.full((nav.shape[0], nav.shape[1], 3), 248, dtype=np.uint8)
    rgb[nav] = (180, 190, 210)
    if max_side is not None:
        return finalize_export_topdown_rgb(rgb, max_side=max_side, min_side=min_map_side)
    return rgb


def habitat_gt_topdown_cropped(
    pathfinder: Any,
    obstacles: Any,
    explored: Any,
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    robot_xy: np.ndarray | tuple[float, float] | None,
    *,
    floor_y: float,
    margin_cells: int = 8,
    max_side: int = 1280,
    min_map_side: int = 1024,
    trajectory_xyt: list[tuple[float, float, float] | list[float]] | None = None,
    filter_islands: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-grid GT raster + cropped RGB (same bbox as eval map). Returns ``(nav_full, gt_rgb)``."""
    from emet.visualization.map_grid import prune_explored_islands
    from emet.visualization.map_snapshot import explored_crop_indices

    exp = np.asarray(explored, dtype=bool)
    if filter_islands:
        exp = prune_explored_islands(
            exp,
            grid_origin_xy=grid_origin_xy,
            grid_resolution=grid_resolution,
            robot_xy=robot_xy,
            trajectory_xyt=trajectory_xyt,
        )
    h, w = exp.shape
    nav_full = rasterize_habitat_navmesh_grid(
        pathfinder,
        (h, w),
        grid_origin_xy,
        grid_resolution,
        floor_y=floor_y,
    )
    bbox = explored_crop_indices(
        exp,
        robot_xy,
        grid_origin_xy,
        grid_resolution,
        (h, w),
        margin_cells=margin_cells,
    )
    if bbox is None:
        return nav_full, habitat_gt_topdown_rgb(nav_full, max_side=max_side, min_map_side=min_map_side)
    i0, i1, j0, j1 = bbox
    gt_rgb = habitat_gt_topdown_rgb(nav_full, crop_slice=bbox, max_side=max_side, min_map_side=min_map_side)
    return nav_full, gt_rgb

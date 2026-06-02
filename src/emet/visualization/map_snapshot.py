# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Top-down 2D map RGB snapshots + text stats for agent tools / Rerun / Discord."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _to_numpy_bool_2d(x: Any) -> np.ndarray:
    if x is None:
        return np.zeros((1, 1), dtype=bool)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D map, got shape {arr.shape}")
    return arr.astype(bool)


def _grid_origin_xy(grid_origin: Any) -> np.ndarray:
    if isinstance(grid_origin, torch.Tensor):
        g = grid_origin.detach().cpu().numpy().reshape(-1)
    else:
        g = np.asarray(grid_origin, dtype=np.float64).reshape(-1)
    if g.size >= 2:
        return g[:2].astype(np.float64)
    return np.zeros(2, dtype=np.float64)


def world_xy_to_grid_ij(
    xy: np.ndarray | tuple[float, float],
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    shape_hw: tuple[int, int],
) -> tuple[int, int]:
    """Map world (x, y) to integer grid indices (row, col) matching voxel 2D maps."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1)[:2]
    go = np.asarray(grid_origin_xy, dtype=np.float64).reshape(-1)[:2]
    res = float(grid_resolution)
    if res <= 0:
        res = 1e-6
    g = xy / res + go
    i = int(np.floor(g[0]))
    j = int(np.floor(g[1]))
    h, w = shape_hw
    i = max(0, min(h - 1, i))
    j = max(0, min(w - 1, j))
    return i, j


def render_topdown_map_rgb(
    obstacles: Any,
    explored: Any,
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    robot_xy: np.ndarray | tuple[float, float] | None = None,
    *,
    max_side: int | None = 640,
) -> np.ndarray:
    """Render obstacles / explored / optional robot pose as uint8 HxWx3 (top-down, grid indices = image rows/cols)."""
    obs = _to_numpy_bool_2d(obstacles)
    exp = _to_numpy_bool_2d(explored)
    if obs.shape != exp.shape:
        raise ValueError(f"obstacles shape {obs.shape} != explored shape {exp.shape}")
    h, w = obs.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Unknown: dark gray
    rgb[:, :] = (28, 28, 36)
    # Explored free
    free = exp & ~obs
    rgb[free] = (50, 160, 80)
    # Obstacles
    rgb[obs] = (200, 55, 55)
    # Explored obstacle (rare): keep obstacle color
    if robot_xy is not None:
        ri, rj = world_xy_to_grid_ij(robot_xy, grid_origin_xy, grid_resolution, (h, w))
        r = 3
        i0, i1 = max(0, ri - r), min(h, ri + r + 1)
        j0, j1 = max(0, rj - r), min(w, rj + r + 1)
        rgb[i0:i1, j0:j1] = np.maximum(rgb[i0:i1, j0:j1], np.uint8([255, 255, 255]))
        rgb[ri, rj] = (255, 255, 0)
    if max_side is not None:
        m = max(h, w)
        if m > max_side and m > 0:
            step = int(np.ceil(m / max_side))
            rgb = rgb[::step, ::step].copy()
    return rgb


def downsample_topdown_rgb_max_side(rgb: np.ndarray, max_side: int) -> np.ndarray:
    """Uniform grid stride so max(H,W) <= max_side (same rule as ``render_topdown_map_rgb``)."""
    h, w = rgb.shape[0], rgb.shape[1]
    m = max(h, w)
    if m <= max_side or m == 0:
        return rgb
    step = int(np.ceil(m / max_side))
    return rgb[::step, ::step].copy()


def crop_topdown_rgb_to_explored(
    rgb: np.ndarray,
    explored: Any,
    robot_xy: np.ndarray | tuple[float, float] | None,
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    *,
    margin_cells: int = 16,
    robot_radius_cells: int = 5,
) -> np.ndarray:
    """Crop top-down RGB to the bounding box of explored cells (plus margin and a small robot neighborhood)."""
    exp = _to_numpy_bool_2d(explored)
    if rgb.shape[0] != exp.shape[0] or rgb.shape[1] != exp.shape[1]:
        return np.ascontiguousarray(rgb)
    mask = exp.copy()
    if robot_xy is not None:
        ri, rj = world_xy_to_grid_ij(robot_xy, grid_origin_xy, grid_resolution, exp.shape)
        rr = int(robot_radius_cells)
        i0, i1 = max(0, ri - rr), min(exp.shape[0], ri + rr + 1)
        j0, j1 = max(0, rj - rr), min(exp.shape[1], rj + rr + 1)
        mask[i0:i1, j0:j1] = True
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.ascontiguousarray(rgb)
    h, w = rgb.shape[0], rgb.shape[1]
    mc = int(margin_cells)
    i0, i1 = max(0, int(ys.min()) - mc), min(h, int(ys.max()) + 1 + mc)
    j0, j1 = max(0, int(xs.min()) - mc), min(w, int(xs.max()) + 1 + mc)
    if i1 <= i0 or j1 <= j0:
        return np.ascontiguousarray(rgb)
    return np.ascontiguousarray(rgb[i0:i1, j0:j1])


def share_topdown_map_rgb(
    obstacles: Any,
    explored: Any,
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    robot_xy: np.ndarray | tuple[float, float] | None,
    *,
    max_side: int = 640,
    margin_cells: int = 16,
) -> np.ndarray:
    """Full-resolution render, crop to explored region, then downsample (Rerun / Discord / sharing)."""
    rgb_full = render_topdown_map_rgb(
        obstacles,
        explored,
        grid_origin_xy,
        grid_resolution,
        robot_xy,
        max_side=None,
    )
    cropped = crop_topdown_rgb_to_explored(
        rgb_full,
        explored,
        robot_xy,
        grid_origin_xy,
        grid_resolution,
        margin_cells=margin_cells,
    )
    return downsample_topdown_rgb_max_side(cropped, max_side)


def discord_share_map_rgb(
    obstacles: Any,
    explored: Any,
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    robot_xy: np.ndarray | tuple[float, float] | None,
    *,
    max_side: int = 640,
    margin_cells: int = 16,
) -> np.ndarray:
    """Alias for :func:`share_topdown_map_rgb` (Discord map posts)."""
    return share_topdown_map_rgb(
        obstacles,
        explored,
        grid_origin_xy,
        grid_resolution,
        robot_xy,
        max_side=max_side,
        margin_cells=margin_cells,
    )


def build_map_stats(
    obstacles: Any,
    explored: Any,
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    robot_xy: np.ndarray | tuple[float, float] | None,
) -> dict[str, Any]:
    obs = _to_numpy_bool_2d(obstacles)
    exp = _to_numpy_bool_2d(explored)
    h, w = obs.shape
    explored_cells = int(np.count_nonzero(exp))
    obstacle_cells = int(np.count_nonzero(obs))
    free_explored = int(np.count_nonzero(exp & ~obs))
    stats: dict[str, Any] = {
        "grid_shape_hw": (h, w),
        "explored_cells": explored_cells,
        "obstacle_cells": obstacle_cells,
        "free_explored_cells": free_explored,
        "grid_resolution": float(grid_resolution),
        "grid_origin_xy": grid_origin_xy.tolist() if isinstance(grid_origin_xy, np.ndarray) else list(grid_origin_xy),
        "map_nonempty": explored_cells > 0 or obstacle_cells > 0,
    }
    if robot_xy is not None:
        xy = np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2]
        stats["base_xy"] = (float(xy[0]), float(xy[1]))
        gi, gj = world_xy_to_grid_ij(xy, grid_origin_xy, grid_resolution, (h, w))
        stats["base_grid_ij"] = (gi, gj)
        on_obstacle = bool(obs[gi, gj])
        stats["base_on_obstacle_cell"] = on_obstacle
        stats["base_on_explored_cell"] = bool(exp[gi, gj])
    else:
        stats["base_xy"] = None
        stats["base_grid_ij"] = None
        stats["base_on_obstacle_cell"] = None
        stats["base_on_explored_cell"] = None
    lines = [
        f"2D map shape (H,W)=({h},{w}), resolution={float(grid_resolution):.4f} m/cell.",
        f"Explored cells={explored_cells}, obstacle cells={obstacle_cells}, free explored={free_explored}.",
    ]
    if stats.get("base_xy") is not None:
        bx, by = stats["base_xy"]
        lines.append(f"Base pose (world xy) ≈ ({bx:.3f}, {by:.3f}); grid index {stats['base_grid_ij']}.")
        if stats.get("base_on_obstacle_cell"):
            lines.append("Base cell is marked obstacle (common if dilated map or pose inside wall).")
        elif not stats.get("base_on_explored_cell"):
            lines.append("Base cell not yet explored (map may still be empty or frame mismatch).")
    if free_explored == 0 and explored_cells == 0:
        lines.append("No explored cells yet — need motion + valid depth before exploration can score frontiers.")
    stats["summary_lines"] = lines
    return stats


def format_navigation_report(stats: dict[str, Any], *, explore_ok: bool | None = None) -> str:
    """Single string for tool results / LLM."""
    parts = list(stats.get("summary_lines", []))
    if explore_ok is True:
        parts.append("Last explore command: reported success by executor.")
    elif explore_ok is False:
        parts.append("Last explore command: executor reported failure (no frontier / non-navigable start / empty map).")
    return " ".join(parts)


def snapshot_from_voxel_map(
    voxel_map: Any,
    robot_xy: np.ndarray | tuple[float, float] | None,
    *,
    max_side: int = 640,
) -> tuple[np.ndarray | None, dict[str, Any], np.ndarray | None]:
    """Build RGB snapshot + stats from a SparseVoxelMap-like object.

    Returns ``(img, stats, img_share)``. Both images are cropped to the explored region (plus margin)
    then downsampled to ``max_side`` (same pipeline as Discord sharing). If no map, all three are
    ``None`` / empty stats.
    """
    if voxel_map is None or not hasattr(voxel_map, "get_2d_map"):
        empty: dict[str, Any] = {
            "summary_lines": ["No voxel map attached (get_voxel_map unavailable)."],
            "map_nonempty": False,
        }
        return None, empty, None
    obstacles, explored = voxel_map.get_2d_map()
    go = _grid_origin_xy(getattr(voxel_map, "grid_origin", np.zeros(2)))
    res = float(getattr(voxel_map, "grid_resolution", 0.1) or 0.1)
    stats = build_map_stats(obstacles, explored, go, res, robot_xy)
    img = share_topdown_map_rgb(obstacles, explored, go, res, robot_xy, max_side=max_side)
    return img, stats, img

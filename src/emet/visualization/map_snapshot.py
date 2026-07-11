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


def upscale_topdown_rgb_min_side(rgb: np.ndarray, min_side: int) -> np.ndarray:
    """Nearest-neighbor upscale so max(H,W) >= min_side (eval exports stay readable)."""
    if min_side <= 0:
        return rgb
    h, w = rgb.shape[0], rgb.shape[1]
    m = max(h, w)
    if m >= min_side or m == 0:
        return rgb
    from PIL import Image

    scale = float(min_side) / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(rgb[:, :, :3].astype(np.uint8), mode="RGB")
    return np.asarray(img.resize((new_w, new_h), resample=Image.Resampling.NEAREST), dtype=np.uint8)


def finalize_export_topdown_rgb(
    rgb: np.ndarray,
    *,
    max_side: int = 1280,
    min_side: int = 1024,
) -> np.ndarray:
    """Upscale small crops, then downsample if above max_side."""
    out = upscale_topdown_rgb_min_side(rgb, min_side)
    return downsample_topdown_rgb_max_side(out, max_side)


def explored_crop_indices(
    explored: Any,
    robot_xy: np.ndarray | tuple[float, float] | None,
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    shape_hw: tuple[int, int],
    *,
    margin_cells: int = 16,
    robot_radius_cells: int = 5,
    trajectory_xyt: list[tuple[float, float, float] | list[float]] | None = None,
) -> tuple[int, int, int, int] | None:
    """Row/col slice ``(i0, i1, j0, j1)`` around explored cells (+ robot neighborhood).

    Same bounding box as :func:`crop_topdown_rgb_to_explored` / Discord share maps.
    Returns ``None`` when no explored cells (and no robot) are present.
    When ``trajectory_xyt`` is set, the path corridor is included so crops stay tight
    to where the agent actually drove.
    """
    exp = _to_numpy_bool_2d(explored)
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if exp.shape[0] != h or exp.shape[1] != w:
        h, w = exp.shape[0], exp.shape[1]
    mask = exp.copy()
    if robot_xy is not None:
        ri, rj = world_xy_to_grid_ij(robot_xy, grid_origin_xy, grid_resolution, (h, w))
        rr = int(robot_radius_cells)
        ri0, ri1 = max(0, ri - rr), min(h, ri + rr + 1)
        rj0, rj1 = max(0, rj - rr), min(w, rj + rr + 1)
        mask[ri0:ri1, rj0:rj1] = True
    if trajectory_xyt:
        from emet.visualization.map_grid import build_trajectory_corridor_mask

        corridor = build_trajectory_corridor_mask(
            (h, w),
            grid_origin_xy,
            grid_resolution,
            robot_xy=robot_xy,
            trajectory_xyt=trajectory_xyt,
            radius_cells=max(2, int(robot_radius_cells)),
        )
        mask |= corridor
        # Prefer a tight crop along the driven path when depth exploration is sparse.
        exp_frac = float(exp.sum()) / float(max(1, mask.sum()))
        if exp_frac < 0.85:
            mask = corridor.copy()
            if robot_xy is not None:
                ri, rj = world_xy_to_grid_ij(robot_xy, grid_origin_xy, grid_resolution, (h, w))
                rr = int(robot_radius_cells)
                ri0, ri1 = max(0, ri - rr), min(h, ri + rr + 1)
                rj0, rj1 = max(0, rj - rr), min(w, rj + rr + 1)
                mask[ri0:ri1, rj0:rj1] = True
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    mc = int(margin_cells)
    i0, i1 = max(0, int(ys.min()) - mc), min(h, int(ys.max()) + 1 + mc)
    j0, j1 = max(0, int(xs.min()) - mc), min(w, int(xs.max()) + 1 + mc)
    if i1 <= i0 or j1 <= j0:
        return None
    return i0, i1, j0, j1


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
    bbox = explored_crop_indices(
        explored,
        robot_xy,
        grid_origin_xy,
        grid_resolution,
        exp.shape,
        margin_cells=margin_cells,
        robot_radius_cells=robot_radius_cells,
    )
    if bbox is None:
        return np.ascontiguousarray(rgb)
    i0, i1, j0, j1 = bbox
    return np.ascontiguousarray(rgb[i0:i1, j0:j1])


def _dedupe_trajectory_xyt(
    trajectory_xyt: list[tuple[float, float, float] | list[float]],
    *,
    min_step_m: float = 0.02,
) -> list[tuple[float, float, float]]:
    """Drop consecutive poses closer than ``min_step_m`` (spin-in-place clutter)."""
    out: list[tuple[float, float, float]] = []
    for raw in trajectory_xyt:
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.size < 2:
            continue
        theta = float(arr[2]) if arr.size >= 3 else 0.0
        pose = (float(arr[0]), float(arr[1]), theta)
        if out:
            px, py, _ = out[-1]
            if float(np.hypot(pose[0] - px, pose[1] - py)) < min_step_m:
                out[-1] = pose
                continue
        out.append(pose)
    return out


def _subsample_trajectory_arrow_indices(
    trajectory_xyt: list[tuple[float, float, float]],
    *,
    min_dist_m: float = 0.15,
    max_arrows: int = 24,
) -> list[int]:
    """Indices for heading arrows along a trajectory (always first + last)."""
    if not trajectory_xyt:
        return []
    if len(trajectory_xyt) == 1:
        return [0]
    picks = [0]
    last_xy = trajectory_xyt[0][:2]
    for idx in range(1, len(trajectory_xyt) - 1):
        xy = trajectory_xyt[idx][:2]
        if float(np.hypot(xy[0] - last_xy[0], xy[1] - last_xy[1])) >= min_dist_m:
            picks.append(idx)
            last_xy = xy
    if picks[-1] != len(trajectory_xyt) - 1:
        picks.append(len(trajectory_xyt) - 1)
    if len(picks) > max_arrows:
        stride = int(np.ceil(len(picks) / max_arrows))
        thinned = picks[::stride]
        if thinned[-1] != picks[-1]:
            thinned.append(picks[-1])
        picks = thinned
    return picks


def _draw_line_rgb(
    rgb: np.ndarray,
    i0: int,
    j0: int,
    i1: int,
    j1: int,
    color: tuple[int, int, int],
) -> None:
    """Bresenham line on ``rgb`` (row=i, col=j)."""
    h, w = rgb.shape[0], rgb.shape[1]
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    err = di - dj
    i, j = i0, j0
    while True:
        if 0 <= i < h and 0 <= j < w:
            rgb[i, j] = np.uint8(color)
        if i == i1 and j == j1:
            break
        e2 = 2 * err
        if e2 > -dj:
            err -= dj
            i += si
        if e2 < di:
            err += di
            j += sj


def _draw_disk_rgb(rgb: np.ndarray, ri: int, rj: int, radius: int, color: tuple[int, int, int]) -> None:
    h, w = rgb.shape[0], rgb.shape[1]
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di * di + dj * dj > radius * radius:
                continue
            ii, jj = ri + di, rj + dj
            if 0 <= ii < h and 0 <= jj < w:
                rgb[ii, jj] = np.uint8(color)


def _draw_heading_arrow_rgb(
    rgb: np.ndarray,
    ri: int,
    rj: int,
    theta: float,
    grid_resolution: float,
    *,
    arrow_len_m: float = 0.35,
    color: tuple[int, int, int] = (220, 50, 50),
) -> None:
    """Draw a small heading arrow; ``theta`` is world yaw (radians)."""
    res = float(grid_resolution)
    if res <= 0:
        res = 0.1
    length_cells = max(2.0, arrow_len_m / res)
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    ti = int(round(ri + ct * length_cells))
    tj = int(round(rj + st * length_cells))
    _draw_line_rgb(rgb, ri, rj, ti, tj, color)
    head_len = max(1.5, length_cells * 0.35)
    for ang in (2.4, -2.4):
        hi = int(round(ti - head_len * np.cos(theta + ang)))
        hj = int(round(tj - head_len * np.sin(theta + ang)))
        _draw_line_rgb(rgb, ti, tj, hi, hj, color)


def overlay_trajectory_on_map_rgb(
    rgb: np.ndarray,
    trajectory_xyt: list[tuple[float, float, float] | list[float]],
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    *,
    crop_offset_ij: tuple[int, int] = (0, 0),
    full_shape_hw: tuple[int, int] | None = None,
    arrow_min_dist_m: float = 0.15,
    max_arrows: int = 24,
) -> np.ndarray:
    """Paint deduped path + heading arrows onto an eval/share top-down RGB image."""
    if rgb is None or not trajectory_xyt:
        return rgb
    path = _dedupe_trajectory_xyt(trajectory_xyt)
    if len(path) < 1:
        return rgb
    out = np.ascontiguousarray(rgb)
    h, w = out.shape[0], out.shape[1]
    if full_shape_hw is None:
        full_shape_hw = (h, h) if h == w else (h + int(crop_offset_ij[0]), w + int(crop_offset_ij[1]))
    i_off, j_off = int(crop_offset_ij[0]), int(crop_offset_ij[1])

    def to_ij(x: float, y: float) -> tuple[int, int]:
        ri, rj = world_xy_to_grid_ij((x, y), grid_origin_xy, grid_resolution, full_shape_hw)
        return ri - i_off, rj - j_off

    path_color = (30, 90, 230)
    prev: tuple[int, int] | None = None
    for x, y, _ in path:
        ij = to_ij(x, y)
        if prev is not None:
            _draw_line_rgb(out, prev[0], prev[1], ij[0], ij[1], path_color)
        prev = ij

    for idx in _subsample_trajectory_arrow_indices(
        path, min_dist_m=arrow_min_dist_m, max_arrows=max_arrows
    ):
        x, y, theta = path[idx]
        ri, rj = to_ij(x, y)
        if 0 <= ri < h and 0 <= rj < w:
            _draw_heading_arrow_rgb(out, ri, rj, theta, grid_resolution)

    if path:
        sx, sy, _ = path[0]
        ex, ey, _ = path[-1]
        si, sj = to_ij(sx, sy)
        ei, ej = to_ij(ex, ey)
        if 0 <= si < h and 0 <= sj < w:
            _draw_disk_rgb(out, si, sj, 2, (40, 200, 60))
        if (ei, ej) != (si, sj) and 0 <= ei < h and 0 <= ej < w:
            _draw_disk_rgb(out, ei, ej, 2, (255, 140, 0))
    return out


def eval_topdown_map_rgb(
    obstacles: Any,
    explored: Any,
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    robot_xy: np.ndarray | tuple[float, float] | None,
    *,
    max_side: int = 1280,
    min_map_side: int = 1024,
    margin_cells: int = 24,
    trajectory_xyt: list[tuple[float, float, float] | list[float]] | None = None,
    filter_islands: bool = False,
    stamp_trajectory_corridor: bool = True,
) -> np.ndarray:
    """Eval/diagnostics export: crop to explored footprint, white background, only paint explored cells.

    Unlike :func:`share_topdown_map_rgb`, unmapped margin pixels stay white (not dark gray) so small
    Habitat/OVMM maps remain readable on a 1024×1024 grid.

    When ``trajectory_xyt`` is provided and ``stamp_trajectory_corridor`` is true, cells along the
    driven path are treated as explored free space so exports do not show white gaps between sparse
    depth-mapping blobs.
    """
    obs = _to_numpy_bool_2d(obstacles)
    exp = _to_numpy_bool_2d(explored)
    path = _dedupe_trajectory_xyt(trajectory_xyt) if trajectory_xyt else None
    if stamp_trajectory_corridor and path:
        from emet.visualization.map_grid import merge_trajectory_corridor_explored

        exp = merge_trajectory_corridor_explored(
            exp,
            obs,
            grid_origin_xy,
            grid_resolution,
            robot_xy=robot_xy,
            trajectory_xyt=path,
        )
    if filter_islands:
        from emet.visualization.map_grid import prune_explored_islands

        exp = prune_explored_islands(
            exp,
            grid_origin_xy=grid_origin_xy,
            grid_resolution=grid_resolution,
            robot_xy=robot_xy,
            trajectory_xyt=path,
        )
    h, w = obs.shape
    bbox = explored_crop_indices(
        exp,
        robot_xy,
        grid_origin_xy,
        grid_resolution,
        (h, w),
        margin_cells=margin_cells,
        trajectory_xyt=path,
    )
    if bbox is None:
        rgb = render_topdown_map_rgb(
            obstacles,
            explored,
            grid_origin_xy,
            grid_resolution,
            robot_xy,
            max_side=None,
        )
        if trajectory_xyt:
            rgb = overlay_trajectory_on_map_rgb(
                rgb,
                trajectory_xyt,
                grid_origin_xy,
                grid_resolution,
                full_shape_hw=(h, w),
            )
        return finalize_export_topdown_rgb(rgb, max_side=max_side, min_side=min_map_side)
    i0, i1, j0, j1 = bbox
    exp_c = exp[i0:i1, j0:j1]
    obs_c = obs[i0:i1, j0:j1]
    rgb = np.full((exp_c.shape[0], exp_c.shape[1], 3), 248, dtype=np.uint8)
    free = exp_c & ~obs_c
    rgb[free] = (50, 160, 80)
    rgb[exp_c & obs_c] = (200, 55, 55)
    if robot_xy is not None:
        ri, rj = world_xy_to_grid_ij(robot_xy, grid_origin_xy, grid_resolution, (h, w))
        ri -= i0
        rj -= j0
        ch, cw = rgb.shape[0], rgb.shape[1]
        if 0 <= ri < ch and 0 <= rj < cw:
            r = 3
            i_lo, i_hi = max(0, ri - r), min(ch, ri + r + 1)
            j_lo, j_hi = max(0, rj - r), min(cw, rj + r + 1)
            rgb[i_lo:i_hi, j_lo:j_hi] = np.maximum(
                rgb[i_lo:i_hi, j_lo:j_hi], np.uint8([255, 255, 255])
            )
            rgb[ri, rj] = (255, 255, 0)
    if trajectory_xyt:
        rgb = overlay_trajectory_on_map_rgb(
            rgb,
            trajectory_xyt,
            grid_origin_xy,
            grid_resolution,
            crop_offset_ij=(i0, j0),
            full_shape_hw=(h, w),
        )
    return finalize_export_topdown_rgb(rgb, max_side=max_side, min_side=min_map_side)


def _alpha_blend_rgb(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    """Blend ``overlay`` onto ``base`` where overlay is not white background."""
    out = np.ascontiguousarray(base)
    a = float(np.clip(alpha, 0.0, 1.0))
    mask = np.any(overlay != np.uint8([248, 248, 248]), axis=-1)
    if not mask.any():
        return out
    blended = (
        (1.0 - a) * out[mask].astype(np.float32) + a * overlay[mask].astype(np.float32)
    ).astype(np.uint8)
    out[mask] = blended
    return out


def eval_topdown_overlay_rgb(
    obstacles: Any,
    explored: Any,
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    robot_xy: np.ndarray | tuple[float, float] | None,
    *,
    gt_navigable: Any | None = None,
    max_side: int = 1280,
    min_map_side: int = 1024,
    margin_cells: int = 24,
    trajectory_xyt: list[tuple[float, float, float] | list[float]] | None = None,
    filter_islands: bool = True,
) -> np.ndarray:
    """Composite GT navmesh (slate) + agent explored/obstacles + trajectory path."""
    obs = _to_numpy_bool_2d(obstacles)
    exp = _to_numpy_bool_2d(explored)
    path = _dedupe_trajectory_xyt(trajectory_xyt) if trajectory_xyt else None
    from emet.visualization.map_grid import merge_trajectory_corridor_explored

    exp = merge_trajectory_corridor_explored(
        exp,
        obs,
        grid_origin_xy,
        grid_resolution,
        robot_xy=robot_xy,
        trajectory_xyt=path,
    )
    if filter_islands:
        from emet.visualization.map_grid import prune_explored_islands

        exp = prune_explored_islands(
            exp,
            grid_origin_xy=grid_origin_xy,
            grid_resolution=grid_resolution,
            robot_xy=robot_xy,
            trajectory_xyt=path,
        )
    h, w = obs.shape
    bbox = explored_crop_indices(
        exp,
        robot_xy,
        grid_origin_xy,
        grid_resolution,
        (h, w),
        margin_cells=margin_cells,
        trajectory_xyt=path,
    )
    if bbox is None:
        agent = eval_topdown_map_rgb(
            obs,
            exp,
            grid_origin_xy,
            grid_resolution,
            robot_xy,
            max_side=max_side,
            margin_cells=margin_cells,
            trajectory_xyt=trajectory_xyt,
            filter_islands=False,
        )
        return agent
    i0, i1, j0, j1 = bbox
    exp_c = exp[i0:i1, j0:j1]
    obs_c = obs[i0:i1, j0:j1]
    if gt_navigable is not None:
        from emet.habitat.navmesh_topdown import habitat_gt_topdown_rgb

        gt_c = np.asarray(gt_navigable, dtype=bool)[i0:i1, j0:j1]
        rgb = habitat_gt_topdown_rgb(gt_c, crop_slice=None, max_side=None)
    else:
        rgb = np.full((exp_c.shape[0], exp_c.shape[1], 3), 248, dtype=np.uint8)
    free = exp_c & ~obs_c
    rgb[free] = (50, 160, 80)
    rgb[exp_c & obs_c] = (200, 55, 55)
    if robot_xy is not None:
        ri, rj = world_xy_to_grid_ij(robot_xy, grid_origin_xy, grid_resolution, (h, w))
        ri -= i0
        rj -= j0
        ch, cw = rgb.shape[0], rgb.shape[1]
        if 0 <= ri < ch and 0 <= rj < cw:
            rgb[ri, rj] = (255, 255, 0)
    if trajectory_xyt:
        rgb = overlay_trajectory_on_map_rgb(
            rgb,
            trajectory_xyt,
            grid_origin_xy,
            grid_resolution,
            crop_offset_ij=(i0, j0),
            full_shape_hw=(h, w),
        )
    return finalize_export_topdown_rgb(rgb, max_side=max_side, min_side=min_map_side)


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
        if not stats["base_on_explored_cell"] and explored_cells > 0:
            # Distance from base to nearest explored cell (m) — large values → pose frame mismatch.
            exp_idx = np.argwhere(exp)
            d_cells = np.hypot(exp_idx[:, 0].astype(np.float64) - gi, exp_idx[:, 1].astype(np.float64) - gj)
            nearest_m = float(np.min(d_cells) * float(grid_resolution))
            stats["nearest_explored_m"] = nearest_m
        else:
            stats["nearest_explored_m"] = 0.0 if stats["base_on_explored_cell"] else None
    else:
        stats["base_xy"] = None
        stats["base_grid_ij"] = None
        stats["base_on_obstacle_cell"] = None
        stats["base_on_explored_cell"] = None
        stats["nearest_explored_m"] = None
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
            nearest = stats.get("nearest_explored_m")
            if nearest is not None and explored_cells > 0:
                lines.append(
                    f"Base cell not yet explored (nearest explored ≈ {nearest:.2f} m — "
                    "large gap often means gps vs world frame mismatch)."
                )
            else:
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


def snapshot_eval_from_voxel_map(
    voxel_map: Any,
    robot_xy: np.ndarray | tuple[float, float] | None,
    *,
    max_side: int = 1280,
    trajectory_xyt: list[tuple[float, float, float] | list[float]] | None = None,
    filter_islands: bool = False,
    gt_navigable: Any | None = None,
    min_map_side: int = 1024,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Build eval/diagnostics top-down map (white background, explored-only coloring)."""
    if voxel_map is None or not hasattr(voxel_map, "get_2d_map"):
        empty: dict[str, Any] = {
            "summary_lines": ["No voxel map attached (get_voxel_map unavailable)."],
            "map_nonempty": False,
        }
        return None, empty
    obstacles, explored = voxel_map.get_2d_map()
    go = _grid_origin_xy(getattr(voxel_map, "grid_origin", np.zeros(2)))
    res = float(getattr(voxel_map, "grid_resolution", 0.1) or 0.1)
    stats = build_map_stats(obstacles, explored, go, res, robot_xy)
    img = eval_topdown_map_rgb(
        obstacles,
        explored,
        go,
        res,
        robot_xy,
        max_side=max_side,
        min_map_side=min_map_side,
        trajectory_xyt=trajectory_xyt,
        filter_islands=filter_islands,
    )
    if gt_navigable is not None:
        stats["overlay_available"] = True
    return img, stats


def snapshot_eval_overlay_from_voxel_map(
    voxel_map: Any,
    robot_xy: np.ndarray | tuple[float, float] | None,
    *,
    max_side: int = 1280,
    trajectory_xyt: list[tuple[float, float, float] | list[float]] | None = None,
    gt_navigable: Any | None = None,
    filter_islands: bool = True,
    min_map_side: int = 1024,
) -> np.ndarray | None:
    """GT navmesh + agent map + trajectory composite for diagnostics export."""
    if voxel_map is None or not hasattr(voxel_map, "get_2d_map"):
        return None
    obstacles, explored = voxel_map.get_2d_map()
    go = _grid_origin_xy(getattr(voxel_map, "grid_origin", np.zeros(2)))
    res = float(getattr(voxel_map, "grid_resolution", 0.1) or 0.1)
    return eval_topdown_overlay_rgb(
        obstacles,
        explored,
        go,
        res,
        robot_xy,
        gt_navigable=gt_navigable,
        max_side=max_side,
        min_map_side=min_map_side,
        trajectory_xyt=trajectory_xyt,
        filter_islands=filter_islands,
    )


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

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class WorldMapRaster:
    """Explored / obstacle booleans on a fixed world XY grid (row=y, col=x)."""

    explored: np.ndarray
    obstacles: np.ndarray
    resolution_m: float
    origin_xy: tuple[float, float]  # world (x,y) of raster index (0,0) center or corner
    clip_rect: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax


def inset_clip_rect(
    rect: tuple[float, float, float, float],
    inset: float,
) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = rect
    if xmax - xmin < 2 * inset + 0.75 or ymax - ymin < 2 * inset + 0.75:
        return rect
    return (xmin + inset, xmax - inset, ymin + inset, ymax - inset)


def world_raster_from_voxel_map(
    voxel_map: Any,
    clip_rect: tuple[float, float, float, float],
    *,
    resolution_m: float = 0.1,
    inset_m: float = 0.0,
) -> WorldMapRaster:
    """Sample voxel 2D map onto a world-aligned grid covering *clip_rect*."""
    xmin, xmax, ymin, ymax = clip_rect
    if inset_m > 0:
        xmin, xmax, ymin, ymax = inset_clip_rect(clip_rect, inset_m)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"invalid clip_rect {clip_rect!r}")

    nx = max(1, int(np.ceil((xmax - xmin) / resolution_m)))
    ny = max(1, int(np.ceil((ymax - ymin) / resolution_m)))
    xs = xmin + (np.arange(nx, dtype=np.float64) + 0.5) * resolution_m
    ys = ymin + (np.arange(ny, dtype=np.float64) + 0.5) * resolution_m
    xx, yy = np.meshgrid(xs, ys)
    xy = np.stack([xx.ravel(), yy.ravel()], axis=1)

    obstacles, explored = voxel_map.get_2d_map()
    grid = voxel_map.grid
    device = obstacles.device
    xy_t = torch.tensor(xy, dtype=torch.float32, device=device)
    grid_xy = grid.xy_to_grid_coords_clamped(xy_t)
    rows = torch.floor(grid_xy[:, 0]).long().clamp(0, obstacles.shape[0] - 1)
    cols = torch.floor(grid_xy[:, 1]).long().clamp(0, obstacles.shape[1] - 1)

    exp_flat = explored[rows, cols].detach().cpu().numpy().astype(bool)
    obs_flat = obstacles[rows, cols].detach().cpu().numpy().astype(bool)
    return WorldMapRaster(
        explored=exp_flat.reshape(ny, nx),
        obstacles=obs_flat.reshape(ny, nx),
        resolution_m=float(resolution_m),
        origin_xy=(float(xmin), float(ymin)),
        clip_rect=(float(xmin), float(xmax), float(ymin), float(ymax)),
    )


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union for boolean masks of the same shape."""
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    if aa.shape != bb.shape:
        raise ValueError(f"mask shape mismatch {aa.shape} vs {bb.shape}")
    inter = int(np.count_nonzero(aa & bb))
    union = int(np.count_nonzero(aa | bb))
    if union == 0:
        return 1.0
    return float(inter / union)


@dataclass(frozen=True)
class WorldMapSimilarity:
    explored_iou: float
    obstacle_iou: float
    explored_a: int
    explored_b: int
    obstacles_a: int
    obstacles_b: int


def compare_world_rasters(a: WorldMapRaster, b: WorldMapRaster) -> WorldMapSimilarity:
    """Compare two world rasters (must share shape / clip)."""
    if a.explored.shape != b.explored.shape:
        raise ValueError(f"raster shape mismatch {a.explored.shape} vs {b.explored.shape}")
    return WorldMapSimilarity(
        explored_iou=binary_iou(a.explored, b.explored),
        obstacle_iou=binary_iou(a.obstacles, b.obstacles),
        explored_a=int(np.count_nonzero(a.explored)),
        explored_b=int(np.count_nonzero(b.explored)),
        obstacles_a=int(np.count_nonzero(a.obstacles)),
        obstacles_b=int(np.count_nonzero(b.obstacles)),
    )


def compare_world_rasters_in_shared_view(a: WorldMapRaster, b: WorldMapRaster) -> WorldMapSimilarity:
    """Compare maps where either robot reported explored free space (overlap-friendly).

    Static walls should agree in this region even when spawn positions differ and each
    robot only scanned part of the room.
    """
    if a.explored.shape != b.explored.shape:
        raise ValueError(f"raster shape mismatch {a.explored.shape} vs {b.explored.shape}")
    view = np.asarray(a.explored | b.explored, dtype=bool)
    if not np.any(view):
        return compare_world_rasters(a, b)
    exp_a = a.explored & view
    exp_b = b.explored & view
    obs_a = a.obstacles & view
    obs_b = b.obstacles & view
    return WorldMapSimilarity(
        explored_iou=binary_iou(exp_a, exp_b),
        obstacle_iou=binary_iou(obs_a, obs_b),
        explored_a=int(np.count_nonzero(exp_a)),
        explored_b=int(np.count_nonzero(exp_b)),
        obstacles_a=int(np.count_nonzero(obs_a)),
        obstacles_b=int(np.count_nonzero(obs_b)),
    )


def format_similarity_table(
    robots: list[str],
    rasters: dict[str, WorldMapRaster],
    *,
    reference: str,
    compare_fn=compare_world_rasters_in_shared_view,
) -> str:
    """Text table of explored/obstacle counts and IoU vs *reference*."""
    if reference not in rasters:
        raise KeyError(f"reference robot {reference!r} missing from rasters")
    ref = rasters[reference]
    lines = [
        f"{'robot':<12} {'explored':>9} {'obstacle':>9} {'exp_iou':>8} {'obs_iou':>8}",
        "(IoU in union of explored-free cells vs reference)",
        "-" * 52,
    ]
    for robot in robots:
        r = rasters[robot]
        if robot == reference:
            exp_iou, obs_iou = 1.0, 1.0
        else:
            sim = compare_fn(ref, rasters[robot])
            exp_iou, obs_iou = sim.explored_iou, sim.obstacle_iou
        lines.append(
            f"{robot:<12} {int(np.count_nonzero(r.explored)):>9} "
            f"{int(np.count_nonzero(r.obstacles)):>9} {exp_iou:>8.3f} {obs_iou:>8.3f}"
        )
    return "\n".join(lines)


def render_world_raster_ascii(raster: WorldMapRaster, *, max_side: int = 320) -> str:
    """ASCII view of a world raster (same symbols as navgrid debug)."""
    from emet.mapping.debug_navgrid_ascii import _downsample_maxpool, navgrid_max_side

    side = max_side if max_side > 0 else navgrid_max_side()
    ds_obs, ds_exp, step = _downsample_maxpool(raster.obstacles, raster.explored, side)
    lines: list[str] = [
        f"world_raster: {ds_obs.shape[1]}x{ds_obs.shape[0]} (~{raster.resolution_m * step:.2f}m/char stride={step})",
        "navgrid_key: '#'=obstacle '.'=explored_free ' '=unknown",
    ]
    for gr in range(ds_obs.shape[0]):
        row: list[str] = []
        for gc in range(ds_obs.shape[1]):
            if ds_obs[gr, gc]:
                row.append("#")
            elif ds_exp[gr, gc]:
                row.append(".")
            else:
                row.append(" ")
        lines.append("".join(row))
    return "\n".join(lines)

# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Explored-floor metrics for cross-robot Dynagraph export comparison.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FLOOR_METRICS_JSON = "floor_metrics.json"


def _unwrap_voxel_map(voxel_map: Any) -> Any:
    if voxel_map is None:
        return None
    if hasattr(voxel_map, "get_2d_map"):
        return voxel_map
    inner = getattr(voxel_map, "voxel_map", None)
    if inner is not None and hasattr(inner, "get_2d_map"):
        return inner
    return voxel_map


def _to_numpy_bool_2d(arr: Any) -> np.ndarray | None:
    if arr is None:
        return None
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    out = np.asarray(arr)
    if out.ndim != 2:
        return None
    return out.astype(bool)


def compute_explored_floor_metrics(
    voxel_map: Any,
    *,
    robot: str | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize the 2D explored mask from a DynaMem / voxel map."""
    vm = _unwrap_voxel_map(voxel_map)
    grid_resolution = float(getattr(vm, "grid_resolution", 0.05)) if vm is not None else 0.05
    grid_origin = getattr(vm, "grid_origin", None) if vm is not None else None
    if grid_origin is not None and hasattr(grid_origin, "cpu"):
        grid_origin = grid_origin.cpu().numpy()
    grid_origin_xy: list[float] | None = None
    if grid_origin is not None and np.size(grid_origin) >= 2:
        grid_origin_xy = [float(grid_origin[0]), float(grid_origin[1])]

    explored_2d: np.ndarray | None = None
    obstacles_2d: np.ndarray | None = None
    obstacle_cell_count = 0
    if vm is not None and hasattr(vm, "get_2d_map"):
        try:
            obstacles, explored = vm.get_2d_map()
            explored_2d = _to_numpy_bool_2d(explored)
            obstacles_2d = _to_numpy_bool_2d(obstacles)
            if obstacles_2d is not None:
                obstacle_cell_count = int(obstacles_2d.sum())
        except Exception:
            explored_2d = None
            obstacles_2d = None

    if explored_2d is None:
        return {
            "explored_cell_count": 0,
            "explored_area_m2": 0.0,
            "free_floor_cell_count": 0,
            "free_floor_area_m2": 0.0,
            "obstacle_cell_count": obstacle_cell_count,
            "grid_resolution_m": grid_resolution,
            "grid_origin_xy": grid_origin_xy,
            "explored_grid_shape": None,
            "explored_bounds_world_xy": None,
            "robot": robot,
            "environment": environment,
        }

    cell_count = int(explored_2d.sum())
    cell_area = grid_resolution * grid_resolution
    area_m2 = float(cell_count * cell_area)

    # Free floor = explored cells that are not obstacles (walkable footprint).
    if obstacles_2d is not None and obstacles_2d.shape == explored_2d.shape:
        free = explored_2d & ~obstacles_2d
        free_floor_cells = int(free.sum())
    else:
        free_floor_cells = cell_count
    free_floor_area_m2 = float(free_floor_cells * cell_area)

    bounds_world: list[list[float]] | None = None
    rows, cols = np.where(explored_2d)
    if rows.size > 0 and grid_origin_xy is not None:
        j_min, j_max = int(cols.min()), int(cols.max())
        i_min, i_max = int(rows.min()), int(rows.max())
        ox, oy = grid_origin_xy
        x_min = (j_min - ox) * grid_resolution
        x_max = (j_max - ox) * grid_resolution
        y_min = (i_min - oy) * grid_resolution
        y_max = (i_max - oy) * grid_resolution
        bounds_world = [[x_min, y_min], [x_max, y_max]]

    return {
        "explored_cell_count": cell_count,
        "explored_area_m2": area_m2,
        "free_floor_cell_count": free_floor_cells,
        "free_floor_area_m2": free_floor_area_m2,
        "obstacle_cell_count": obstacle_cell_count,
        "grid_resolution_m": grid_resolution,
        "grid_origin_xy": grid_origin_xy,
        "explored_grid_shape": [int(explored_2d.shape[0]), int(explored_2d.shape[1])],
        "explored_bounds_world_xy": bounds_world,
        "robot": robot,
        "environment": environment,
    }


def write_floor_metrics_json(export_dir: str | Path, metrics: dict[str, Any]) -> Path:
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / FLOOR_METRICS_JSON
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_floor_metrics(export_dir: str | Path) -> dict[str, Any]:
    path = Path(export_dir) / FLOOR_METRICS_JSON
    if not path.is_file():
        raise FileNotFoundError(f"Missing {FLOOR_METRICS_JSON} in {export_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def merge_spawn_floor_map(
    metrics: dict[str, Any],
    spawn_floor_map: dict[str, Any] | None,
) -> dict[str, Any]:
    if spawn_floor_map:
        metrics = dict(metrics)
        metrics["spawn_floor_map"] = spawn_floor_map
    return metrics


def explored_vs_spawn_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compare Dynagraph explored area to spawner walkable map (same grid when possible)."""
    explored_area = float(metrics.get("explored_area_m2", 0.0))
    explored_cells = int(metrics.get("explored_cell_count", 0))
    spawn = metrics.get("spawn_floor_map") or {}
    spawn_area = spawn.get("spawn_walkable_area_m2")
    spawn_cells = spawn.get("spawn_walkable_cell_count")
    scene_area = spawn.get("scene_walkable_area_m2")
    scene_cells = spawn.get("scene_walkable_cell_count")
    out: dict[str, Any] = {
        "explored_area_m2": explored_area,
        "explored_cell_count": explored_cells,
        "spawn_walkable_area_m2": spawn_area,
        "spawn_walkable_cell_count": spawn_cells,
        "scene_walkable_area_m2": scene_area,
        "scene_walkable_cell_count": scene_cells,
    }
    ref_area = scene_area if scene_area is not None else spawn_area
    ref_cells = scene_cells if scene_cells is not None else spawn_cells
    if ref_area is not None and float(ref_area) > 1e-9:
        out["explored_fraction_of_scene_walkable"] = float(explored_area / float(ref_area))
    if ref_cells is not None and int(ref_cells) > 0:
        out["explored_cell_fraction_of_scene_walkable"] = float(explored_cells / int(ref_cells))
    if spawn_area is not None and float(spawn_area) > 1e-9:
        out["explored_fraction_of_spawn"] = float(explored_area / float(spawn_area))
    if spawn_cells is not None and int(spawn_cells) > 0:
        out["explored_cell_fraction_of_spawn"] = float(explored_cells / int(spawn_cells))
    return out


def format_floor_metrics_summary(metrics: dict[str, Any]) -> str:
    cells = metrics.get("explored_cell_count", 0)
    area = metrics.get("explored_area_m2", 0.0)
    res = metrics.get("grid_resolution_m", 0.05)
    robot = metrics.get("robot")
    prefix = f"robot={robot!r} " if robot else ""
    lines = [(f"{prefix}explored floor: {cells} cells, {area:.3f} m² (grid_resolution={res:.3f} m/cell)")]
    free_cells = metrics.get("free_floor_cell_count")
    free_area = metrics.get("free_floor_area_m2")
    if free_cells is not None and free_area is not None:
        lines.append(f"free floor (explored ∩ ¬obstacle): {free_cells} cells, {float(free_area):.3f} m²")
    spawn = metrics.get("spawn_floor_map") or {}
    if spawn.get("scene_walkable_area_m2") is not None:
        sa = float(spawn["scene_walkable_area_m2"])
        sc = spawn.get("scene_walkable_cell_count", "?")
        lines.append(f"spawner scene walkable map: {sc} cells, {sa:.3f} m² (grid={spawn.get('grid_resolution_m')} m)")
    if spawn.get("spawn_walkable_area_m2") is not None:
        sa = float(spawn["spawn_walkable_area_m2"])
        sc = spawn.get("spawn_walkable_cell_count", "?")
        lines.append(f"spawner walkable map: {sc} cells, {sa:.3f} m² (clip_eroded={spawn.get('clip_eroded_area_m2')})")
        vs = explored_vs_spawn_summary(metrics)
        if "explored_fraction_of_scene_walkable" in vs:
            lines.append(f"explored / scene walkable: {100.0 * vs['explored_fraction_of_scene_walkable']:.1f}%")
        elif "explored_fraction_of_spawn" in vs:
            lines.append(f"explored / spawn walkable: {100.0 * vs['explored_fraction_of_spawn']:.1f}%")
    return "\n".join(lines)


def compare_explored_floor_metrics(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    atol_cells: int = 0,
    rtol_area: float = 0.05,
) -> dict[str, Any]:
    """Compare two floor-metric dicts; area/cells should match for same exploration on same scene."""
    lc = int(left.get("explored_cell_count", 0))
    rc = int(right.get("explored_cell_count", 0))
    la = float(left.get("explored_area_m2", 0.0))
    ra = float(right.get("explored_area_m2", 0.0))

    cell_delta = abs(lc - rc)
    area_ok = True
    if max(la, ra) > 1e-9:
        area_ok = abs(la - ra) / max(la, ra) <= rtol_area
    else:
        area_ok = abs(la - ra) <= 1e-6

    cells_ok = cell_delta <= atol_cells
    same_grid = (
        left.get("explored_grid_shape") == right.get("explored_grid_shape")
        and left.get("grid_origin_xy") == right.get("grid_origin_xy")
        and left.get("grid_resolution_m") == right.get("grid_resolution_m")
    )

    return {
        "cells_match": cells_ok,
        "area_match": area_ok,
        "same_grid_frame": same_grid,
        "cell_delta": cell_delta,
        "area_delta_m2": float(la - ra),
        "left": {"cells": lc, "area_m2": la, "robot": left.get("robot")},
        "right": {"cells": rc, "area_m2": ra, "robot": right.get("robot")},
        "match": cells_ok and area_ok,
    }

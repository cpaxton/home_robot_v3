# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Collision checks for arm configurations against the agent's voxel / 2D obstacle map.

Uses the same world model as base A* nav (``get_2d_map``), not MJCF/CuRobo geometry, so
planning stays consistent between sim and real robots.

**Grid indexing** matches ``GridParams`` / ``SparseVoxelMap`` by default:

    grid_i = floor(world_x / resolution + grid_origin[0])
    grid_j = floor(world_y / resolution + grid_origin[1])

where ``grid_origin`` is the map center in **cell indices**. Synthetic tests may pass a
world-meter origin with ``convention="world_offset"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import mujoco
import numpy as np

GridConvention = Literal["grid_params", "world_offset"]


def world_xy_to_grid(
    x: float,
    y: float,
    *,
    grid_origin: np.ndarray,
    resolution: float,
    convention: GridConvention = "grid_params",
) -> tuple[int, int]:
    """World XY → obstacle-grid indices."""
    go = np.asarray(grid_origin, dtype=np.float64).reshape(-1)
    res = float(resolution)
    if convention == "world_offset":
        gi = int(np.floor((float(x) - float(go[0])) / res))
        gj = int(np.floor((float(y) - float(go[1])) / res))
    else:
        gi = int(np.floor(float(x) / res + float(go[0])))
        gj = int(np.floor(float(y) / res + float(go[1])))
    return gi, gj


def _xy_to_grid(x: float, y: float, *, grid_origin: np.ndarray, resolution: float) -> tuple[int, int]:
    """Backward-compatible alias (world_offset) used by older unit tests."""
    return world_xy_to_grid(x, y, grid_origin=grid_origin, resolution=resolution, convention="world_offset")


def link_samples_collide_2d(
    obstacles: np.ndarray,
    *,
    grid_origin: np.ndarray,
    resolution: float,
    sample_xy: Sequence[tuple[float, float]],
    inflate_cells: int = 0,
    convention: GridConvention = "world_offset",
) -> bool:
    """True if any sample XY falls in an occupied (optionally dilated) cell."""
    obs = obstacles
    if hasattr(obstacles, "detach"):
        obs = obstacles.detach().cpu().numpy()
    obs = np.asarray(obs, dtype=bool)
    h, w = obs.shape[:2]
    pad = max(int(inflate_cells), 0)
    for x, y in sample_xy:
        gi, gj = world_xy_to_grid(
            float(x), float(y), grid_origin=grid_origin, resolution=resolution, convention=convention
        )
        for di in range(-pad, pad + 1):
            for dj in range(-pad, pad + 1):
                ii, jj = gi + di, gj + dj
                if 0 <= ii < h and 0 <= jj < w and bool(obs[ii, jj]):
                    return True
    return False


def fk_link_xy_samples(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_names: Sequence[str],
) -> list[tuple[float, float]]:
    """World XY of named bodies after ``mj_forward`` (caller must set qpos)."""
    mujoco.mj_forward(model, data)
    out: list[tuple[float, float]] = []
    for name in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
        if bid < 0:
            continue
        xpos = data.body(bid).xpos
        out.append((float(xpos[0]), float(xpos[1])))
    return out


def _as_numpy_bool_obstacles(obstacles: Any) -> np.ndarray:
    if hasattr(obstacles, "detach"):
        return np.asarray(obstacles.detach().cpu().numpy(), dtype=bool)
    return np.asarray(obstacles, dtype=bool)


def _grid_params_from_voxel_map(voxel_map: Any) -> tuple[np.ndarray, float, GridConvention] | None:
    """Return (grid_origin, resolution, convention) for a voxel map."""
    grid = getattr(voxel_map, "grid", None)
    if grid is not None and hasattr(grid, "grid_origin"):
        go = grid.grid_origin
        if hasattr(go, "detach"):
            go = go.detach().cpu().numpy()
        go = np.asarray(go, dtype=np.float64).reshape(-1)[:2]
        res = getattr(voxel_map, "grid_resolution", None)
        if res is None:
            res = getattr(grid, "resolution", None)
        if res is None:
            return None
        return go, float(res), "grid_params"

    origin = getattr(voxel_map, "origin", None) or getattr(voxel_map, "grid_origin", None)
    resolution = (
        getattr(voxel_map, "voxel_size", None)
        or getattr(voxel_map, "grid_resolution", None)
        or getattr(voxel_map, "resolution", None)
    )
    if origin is None or resolution is None:
        return None
    if hasattr(origin, "detach"):
        origin = origin.detach().cpu().numpy()
    return np.asarray(origin, dtype=np.float64).reshape(-1)[:2], float(resolution), "world_offset"


class VoxelMapArmCollisionChecker:
    """Validate arm FK samples against a 2D obstacle grid from the agent voxel map."""

    def __init__(
        self,
        obstacles: np.ndarray,
        *,
        grid_origin: np.ndarray | list[float],
        resolution: float,
        link_bodies: Sequence[str],
        inflate_cells: int = 1,
        convention: GridConvention = "world_offset",
    ) -> None:
        self.obstacles = _as_numpy_bool_obstacles(obstacles)
        self.grid_origin = np.asarray(grid_origin, dtype=np.float64).reshape(-1)[:2]
        self.resolution = float(resolution)
        self.link_bodies = list(link_bodies)
        self.inflate_cells = int(inflate_cells)
        self.convention: GridConvention = convention

    @classmethod
    def from_voxel_map(
        cls,
        voxel_map: Any,
        *,
        link_bodies: Sequence[str],
        inflate_cells: int = 1,
    ) -> VoxelMapArmCollisionChecker | None:
        """Build from a map that implements ``get_2d_map()`` → (obstacles, explored, …).

        Prefers ``SparseVoxelMap`` / ``GridParams`` (``grid.grid_origin`` in cells). Falls back to
        attributes ``origin`` / ``voxel_size`` with world-meter origin.
        """
        get_2d = getattr(voxel_map, "get_2d_map", None)
        if not callable(get_2d):
            return None
        result = get_2d()
        if result is None:
            return None
        if isinstance(result, dict):
            obstacles = result.get("obstacles")
            meta = _grid_params_from_voxel_map(voxel_map)
            if meta is None:
                origin = result.get("origin") or result.get("grid_origin")
                resolution = result.get("resolution") or result.get("voxel_size")
                if obstacles is None or origin is None or resolution is None:
                    return None
                return cls(
                    obstacles,
                    grid_origin=origin,
                    resolution=float(resolution),
                    link_bodies=link_bodies,
                    inflate_cells=inflate_cells,
                    convention="world_offset",
                )
            origin, resolution, convention = meta
        else:
            obstacles = result[0] if len(result) > 0 else None
            meta = _grid_params_from_voxel_map(voxel_map)
            if obstacles is None or meta is None:
                return None
            origin, resolution, convention = meta
        return cls(
            obstacles,
            grid_origin=origin,
            resolution=resolution,
            link_bodies=link_bodies,
            inflate_cells=inflate_cells,
            convention=convention,
        )

    def configuration_collides(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        samples = fk_link_xy_samples(model, data, self.link_bodies)
        return link_samples_collide_2d(
            self.obstacles,
            grid_origin=self.grid_origin,
            resolution=self.resolution,
            sample_xy=samples,
            inflate_cells=self.inflate_cells,
            convention=self.convention,
        )

    def trajectory_collides(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        joint_names: Sequence[str],
        arm_waypoints: Sequence[np.ndarray],
    ) -> int | None:
        """Return index of first colliding waypoint, or None if clear."""
        from emet.motion.mujoco_arm_ik import joint_qpos_addrs

        qadr = joint_qpos_addrs(model, list(joint_names))
        for i, q in enumerate(arm_waypoints):
            qq = np.asarray(q, dtype=np.float64).reshape(-1)
            for a, v in zip(qadr, qq, strict=True):
                data.qpos[a] = float(v)
            if self.configuration_collides(model, data):
                return i
        return None

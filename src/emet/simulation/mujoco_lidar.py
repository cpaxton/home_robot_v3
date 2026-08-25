# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Read replicated MuJoCo base_lidar rangefinders into ZMQ ``lidar_points``."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


def base_lidar_sensor_names(model: mujoco.MjModel, *, prefix: str = "base_lidar") -> list[str]:
    names: list[str] = []
    for i in range(int(model.nsensor)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        if name and name.startswith(prefix):
            names.append(name)
    names.sort()
    return names


def model_has_base_lidar(model: mujoco.MjModel, *, prefix: str = "base_lidar") -> bool:
    return bool(base_lidar_sensor_names(model, prefix=prefix))


def read_base_lidar_ranges(
    mjdata: mujoco.MjData,
    model: mujoco.MjModel,
    *,
    prefix: str = "base_lidar",
) -> np.ndarray | None:
    names = base_lidar_sensor_names(model, prefix=prefix)
    if not names:
        return None
    return np.array([float(mjdata.sensor(name).data[0]) for name in names], dtype=np.float64)


def lidar_ranges_to_points(
    ranges: np.ndarray,
    *,
    max_range: float = 10.0,
) -> np.ndarray:
    """Convert planar rangefinder hits to Nx2 points in the lidar frame (+X forward at ray 0)."""
    ranges = np.asarray(ranges, dtype=np.float64).reshape(-1)
    n = int(ranges.size)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    angles = np.arange(n, dtype=np.float64) * (2.0 * np.pi / n)
    cleaned = ranges.copy()
    cleaned[~np.isfinite(cleaned)] = max_range
    valid = (cleaned > 1e-3) & (cleaned < max_range - 1e-3)
    r = cleaned[valid]
    a = angles[valid]
    if r.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.column_stack((r * np.cos(a), r * np.sin(a)))


def attach_lidar_to_zmq_message(
    message: dict[str, Any],
    model: mujoco.MjModel,
    mjdata: mujoco.MjData,
    *,
    max_range: float = 10.0,
) -> None:
    """Populate ``lidar_points`` / ``lidar_timestamp`` when the MJCF defines ``base_lidar*`` sensors."""
    ranges = read_base_lidar_ranges(mjdata, model)
    if ranges is None:
        return
    message["lidar_points"] = lidar_ranges_to_points(ranges, max_range=max_range)
    message["lidar_timestamp"] = int(float(mjdata.time) * 1e9)

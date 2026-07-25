# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Helpers to face a world XY point (CHAT ``face_toward`` tool)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def signed_yaw_delta_rad(current_yaw: float, target_yaw: float) -> float:
    """Smallest signed yaw change (rad): positive = left / CCW."""
    return (float(target_yaw) - float(current_yaw) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_face_xy(
    base_xyt: list[float] | np.ndarray,
    target_xy: list[float] | np.ndarray,
) -> tuple[float, float]:
    """Return ``(delta_yaw_rad, bearing_rad)`` to face ``target_xy`` from ``base_xyt``."""
    pose = np.asarray(base_xyt, dtype=np.float64).reshape(-1)
    tgt = np.asarray(target_xy, dtype=np.float64).reshape(-1)
    if pose.size < 3 or tgt.size < 2:
        raise ValueError("base_xyt needs x,y,theta and target_xy needs x,y")
    bearing = math.atan2(float(tgt[1]) - float(pose[1]), float(tgt[0]) - float(pose[0]))
    return signed_yaw_delta_rad(float(pose[2]), bearing), bearing


def resolve_object_xy(agent: Any, label: str) -> tuple[np.ndarray | None, str]:
    """Best-effort world XY for *label* from graph memory or voxel localize_text."""
    query = (label or "").strip()
    if not query:
        return None, "empty label"
    if agent is not None and hasattr(agent, "_localize_point_from_graph_memory"):
        try:
            pt = agent._localize_point_from_graph_memory(query)
            if pt is not None:
                arr = np.asarray(pt, dtype=np.float64).reshape(-1)
                if arr.size >= 2 and np.isfinite(arr[:2]).all():
                    return arr[:2].copy(), "graph"
        except Exception:
            pass
    if agent is not None and hasattr(agent, "get_voxel_map"):
        try:
            vm = agent.get_voxel_map()
            if vm is not None and hasattr(vm, "localize_text"):
                result = vm.localize_text(query, return_debug=True)
                point = result[0] if isinstance(result, (list, tuple)) else result
                if point is not None:
                    coords = np.asarray(point, dtype=np.float64).reshape(-1)
                    if coords.size >= 2 and np.isfinite(coords[:2]).all():
                        return coords[:2].copy(), "voxel"
        except Exception:
            pass
    return None, "not found"

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared planar geometry for graph memory, rooms, RAG, and frontiers."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class UnionFind:
    """Disjoint-set for clustering node ids."""

    def __init__(self, ids: Sequence[int]) -> None:
        self.parent = {int(i): int(i) for i in ids}

    def find(self, x: int) -> int:
        x = int(x)
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def xy(node: Any) -> tuple[float, float]:
    """Planar XY from a graph node or xyz-like value."""
    xyz_val = getattr(node, "xyz", node)
    return float(xyz_val[0]), float(xyz_val[1])


def planar_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def xyz(value: Any) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    return (
        float(arr[0]) if arr.size > 0 else 0.0,
        float(arr[1]) if arr.size > 1 else 0.0,
        float(arr[2]) if arr.size > 2 else 0.0,
    )


_ROOM_WORDS = frozenset(
    {
        "room",
        "kitchen",
        "bedroom",
        "bathroom",
        "living room",
        "dining room",
        "hallway",
        "office",
        "garage",
        "sunroom",
    }
)


def _near(p1: np.ndarray, p2: np.ndarray, max_dist: float = 1.5) -> bool:
    return float(np.linalg.norm(p1[:2] - p2[:2])) <= max_dist


def _on(p_lower: np.ndarray, p_upper: np.ndarray, z_thresh: float = 0.15) -> bool:
    """Heuristic: lower object is 'on' upper if roughly below and close in xy."""
    if p_lower[2] >= p_upper[2]:
        return False
    return abs(p_lower[2] - p_upper[2]) <= z_thresh + 0.2 and float(np.linalg.norm(p_lower[:2] - p_upper[:2])) < 0.5


def _on_floor(p: np.ndarray, floor_z: float = 0.05) -> bool:
    return float(p[2]) <= floor_z


def _node_is_room(node: Any) -> bool:
    text = " ".join(node.labels).lower()
    return any(word in text for word in _ROOM_WORDS)


def _inside_bounds(point: np.ndarray, bounds: dict[str, list[float]] | None) -> bool:
    if not bounds or "min" not in bounds or "max" not in bounds:
        return False
    xyz = np.asarray(point, dtype=float).reshape(-1)[:3]
    lower = np.asarray(bounds["min"], dtype=float).reshape(-1)[:3]
    upper = np.asarray(bounds["max"], dtype=float).reshape(-1)[:3]
    return bool(np.all(xyz >= lower) and np.all(xyz <= upper))


near = _near
on = _on
on_floor = _on_floor
node_is_room = _node_is_room
inside_bounds = _inside_bounds
_xy = xy
_xyz = xyz
_planar_dist = planar_dist
_UnionFind = UnionFind

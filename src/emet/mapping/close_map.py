# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""2D close-look map on the occupancy XY plane.

Each occupancy cell stores the nearest camera-to-surface range that hit it and
whether the optical axis was aimed at that hit. ``resolved`` means we actually
got a close, on-axis look — occupancy exploration is not enough for 4 cm objects.

Used by agentic find (stay on a place card until close or escape) and TAMP /
CHAT as a shared “have we resolved this XY?” signal.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_R_CLOSE_M = 0.55
DEFAULT_AIM_DEG = 25.0
DEFAULT_QUERY_RADIUS_M = 0.35
DEFAULT_ESCAPE_ATTEMPTS = 4
DEFAULT_CHAT_ESCAPE_ATTEMPTS = 2

_TRUE = frozenset({"1", "true", "yes", "on"})


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    return int(raw)


def close_map_r_close_m(override: float | None = None) -> float:
    if override is not None:
        return float(override)
    return _env_float("EMET_CLOSE_MAP_R_M", DEFAULT_R_CLOSE_M)


def close_map_aim_deg(override: float | None = None) -> float:
    if override is not None:
        return float(override)
    return _env_float("EMET_CLOSE_MAP_AIM_DEG", DEFAULT_AIM_DEG)


def close_map_query_radius_m(override: float | None = None) -> float:
    if override is not None:
        return float(override)
    return _env_float("EMET_CLOSE_MAP_QUERY_RADIUS_M", DEFAULT_QUERY_RADIUS_M)


def close_map_escape_attempts(*, is_chat: bool = False, override: int | None = None) -> int:
    if override is not None:
        return max(1, int(override))
    if is_chat:
        return max(1, _env_int("EMET_CLOSE_MAP_CHAT_ESCAPE_ATTEMPTS", DEFAULT_CHAT_ESCAPE_ATTEMPTS))
    return max(1, _env_int("EMET_CLOSE_MAP_ESCAPE_ATTEMPTS", DEFAULT_ESCAPE_ATTEMPTS))


@dataclass(frozen=True)
class CloseLookQuery:
    """Neighborhood stats around a world XY."""

    x: float
    y: float
    radius_m: float
    n_hit_cells: int
    n_resolved_cells: int
    min_cam_dist_m: float | None
    aimed_hit: bool
    resolved: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "radius_m": float(self.radius_m),
            "n_hit_cells": int(self.n_hit_cells),
            "n_resolved_cells": int(self.n_resolved_cells),
            "min_cam_dist_m": (None if self.min_cam_dist_m is None else float(self.min_cam_dist_m)),
            "aimed_hit": bool(self.aimed_hit),
            "resolved": bool(self.resolved),
        }


@dataclass(frozen=True)
class CloseLookDecision:
    """Stay on this XY for another close look, or escape (unreachable / budget)."""

    stay: bool
    escape: bool
    reason: str
    query: CloseLookQuery | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "stay": bool(self.stay),
            "escape": bool(self.escape),
            "reason": str(self.reason),
        }
        if self.query is not None:
            out["query"] = self.query.as_dict()
        return out


def decide_close_look(
    query: CloseLookQuery | None,
    *,
    approaches_left: int,
    nav_blocked: bool,
    attempts: int,
    is_chat: bool = False,
    max_attempts: int | None = None,
) -> CloseLookDecision:
    """Policy: stay while unresolved and reachable; escape when stuck.

    Chat / agent mode uses a tighter attempt cap so the robot can leave
    unreachable furniture instead of orbiting forever.
    """
    cap = close_map_escape_attempts(is_chat=is_chat, override=max_attempts)
    if query is None:
        return CloseLookDecision(stay=False, escape=False, reason="no_map")
    if query.resolved:
        return CloseLookDecision(stay=False, escape=False, reason="resolved", query=query)
    if nav_blocked:
        return CloseLookDecision(stay=False, escape=True, reason="escape_unreachable", query=query)
    if int(attempts) >= int(cap) or int(approaches_left) <= 0:
        reason = "escape_chat" if is_chat else "escape_exhausted"
        return CloseLookDecision(stay=False, escape=True, reason=reason, query=query)
    return CloseLookDecision(stay=True, escape=False, reason="unresolved_stay", query=query)


class CloseDistanceMap:
    """Min camera range + aimed flag per occupancy cell."""

    def __init__(
        self,
        grid_size: tuple[int, int],
        origin_xy: np.ndarray | list[float],
        resolution_m: float,
        *,
        r_close_m: float | None = None,
        aim_deg: float | None = None,
        query_radius_m: float | None = None,
    ) -> None:
        h, w = int(grid_size[0]), int(grid_size[1])
        self.grid_size = (h, w)
        self.origin_xy = np.asarray(origin_xy, dtype=np.float64).reshape(-1)[:2].copy()
        self.resolution_m = float(resolution_m)
        self.r_close_m = close_map_r_close_m(r_close_m)
        self.aim_deg = close_map_aim_deg(aim_deg)
        self.query_radius_m = close_map_query_radius_m(query_radius_m)
        self._aim_cos = math.cos(math.radians(self.aim_deg))
        self.min_cam_dist = np.full((h, w), np.inf, dtype=np.float32)
        self.aimed = np.zeros((h, w), dtype=bool)
        self.n_hits = np.zeros((h, w), dtype=np.uint16)
        self.n_updates = 0

    def reset(self) -> None:
        self.min_cam_dist.fill(np.inf)
        self.aimed.fill(False)
        self.n_hits.fill(0)
        self.n_updates = 0

    def world_xy_to_ij(self, x: float, y: float) -> tuple[int, int] | None:
        i = int(np.rint(float(x) / self.resolution_m + self.origin_xy[0]))
        j = int(np.rint(float(y) / self.resolution_m + self.origin_xy[1]))
        h, w = self.grid_size
        if i < 0 or j < 0 or i >= h or j >= w:
            return None
        return i, j

    def update_from_view(
        self,
        camera_pose: np.ndarray,
        world_xyz: np.ndarray,
        valid: np.ndarray | None = None,
        *,
        stride: int = 1,
    ) -> int:
        """Project valid world points into the grid. Returns cells touched this view."""
        pose = np.asarray(camera_pose, dtype=np.float64)
        if pose.shape != (4, 4):
            return 0
        pts = np.asarray(world_xyz, dtype=np.float64)
        if pts.size == 0:
            return 0
        if pts.ndim == 3:
            pts = pts.reshape(-1, 3)
        elif pts.ndim != 2 or pts.shape[-1] < 3:
            return 0
        if valid is not None:
            mask = np.asarray(valid, dtype=bool).reshape(-1)
            if mask.shape[0] != pts.shape[0]:
                return 0
            pts = pts[mask]
        if pts.shape[0] == 0:
            return 0
        stride = max(1, int(stride))
        if pts.shape[0] > 40000 and stride == 1:
            stride = 2
        if stride > 1:
            pts = pts[::stride]
        cam = pose[:3, 3]
        fwd = pose[:3, 2]
        fn = float(np.linalg.norm(fwd))
        if fn < 1e-9:
            return 0
        fwd = fwd / fn
        vec = pts[:, :3] - cam.reshape(1, 3)
        dist = np.linalg.norm(vec, axis=1)
        good = np.isfinite(dist) & (dist > 1e-4)
        if not np.any(good):
            return 0
        pts = pts[good]
        dist = dist[good]
        vec = vec[good]
        vn = dist.reshape(-1, 1)
        aimed = (vec * (1.0 / vn)).dot(fwd) >= self._aim_cos
        gi = np.rint(pts[:, 0] / self.resolution_m + self.origin_xy[0]).astype(np.int32)
        gj = np.rint(pts[:, 1] / self.resolution_m + self.origin_xy[1]).astype(np.int32)
        h, w = self.grid_size
        inside = (gi >= 0) & (gj >= 0) & (gi < h) & (gj < w)
        if not np.any(inside):
            return 0
        gi, gj, dist, aimed = gi[inside], gj[inside], dist[inside].astype(np.float32), aimed[inside]
        np.minimum.at(self.min_cam_dist, (gi, gj), dist)
        self.aimed[gi, gj] |= aimed
        ones = np.ones(gi.shape[0], dtype=np.uint16)
        np.add.at(self.n_hits, (gi, gj), ones)
        self.n_hits = np.minimum(self.n_hits, 60000, out=self.n_hits)
        self.n_updates += 1
        return int(np.unique(gi.astype(np.int64) * w + gj).size)

    def query_xy(
        self,
        x: float,
        y: float,
        *,
        radius_m: float | None = None,
    ) -> CloseLookQuery:
        rad = float(self.query_radius_m if radius_m is None else radius_m)
        h, w = self.grid_size
        i0 = int(np.rint(float(x) / self.resolution_m + self.origin_xy[0]))
        j0 = int(np.rint(float(y) / self.resolution_m + self.origin_xy[1]))
        n_cells = max(0, int(np.ceil(rad / self.resolution_m)))
        i_lo, i_hi = max(0, i0 - n_cells), min(h, i0 + n_cells + 1)
        j_lo, j_hi = max(0, j0 - n_cells), min(w, j0 + n_cells + 1)
        if i_lo >= i_hi or j_lo >= j_hi:
            return CloseLookQuery(
                x=float(x),
                y=float(y),
                radius_m=rad,
                n_hit_cells=0,
                n_resolved_cells=0,
                min_cam_dist_m=None,
                aimed_hit=False,
                resolved=False,
            )
        ii, jj = np.ogrid[i_lo:i_hi, j_lo:j_hi]
        wx = (ii.astype(np.float64) - self.origin_xy[0]) * self.resolution_m
        wy = (jj.astype(np.float64) - self.origin_xy[1]) * self.resolution_m
        in_r = (wx - float(x)) ** 2 + (wy - float(y)) ** 2 <= rad * rad
        hits = (self.n_hits[i_lo:i_hi, j_lo:j_hi] > 0) & in_r
        n_hit = int(hits.sum())
        if n_hit == 0:
            return CloseLookQuery(
                x=float(x),
                y=float(y),
                radius_m=rad,
                n_hit_cells=0,
                n_resolved_cells=0,
                min_cam_dist_m=None,
                aimed_hit=False,
                resolved=False,
            )
        dist = self.min_cam_dist[i_lo:i_hi, j_lo:j_hi]
        aimed = self.aimed[i_lo:i_hi, j_lo:j_hi] & hits
        resolved_mask = hits & aimed & (dist <= self.r_close_m)
        n_res = int(resolved_mask.sum())
        min_d = float(np.min(dist[hits]))
        return CloseLookQuery(
            x=float(x),
            y=float(y),
            radius_m=rad,
            n_hit_cells=n_hit,
            n_resolved_cells=n_res,
            min_cam_dist_m=min_d,
            aimed_hit=bool(np.any(aimed)),
            resolved=n_res > 0,
        )

    def summary(self) -> dict[str, Any]:
        hit = self.n_hits > 0
        n_hit = int(hit.sum())
        n_res = int((hit & self.aimed & (self.min_cam_dist <= self.r_close_m)).sum())
        finite = self.min_cam_dist[hit]
        return {
            "r_close_m": float(self.r_close_m),
            "aim_deg": float(self.aim_deg),
            "n_updates": int(self.n_updates),
            "n_hit_cells": n_hit,
            "n_resolved_cells": n_res,
            "min_cam_dist_m": (None if finite.size == 0 else float(np.min(finite))),
        }


def close_map_from_voxel_map(voxel_map: Any) -> CloseDistanceMap | None:
    cm = getattr(voxel_map, "close_map", None) if voxel_map is not None else None
    return cm if isinstance(cm, CloseDistanceMap) else None


def close_map_from_agent(agent: Any) -> CloseDistanceMap | None:
    if agent is None:
        return None
    return close_map_from_voxel_map(getattr(agent, "voxel_map", None))


def format_close_map_hint(
    cm: CloseDistanceMap | None,
    x: float,
    y: float,
    *,
    is_chat: bool = False,
) -> str:
    """One-line hint for CHAT / TAMP: stay, or escape if unreachable."""
    if cm is None:
        return ""
    q = cm.query_xy(x, y)
    if q.resolved:
        d = q.min_cam_dist_m
        d_s = "unknown" if d is None else f"{d:.2f}m"
        return f"Close-look map: resolved at ({x:.2f}, {y:.2f}) min_cam={d_s} (r≤{cm.r_close_m:.2f}m, aimed)."
    d = q.min_cam_dist_m
    if q.n_hit_cells == 0:
        seen = "never seen this XY"
    elif d is None:
        seen = "seen but no range"
    else:
        seen = f"min_cam={d:.2f}m aimed={q.aimed_hit}"
    if is_chat:
        return (
            f"Close-look map: NOT resolved at ({x:.2f}, {y:.2f}) ({seen}; need ≤{cm.r_close_m:.2f}m aimed). "
            "Approach closer if reachable; if nav keeps failing, stop retrying and explore or ask."
        )
    return (
        f"Close-look map: NOT resolved at ({x:.2f}, {y:.2f}) ({seen}; need ≤{cm.r_close_m:.2f}m aimed). "
        "Stay for another approach unless the cell is unreachable."
    )

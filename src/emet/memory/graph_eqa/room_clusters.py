# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Graph-derived room / region clusters for GraphEQA.

Partitions object nodes into connected components using ``near`` edges and a
planar link radius, then names each component with the same vocabulary as the
agentic router ``current_room`` field. Not a Hydra Places layer — deterministic
CC grouping only.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from emet.memory.graph_eqa.agentic_tools import normalize_current_room, room_is_outdoor

DEFAULT_ROOM_LINK_RADIUS_M = 2.0
DEFAULT_ROOM_ASSIGN_MAX_M = 3.0


@dataclass(frozen=True)
class RoomCluster:
    """One connected group of object nodes treated as a room/region."""

    cluster_id: int
    node_ids: tuple[int, ...]
    labels: tuple[str, ...]
    centroid_xy: tuple[float, float]
    room_name: str
    area_proxy: float = 0.0
    is_outdoor: bool = False


class _UnionFind:
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


def _xy(node: Any) -> tuple[float, float]:
    xyz = getattr(node, "xyz", (0.0, 0.0, 0.0))
    return float(xyz[0]), float(xyz[1])


def _planar_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _label_list(node: Any) -> list[str]:
    out: list[str] = []
    for lab in list(getattr(node, "labels", None) or []):
        s = str(lab).strip()
        if s and s.lower() != "frontier":
            out.append(s)
    return out


def _is_object_node(node: Any) -> bool:
    return not bool(getattr(node, "is_frontier", False)) and not bool(getattr(node, "is_viewpoint", False))


def name_cluster_from_labels(labels: Sequence[str]) -> str:
    """Majority vote of normalized room tokens from object labels."""
    votes: Counter[str] = Counter()
    for lab in labels:
        name = normalize_current_room(lab)
        if name != "unknown":
            votes[name] += 1
        # Also try each whitespace token (e.g. "brick patio chair").
        for tok in str(lab).lower().replace("-", " ").split():
            tname = normalize_current_room(tok)
            if tname != "unknown":
                votes[tname] += 1
    if not votes:
        return "unknown"
    # Prefer outdoor/patio over generic indoor when tied with outdoor signal.
    best = votes.most_common()
    top_count = best[0][1]
    tied = [n for n, c in best if c == top_count]
    for prefer in ("patio", "outdoor", "kitchen", "living_room"):
        if prefer in tied:
            return prefer
    return best[0][0]


def cluster_object_nodes(
    nodes: Sequence[Any],
    edges: Sequence[tuple[int, int, str]] | None = None,
    *,
    link_radius_m: float = DEFAULT_ROOM_LINK_RADIUS_M,
) -> list[RoomCluster]:
    """Connected components over object nodes via ``near`` edges and planar radius."""
    objects = [n for n in nodes if _is_object_node(n)]
    if not objects:
        return []
    ids = [int(getattr(n, "node_id", i)) for i, n in enumerate(objects)]
    by_id = {int(getattr(n, "node_id", i)): n for i, n in enumerate(objects)}
    uf = _UnionFind(ids)

    id_set = set(ids)
    for a, b, rel in edges or ():
        if str(rel) != "near":
            continue
        ai, bi = int(a), int(b)
        if ai in id_set and bi in id_set:
            uf.union(ai, bi)

    radius = float(link_radius_m)
    for i, a in enumerate(objects):
        axy = _xy(a)
        aid = ids[i]
        for j in range(i + 1, len(objects)):
            if _planar_dist(axy, _xy(objects[j])) <= radius:
                uf.union(aid, ids[j])

    groups: dict[int, list[Any]] = {}
    for nid in ids:
        groups.setdefault(uf.find(nid), []).append(by_id[nid])

    clusters: list[RoomCluster] = []
    for root, members in groups.items():
        labels: list[str] = []
        xs: list[float] = []
        ys: list[float] = []
        for n in members:
            for lab in _label_list(n):
                if lab not in labels:
                    labels.append(lab)
            x, y = _xy(n)
            xs.append(x)
            ys.append(y)
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        # Bounding-box area proxy (m²) for size ranking in compact summaries.
        if len(xs) >= 2:
            area = max(0.01, (max(xs) - min(xs)) * (max(ys) - min(ys)))
        else:
            area = 0.01
        room_name = name_cluster_from_labels(labels)
        clusters.append(
            RoomCluster(
                cluster_id=int(root),
                node_ids=tuple(int(getattr(n, "node_id", -1)) for n in members),
                labels=tuple(labels[:32]),
                centroid_xy=(float(cx), float(cy)),
                room_name=room_name,
                area_proxy=float(area),
                is_outdoor=room_is_outdoor(room_name),
            )
        )

    # Stable display order: larger first, then cluster_id; renumber 1..N.
    clusters.sort(key=lambda c: (-c.area_proxy, -len(c.node_ids), c.cluster_id))
    out: list[RoomCluster] = []
    for i, c in enumerate(clusters, start=1):
        out.append(
            RoomCluster(
                cluster_id=i,
                node_ids=c.node_ids,
                labels=c.labels,
                centroid_xy=c.centroid_xy,
                room_name=c.room_name,
                area_proxy=c.area_proxy,
                is_outdoor=c.is_outdoor,
            )
        )
    return out


def estimate_room_at_xy(
    clusters: Sequence[RoomCluster],
    robot_xy: tuple[float, float] | Sequence[float],
    *,
    max_dist_m: float = DEFAULT_ROOM_ASSIGN_MAX_M,
) -> str:
    """Nearest cluster room name within ``max_dist_m``, else ``unknown``."""
    if not clusters:
        return "unknown"
    xy = (float(robot_xy[0]), float(robot_xy[1]))
    best: RoomCluster | None = None
    best_d = float("inf")
    for c in clusters:
        d = _planar_dist(xy, c.centroid_xy)
        if d < best_d:
            best_d = d
            best = c
    if best is None or best_d > float(max_dist_m):
        return "unknown"
    return normalize_current_room(best.room_name)


def format_rooms_compact(clusters: Sequence[RoomCluster], *, max_chars: int = 200) -> str:
    """Short ``Rooms: kitchen(12), patio(5), …`` line for router / memory prompts."""
    if not clusters:
        return ""
    parts: list[str] = []
    for c in clusters:
        name = c.room_name if c.room_name != "unknown" else f"region_{c.cluster_id}"
        parts.append(f"{name}({len(c.node_ids)})")
    line = "Rooms: " + ", ".join(parts)
    if len(line) > int(max_chars):
        line = line[: max(0, int(max_chars) - 1)].rstrip() + "…"
    return line


def merge_room_estimates(vlm_room: str | None, graph_room: str | None) -> str:
    """Prefer a known graph estimate over VLM; else fall back to VLM / unknown."""
    g = normalize_current_room(graph_room)
    v = normalize_current_room(vlm_room)
    if g != "unknown":
        return g
    return v

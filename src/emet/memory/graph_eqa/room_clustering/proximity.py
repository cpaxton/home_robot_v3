# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Naive room partition: ``near`` edges plus planar XY radius."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from emet.memory.graph_eqa.room_clustering.config import DEFAULT_LINK_RADIUS_M


@dataclass(frozen=True)
class RoomCluster:
    """One connected group of object nodes treated as a room/region."""

    cluster_id: int
    node_ids: tuple[int, ...]
    labels: tuple[str, ...]
    centroid_xy: tuple[float, float]
    room_name: str
    area_proxy: float = 0.0
    room_id: str = ""


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


def cluster_object_nodes(
    nodes: Sequence[Any],
    edges: Sequence[tuple[int, int, str]] | None = None,
    *,
    link_radius_m: float = DEFAULT_LINK_RADIUS_M,
    connectivity_fn: Any | None = None,
) -> list[RoomCluster]:
    """Connected components over object nodes via ``near`` edges and planar radius."""
    from emet.memory.graph_eqa.room_clusters import hypothesize_room_name

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
            if connectivity_fn is None or bool(connectivity_fn(_xy(by_id[ai]), _xy(by_id[bi]))):
                uf.union(ai, bi)

    radius = float(link_radius_m)
    for i, a in enumerate(objects):
        axy = _xy(a)
        aid = ids[i]
        for j in range(i + 1, len(objects)):
            bxy = _xy(objects[j])
            if _planar_dist(axy, bxy) <= radius and (connectivity_fn is None or bool(connectivity_fn(axy, bxy))):
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
        if len(xs) >= 2:
            area = max(0.01, (max(xs) - min(xs)) * (max(ys) - min(ys)))
        else:
            area = 0.01
        room_name = hypothesize_room_name(labels)
        clusters.append(
            RoomCluster(
                cluster_id=int(root),
                node_ids=tuple(int(getattr(n, "node_id", -1)) for n in members),
                labels=tuple(labels[:32]),
                centroid_xy=(float(cx), float(cy)),
                room_name=room_name,
                area_proxy=float(area),
            )
        )

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
                room_id=c.room_id,
            )
        )
    return out

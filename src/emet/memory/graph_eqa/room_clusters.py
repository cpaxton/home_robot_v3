# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Graph-derived room / region clusters for GraphEQA.

Partitions object nodes into connected components using ``near`` edges and a
planar link radius, then hypothesizes a room name from object labels (not a
Hydra Places layer). Room names are ordinary labels — patio/outdoor are rooms
like kitchen, not a separate policy class.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from emet.memory.graph_eqa.agentic_tools import normalize_current_room

DEFAULT_ROOM_LINK_RADIUS_M = 2.0
DEFAULT_ROOM_ASSIGN_MAX_M = 3.0

# Object-label cues → hypothesized room (majority vote with normalize_current_room).
_OBJECT_ROOM_HINTS: dict[str, str] = {
    "stove": "kitchen",
    "oven": "kitchen",
    "fridge": "kitchen",
    "refrigerator": "kitchen",
    "microwave": "kitchen",
    "dishwasher": "kitchen",
    "counter": "kitchen",
    "cabinet": "kitchen",
    "sink": "kitchen",
    "sofa": "living_room",
    "couch": "living_room",
    "tv": "living_room",
    "television": "living_room",
    "fireplace": "living_room",
    "coffee": "living_room",
    "armchair": "living_room",
    "dining": "dining_room",
    "table": "dining_room",
    "chair": "dining_room",
    "bed": "bedroom",
    "nightstand": "bedroom",
    "wardrobe": "bedroom",
    "toilet": "bathroom",
    "bathtub": "bathroom",
    "shower": "bathroom",
    "vanity": "bathroom",
    "grill": "patio",
    "lawn": "outdoor",
    "grass": "outdoor",
    "fence": "outdoor",
    "deck": "patio",
    "porch": "patio",
    "patio": "patio",
    "garden": "outdoor",
    "yard": "outdoor",
    "brick": "patio",
}


@dataclass(frozen=True)
class RoomCluster:
    """One connected group of object nodes treated as a room/region."""

    cluster_id: int
    node_ids: tuple[int, ...]
    labels: tuple[str, ...]
    centroid_xy: tuple[float, float]
    room_name: str
    area_proxy: float = 0.0


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


def hypothesize_room_name(labels: Sequence[str]) -> str:
    """Hypothesize a canonical room name from object / place labels in a cluster."""
    votes: Counter[str] = Counter()
    for lab in labels:
        name = normalize_current_room(lab)
        if name != "unknown":
            votes[name] += 2  # explicit room words weigh more
        blob = str(lab).lower().replace("-", " ")
        for tok in blob.split():
            tname = normalize_current_room(tok)
            if tname != "unknown":
                votes[tname] += 1
            hint = _OBJECT_ROOM_HINTS.get(tok)
            if hint:
                votes[normalize_current_room(hint)] += 1
        for key, hint in _OBJECT_ROOM_HINTS.items():
            if " " in key and key in blob:
                votes[normalize_current_room(hint)] += 1
    if not votes:
        return "unknown"
    best = votes.most_common()
    top_count = best[0][1]
    tied = [n for n, c in best if c == top_count]
    for prefer in (
        "kitchen",
        "living_room",
        "dining_room",
        "bedroom",
        "bathroom",
        "patio",
        "outdoor",
        "hallway",
        "garage",
    ):
        if prefer in tied:
            return prefer
    return best[0][0]


# Back-compat alias used by older tests / imports.
name_cluster_from_labels = hypothesize_room_name


def question_target_rooms(question: str) -> set[str]:
    """Canonical rooms the question is about (MCQ landmarks + room words in stem)."""
    targets: set[str] = set()
    q = str(question or "")
    try:
        from emet.memory.graph_eqa.graph_memory import location_mcq_landmark_phrases

        for lm in location_mcq_landmark_phrases(q):
            name = normalize_current_room(lm)
            if name != "unknown":
                targets.add(name)
            for tok in str(lm).lower().replace("-", " ").split():
                t = normalize_current_room(tok)
                if t != "unknown":
                    targets.add(t)
                hint = _OBJECT_ROOM_HINTS.get(tok)
                if hint:
                    targets.add(normalize_current_room(hint))
    except Exception:
        pass
    try:
        from emet.memory.graph_eqa.graph_memory import question_stem_for_keywords

        stem = question_stem_for_keywords(q)
    except Exception:
        stem = q
    for tok in str(stem).lower().replace("-", " ").split():
        t = normalize_current_room(tok)
        if t != "unknown":
            targets.add(t)
    return {t for t in targets if t != "unknown"}


def room_mismatches_question(current_room: str | None, question: str) -> bool:
    """True when current room is known and not among question target rooms.

    Uniform explore nudge (leave the wrong room) — no outdoor special case.
    """
    current = normalize_current_room(current_room)
    if current == "unknown":
        return False
    targets = question_target_rooms(question)
    if not targets:
        return False
    return current not in targets


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


def paint_room_labels(
    rgb: np.ndarray,
    clusters: Sequence[RoomCluster],
    *,
    grid_origin_xy: np.ndarray | Sequence[float],
    grid_resolution: float,
    full_shape_hw: tuple[int, int],
    crop_offset_ij: tuple[int, int] = (0, 0),
    crop_shape_hw: tuple[int, int] | None = None,
    font_size: int = 22,
) -> np.ndarray:
    """Draw hypothesized room names at centroids on a top-down RGB (crop or export size).

    When ``crop_shape_hw`` is the pre-resize crop and ``rgb`` is the finalized export,
    centroids are mapped through crop → export scale so text stays crisp.
    """
    from PIL import Image, ImageDraw, ImageFont

    from emet.visualization.map_snapshot import world_xy_to_grid_ij

    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3 or not clusters:
        return arr
    img = Image.fromarray(arr[:, :, :3].copy(), mode="RGB")
    draw = ImageDraw.Draw(img)
    font = None
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ):
        try:
            font = ImageFont.truetype(path, int(font_size))
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    h_full, w_full = int(full_shape_hw[0]), int(full_shape_hw[1])
    i0, j0 = int(crop_offset_ij[0]), int(crop_offset_ij[1])
    eh, ew = img.size[1], img.size[0]
    if crop_shape_hw is not None:
        ch, cw = int(crop_shape_hw[0]), int(crop_shape_hw[1])
    else:
        ch, cw = eh, ew
    scale_y = float(eh) / float(max(1, ch))
    scale_x = float(ew) / float(max(1, cw))
    go = np.asarray(grid_origin_xy, dtype=float).reshape(-1)[:2]
    res = float(grid_resolution) or 0.1
    for c in clusters:
        ri, rj = world_xy_to_grid_ij(c.centroid_xy, go, res, (h_full, w_full))
        cy = (ri - i0) * scale_y
        cx = (rj - j0) * scale_x
        py, px = int(round(cy)), int(round(cx))
        if not (0 <= py < eh and 0 <= px < ew):
            continue
        label = str(c.room_name if c.room_name != "unknown" else f"region_{c.cluster_id}").replace("_", " ")
        # Anchor roughly at centroid center.
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 8 * len(label), int(font_size)
        tx, ty = px - tw // 2, py - th // 2
        # Opaque pill behind text for contrast on observed RGB.
        pad = 3
        draw.rectangle(
            [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
            fill=(20, 20, 24),
            outline=(255, 220, 80),
        )
        draw.text((tx, ty), label, fill=(255, 240, 80), font=font)
    return np.asarray(img, dtype=np.uint8)

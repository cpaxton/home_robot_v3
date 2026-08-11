# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Graph-derived room / region clusters for GraphEQA.

Partitions object nodes into connected components using ``near`` edges and a
planar link radius. Room *names* come from VLM stamps (router ``current_room``
and close-look investigate evidence) and explicit room words in labels — not
furniture-class heuristics (chair/table → dining).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from emet.memory.graph_eqa.agentic_tools import (
    coerce_room_label,
    normalize_current_room,
    room_is_outdoor,
    sanitize_room_phrase,
)

DEFAULT_ROOM_LINK_RADIUS_M = 2.0
DEFAULT_ROOM_ASSIGN_MAX_M = 3.0
ROOM_POLICY_CANONICAL = "canonical"
ROOM_POLICY_LLM = "llm"
ROOM_POLICIES = frozenset({ROOM_POLICY_CANONICAL, ROOM_POLICY_LLM})

# Canonical-policy targets / mismatch only (metrics + optional leave hint).
# Do not invent a frozen "question_area" from MCQ options — explore asks the VLM
# the real question each frontier pick.
_OBJECT_ROOM_HINTS: dict[str, str] = {
    "stove": "kitchen",
    "oven": "kitchen",
    "fridge": "kitchen",
    "refrigerator": "kitchen",
    "microwave": "kitchen",
    "dishwasher": "kitchen",
    "counter": "kitchen",
    "sink": "kitchen",
    "sofa": "living_room",
    "couch": "living_room",
    "tv": "living_room",
    "television": "living_room",
    "fireplace": "living_room",
    "dining": "dining_room",
    "bed": "bedroom",
    "toilet": "bathroom",
    "bathtub": "bathroom",
    "shower": "bathroom",
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
    """Cold-start name from *explicit* room words in labels only (no furniture→room).

    Examples: ``kitchen island`` → kitchen, ``brick patio`` → patio.
    ``chair`` / ``table`` / ``stove`` alone → unknown (VLM stamp later).
    """
    votes: Counter[str] = Counter()
    for lab in labels:
        name = normalize_current_room(lab)
        if name != "unknown":
            votes[name] += 2
        blob = str(lab).lower().replace("-", " ")
        for tok in blob.split():
            tname = normalize_current_room(tok)
            if tname != "unknown":
                votes[tname] += 1
    if not votes:
        return "unknown"
    return votes.most_common(1)[0][0]


def room_from_observation_labels(labels: Sequence[str]) -> str:
    """Room from explicit room words, else sparse landmark hints (toilet/bed/stove…).

    Unlike :func:`hypothesize_room_name`, allows a small object→room map for close-look
    stamps (investigate) — still no chair/table→dining invention.
    """
    name = hypothesize_room_name(labels)
    if name != "unknown":
        return name
    votes: Counter[str] = Counter()
    for lab in labels:
        blob = str(lab).lower().replace("-", " ")
        for tok in blob.split():
            hint = _OBJECT_ROOM_HINTS.get(tok)
            if not hint:
                continue
            tname = normalize_current_room(hint)
            if tname != "unknown":
                votes[tname] += 1
    if not votes:
        return "unknown"
    return votes.most_common(1)[0][0]


def labels_corroborate_outdoor(labels: Sequence[str]) -> bool:
    """True when labels mention outdoor/patio/yard (or normalize to outdoor)."""
    for lab in labels:
        if room_is_outdoor(str(lab)):
            return True
        blob = str(lab).lower()
        if any(w in blob for w in ("outdoor", "patio", "yard", "deck", "porch", "lawn", "fence")):
            return True
    return False


def _is_named_indoor_room(room: str | None) -> bool:
    n = normalize_current_room(room)
    if n != "unknown" and not room_is_outdoor(n):
        return True
    # Free-text that did not bucket but is clearly not outdoor (e.g. rare phrases).
    s = sanitize_room_phrase(room)
    if s == "unknown":
        return False
    return not room_is_outdoor(s)


def should_apply_room_stamp(
    existing: str | None,
    proposed: str | None,
    *,
    cluster_labels: Sequence[str] | None = None,
    corroborating_labels: Sequence[str] | None = None,
) -> bool:
    """Whether ``proposed`` may replace ``existing`` on a cluster.

    Blocks outdoor/patio from overwriting a named indoor room (or a cluster whose
    labels already look indoor) unless outdoor evidence appears in labels.
    """
    name = sanitize_room_phrase(proposed)
    if name == "unknown":
        return False
    labs = [str(x) for x in list(cluster_labels or ()) + list(corroborating_labels or ()) if str(x).strip()]
    if not room_is_outdoor(name):
        return True
    if labels_corroborate_outdoor(labs):
        return True
    if _is_named_indoor_room(existing):
        return False
    # Unknown / region cluster: still refuse outdoor when landmark labels look indoor.
    from_lab = room_from_observation_labels(labs)
    if from_lab != "unknown" and not room_is_outdoor(from_lab):
        return False
    return True


def resolve_investigate_room_stamp(
    *,
    labels: Sequence[str],
    current_room: str | None,
    room_policy: str = ROOM_POLICY_CANONICAL,
) -> str:
    """Room to stamp after a close look from local label evidence only.

    ``current_room`` is accepted for API compatibility but is **not** used as a
    fallback — sticky estimates re-painted kitchens when labels were empty.
    Callers skip the stamp when this returns ``unknown``.
    """
    _ = current_room  # API compat; do not fall back to sticky estimates.
    policy = resolve_room_policy(room_policy)
    label_room = room_from_observation_labels(labels)
    if label_room != "unknown":
        return coerce_room_label(label_room, room_policy=policy)
    return "unknown"


# Back-compat alias used by older tests / imports.
name_cluster_from_labels = hypothesize_room_name


def resolve_room_policy(raw: Any) -> str:
    """Return ``canonical`` or ``llm`` (default canonical)."""
    s = str(raw or "").strip().lower()
    if s in ROOM_POLICIES:
        return s
    return ROOM_POLICY_CANONICAL


def question_target_rooms(question: str) -> set[str]:
    """Canonical rooms the question is about (MCQ landmarks + room words in stem).

    NOTE (OVMM find interpretation): room targets are a strong cue for **receptacle**
    localization (receptacles are room-typical fixtures: table/counter/cab), but a
    weak cue for **object** localization — a target object (``jar``, ``red cylinder``)
    can sit on any receptacle in any room. Consumers that gate explore/escape on room
    mismatch should treat a room signal as receptacle evidence, not object evidence.
    """
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

    Canonical-policy explore nudge (leave the wrong room) — no outdoor special case.
    """
    current = normalize_current_room(current_room)
    if current == "unknown":
        return False
    targets = question_target_rooms(question)
    if not targets:
        return False
    return current not in targets


def room_leave_needed(
    *,
    room_policy: str,
    current_room: str | None,
    question: str,
    in_target_area: bool | None,
) -> bool:
    """Whether explore should soft-bias away from the current place."""
    policy = resolve_room_policy(room_policy)
    if policy == ROOM_POLICY_LLM:
        cur = sanitize_room_phrase(current_room)
        if cur == "unknown" or in_target_area is None:
            return False
        return in_target_area is False
    return room_mismatches_question(current_room, question)


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
    """Nearest cluster room name within ``max_dist_m``, else ``unknown``.

    Returns the stored cluster phrase as-is (light sanitize). Callers applying
    canonical policy should run :func:`normalize_current_room` / ``coerce_room_label``.
    """
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
    return sanitize_room_phrase(best.room_name)


def stamp_room_at_xy(
    clusters: Sequence[RoomCluster],
    robot_xy: tuple[float, float] | Sequence[float],
    room: str | None,
    *,
    max_dist_m: float = DEFAULT_ROOM_ASSIGN_MAX_M,
    protect_indoor_from_outdoor: bool = False,
    corroborating_labels: Sequence[str] | None = None,
) -> list[RoomCluster]:
    """Set nearest cluster's ``room_name`` (immutable list copy).

    ``room`` should already be policy-coerced by the caller (canonical bucket or
    free-text phrase). This only light-sanitizes and rejects ``unknown``.

    When ``protect_indoor_from_outdoor`` is set, refuse outdoor/patio stamps that
    would clobber a named indoor room (or indoor-looking cluster labels) without
    outdoor corroboration in ``corroborating_labels`` / cluster labels.
    """
    name = sanitize_room_phrase(room)
    if name == "unknown" or not clusters:
        return list(clusters)
    xy = (float(robot_xy[0]), float(robot_xy[1]))
    best_i = -1
    best_d = float("inf")
    for i, c in enumerate(clusters):
        d = _planar_dist(xy, c.centroid_xy)
        if d < best_d:
            best_d = d
            best_i = i
    if best_i < 0 or best_d > float(max_dist_m):
        return list(clusters)
    existing = clusters[best_i]
    if protect_indoor_from_outdoor and not should_apply_room_stamp(
        existing.room_name,
        name,
        cluster_labels=existing.labels,
        corroborating_labels=corroborating_labels,
    ):
        return list(clusters)
    out = list(clusters)
    out[best_i] = replace(out[best_i], room_name=name)
    return out


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


def merge_room_estimates(
    vlm_room: str | None,
    graph_room: str | None,
    *,
    room_policy: str = ROOM_POLICY_CANONICAL,
) -> str:
    """Prefer a known VLM estimate; fall back to graph/stamp when VLM is unknown."""
    policy = resolve_room_policy(room_policy)
    if policy == ROOM_POLICY_LLM:
        v = sanitize_room_phrase(vlm_room)
        g = sanitize_room_phrase(graph_room)
        if v != "unknown":
            return v
        return g
    g = normalize_current_room(graph_room)
    v = normalize_current_room(vlm_room)
    if v != "unknown":
        return v
    return g


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
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 8 * len(label), int(font_size)
        tx, ty = px - tw // 2, py - th // 2
        pad = 3
        draw.rectangle(
            [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
            fill=(20, 20, 24),
            outline=(255, 220, 80),
        )
        draw.text((tx, ty), label, fill=(255, 240, 80), font=font)
    return np.asarray(img, dtype=np.uint8)

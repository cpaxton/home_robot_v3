# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""GT object selection for OVMM probes (no sim, no YOLOE).

PickPlace ``obj_main`` is often a tiny/ambiguous food (sugar cube, garlic). Live
find and ``probe-verify`` should aim at a jar/bottle/bowl-class body instead,
and should not query the word ``cab``. Offline ``probe-map`` still uses dump
phrases including ``cab`` so we can see substring hits on cached graphs.

This module is CPU-only: category policy, placement catalog, viewpoint targets.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from emet.eval.ovmm_find_phase import category_matches, semantic_label_from_instance

DEFAULT_OBJECT_BODY = "obj_main"

# Open-vocab findables. Rank order is the pick preference.
PREFERRED_FIND_CATS = (
    "jar",
    "bottle",
    "bowl",
    "cup",
    "mug",
    "can",
    "plate",
    "apple",
    "banana",
    "bread",
    "milk",
    "box",
    "pot",
    "pan",
    "kettle",
)

# Tiny / visually ambiguous Robocasa foods — not a fair open-vocab find target.
SKIP_FIND_CATS = frozenset(
    {
        "sugar cube",
        "marshmallow",
        "garlic",
        "muffins",
        "ping pong ball",
    }
)

_FIXTURE_NEEDLES = (
    "cabinet",
    "counter",
    "sink",
    "stove",
    "fridge",
    "refrigerator",
    "dishwasher",
    "hood",
    "wall",
    "floor",
    "cab",
    "microwave",
    "oven",
    "range",
    "door",
    "window",
)
_CABINET_NEEDLES = ("kitchen cabinet", "cabinet")
_COUNTER_NEEDLES = ("kitchen counter", "counter")

DEFAULT_RECEP_QUERIES = (
    "cabinet",
    "kitchen cabinet",
    "counter",
    "kitchen counter",
)

# Offline map dump language (includes ``cab`` / ``jar`` to diagnose substring matching).
DEFAULT_MAP_QUERIES = (
    "jar",
    "bottle",
    "cab",
    "cabinet",
    "kitchen cabinet",
    "counter",
    "kitchen counter",
)


def xyz_from_placement(info: dict[str, Any]) -> np.ndarray | None:
    """Return a finite world XYZ from a placement dict, or None.

    Accepts numpy ``pos`` (live ``read_sim_object_placements``) or ``xyz``.
    Do not use ``info.get("pos") or …`` — a non-empty ndarray is ambiguous in boolean context.
    """
    pos = info.get("pos")
    if pos is None:
        pos = info.get("xyz")
    if pos is None:
        return None
    arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    if arr.size < 3 or not np.isfinite(arr[:3]).all():
        return None
    return arr[:3].copy()


def norm_cat(cat: str) -> str:
    """Lowercase display category: underscores to spaces, instance hashes stripped."""
    return semantic_label_from_instance(str(cat).replace("_", " ")).strip().lower()


def is_fixture_cat(cat: str) -> bool:
    """True for cabinets, counters, appliances, walls — not manipulable find targets."""
    key = norm_cat(cat)
    return any(category_matches(n, key) for n in _FIXTURE_NEEDLES)


def is_skip_find_cat(cat: str) -> bool:
    """True for sugar-cube-scale foods that are a bad open-vocab find target."""
    return norm_cat(cat) in SKIP_FIND_CATS


def pick_find_object(
    placements: dict[str, dict[str, Any]],
    *,
    object_body: str = DEFAULT_OBJECT_BODY,
) -> dict[str, Any] | None:
    """Pick a jar/bottle/bowl-class body from sim GT; skip sugar cube and fixtures.

    If ``object_body`` (default ``obj_main``) is itself a preferred category, it
    wins. Otherwise the lowest index in ``PREFERRED_FIND_CATS`` among remaining
    non-fixture bodies is used (e.g. a distractor bottle when ``obj_main`` is a
    sugar cube). Returns None when nothing findable is in the scene.

    The result dict has ``id="object"``, ``body``, ``cat``, ``xyz`` (list), and
    ``preferred`` (True when the cat matched ``PREFERRED_FIND_CATS``).
    """
    rows: list[tuple[int, float, str, str, np.ndarray]] = []
    for body, info in placements.items():
        if not isinstance(info, dict):
            continue
        cat = str(info.get("cat") or body)
        if is_fixture_cat(cat):
            continue
        xyz = xyz_from_placement(info)
        if xyz is None:
            continue
        if is_skip_find_cat(cat):
            continue
        rank = 999
        for i, pref in enumerate(PREFERRED_FIND_CATS):
            if category_matches(pref, cat):
                rank = i
                break
        rows.append((rank, abs(float(xyz[2]) - 0.9), body, cat, xyz))
    if not rows:
        return None
    hinted = [r for r in rows if r[2] == object_body]
    if hinted and hinted[0][0] < 900:
        rows = hinted
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    rank, _z, body, cat, xyz = rows[0]
    return {
        "id": "object",
        "body": body,
        "cat": cat,
        "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
        "preferred": rank < 900,
    }


def placement_catalog(placements: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """JSON-safe list of every GT body: cat, xyz, fixture/skip flags."""
    rows: list[dict[str, Any]] = []
    for body in sorted(placements):
        info = placements[body]
        if not isinstance(info, dict):
            continue
        cat = str(info.get("cat") or body)
        xyz = xyz_from_placement(info)
        rows.append(
            {
                "body": body,
                "cat": cat,
                "xyz": None if xyz is None else [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "fixture": is_fixture_cat(cat),
                "skip_find": is_skip_find_cat(cat),
            }
        )
    return rows


def nearest_category(
    placements: dict[str, dict[str, Any]],
    needles: tuple[str, ...],
    *,
    near_xyz: np.ndarray | None,
    skip: set[str],
) -> dict[str, Any] | None:
    """Nearest placement whose ``cat`` matches any needle (substring, live-find rules)."""
    hits: list[tuple[float, str, dict[str, Any], np.ndarray]] = []
    for body, info in placements.items():
        if body in skip or not isinstance(info, dict):
            continue
        cat = str(info.get("cat") or body)
        if not any(category_matches(n, cat) for n in needles):
            continue
        xyz = xyz_from_placement(info)
        if xyz is None:
            continue
        dist = float(np.linalg.norm(xyz[:2] - near_xyz[:2])) if near_xyz is not None else 0.0
        hits.append((dist, body, info, xyz))
    if not hits:
        return None
    hits.sort(key=lambda row: (row[0], row[1]))
    _dist, body, info, xyz = hits[0]
    return {
        "id": needles[0].replace(" ", "_"),
        "body": body,
        "cat": str(info.get("cat") or body),
        "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
    }


def pick_view_targets(
    placements: dict[str, dict[str, Any]],
    *,
    object_body: str = DEFAULT_OBJECT_BODY,
    include_counter: bool = False,
) -> list[dict[str, Any]]:
    """Findable object plus nearest cabinet. Counter is optional.

    Cabinet/counter rows set ``yaw_only=True``: the live probe should face them
    from the current floor pose, not drive onto the fixture body origin.
    """
    targets: list[dict[str, Any]] = []
    skip: set[str] = set()
    near: np.ndarray | None = None
    obj = pick_find_object(placements, object_body=object_body)
    if obj is not None:
        near = np.asarray(obj["xyz"], dtype=np.float64)
        skip.add(str(obj["body"]))
        targets.append(obj)
    cab = nearest_category(placements, _CABINET_NEEDLES, near_xyz=near, skip=skip)
    if cab is not None:
        skip.add(str(cab["body"]))
        cab["id"] = "cabinet"
        cab["yaw_only"] = True
        targets.append(cab)
    if include_counter:
        counter = nearest_category(placements, _COUNTER_NEEDLES, near_xyz=near, skip=skip)
        if counter is not None:
            counter["id"] = "counter"
            counter["yaw_only"] = True
            targets.append(counter)
    return targets


def resolve_phrases(
    queries: list[str] | tuple[str, ...] | None,
    placements: dict[str, dict[str, Any]] | None,
    *,
    object_body: str = DEFAULT_OBJECT_BODY,
    object_cat: str | None = None,
) -> list[str]:
    """Phrases to score: explicit ``queries``, or findable cat + receptacle aliases.

    Sugar-cube-scale names are dropped even if the caller passed them.
    """
    if queries:
        phrases = [str(q).strip() for q in queries if str(q).strip()]
    else:
        phrases = list(DEFAULT_RECEP_QUERIES)
        cat = object_cat
        if cat is None and placements:
            picked = pick_find_object(placements, object_body=object_body)
            cat = None if picked is None else str(picked.get("cat") or "")
        if cat and not is_skip_find_cat(cat):
            label = semantic_label_from_instance(str(cat).replace("_", " "))
            if label and label.lower() not in {p.lower() for p in phrases}:
                phrases.insert(0, label)
    return [p for p in phrases if p and not is_skip_find_cat(p)]

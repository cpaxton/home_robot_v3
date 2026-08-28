# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""DynaMem ``localize_text`` → world XYZ (YoloE hit or high SigLIP cosine).

This is the same call Stretch used in the original voxel map: detector first,
then a cosine gate. Agentic find and OVMM scoring should use this point as
the object location — not camera pose, and not a raw SigLIP argmax.

The first successful live query **pins** phrase → XYZ on the voxel map. Later
calls reuse that snapshot so explore / extra voxel updates cannot erase a
mapping-time hit. Pass ``refresh=True`` to query the live map again.
"""

from __future__ import annotations

from typing import Any

import numpy as np

VOXEL_HYP_OBS_BASE = -3_000_000
# verify_siglip: live camera now. Not a stored obs and not a voxel handle.
CURRENT_VIEW_OBS_ID = -1
LOCALIZE_PINS_ATTR = "_emet_localize_pins"


def is_current_view_sentinel(obs_id: Any) -> bool:
    """True for ``verify_siglip`` ``obs_id=-1`` (current camera frame)."""
    try:
        return int(obs_id) == int(CURRENT_VIEW_OBS_ID)
    except (TypeError, ValueError):
        return False


def is_proposal_handle(obs_id: Any) -> bool:
    """True for voxel ``localize_text`` handles (``obs_id <= VOXEL_HYP_OBS_BASE``).

    Other negative ids are not detections: SigLIP seeds (``-2_000_000``),
    frontier keys (``-1_000_000``), and ``CURRENT_VIEW_OBS_ID`` (``-1``).
    """
    try:
        return int(obs_id) <= int(VOXEL_HYP_OBS_BASE)
    except (TypeError, ValueError):
        return False


def voxel_proposal_id(index: int = 0) -> int:
    """Stable investigate() handle for the ``index``-th voxel detection this recall."""
    return int(VOXEL_HYP_OBS_BASE) - int(index)


def localize_confidence(stats: dict[str, Any] | None) -> float | None:
    """1.0 on YoloE hit, else gated cosine if present."""
    if not isinstance(stats, dict):
        return None
    if stats.get("yoloe_hit"):
        return 1.0
    cosine = stats.get("max_cosine")
    try:
        if cosine is None:
            return None
        return float(cosine)
    except (TypeError, ValueError):
        return None


def _as_xyz(target: Any) -> np.ndarray | None:
    """Return a finite (3,) world XYZ, or None if ``target`` is not a point."""
    if target is None:
        return None
    if hasattr(target, "detach"):
        try:
            target = target.detach().cpu().numpy()
        except Exception:
            return None
    try:
        arr = np.asarray(target, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 3 or not np.isfinite(arr[:3]).all():
        return None
    return arr[:3].copy()


def voxel_map_from_agent(agent: Any, *, require_localize: bool = True) -> Any | None:
    """Voxel map on a controller / executor agent.

    Prefer ``agent.voxel_map`` when it has ``localize_text``; otherwise
    ``get_voxel_map()`` (instance-memory proxy). Occupancy-only maps are
    returned when ``require_localize`` is False.
    """
    if agent is None:
        return None
    vm = getattr(agent, "voxel_map", None)
    getter = getattr(agent, "get_voxel_map", None)
    getter_vm = None
    if vm is None or (require_localize and not hasattr(vm, "localize_text")):
        if callable(getter):
            try:
                getter_vm = getter()
            except Exception:
                getter_vm = None
    for cand in (vm, getter_vm):
        if cand is None:
            continue
        if not require_localize or hasattr(cand, "localize_text"):
            return cand
    return None


def _pin_store(voxel_map: Any) -> dict[str, dict[str, Any]]:
    store = getattr(voxel_map, LOCALIZE_PINS_ATTR, None)
    if not isinstance(store, dict):
        store = {}
        setattr(voxel_map, LOCALIZE_PINS_ATTR, store)
    return store


def pin_localize_xyz(
    voxel_map: Any,
    phrase: str,
    xyz: Any,
    stats: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """Remember a mapping-time ``localize_text`` hit. First pin for a phrase wins."""
    query = str(phrase or "").strip().lower()
    point = _as_xyz(xyz)
    if voxel_map is None or not query or point is None:
        return None
    store = _pin_store(voxel_map)
    if query in store:
        existing = _as_xyz(store[query].get("xyz"))
        return existing.copy() if existing is not None else None
    payload: dict[str, Any] = {"xyz": point.copy(), "stats": dict(stats or {})}
    store[query] = payload
    return point.copy()


def pinned_localize_xyz(
    voxel_map: Any,
    phrase: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return a previously pinned XYZ for ``phrase``, if any."""
    query = str(phrase or "").strip().lower()
    empty: dict[str, Any] = {"query": str(phrase or "").strip(), "from_pin": True}
    if voxel_map is None or not query:
        return None, empty
    store = getattr(voxel_map, LOCALIZE_PINS_ATTR, None)
    if not isinstance(store, dict):
        return None, empty
    payload = store.get(query)
    if not isinstance(payload, dict):
        return None, empty
    point = _as_xyz(payload.get("xyz"))
    if point is None:
        return None, empty
    raw = payload.get("stats")
    stats = dict(raw) if isinstance(raw, dict) else {}
    stats["from_pin"] = True
    stats.setdefault("query", str(phrase or "").strip())
    return point.copy(), stats


def clear_localize_pins(voxel_map: Any) -> None:
    """Drop mapping-time pins (tests / new map object)."""
    if voxel_map is None:
        return
    setattr(voxel_map, LOCALIZE_PINS_ATTR, {})


def unpin_localize_xyz(voxel_map: Any, phrase: str) -> bool:
    """Remove the pin for ``phrase`` — a close ABSENT disproves that XYZ."""
    query = str(phrase or "").strip().lower()
    if voxel_map is None or not query:
        return False
    store = _pin_store(voxel_map)
    if query not in store:
        return False
    store.pop(query, None)
    return True


def pin_phrases_after_mapping(
    voxel_map: Any,
    phrases: list[str] | tuple[str, ...] | None,
) -> dict[str, bool]:
    """Live-query each phrase once (pytest / mapping probe). Not the OVMM harness."""
    hits: dict[str, bool] = {}
    seen: set[str] = set()
    for raw in phrases or ():
        phrase = str(raw or "").strip()
        key = phrase.lower()
        if not phrase or key in seen:
            continue
        seen.add(key)
        xyz, _stats = localize_text_xyz(voxel_map, phrase, refresh=False)
        hits[phrase] = xyz is not None
    return hits


def localize_text_xyz(
    voxel_map: Any,
    text: str,
    *,
    refresh: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Run ``voxel_map.localize_text`` and return ``(xyz, stats)``.

    ``stats`` copies ``_last_localize_stats`` when present (max_cosine, yoloe_hit).
    After the first live hit, later calls return that pin unless ``refresh=True``.
    """
    query = str(text or "").strip()
    empty: dict[str, Any] = {
        "query": query,
        "max_cosine": None,
        "yoloe_hit": False,
        "from_pin": False,
    }
    if voxel_map is None or not query:
        return None, empty
    if not refresh:
        pinned, pin_stats = pinned_localize_xyz(voxel_map, query)
        if pinned is not None:
            pin_stats.setdefault("from_pin", True)
            voxel_map._last_localize_stats = dict(pin_stats)
            return pinned, pin_stats
    if not hasattr(voxel_map, "localize_text"):
        return None, empty
    result = voxel_map.localize_text(query, debug=False, return_debug=True)
    target = result[0] if isinstance(result, (list, tuple)) else result
    xyz = _as_xyz(target)
    raw = getattr(voxel_map, "_last_localize_stats", None)
    stats = dict(raw) if isinstance(raw, dict) else dict(empty)
    stats.setdefault("query", query)
    stats["from_pin"] = False
    if xyz is not None:
        pin_localize_xyz(voxel_map, query, xyz, stats)
        return xyz, stats
    pinned, pin_stats = pinned_localize_xyz(voxel_map, query)
    if pinned is not None:
        pin_stats.setdefault("from_pin", True)
        voxel_map._last_localize_stats = dict(pin_stats)
        return pinned, pin_stats
    return None, stats


def localize_text_xyz_from_phrases(
    voxel_map: Any,
    phrases: list[str] | tuple[str, ...] | None,
    *,
    refresh: bool = False,
) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
    """First successful ``localize_text`` over ``phrases`` (full phrase before tokens).

    Mapping-time pins are tried before a live query so a later miss does not
    drop a phrase that already hit after mapping.
    """
    seen: set[str] = set()
    last_stats: dict[str, Any] = {}
    ordered: list[str] = []
    for raw in phrases or ():
        phrase = str(raw or "").strip()
        key = phrase.lower()
        if not phrase or key in seen:
            continue
        seen.add(key)
        ordered.append(phrase)
    if not refresh:
        for phrase in ordered:
            pinned, pin_stats = pinned_localize_xyz(voxel_map, phrase)
            if pinned is not None:
                pin_stats.setdefault("from_pin", True)
                if voxel_map is not None:
                    voxel_map._last_localize_stats = dict(pin_stats)
                return pinned, phrase, pin_stats
    for phrase in ordered:
        xyz, stats = localize_text_xyz(voxel_map, phrase, refresh=refresh)
        last_stats = stats
        if xyz is not None:
            return xyz, phrase, stats
    return None, None, last_stats


def pinned_xyz_from_phrases(
    voxel_map: Any,
    phrases: list[str] | tuple[str, ...] | None,
) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
    """Pin-store lookup only. Never runs live ``localize_text`` (encoder may be gone)."""
    seen: set[str] = set()
    for raw in phrases or ():
        phrase = str(raw or "").strip()
        key = phrase.lower()
        if not phrase or key in seen:
            continue
        seen.add(key)
        pinned, pin_stats = pinned_localize_xyz(voxel_map, phrase)
        if pinned is not None:
            pin_stats.setdefault("from_pin", True)
            if voxel_map is not None:
                voxel_map._last_localize_stats = dict(pin_stats)
            return pinned, phrase, pin_stats
    return None, None, {}

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""DynaMem ``localize_text`` → world XYZ (YoloE hit or high SigLIP cosine).

This is the same call Stretch used in the original voxel map: detector first,
then a cosine gate. Agentic find and OVMM scoring should use this point as
the object location — not camera pose, and not a raw SigLIP argmax.
"""

from __future__ import annotations

from typing import Any

import numpy as np

VOXEL_HYP_OBS_BASE = -3_000_000


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


def voxel_map_from_agent(agent: Any) -> Any | None:
    """Voxel map on a controller / executor agent, if ``localize_text`` exists."""
    if agent is None:
        return None
    vm = getattr(agent, "voxel_map", None)
    if vm is not None and hasattr(vm, "localize_text"):
        return vm
    getter = getattr(agent, "get_voxel_map", None)
    if not callable(getter):
        return None
    try:
        vm = getter()
    except Exception:
        return None
    if vm is not None and hasattr(vm, "localize_text"):
        return vm
    return None


def localize_text_xyz(
    voxel_map: Any,
    text: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Run ``voxel_map.localize_text`` and return ``(xyz, stats)``.

    ``stats`` copies ``_last_localize_stats`` when present (max_cosine, yoloe_hit).
    """
    query = str(text or "").strip()
    empty: dict[str, Any] = {"query": query, "max_cosine": None, "yoloe_hit": False}
    if voxel_map is None or not query or not hasattr(voxel_map, "localize_text"):
        return None, empty
    result = voxel_map.localize_text(query, debug=False, return_debug=True)
    target = result[0] if isinstance(result, (list, tuple)) else result
    xyz = _as_xyz(target)
    raw = getattr(voxel_map, "_last_localize_stats", None)
    stats = dict(raw) if isinstance(raw, dict) else dict(empty)
    stats.setdefault("query", query)
    return xyz, stats


def localize_text_xyz_from_phrases(
    voxel_map: Any,
    phrases: list[str] | tuple[str, ...] | None,
) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
    """First successful ``localize_text`` over ``phrases`` (full phrase before tokens)."""
    seen: set[str] = set()
    last_stats: dict[str, Any] = {}
    for raw in phrases or ():
        phrase = str(raw or "").strip()
        key = phrase.lower()
        if not phrase or key in seen:
            continue
        seen.add(key)
        xyz, stats = localize_text_xyz(voxel_map, phrase)
        last_stats = stats
        if xyz is not None:
            return xyz, phrase, stats
    return None, None, last_stats

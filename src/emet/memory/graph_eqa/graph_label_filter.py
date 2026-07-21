# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Scene-aware open-vocab label filters for Dynagraph graph attach.

YoloE defaults to ScanNet-200 classes. In Robocasa kitchens that vocabulary
frequently emits bathroom fixtures (``bathroom stall``, ``toilet``, …) as
high-confidence false positives — wrong intermediate graph state even when
merge/growth metrics look healthy.
"""

from __future__ import annotations

import re
from typing import Any

# Bathroom / hygiene ScanNet classes that should not become kitchen graph nodes.
_KITCHEN_DENY_EXACT: frozenset[str] = frozenset(
    {
        "bathroom stall",
        "bathroom vanity",
        "bathtub",
        "shower",
        "shower head",
        "toilet",
        "toilet paper",
        "toilet paper dispenser",
        "toilet seat cover dispenser",
        "plunger",
        "body wash",
        "face wash",
        "alcohol disinfection",
        "hand sanitizer",
        "inhaler",
        "laundry detergent",
        "clothes dryer",
        "ironing board",
        # ScanNet electronics / clutter that frequently false-positives in Robocasa kitchens
        "adapter",
        "power strip",
        "charger",
    }
)

_KITCHEN_DENY_SUBSTR: tuple[str, ...] = (
    "bathroom",
    "toilet",
    "shower",
    "bathtub",
)

_GENERIC_DROP: frozenset[str] = frozenset(
    {
        "object",
        "thing",
        "item",
        "unknown",
        "n/a",
        "na",
        "none",
    }
)


def _norm(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "").strip().lower())


def resolve_graph_scene_profile(
    *,
    robot: Any | None = None,
    parameters: Any | None = None,
    session: dict[str, Any] | None = None,
) -> str:
    """Return ``kitchen``, ``indoor``, or ``none`` (disable filters).

    Resolution order:
    1. Explicit ``graph_eqa_label_filter.scene_profile`` / ``graph_label_scene_profile``
    2. ``emet_session["environment"]["kind"]`` (``robocasa`` → kitchen)
    3. Default ``indoor`` (no bathroom deny; still drops empty/generic labels)
    """
    params = parameters
    if params is not None:
        explicit = None
        try:
            blk = params.get("graph_eqa_label_filter")
            if isinstance(blk, dict) and blk.get("scene_profile") is not None:
                explicit = str(blk.get("scene_profile"))
            elif params.get("graph_label_scene_profile") is not None:
                explicit = str(params.get("graph_label_scene_profile"))
        except Exception:
            explicit = None
        if explicit:
            key = explicit.strip().lower()
            if key in ("kitchen", "robocasa"):
                return "kitchen"
            if key in ("none", "off", "disable", "disabled"):
                return "none"
            if key in ("indoor", "habitat", "home"):
                return "indoor"
            # ``auto`` falls through to session / default below.

    sess = session
    if sess is None and robot is not None:
        get_sess = getattr(robot, "get_emet_session", None)
        if callable(get_sess):
            try:
                sess = get_sess()
            except Exception:
                sess = None
    if isinstance(sess, dict):
        env = sess.get("environment")
        kind = None
        if isinstance(env, dict):
            kind = env.get("kind")
        if kind is None:
            kind = sess.get("kind")
        if str(kind or "").strip().lower() == "robocasa":
            return "kitchen"

    return "indoor"


def is_graph_label_allowed(label: str, *, scene_profile: str = "indoor") -> bool:
    """Whether ``label`` may create/merge a Dynagraph object node."""
    profile = (scene_profile or "indoor").strip().lower()
    if profile in ("none", "off", "disable", "disabled"):
        return True
    norm = _norm(label)
    if not norm or norm in _GENERIC_DROP:
        return False
    if profile == "kitchen":
        if norm in _KITCHEN_DENY_EXACT:
            return False
        if any(s in norm for s in _KITCHEN_DENY_SUBSTR):
            return False
    return True


def filter_graph_labels(labels: list[str] | None, *, scene_profile: str = "indoor") -> list[str]:
    """Keep only allowed labels (order preserved, de-duplicated by normalized form)."""
    if not labels:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in labels:
        s = str(raw).strip()
        if not s or not is_graph_label_allowed(s, scene_profile=scene_profile):
            continue
        key = _norm(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Canonical / free-text room labels for graph memory and the agentic router."""

from __future__ import annotations

from typing import Any

# Canonical room labels for router current_room (aliases normalize into these).
ROOM_CANONICAL = frozenset(
    {
        "patio",
        "outdoor",
        "kitchen",
        "living_room",
        "dining_room",
        "bedroom",
        "bathroom",
        "hallway",
        "garage",
        "unknown",
    }
)
_OUTDOOR_ROOMS = frozenset({"patio", "outdoor"})
_OUTDOOR_ALIASES = frozenset(
    {
        "patio",
        "outdoor",
        "outdoors",
        "outside",
        "yard",
        "deck",
        "porch",
        "garden",
        "brick_patio",
        "courtyard",
    }
)
_INDOOR_QUESTION_CUES = frozenset(
    {
        "clock",
        "wall clock",
        "kitchen",
        "living",
        "living room",
        "dining",
        "bedroom",
        "bathroom",
        "bowl",
        "fruit bowl",
        "microwave",
        "refrigerator",
        "fridge",
        "sofa",
        "couch",
        "fireplace",
        "cabinet",
        "indoor",
        "inside",
    }
)


def sanitize_room_phrase(raw: Any, *, max_chars: int = 48) -> str:
    """Light cleanup for room labels. Preserves canonical buckets; free-text stays phrases."""
    if raw is None:
        return "unknown"
    s = str(raw).strip().lower()
    if not s or s in {"unknown", "none", "n/a", "null"}:
        return "unknown"
    token = "_".join(s.replace("-", " ").replace("/", " ").replace("_", " ").split())
    if token in ROOM_CANONICAL:
        return token
    phrase = " ".join(s.replace("_", " ").replace("-", " ").replace("/", " ").split())
    if not phrase:
        return "unknown"
    if len(phrase) > int(max_chars):
        phrase = phrase[: max(0, int(max_chars) - 1)].rstrip() + "…"
    return phrase


def normalize_current_room(raw: Any) -> str:
    """Map free-text router ``current_room`` onto a small vocabulary (canonical policy / metrics)."""
    if raw is None:
        return "unknown"
    s = str(raw).strip().lower().replace("-", " ").replace("/", " ")
    s = "_".join(s.split())
    if not s:
        return "unknown"
    if s in ROOM_CANONICAL:
        return s
    if s in _OUTDOOR_ALIASES or any(a in s for a in ("patio", "outdoor", "yard", "deck", "porch")):
        return "outdoor" if "patio" not in s else "patio"
    if "living" in s:
        return "living_room"
    if "dining" in s:
        return "dining_room"
    if "kitchen" in s:
        return "kitchen"
    if "bed" in s:
        return "bedroom"
    if "bath" in s:
        return "bathroom"
    if "hall" in s or "corridor" in s:
        return "hallway"
    if "garage" in s:
        return "garage"
    return "unknown"


# Metrics / histogram alias — same as normalize; not used for LLM-policy decisions.
room_bucket = normalize_current_room


def coerce_room_label(raw: Any, *, room_policy: str = "canonical") -> str:
    """Policy-aware room identity: closed vocab vs free-text phrase."""
    if str(room_policy or "").strip().lower() == "llm":
        return sanitize_room_phrase(raw)
    return normalize_current_room(raw)


def room_is_outdoor(room: str) -> bool:
    return normalize_current_room(room) in _OUTDOOR_ROOMS


def question_implies_indoor(question: str) -> bool:
    """True when the embodied question is likely about an indoor place/object."""
    q = str(question or "").strip().lower()
    if not q:
        return False
    if any(cue in q for cue in _INDOOR_QUESTION_CUES):
        return True
    try:
        from emet.memory.graph_eqa.labels import location_mcq_landmark_phrases

        landmarks = location_mcq_landmark_phrases(question)
    except Exception:
        landmarks = []
    indoor_landmarks = ("kitchen", "living", "dining", "bedroom", "bathroom", "hall")
    return any(any(tok in str(lm).lower() for tok in indoor_landmarks) for lm in landmarks)

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Count-MCQ target extraction and instance collapsing."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from emet.memory.graph_eqa.types import GraphNode

_COUNT_TARGET_BOUNDARY_RE = re.compile(
    r"\b(?:are|is|was|were|am|did|do|does|can|could|"
    r"have|has|had|i|you|it|we|they|there|at|on|in|by|for|with|"
    r"under|over|next|left|put|placed|leave|behind|above|below|near|"
    r"beside|around|inside|outside|standing|sitting|hanging)\b|\?",
    re.IGNORECASE,
)

_COUNT_QUANTITY_WRAPPERS = frozenset(
    {
        "set",
        "sets",
        "piece",
        "pieces",
        "pair",
        "pairs",
        "couple",
        "bunch",
        "lot",
        "lots",
        "number",
    }
)

_COUNT_LEADING_WORDS = frozenset({"a", "an", "the", "of", "some"})

_COUNT_WORD_ALIASES: dict[str, frozenset[str]] = {
    "trash": frozenset({"garbage", "rubbish", "waste", "recycle", "bin"}),
    "garbage": frozenset({"trash", "rubbish", "waste", "recycle", "bin"}),
    "rubbish": frozenset({"trash", "garbage", "waste", "recycle", "bin"}),
    "waste": frozenset({"trash", "garbage", "rubbish", "recycle", "bin"}),
    "recycle": frozenset({"trash", "garbage", "rubbish", "waste", "bin"}),
    "bin": frozenset({"trash", "garbage", "rubbish", "waste", "recycle", "can"}),
    "can": frozenset({"bin"}),
    "nightstand": frozenset({"bedside"}),
    "bedside": frozenset({"nightstand"}),
    "stool": frozenset({"stools", "barstool", "barstools"}),
    "stools": frozenset({"stool", "barstool", "barstools"}),
    "barstool": frozenset({"stool", "stools", "bar", "barstools"}),
    "barstools": frozenset({"stool", "stools", "barstool"}),
    "mat": frozenset({"mats", "rug", "rugs", "doormat"}),
    "mats": frozenset({"mat", "rug", "rugs", "doormat"}),
    "rug": frozenset({"mat", "mats", "rugs", "doormat"}),
    "lamp": frozenset({"lamps", "light", "lights"}),
    "lamps": frozenset({"lamp", "light", "lights"}),
    "pillow": frozenset({"pillows", "cushion", "cushions"}),
    "pillows": frozenset({"pillow", "cushion", "cushions"}),
}

_COUNT_PHRASE_ALIASES: dict[tuple[str, ...], tuple[tuple[str, ...], ...]] = {
    ("bedside", "tables"): (("nightstand",), ("bedside", "table")),
    ("bedside", "table"): (("nightstand",),),
    ("bar", "stools"): (("stool",), ("stools",), ("barstool",), ("barstools",)),
    ("bar", "stool"): (("stool",), ("stools",)),
}

_COUNT_ROOM_PHRASES = (
    "living room",
    "dining room",
    "bedroom",
    "bathroom",
    "kitchen",
    "hallway",
    "office",
    "garage",
    "sunroom",
)

_COUNT_SCOPE_PREPOSITION_RE = re.compile(
    r"\b(?:in|inside|within|at|on|near|by|under|over|next\s+to|beside|behind|"
    r"above|below|around)\s+(?:the\s+)?(.+)$",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class CountTarget:
    """A parsed count-MCQ target and optional grounded scope."""

    tokens: tuple[str, ...]
    scope_tokens: tuple[str, ...] = ()

    @property
    def phrase(self) -> str:
        return " ".join(self.tokens)

    @property
    def scope_phrase(self) -> str:
        return " ".join(self.scope_tokens)


def _count_tokens(text: str) -> list[str]:
    """Tokenize count targets/scopes without allowing punctuation into matches."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _strip_count_wrappers(tokens: list[str]) -> list[str]:
    """Drop quantity wrappers while preserving valid counted nouns such as ``bowl``."""
    out = list(tokens)
    while out and out[0] in _COUNT_LEADING_WORDS:
        out.pop(0)
    if len(out) >= 2 and out[0] in _COUNT_QUANTITY_WRAPPERS and out[1] == "of":
        out = out[2:]
        while out and out[0] in _COUNT_LEADING_WORDS:
            out.pop(0)
    return out


def _count_room_scope_tokens(text: str) -> tuple[str, ...]:
    normalized = re.sub(r"[_-]+", " ", (text or "").lower())
    for phrase in sorted(_COUNT_ROOM_PHRASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", normalized):
            return tuple(phrase.split())
    return ()


def _count_target_from_stem(stem: str) -> CountTarget | None:
    """Parse common count stems and retain any trailing room scope separately."""
    match = re.search(
        r"\bhow\s+many\b(?P<rest>.*)$|"
        r"\b(?:number|count)\s+of\s+(?P<rest_alt>.*)$",
        stem or "",
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    rest = (match.group("rest") or match.group("rest_alt") or "").strip()
    boundary = _COUNT_TARGET_BOUNDARY_RE.search(rest)
    target_text = rest[: boundary.start()] if boundary is not None else rest
    tail = rest[boundary.start() :] if boundary is not None else ""
    tokens = _strip_count_wrappers(_count_tokens(target_text))
    if not tokens:
        return None
    scope_tokens: tuple[str, ...] = ()
    scope = _COUNT_SCOPE_PREPOSITION_RE.search(tail)
    if scope is not None:
        scope_tokens = _count_room_scope_tokens(scope.group(1))
    return CountTarget(tokens=tuple(tokens), scope_tokens=scope_tokens)


def _count_word_forms(token: str) -> set[str]:
    """Return conservative singular/plural forms for a count-label token."""
    word = str(token or "").lower()
    forms = {word}
    if word.endswith("ies") and len(word) > 4:
        forms.add(word[:-3] + "y")
    if word.endswith("ves") and len(word) > 4:
        # Covers both ``shelves`` → ``shelf`` and ``knives`` → ``knife``.
        forms.add(word[:-3] + "f")
        forms.add(word[:-3] + "fe")
    if word.endswith(("ches", "shes", "xes", "zes", "ses")) and len(word) > 3:
        forms.add(word[:-2])
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        forms.add(word[:-1])
    return {form for form in forms if form}


def _count_word_matches(target: str, label: str) -> bool:
    """Match whole label tokens, never substrings such as ``art`` in ``cart``."""
    target_forms = _count_word_forms(target)
    label_forms = _count_word_forms(label)
    if target_forms & label_forms:
        return True
    target_aliases = {
        form for alias in _COUNT_WORD_ALIASES.get(str(target or "").lower(), ()) for form in _count_word_forms(alias)
    }
    return bool(target_aliases & label_forms)


def _collapse_count_nodes_spatially(
    nodes: list[GraphNode],
    *,
    min_xy_m: float = 0.65,
) -> list[GraphNode]:
    """Collapse duplicate instance nodes that missed graph merge (same label, nearby XY)."""
    if len(nodes) <= 1 or min_xy_m <= 0.0:
        return list(nodes)
    ranked = sorted(nodes, key=lambda node: (-int(node.support_count), int(node.node_id)))
    kept: list[GraphNode] = []
    for node in ranked:
        nxy = np.asarray(node.xyz, dtype=np.float64).reshape(3)[:2]
        if any(
            float(np.linalg.norm(nxy - np.asarray(kept_node.xyz, dtype=np.float64).reshape(3)[:2])) < min_xy_m
            for kept_node in kept
        ):
            continue
        kept.append(node)
    return kept


def _count_phrase_matches(target_tokens: tuple[str, ...], label: str) -> bool:
    label_tokens = re.findall(r"[a-z0-9]+", (label or "").lower())
    if not label_tokens or not target_tokens:
        return False
    if len(target_tokens) == 1:
        return any(_count_word_matches(target_tokens[0], token) for token in label_tokens)
    width = len(target_tokens)
    return any(
        all(
            _count_word_matches(target, label_token)
            for target, label_token in zip(target_tokens, label_tokens[i : i + width], strict=True)
        )
        for i in range(len(label_tokens) - width + 1)
    )

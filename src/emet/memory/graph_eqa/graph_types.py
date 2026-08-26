# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""GraphEQA dataclasses, keyword helpers, and spatial predicates.

Callers keep importing these from ``graph_memory``; this module holds the
implementation so the facade stays ingestible by the agent host.
"""

from __future__ import annotations

import re
from dataclasses import (
    dataclass,
    field,
)
from typing import Any

import numpy as np

from emet.utils.logger import Logger

_logger = Logger(__name__)

# Min SigLIP cosine for open-vocab text vs **voxel point** features (DynaMem
# ``verify_point``). Higher precision than image-space scoring. Agentic RGB /
# dense-patch verify uses a lower high-recall bar — see
# ``SIGLIP_IMAGE_PRESENT_THRESHOLD`` in ``agentic_eqa.py`` and
# docs/experiments/agentic_scale.md § SigLIP role.
SIGLIP_PRESENT_THRESHOLD = 0.21
# Stronger bar before SigLIP-only evidence may override the VLM or finalize confidence.
SIGLIP_CONFIRM_THRESHOLD = 0.28
# Question tokens that retrieve garbage in SigLIP (clock questions say "time").
_WEAK_SIGLIP_FIND_TOKENS = frozenset({"time", "hour", "now", "today", "moment"})


# Source tiers for hyp *recall* only (not a VLM decision policy).
_RECALL_SOURCE_TIER: dict[str, float] = {
    "graph": 300.0,
    "confirmed": 200.0,
    "siglip": 100.0,
    "frontier": 0.0,
}


@dataclass(frozen=True)
class NavHypothesis:
    """Retrieved navigation evidence card (graph / CONFIRMED_MEMORY / SigLIP / frontier).

    ``score`` is an internal recall rank key for top-K packing and ROUTER=0 fallback
    order — not a policy signal for the VLM router.
    """

    phrase: str
    obs_id: int
    xyz: np.ndarray
    score: float
    source: str  # "graph" | "confirmed" | "siglip" | "frontier"
    answerability_gain: float = 0.0
    belief_reduction: float = 0.0
    revisit_change_value: float = 0.0
    path_cost: float = 0.0
    failure_risk: float = 0.0
    siglip_sim: float | None = None


@dataclass(frozen=True)
class RelationBelief:
    """Timestamped uncertain context relation."""

    source_id: int
    target_id: int
    relation: str
    confidence: float
    last_evidence_step: int
    contradiction_count: int = 0


@dataclass(frozen=True)
class VerifyResult:
    """SigLIP (+ optional graph-label) verification of a phrase at an observation."""

    # "UNAVAILABLE" means the encoder was gone (released for the VLM), not evidence of absence.
    status: str  # "PRESENT" | "CANDIDATE" | "ABSENT" | "UNAVAILABLE"
    sim: float
    obs_id: int
    phrase: str
    ok: bool = False
    text_feat: np.ndarray | None = None
    img_feat: np.ndarray | None = None


# Expand HM-EQA object phrases onto common caption synonyms (trash can ↔ recycle bin).
_OBJECT_LABEL_ALIASES: dict[str, frozenset[str]] = {
    "trash": frozenset({"recycle", "bin", "garbage", "waste", "rubbish"}),
    "garbage": frozenset({"recycle", "bin", "trash", "waste"}),
    "bin": frozenset({"recycle", "trash", "garbage", "waste"}),
    "recycle": frozenset({"bin", "trash", "garbage"}),
    "mat": frozenset({"exercise", "yoga", "workout"}),
    "curtain": frozenset({"curtains", "drape", "drapes", "shade"}),
}

_QUESTION_STOPWORDS = frozenset(
    {
        "is",
        "the",
        "a",
        "an",
        "on",
        "in",
        "at",
        "to",
        "or",
        "and",
        "did",
        "i",
        "any",
        "there",
        "which",
        "where",
        "what",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "my",
        "me",
        "it",
        "its",
        "this",
        "that",
        "with",
        "for",
        "of",
        "by",
        "from",
        "left",
        "next",
        "under",
        "over",
        "one",
        "two",
        "not",
        "all",
        "can",
        "you",
        "your",
        "how",
        "when",
        "who",
        "why",
        "fold",
        "turned",
        "mounted",
        "standing",
        "covered",
        "color",
        "objects",
        "object",
        "see",
        "things",
        "thing",
        "room",
        "area",
        "items",
        "item",
        "anywhere",
    }
)

# Stem verbs / fillers that must not win SigLIP phrase[0] over the object noun
# (``trying remember placed`` beating ``large wall clock``).
_QUESTION_VERB_FILLERS = frozenset(
    {
        "trying",
        "try",
        "remember",
        "looking",
        "look",
        "placed",
        "place",
        "find",
        "finding",
        "found",
        "seek",
        "seeking",
        "recall",
        "tell",
        "know",
        "think",
        "want",
        "need",
        "help",
        "like",
        "get",
        "got",
        "put",
        "keep",
        "leave",
        "locate",
        "locating",
        "search",
        "searching",
        "show",
        "showing",
        "please",
    }
)

# Count-MCQ target extraction ("How many <target> …"). The match key must be the
# noun phrase right after "How many", not every stem token — "How many chairs are
# in the dining room?" otherwise matches a dining-table node via "dining".
_COUNT_TARGET_BOUNDARY_RE = re.compile(
    r"\b(?:are|is|was|were|am|did|do|does|can|could|"
    r"have|has|had|i|you|it|we|they|there|at|on|in|by|for|with|"
    r"under|over|next|left|put|placed|leave|behind|above|below|near|"
    r"beside|around|inside|outside|standing|sitting|hanging)\b|\?",
    re.IGNORECASE,
)
# Quantity wrappers that are not the object being counted ("sets of utensils").
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


def question_stem_for_keywords(question: str) -> str:
    """Return the object stem of an HM-EQA question (no MCQ options / ``Answer:``).

    Agentic traces pass the full ``… A) … D) … Answer:`` string into phrase
    extract; without stripping, SigLIP verifies ``table sunroom answer`` instead
    of ``fruit bowl`` (failfix5 q105).
    """
    text = (question or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s*Answer:\s*$", "", text, flags=re.IGNORECASE)
    m = re.search(r"\s+[A-D]\)\s+", text, flags=re.IGNORECASE)
    if m:
        text = text[: m.start()]
    if "?" in text:
        text = text.split("?", 1)[0]
    return text.strip()


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


def heuristic_relevant_phrases(question: str, *, max_phrases: int = 4) -> list[str]:
    """Multi-word object phrases from the question stem (e.g. ``woven basket``).

    Prefer later noun compounds over leading verb n-grams so SigLIP verify uses
    ``large wall clock`` / ``fruit bowl`` rather than ``trying remember placed``.
    """
    head = question_stem_for_keywords(question).lower()
    tokens = [tok for tok in re.findall(r"[a-z]+", head) if len(tok) >= 3 and tok not in _QUESTION_STOPWORDS]
    scored: list[tuple[float, int, str]] = []
    for n in range(min(3, len(tokens)), 1, -1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            words = phrase.split()
            score = float(n) * 10.0 + 0.1 * float(i)
            if words[0] in _QUESTION_VERB_FILLERS:
                score -= 50.0
            if words[-1] in _QUESTION_VERB_FILLERS:
                score -= 20.0
            scored.append((-score, i, phrase))
    phrases: list[str] = []
    for _neg_score, _i, phrase in sorted(scored):
        if phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) >= max_phrases:
            break
    return phrases


def _object_match_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(tok) >= 3 and tok not in _QUESTION_STOPWORDS
    }


_ACTION_READ_RE = re.compile(r"^read\s*(?:image\s+)?(\d+)$")
_ACTION_LOOK_RE = re.compile(r"^(?:image\s+)?(\d+)$")


def parse_eqa_action(action: str) -> tuple[str, int | None]:
    """Parse the EQA ``action`` field into ``("", None)``, ``("look", N)``, or ``("read", N)``.

    Only the documented forms are accepted: ``""``, a bare Image id (``2`` / ``Image 2``),
    or ``read N`` (``read 2`` / ``read Image 2`` / ``read2``). Free text that merely
    contains a number (e.g. ``I count 3 lamps``) is rejected as ``("", None)`` so a stray
    digit can never be mistaken for an Image id; callers fall back to the Image-1 FIND view.
    ``read N`` means the answer is written or shown in Image N but not legible — stay on
    that view. A bare image id is ``look`` (navigate / inspect).
    """
    raw = (action or "").strip().lower()
    if not raw:
        return "", None
    m = _ACTION_READ_RE.match(raw)
    if m:
        return "read", int(m.group(1))
    m = _ACTION_LOOK_RE.match(raw)
    if m:
        return "look", int(m.group(1))
    return "", None


def label_matches_relevant_object(obj: str, label: str) -> bool:
    """True when ``label`` plausibly names ``obj`` (handles ``standing fan`` vs ``fan``)."""
    obj_l = (obj or "").strip().lower()
    lab_l = (label or "").strip().lower()
    if not obj_l or not lab_l:
        return False
    if obj_l in lab_l or lab_l in obj_l:
        return True
    obj_tok = _object_match_tokens(obj_l)
    lab_tok = _object_match_tokens(lab_l)
    if not obj_tok or not lab_tok:
        return False
    if obj_tok <= lab_tok or lab_tok <= obj_tok:
        return True
    if obj_tok & lab_tok:
        return True
    # Alias expansion: "trash can" ↔ "recycle bin".
    for tok in obj_tok:
        aliases = _OBJECT_LABEL_ALIASES.get(tok)
        if aliases and (aliases & lab_tok):
            return True
    for tok in lab_tok:
        aliases = _OBJECT_LABEL_ALIASES.get(tok)
        if aliases and (aliases & obj_tok):
            return True
    return False


def countable_primary_label_matches(obj: str, node: GraphNode) -> bool:
    """FIND-candidate matching uses detector-primary labels on instance nodes only."""
    if not getattr(node, "countable_instance", False):
        return False
    labels = [str(l).strip() for l in (node.labels or []) if str(l).strip()]
    if not labels:
        return False
    return label_matches_relevant_object(obj, labels[0])


def node_display_name(node: GraphNode, *, max_len: int = 120) -> str:
    """Close-look Qwen name when present; otherwise detector / VLM labels."""
    looked = str(getattr(node, "close_look_label", None) or "").strip()
    if looked:
        return looked if len(looked) <= max_len else looked[: max_len - 3] + "..."
    labels = [str(lab).strip() for lab in (getattr(node, "labels", None) or []) if str(lab).strip()]
    if getattr(node, "is_viewpoint", False):
        s = ", ".join(labels) if labels else "view"
    elif getattr(node, "is_frontier", False):
        s = ", ".join(labels) if labels else "frontier"
    else:
        s = ", ".join(labels) if labels else "object"
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def finder_label_texts(node: GraphNode) -> list[str]:
    """Close-look Qwen name first, then detector primary — texts used to FIND views."""
    texts: list[str] = []
    looked = str(getattr(node, "close_look_label", None) or "").strip()
    if looked:
        texts.append(looked)
    primary = str(node.labels[0]).strip() if getattr(node, "labels", None) else ""
    if primary and primary.lower() not in {t.lower() for t in texts}:
        texts.append(primary)
    return texts


def format_graph_node_candidates(nodes: list[GraphNode], *, max_nodes: int = 6) -> str:
    """Point at views; use a close-look Qwen label when we have one, never YoloE vocab."""
    bits: list[str] = []
    for node in nodes[:max_nodes]:
        xyz = np.asarray(node.xyz, dtype=np.float64).reshape(-1)
        loc = f"[Image {int(node.obs_id)}] at ({float(xyz[0]):.1f}, {float(xyz[1]):.1f})"
        looked = str(getattr(node, "close_look_label", None) or "").strip()
        if looked:
            bits.append(f"{looked} {loc}")
        else:
            bits.append(loc)
    suffix = " …" if len(nodes) > max_nodes else ""
    return "; ".join(bits) + suffix


_GRAPH_CANDIDATE_COUNT_DISCLAIMER = "list length is not a count; verify in attached images"


def _location_mcq_weak_tokens() -> frozenset[str]:
    return frozenset(
        {
            "next",
            "near",
            "with",
            "from",
            "room",
            "living",
            "between",
            "the",
            "and",
            "by",
            "on",
            "at",
            "of",
            "to",
            "in",
            "under",
            "any",
            "left",
            "leave",
            "placed",
            "somewhere",
            "anywhere",
        }
    )


def distinctive_choice_tokens(choice: str) -> list[str]:
    """Content tokens from an MCQ option (for label↔option matching)."""
    weak = _location_mcq_weak_tokens()
    return [t for t in re.findall(r"[a-z]+", (choice or "").lower()) if len(t) > 3 and t not in weak]


# Generic furniture words that appear in many options/views; do not let them beat
# rare landmarks (refrigerator, treadmill) when ranking Image 1.
_LANDMARK_GENERIC_TOKENS = frozenset(
    {
        "table",
        "tables",
        "chair",
        "chairs",
        "sofa",
        "sofas",
        "couch",
        "couches",
        "door",
        "doors",
        "wall",
        "floor",
        "room",
        "rug",
        "window",
        "cabinet",
        "cabinets",
    }
)

# Question-object → landmark tokens that should win Image 1 over generic furniture.
# For trash: prefer fridge/recycle over sink — sink is often a competing MCQ option.
_QUESTION_LANDMARK_BOOST: dict[str, frozenset[str]] = {
    "trash": frozenset({"refrigerator", "fridge", "recycle", "bin"}),
    "garbage": frozenset({"refrigerator", "fridge", "recycle", "bin"}),
    "bin": frozenset({"refrigerator", "fridge", "recycle"}),
    "mat": frozenset({"treadmill", "elliptical", "bike", "bicycle", "exercise"}),
    "towel": frozenset({"bathroom", "bath", "sink", "toilet", "shower"}),
    "kettle": frozenset({"kitchen", "counter", "stove", "sink"}),
    "pillow": frozenset({"sofa", "couch", "bed", "armchair"}),
    "fruit": frozenset({"dining"}),
    "bowl": frozenset({"dining"}),
}


def location_mcq_landmark_phrases(
    question: str,
    *,
    max_landmarks: int = 4,
) -> list[str]:
    """Landmark phrases from location-MCQ options (e.g. ``kitchen island``).

    Stem heuristics ignore choices, and enrich labels are often empty for a scene —
    without these seeds, hyp recall never builds Investigate cards for option places
    even when the graph already has matching labels.
    """
    try:
        from emet.habitat.metrics import (
            choices_are_location_mcq,
            parse_mcq_choices_from_question,
        )
    except ImportError:
        return []
    choices = parse_mcq_choices_from_question(question or "")
    if not choices or not choices_are_location_mcq(choices):
        return []
    out: list[str] = []
    lead = re.compile(
        r"^(?:on|in|at|by|near|under|beside|behind|inside|outside|"
        r"between|next\s+to)\s+(?:the\s+)?",
        re.IGNORECASE,
    )
    for ch in choices[:4]:
        text = lead.sub("", str(ch or "").strip()).strip().lower()
        text = re.sub(r"\s+", " ", text)
        if len(text) < 3 or text in _QUESTION_STOPWORDS:
            continue
        if text not in out:
            out.append(text)
        if len(out) >= max_landmarks:
            break
    return out


def consolidate_relevant_keywords(
    phrases: list[str],
    extras: list[str],
    *,
    max_items: int = 4,
) -> tuple[list[str], list[str]]:
    """Phrase-first dedupe for CONFIRMED_MEMORY and image selection."""
    phrase_list: list[str] = []
    for item in phrases:
        key = item.strip().lower()
        if key and key not in phrase_list:
            phrase_list.append(key)

    phrase_tokens: set[str] = set()
    for phrase in phrase_list:
        phrase_tokens.update(_object_match_tokens(phrase))

    objects: list[str] = list(phrase_list)
    for item in extras:
        key = item.strip().lower()
        if not key or key in objects:
            continue
        if key in _QUESTION_STOPWORDS:
            continue
        if key in phrase_tokens:
            continue
        if any(key in phrase.split() for phrase in phrase_list):
            continue
        objects.append(key)

    return phrase_list[:max_items], objects[:max_items]


def heuristic_relevant_objects(question: str, *, max_objects: int = 4) -> list[str]:
    """Cheap noun-like tokens from the question stem (before MCQ options).

    Skip verb fillers and prefer later tokens (object of ``looking for X``).
    """
    head = question_stem_for_keywords(question)
    toks: list[str] = []
    for tok in re.findall(r"[a-z]{3,}", head.lower()):
        if tok in _QUESTION_STOPWORDS or tok in _QUESTION_VERB_FILLERS:
            continue
        if tok not in toks:
            toks.append(tok)
    out: list[str] = []
    for tok in reversed(toks):
        if tok not in out:
            out.append(tok)
        if len(out) >= max_objects:
            break
    return out


def labels_are_semantic_graph_hypothesis(labels: list[str] | None) -> bool:
    """
    Whether ``labels`` should become a scene-graph node (vs navigation-only sample).

    Generic VLM fallback ``["object"]`` is not a semantic hypothesis: it would clutter
    the graph with one node per controller step.
    """
    if not labels:
        return False
    if len(labels) == 1 and labels[0].strip().lower() == "object":
        return False
    return True


@dataclass
class GraphNavigationSample:
    """A viewpoint along the run without an object-level graph node (RGB + anchors)."""

    rgb: np.ndarray
    xyz: np.ndarray  # (3,) scene anchor (e.g. depth median in world frame)
    base_xyz: np.ndarray | None = None  # (3,) optional robot base x,y,z for trajectory context


GT_BODY_DESC_PREFIX = "ground_truth:"


@dataclass
class GraphNode:
    """Single node in the scene graph: an object or region with label and position."""

    node_id: int
    labels: list[str]
    xyz: np.ndarray  # (3,) world position
    obs_id: int  # 1-based index into observations list
    description: str | None = None  # optional VLM-generated description
    last_seen: int = 0
    support_count: int = 1
    extent_half: np.ndarray | None = None  # (3,) half-axis sizes in meters; None = point-like
    bbox_xyxy: tuple[int, int, int, int] | None = None  # pixel crop in obs RGB; None = full frame
    is_viewpoint: bool = False  # True = robot/camera vantage (``seen_from`` target), not a detected object
    is_frontier: bool = False  # True = unexplored map frontier cluster (managed by sync_frontier_nodes)
    frontier_cell_count: int = 0  # frontier only: unexplored cells in this cluster (area gain)
    frontier_keyword_score: float = 0.0  # frontier only: question-keyword affinity of nearby hints
    embedding: np.ndarray | None = None  # optional visual embedding (e.g. SigLIP crop)
    bounds_3d: dict[str, list[float]] | None = None  # axis-aligned world bounds {min,max,center,size}
    nav_attempts: int = 0
    nav_failures: int = 0
    last_nav_note: str | None = None
    last_nav_at_step: int = 0
    belief_confidence: float = 0.5
    position_covariance: np.ndarray | None = None
    position_history: list[dict[str, Any]] = field(default_factory=list)
    identity_key: str | None = None
    # True only when this node came from instance-level evidence. Label-only
    # frame summaries are not safe inputs for exact count hints.
    countable_instance: bool = False
    # Qwen caption after a close look / vlm_assess. Preferred over detector class names.
    close_look_label: str | None = None
    change_events: list[dict[str, Any]] = field(default_factory=list)
    expected_absence_count: int = 0
    last_absence_step: int = -1


def is_ground_truth_node(node: GraphNode | None) -> bool:
    """True when ``node.description`` marks sim GT (stable ``body_name`` key)."""
    if node is None:
        return False
    desc = getattr(node, "description", None)
    return isinstance(desc, str) and desc.startswith(GT_BODY_DESC_PREFIX)


@dataclass
class GraphObservation:
    """One observation (image + pose + labels) used to build the graph."""

    obs_id: int  # 1-based
    rgb: np.ndarray  # (H, W, 3)
    xyz: np.ndarray  # (3,) e.g. mean of visible points or camera position
    labels: list[str]
    description: str | None = None  # optional VLM-generated description
    viewer_xyz: np.ndarray | None = None  # (3,) robot base or camera when the image was taken


def _near(p1: np.ndarray, p2: np.ndarray, max_dist: float = 1.5) -> bool:
    return float(np.linalg.norm(p1[:2] - p2[:2])) <= max_dist


def _on(p_lower: np.ndarray, p_upper: np.ndarray, z_thresh: float = 0.15) -> bool:
    """Heuristic: lower object is 'on' upper if roughly below and close in xy."""
    if p_lower[2] >= p_upper[2]:
        return False
    return abs(p_lower[2] - p_upper[2]) <= z_thresh + 0.2 and float(np.linalg.norm(p_lower[:2] - p_upper[:2])) < 0.5


def _on_floor(p: np.ndarray, floor_z: float = 0.05) -> bool:
    return float(p[2]) <= floor_z


_ROOM_WORDS = frozenset(
    {
        "room",
        "kitchen",
        "bedroom",
        "bathroom",
        "living room",
        "dining room",
        "hallway",
        "office",
        "garage",
        "sunroom",
    }
)


def _node_is_room(node: GraphNode) -> bool:
    text = " ".join(node.labels).lower()
    return any(word in text for word in _ROOM_WORDS)


def _inside_bounds(point: np.ndarray, bounds: dict[str, list[float]] | None) -> bool:
    if not bounds or "min" not in bounds or "max" not in bounds:
        return False
    xyz = np.asarray(point, dtype=float).reshape(-1)[:3]
    lower = np.asarray(bounds["min"], dtype=float).reshape(-1)[:3]
    upper = np.asarray(bounds["max"], dtype=float).reshape(-1)[:3]
    return bool(np.all(xyz >= lower) and np.all(xyz <= upper))

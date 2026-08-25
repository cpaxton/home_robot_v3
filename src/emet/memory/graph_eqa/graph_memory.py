# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Graph-based EQA memory: re-implementation inspired by GraphEQA
# (https://arxiv.org/abs/2412.14480). Object-centric scene graph + task-relevant
# images for embodied question answering. No code copied from closed-source repos.

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from PIL import Image

from emet.core.parameters import Parameters
from emet.habitat.metrics import extract_mcq_letter
from emet.memory.graph_eqa.attempt_ledger import (
    AttemptRecord,
    AttemptSource,
    infer_nav_outcome,
    infer_nav_status_code,
    records_from_dicts,
    records_to_dicts,
    summary_bits_for_obs,
)
from emet.memory.graph_eqa.eqa_views import (
    EQA_SAME_OBS_PROGRESS_M,
    eqa_look_is_spent,
    rgb_uint8,
    spread_obs_ids_xy,
    tightest_node_crop,
)
from emet.memory.graph_eqa.human_answer import format_human_eqa_answer
from emet.memory.graph_eqa.mcq_debias import (
    LETTERS,
    answer_is_unknownish,
    count_answer_is_none_or_zero,
    extract_single_letter,
    letter_to_original_index,
    match_freeform_to_choice,
    rotated_choice_order,
    tally_choice_votes,
)
from emet.memory.graph_eqa.world_evidence import (
    WorldEvidenceStore,
    resolve_world_evidence_mode,
)
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
        form
        for alias in _COUNT_WORD_ALIASES.get(str(target or "").lower(), ())
        for form in _count_word_forms(alias)
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
            float(np.linalg.norm(nxy - np.asarray(kept_node.xyz, dtype=np.float64).reshape(3)[:2]))
            < min_xy_m
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


_GRAPH_CANDIDATE_COUNT_DISCLAIMER = (
    "list length is not a count; verify in attached images"
)


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


class GraphEQAMemory:
    """
    Graph-based semantic memory for Embodied Question Answering (EQA).

    Maintains an object-centric scene graph (nodes = objects/regions with labels and
    3D positions; edges = spatial relations). Uses the same EQA query contract as
    the DynaMem voxel map: query_answer(question, xyt, planner) returns
    (reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images).

    Optional **Dynagraph** behavior (parameters ``dynagraph_merge_xy_m``,
    ``dynagraph_staleness_horizon``): spatial merge of nodes with the same primary
    label within XY distance, and ``maintain(current_step)`` to drop stale nodes.
    """

    def __init__(
        self,
        parameters: Parameters | None = None,
        max_near_distance: float = 1.5,
        eqa_client: Callable[..., str] | None = None,
        image_description_client: Callable[..., str] | None = None,
        log_dir: str = "graph_eqa_log",
        defer_llm_clients: bool = False,
    ):
        self.parameters = parameters or {}
        self.max_near_distance = max_near_distance
        self.last_eqa_raw: str = ""
        self.last_eqa_parsed: tuple[str, str, bool, str, str] = ("", "", False, "", "")
        # Model-native output before salvage, memory-location, or equipment overrides.
        # Agentic grounded_v2 arbitration consumes these immutable fields.
        self.last_eqa_model_raw: str = ""
        self.last_eqa_model_parsed: tuple[str, str, bool, str, str] = ("", "", False, "", "")
        self.last_agentic_decision: dict[str, Any] | None = None
        self.last_eqa_obs_ids: list[int] = []
        self.last_eqa_action_obs_id: int | None = None
        # Next query_answer must attach this RGB as Image 1 (Action:N / post-nav look).
        self.last_eqa_look_obs_id: int | None = None
        self.last_eqa_prompt_node_count: int = 0
        self.last_eqa_prompt_regions: int = 0
        self.last_eqa_spatial_rag: dict[str, Any] | None = None
        self.last_room_clusters: list[Any] = []
        self.last_nav_result_note: str = ""
        self.last_eqa_nav_fallback_count: int = 0
        # Frames attached to the last EQA call, kept for salvage / counterfactual re-asks.
        self.last_relevant_images: list[Any] = []
        # Decode-budget health: did the generation reach ``answer:``, and did the terse
        # re-ask have to rescue it?
        self.last_eqa_answer_field_emitted: bool = False
        self.last_eqa_salvage_used: bool = False
        # Model's own confidence before the graph-coverage gate suppresses it (for early-stop).
        self.last_eqa_model_confident: bool = False
        # Per-question distances for each obs_id we navigated to (same-obs look budget).
        self._obs_nav_dists: dict[int, list[float]] = {}
        self._nodes: list[GraphNode] = []
        self._edges: list[tuple[int, int, str]] = []  # (id1, id2, relation)
        self._room_clusters: list[Any] = []
        self._room_connectivity_fn: Callable[[tuple[float, float], tuple[float, float]], bool] | None = None
        self._relation_beliefs: dict[tuple[int, int, str], RelationBelief] = {}
        self._change_events: list[dict[str, Any]] = []
        self._observations: list[GraphObservation] = []
        self._next_obs_id = 1
        self._question: str | None = None
        self._relevant_objects: list[str] | None = None
        # Dynagraph improvements (kept OFF here so GraphEQA stays a clean baseline; the
        # DynagraphController turns them on):
        #  * memory_summary_enabled: prepend the CONFIRMED_MEMORY block to the planner prompt.
        #  * _text_grounder: open-vocab visual grounder (text -> (similarity, xyz)) backed by
        #    the voxel map's SigLIP features, decoupling grounding from brittle caption labels
        #    (e.g. a "woven basket" captioned as "decorative plant").
        self.memory_summary_enabled: bool = False
        #  * mcq_debias_enabled: choice-rotation vote at episode end (see mcq_debias.py).
        self.mcq_debias_enabled: bool = False
        self.last_mcq_debias: dict[str, Any] = {}
        self._text_grounder: Callable[[str], tuple[float, np.ndarray] | None] | None = None
        self._obs_id_grounder: Callable[[str], int | None] | None = None
        self._visual_find_fn: Callable[..., list[Any]] | None = None
        # phrase → top-k (sim, obs_id) from find_all_images / whole-image rank, frozen
        # before prepare_dynagraph_vram_for_eqa drops GPU SigLIP.
        self._visual_find_rank_cache: dict[str, list[tuple[float, int]]] = {}
        self._eqa_highlight_key: tuple[str, int] | None = None
        self._enrich_object_hints: list[str] = []
        self._history_outputs: list[str] = []
        self._relevant_phrases: list[str] = []
        self._confirmed_memory_siglip_encoder: Any | None = None
        self._obs_siglip_features: dict[int, np.ndarray] = {}
        # obs_id → content revision (bumped when merge refreshes RGB/candidate evidence).
        self._obs_revisions: dict[int, int] = {}
        self._last_obs_content_update_id: int | None = None
        self._siglip_phrase_cache: dict[str, tuple[float, np.ndarray, int | None]] = {}

        self.log_dir = log_dir
        self.eqa_client = eqa_client
        self.image_description_client = image_description_client
        self._defer_llm_clients = defer_llm_clients
        self._nav_samples: list[GraphNavigationSample] = []
        self._viewpoint_by_obs_id: dict[int, int] = {}
        # Reuse viewpoint nodes when the camera/base pose is within this radius (m).
        self.viewpoint_merge_m: float = 0.15
        self._record_navigation = True
        self._nav_max = 256
        self._graph_timestep: int = 0
        self._fallback_timestep: int = 0
        self.spatial_merge_m: float = 0.0
        self.staleness_horizon: int = 0
        self.instance_ingest_stats: dict[str, int] = {
            "proposed": 0,
            "rejected_confidence": 0,
            "rejected_support": 0,
            "admitted": 0,
            "merged": 0,
            "created": 0,
        }
        self.frontier_nodes_enabled: bool = True
        self._frontier_max_nodes: int = 12
        self._frontier_min_cluster_cells: int = 3
        self._frontier_keyword_score_weight: float = 1.0
        self.image_nav_min_approach_m: float = 0.35
        # (obs_id, normalized_phrase) claims retracted after close+ABSENT verify —
        # keep the place node, but stop offering that stem-object hyp card.
        self._retracted_nav_claims: set[tuple[int, str]] = set()
        # Distinct view IDs that produced ABSENT evidence; separate from the place
        # IDs above because one place can be verified from multiple fresh stations.
        self._retraction_evidence_views: set[tuple[int, str]] = set()
        # Action-outcome ledger (opt-in via eqa.attempt_ledger / EMET_EQA_ATTEMPT_LEDGER).
        # When off, record_nav_attempt keeps updating GraphNode counters only.
        self._attempt_records: list[AttemptRecord] = []
        self._attempt_ledger_max: int = 512
        self._attempt_ledger_question_id: str | None = None
        # When True, clear_retracted_nav_claims keeps ABSENT blacklists across questions.
        # Default off so HM-EQA paper arms stay unchanged.
        self.persist_absent_claims: bool = False
        # Room-scoped timeline (survives cluster rebuild; agent-facing via format_room_history).
        self._room_events: list[dict[str, Any]] = []
        self._room_events_max: int = 64
        eqa_cfg = self._parameters_dict().get("eqa")
        configured_world_mode = eqa_cfg.get("graph_evidence_mode", "off") if isinstance(eqa_cfg, dict) else "off"
        world_mode = resolve_world_evidence_mode(
            os.environ.get("EMET_EQA_GRAPH_EVIDENCE_MODE", "") or configured_world_mode
        )
        self.world_evidence = WorldEvidenceStore(
            mode=world_mode,
            session_id=os.environ.get("EMET_WORLD_SESSION_ID", ""),
        )
        self._capture_context: dict[str, Any] = {}
        self._load_navigation_settings()
        self._load_dynagraph_settings()
        self._load_frontier_settings()
        self._load_attempt_ledger_settings()

        if not defer_llm_clients and (self.eqa_client is None or self.image_description_client is None):
            self._init_clients()

    def _parameters_dict(self) -> dict[str, Any]:
        p = self.parameters
        if isinstance(p, dict):
            return p
        if hasattr(p, "data") and isinstance(p.data, dict):
            return p.data
        return {}

    def _load_navigation_settings(self) -> None:
        d = self._parameters_dict()
        if not d:
            return
        v = d.get("graph_eqa_record_navigation")
        if v is not None:
            self._record_navigation = bool(v)
        blk = d.get("graph_eqa_extract")
        if isinstance(blk, dict) and blk.get("navigation_samples_max") is not None:
            self._nav_max = max(1, int(blk["navigation_samples_max"]))
        eqa = d.get("eqa")
        if isinstance(eqa, dict) and eqa.get("image_nav_min_approach_m") is not None:
            self.image_nav_min_approach_m = max(0.05, float(eqa["image_nav_min_approach_m"]))

    def _load_dynagraph_settings(self) -> None:
        d = self._parameters_dict()
        if not d:
            return
        if d.get("dynagraph_merge_xy_m") is not None:
            self.spatial_merge_m = float(d["dynagraph_merge_xy_m"])
        if d.get("dynagraph_staleness_horizon") is not None:
            self.staleness_horizon = max(0, int(d["dynagraph_staleness_horizon"]))
        if d.get("dynagraph_viewpoint_merge_m") is not None:
            self.viewpoint_merge_m = max(0.0, float(d["dynagraph_viewpoint_merge_m"]))

    def _load_frontier_settings(self) -> None:
        d = self._parameters_dict()
        blk = d.get("graph_eqa_frontier_nodes")
        if not isinstance(blk, dict):
            eqa = d.get("graph_eqa")
            if isinstance(eqa, dict):
                blk = eqa.get("frontier_nodes")
        if not isinstance(blk, dict):
            return
        if blk.get("enabled") is not None:
            self.frontier_nodes_enabled = bool(blk["enabled"])
        if blk.get("max_nodes") is not None:
            self._frontier_max_nodes = max(1, int(blk["max_nodes"]))
        if blk.get("min_cluster_cells") is not None:
            self._frontier_min_cluster_cells = max(1, int(blk["min_cluster_cells"]))
        if blk.get("keyword_score_weight") is not None:
            self._frontier_keyword_score_weight = max(0.0, float(blk["keyword_score_weight"]))

    def _load_attempt_ledger_settings(self) -> None:
        """Load ``eqa.attempt_ledger`` dict knobs (max_records, persist_absent_claims)."""
        d = self._parameters_dict()
        blk: dict[str, Any] = {}
        eqa = d.get("eqa")
        if isinstance(eqa, dict) and isinstance(eqa.get("attempt_ledger"), dict):
            blk = dict(eqa["attempt_ledger"])
        agent = d.get("agent")
        if isinstance(agent, dict) and isinstance(agent.get("attempt_ledger"), dict):
            blk = {**blk, **agent["attempt_ledger"]}
        if blk.get("max_records") is not None:
            self._attempt_ledger_max = max(32, int(blk["max_records"]))
        if blk.get("persist_absent_claims") is not None:
            self.persist_absent_claims = bool(blk["persist_absent_claims"])
        env_persist = os.environ.get("EMET_ATTEMPT_LEDGER_PERSIST_ABSENT", "").strip().lower()
        if env_persist in ("1", "true", "yes", "on"):
            self.persist_absent_claims = True
        elif env_persist in ("0", "false", "no", "off"):
            self.persist_absent_claims = False
        env_max = os.environ.get("EMET_ATTEMPT_LEDGER_MAX", "").strip()
        if env_max:
            try:
                self._attempt_ledger_max = max(32, int(env_max))
            except ValueError:
                pass

    def attempt_summary_for_obs(self, obs_id: int, *, max_bits: int = 4) -> str:
        """Newest-first compact attempt tags for place cards / diagnostics."""
        return summary_bits_for_obs(self._attempt_records, int(obs_id), max_bits=max_bits)

    def record_close_look_label(self, obs_id: int, label: str) -> None:
        """Store a Qwen close-look / vlm_assess name on graph nodes for this view.

        Tags object nodes whose candidate image is ``obs_id``, and objects linked
        ``seen_from`` a viewpoint with that observation id (fusion may have moved
        the node's own ``obs_id`` to an earlier frame).
        """
        text = str(label or "").strip()
        if not text:
            return
        oid = int(obs_id)
        clipped = text[:80]
        vp_ids = {
            int(node.node_id)
            for node in self._nodes
            if node.is_viewpoint and int(node.obs_id) == oid
        }
        seen_from_ids: set[int] = set()
        for src, dst, rel in self._edges:
            if rel == "seen_from" and int(dst) in vp_ids:
                seen_from_ids.add(int(src))
        for idx, node in enumerate(self._nodes):
            if node.is_frontier or node.is_viewpoint:
                continue
            if int(node.obs_id) != oid and int(node.node_id) not in seen_from_ids:
                continue
            self._nodes[idx] = replace(node, close_look_label=clipped)

    def set_graph_timestep(self, step: int) -> None:
        """Set the discrete time index used for ``last_seen`` and staleness (e.g. controller ``obs_count``)."""
        self._graph_timestep = int(step)

    def _position_update(
        self,
        node: GraphNode,
        measured_xyz: np.ndarray,
        *,
        step: int,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]], float]:
        """Update a track without averaging a contradictory relocation into its centroid."""
        measured = np.asarray(measured_xyz, dtype=float).reshape(-1)[:3]
        current = np.asarray(node.xyz, dtype=float).reshape(-1)[:3]
        history = list(node.position_history)
        if not history:
            history.append(
                {
                    "step": int(node.last_seen),
                    "xyz": current.tolist(),
                    "confidence": float(node.belief_confidence),
                }
            )
        distance = float(np.linalg.norm(measured - current))
        relocation_bar = max(0.45, float(self.spatial_merge_m) * 1.5)
        changes = list(node.change_events)
        confidence = min(0.99, float(node.belief_confidence) + 0.08)
        if distance > relocation_bar:
            event = {
                "type": "position_contradiction",
                "node_id": int(node.node_id),
                "step": int(step),
                "from_xyz": current.tolist(),
                "to_xyz": measured.tolist(),
                "displacement_m": distance,
                "confidence": min(0.99, distance / max(relocation_bar, 1e-6)),
            }
            changes.append(event)
            self._change_events.append(event)
            updated = measured.copy()
            confidence = max(0.2, float(node.belief_confidence) * 0.7)
        else:
            support = max(1, int(node.support_count))
            updated = (current * support + measured) / (support + 1)
        history.append(
            {
                "step": int(step),
                "xyz": measured.tolist(),
                "confidence": confidence,
            }
        )
        samples = np.asarray([entry["xyz"] for entry in history[-20:]], dtype=float)
        covariance = np.cov(samples.T) if samples.shape[0] >= 2 else np.zeros((3, 3), dtype=float)
        return updated, covariance, history[-64:], changes[-32:], confidence

    def observe_visible_labels(
        self,
        labels: list[str],
        viewer_xyz: np.ndarray | None,
        *,
        step: int | None = None,
        viewpoint_tolerance_m: float = 0.75,
        absence_confirmations: int = 2,
        visibility_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Conservatively detect disappeared objects from repeated same-view contradictions.

        A node is expected only when the camera revisits approximately the viewpoint
        that originally saw it. This avoids treating out-of-FOV objects as absent.
        Ground-truth placements are never consulted.
        """
        if viewer_xyz is None:
            return []
        evidence_view_id = ""
        if self.world_evidence.enabled:
            context = dict(visibility_context or {})
            required = (
                bool(context.get("camera_frustum_ok")),
                bool(context.get("depth_coverage_ok")),
                bool(context.get("detector_ran")),
                bool(context.get("occlusion_free")),
            )
            evidence_view_id = str(context.get("evidence_view_id") or "")
            if not all(required) or not evidence_view_id:
                return []
        now = self._effective_timestep() if step is None else int(step)
        viewer = np.asarray(viewer_xyz, dtype=float).reshape(-1)[:3]
        visible = [str(label) for label in labels if str(label).strip()]
        events: list[dict[str, Any]] = []
        for index, node in enumerate(self._nodes):
            if node.is_viewpoint or node.is_frontier or is_ground_truth_node(node):
                continue
            original = self._observation_by_id(int(node.obs_id))
            expected_view = getattr(original, "viewer_xyz", None) if original is not None else None
            if expected_view is None:
                continue
            distance = float(np.linalg.norm(np.asarray(expected_view, dtype=float).reshape(-1)[:2] - viewer[:2]))
            if distance > float(viewpoint_tolerance_m):
                continue
            seen = any(label_matches_relevant_object(node.labels[0], label) for label in visible)
            if seen:
                self._nodes[index] = replace(
                    node,
                    expected_absence_count=0,
                    belief_confidence=min(0.99, float(node.belief_confidence) + 0.1),
                )
                continue
            if self.world_evidence.enabled:
                entity = self.world_evidence.entity_for_node(int(node.node_id))
                if entity is not None:
                    self.world_evidence.record_event(
                        subject_kind="entity",
                        subject_id=entity.entity_id,
                        predicate="visible_in_view",
                        polarity="negative",
                        source="detector",
                        confidence=0.6,
                        step=now,
                        view_id=evidence_view_id,
                        place_id=entity.place_id,
                        payload={
                            "claim_view_id": self.view_id_for_obs(int(node.obs_id)),
                            "viewpoint_distance_m": distance,
                            "visibility_qualified": True,
                            "frustum_checked": True,
                            "depth_coverage_checked": True,
                            "occlusion_checked": True,
                        },
                    )
            consecutive = (
                int(node.expected_absence_count) + 1
                if int(node.last_absence_step) < 0 or now - int(node.last_absence_step) <= 2
                else 1
            )
            changes = list(node.change_events)
            if consecutive >= int(absence_confirmations):
                event = {
                    "type": "expected_object_missing",
                    "node_id": int(node.node_id),
                    "obs_id": int(node.obs_id),
                    "step": now,
                    "last_xyz": np.asarray(node.xyz, dtype=float).tolist(),
                    "viewpoint_distance_m": distance,
                    "confirmations": consecutive,
                    "confidence": min(0.95, 0.45 + 0.2 * consecutive),
                }
                if not changes or changes[-1].get("type") != event["type"]:
                    changes.append(event)
                    self._change_events.append(event)
                    events.append(event)
            self._nodes[index] = replace(
                node,
                expected_absence_count=consecutive,
                last_absence_step=now,
                belief_confidence=max(0.05, float(node.belief_confidence) * 0.65),
                change_events=changes[-32:],
            )
        return events

    def get_change_events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._change_events]

    def set_navigation_samples_max(self, n: int) -> None:
        """Raise or lower the cap on stored navigation viewpoint samples (default from config)."""
        self._nav_max = max(1, int(n))

    @property
    def navigation_samples_max(self) -> int:
        return int(self._nav_max)

    def _effective_timestep(self) -> int:
        if self._graph_timestep > 0:
            return self._graph_timestep
        self._fallback_timestep += 1
        return self._fallback_timestep

    def clear_eqa_working_memory(self) -> None:
        """Drop cached EQA / CONFIRMED_MEMORY state after a known world change.

        Forces the next planner call to re-ground from the live graph instead of
        reusing provisional memory summaries and Image-N selections from before
        objects moved.
        """
        self.last_eqa_raw = ""
        self.last_eqa_parsed = ("", "", False, "", "")
        self.last_eqa_model_raw = ""
        self.last_eqa_model_parsed = ("", "", False, "", "")
        self.last_agentic_decision = None
        self.last_eqa_obs_ids = []
        self.last_eqa_action_obs_id = None
        self.last_eqa_look_obs_id = None
        self.last_eqa_prompt_node_count = 0
        self.last_eqa_prompt_regions = 0
        self.last_eqa_spatial_rag = None
        self.last_eqa_nav_fallback_count = 0
        self.last_eqa_model_confident = False
        self.last_relevant_images = []
        self.last_eqa_answer_field_emitted = False
        self.last_eqa_salvage_used = False
        self._obs_nav_dists.clear()
        self.clear_room_events()

    def invalidate_nodes_near(
        self,
        xyz: np.ndarray | list[float] | tuple[float, ...],
        *,
        radius_m: float = 0.75,
        current_step: int | None = None,
        prune: bool = True,
    ) -> tuple[int, int]:
        """Age object nodes near ``xyz`` so staleness pruning can drop them.

        Used after scripted body relocations (dynamic world-change / lifelong fuzz)
        when the old pose is known. Nodes keep their identity until ``maintain``
        runs (or immediately when ``prune=True``).

        Returns:
            ``(n_aged, n_pruned)``.
        """
        if not self._nodes:
            return 0, 0
        target = np.asarray(xyz, dtype=np.float64).reshape(-1)
        if target.size < 2:
            return 0, 0
        cur = int(current_step if current_step is not None else self._effective_timestep())
        horizon = max(0, int(self.staleness_horizon))
        aged_last_seen = cur - horizon - 1 if horizon > 0 else cur - 10_000
        radius = float(radius_m)
        n_aged = 0
        for i, n in enumerate(self._nodes):
            if is_ground_truth_node(n) or n.is_frontier or n.is_viewpoint:
                continue
            node_xy = np.asarray(n.xyz, dtype=np.float64).reshape(-1)
            if node_xy.size < 2:
                continue
            if float(np.linalg.norm(node_xy[:2] - target[:2])) > radius:
                continue
            self._nodes[i] = replace(n, last_seen=int(aged_last_seen))
            n_aged += 1
        n_pruned = 0
        if prune and n_aged > 0:
            if horizon > 0:
                n_pruned = int(self.maintain(cur))
            else:
                # Staleness disabled: still drop explicitly invalidated object nodes.
                n_pruned = int(self._drop_nodes_near(target, radius_m=radius))
        return n_aged, n_pruned

    def _drop_nodes_near(self, xyz: np.ndarray, *, radius_m: float) -> int:
        """Remove non-GT object nodes within ``radius_m`` of ``xyz`` (xy)."""
        target = np.asarray(xyz, dtype=np.float64).reshape(-1)
        to_drop = [
            n
            for n in self._nodes
            if not is_ground_truth_node(n)
            and not n.is_frontier
            and not n.is_viewpoint
            and float(np.linalg.norm(np.asarray(n.xyz, dtype=np.float64).reshape(-1)[:2] - target[:2]))
            <= float(radius_m)
        ]
        if not to_drop:
            return 0
        drop_obs = {n.obs_id for n in to_drop}
        drop_node_ids = {n.node_id for n in to_drop}
        drop_node_ids |= {n.node_id for n in self._nodes if n.is_viewpoint and int(n.obs_id) in drop_obs}
        self._nodes = [n for n in self._nodes if n.node_id not in drop_node_ids]
        self._observations = [o for o in self._observations if o.obs_id not in drop_obs]
        for i, n in enumerate(self._nodes, start=1):
            self._nodes[i - 1] = replace(n, node_id=i)
        self._reindex_world_entities()
        self._rebuild_viewpoint_index()
        self._update_edges()
        return len(to_drop)

    def maintain(self, current_step: int) -> int:
        """
        Drop stale nodes (and their observations) when ``staleness_horizon`` > 0,
        then renumber ``node_id`` to 1..N and rebuild edges.

        Returns:
            Number of nodes removed.
        """
        if self.staleness_horizon <= 0 or not self._nodes:
            return 0
        cur = int(current_step)
        to_drop: list[GraphNode] = [
            n
            for n in self._nodes
            if not is_ground_truth_node(n) and not n.is_frontier and cur - int(n.last_seen) > self.staleness_horizon
        ]
        if not to_drop:
            return 0
        drop_obs = {n.obs_id for n in to_drop if not n.is_viewpoint}
        drop_node_ids = {n.node_id for n in to_drop}
        drop_node_ids |= {n.node_id for n in self._nodes if n.is_viewpoint and int(n.obs_id) in drop_obs}
        self._nodes = [n for n in self._nodes if n.node_id not in drop_node_ids]
        self._observations = [o for o in self._observations if o.obs_id not in drop_obs]
        for i, n in enumerate(self._nodes, start=1):
            self._nodes[i - 1] = replace(n, node_id=i)
        self._reindex_world_entities()
        self._rebuild_viewpoint_index()
        self._update_edges()
        return len(to_drop)

    def _ensure_llm_clients(self) -> None:
        """Load shared Qwen3.5 multimodal on first use when defer_llm_clients=True."""
        if self.eqa_client is not None and self.image_description_client is not None:
            return
        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize EQA + keyword helper (one shared VLM: gemma4 / Qwen-VL / Qwen3.5)."""
        try:
            from emet.llms.eqa_vl_settings import get_eqa_vl_int
            from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients

            kw = get_eqa_vl_int(self.parameters, "graph_keyword_max_tokens", 64)
            self.image_description_client, self.eqa_client = build_graph_eqa_vlm_clients(
                parameters=self.parameters,
                keyword_max_tokens=kw,
            )
        except ImportError as e:
            raise ImportError(
                "GraphEQA memory requires emet.llms for EQA. Install GPU extras (torch, transformers)."
            ) from e

    def obs_revision(self, obs_id: int) -> int:
        """Content generation for *obs_id* (advances when candidate RGB is refreshed)."""
        return int(self._obs_revisions.get(int(obs_id), 0))

    def set_capture_context(
        self,
        *,
        camera_pose_world: Any = None,
        base_pose_world: Any = None,
        session_id: str | None = None,
    ) -> None:
        """Attach immutable sensor-pose provenance to subsequent object views."""
        self._capture_context = {
            "camera_pose_world": (
                np.asarray(camera_pose_world, dtype=float).reshape(4, 4).copy()
                if camera_pose_world is not None
                else None
            ),
            "base_pose_world": (
                np.asarray(base_pose_world, dtype=float).reshape(-1)[:3].copy() if base_pose_world is not None else None
            ),
        }
        if session_id:
            self.world_evidence.session_id = str(session_id)

    def clear_capture_context(self) -> None:
        self._capture_context = {}

    def _record_world_view_for_obs(self, obs_id: int) -> str:
        if not self.world_evidence.enabled:
            return ""
        oid = int(obs_id)
        node = next(
            (
                item
                for item in self._nodes
                if int(item.obs_id) == oid and not item.is_frontier and not item.is_viewpoint
            ),
            None,
        )
        obs = self._observation_by_id(oid)
        if node is None or obs is None:
            return ""
        entity = self.world_evidence.ensure_entity(
            identity_key=str(node.identity_key or f"obs:{oid}"),
            node_id=int(node.node_id),
            labels=list(node.labels),
            xyz=node.xyz,
            step=self._effective_timestep(),
        )
        if entity is None:
            return ""
        context = dict(self._capture_context or {})
        view = self.world_evidence.append_view(
            obs_id=oid,
            revision=self.obs_revision(oid),
            rgb=obs.rgb,
            object_xyz=obs.xyz,
            labels=list(obs.labels),
            description=obs.description,
            entity_id=entity.entity_id,
            place_id=entity.place_id,
            captured_step=self._effective_timestep(),
            camera_pose_world=context.get("camera_pose_world"),
            base_pose_world=context.get("base_pose_world", obs.viewer_xyz),
        )
        if view is not None:
            self.world_evidence.record_event(
                subject_kind="entity",
                subject_id=entity.entity_id,
                predicate="observed",
                polarity="positive",
                source="graph_observation",
                confidence=float(getattr(node, "belief_confidence", 0.5) or 0.5),
                step=self._effective_timestep(),
                view_id=view.view_id,
                place_id=entity.place_id,
                payload={
                    "labels": list(node.labels),
                    "legacy_obs_id": oid,
                    "revision": self.obs_revision(oid),
                },
            )
        return view.view_id if view is not None else ""

    def view_id_for_obs(self, obs_id: int) -> str:
        return self.world_evidence.view_id_for_obs(int(obs_id))

    def bind_episode_context(
        self,
        *,
        question_id: str | int | None,
        session_id: str | int | None,
    ) -> None:
        """Bind IDs before any episode-specific graph/evidence append."""
        self._attempt_ledger_question_id = str(question_id) if question_id is not None else None
        self.world_evidence.set_context(question_id=question_id, session_id=session_id)

    def observation_room(self, obs_id: int) -> tuple[str | None, str]:
        """Return the room persisted on an observation's stable view/place."""
        view = self.world_evidence.view_for_obs(int(obs_id))
        if view is None or not view.place_id:
            return None, "unknown"
        place = self.world_evidence.places.get(view.place_id)
        if place is None or not place.room_id:
            return None, "unknown"
        room = self.world_evidence.rooms.get(place.room_id)
        return place.room_id, room.room_name if room is not None else "unknown"

    def record_agentic_evidence(
        self,
        *,
        stage: str,
        outcome: str,
        obs_id: int,
        phrase: str,
        confidence: float,
        source: str,
        agent_round: int | None = None,
        score: float | None = None,
        threshold: float | None = None,
        supporting_event_ids: tuple[str, ...] = (),
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Persist one agentic evidence stage and return its durable event ID."""
        oid = int(obs_id)
        room_id, room_name = self.observation_room(oid)
        event = self.world_evidence.record_agentic_evidence(
            stage=stage,
            outcome=outcome,
            confidence=float(confidence),
            step=self._effective_timestep(),
            obs_id=oid,
            phrase=phrase,
            source=source,
            view_id=self.view_id_for_obs(oid) or None,
            room_id=room_id,
            room_name=room_name,
            agent_round=agent_round,
            score=score,
            threshold=threshold,
            supporting_event_ids=supporting_event_ids,
            payload=payload,
        )
        return event.event_id if event is not None else ""

    def durable_confirmation_event_ids(self, *, obs_id: int, phrase: str = "") -> tuple[str, ...]:
        return self.world_evidence.durable_confirmation_event_ids(
            obs_id=int(obs_id),
            phrase=phrase,
        )

    def latest_world_view_id(self) -> str:
        if not self.world_evidence.views:
            return ""
        return max(
            self.world_evidence.views.values(),
            key=lambda view: (view.captured_step, view.obs_id, view.revision),
        ).view_id

    def _reindex_world_entities(self) -> None:
        self.world_evidence.reindex_entities(
            self._nodes,
            step=self._effective_timestep(),
        )

    def _bump_obs_revision(self, obs_id: int) -> int:
        oid = int(obs_id)
        nxt = int(self._obs_revisions.get(oid, 0)) + 1
        self._obs_revisions[oid] = nxt
        self._last_obs_content_update_id = oid
        # Stale SigLIP features would disagree with the refreshed RGB candidate.
        self._obs_siglip_features.pop(oid, None)
        return nxt

    def refresh_observation_candidate(
        self,
        obs_id: int,
        rgb: np.ndarray | Image.Image,
        *,
        xyz: np.ndarray | None = None,
        labels: list[str] | None = None,
        description: str | None = None,
        viewer_xyz: np.ndarray | None = None,
    ) -> bool:
        """Update the stored RGB/candidate for an existing graph observation.

        Graph nodes keep a stable ``obs_id`` under spatial merge; revisits must still
        refresh the evidence image (and invalidate caches) so verify/EQA see the
        better view rather than the first frame forever.
        """
        if isinstance(rgb, Image.Image):
            rgb = np.array(rgb)
        rgb_a = np.asarray(rgb)
        oid = int(obs_id)
        viewer_a: np.ndarray | None = None
        if viewer_xyz is not None:
            viewer_a = np.asarray(viewer_xyz, dtype=float).reshape(-1)[:3].copy()
        xyz_a = None
        if xyz is not None:
            xyz_a = np.asarray(xyz, dtype=float).reshape(-1)[:3].copy()
        for o in self._observations:
            if int(o.obs_id) != oid:
                continue
            o.rgb = rgb_a.copy()
            if xyz_a is not None:
                o.xyz = xyz_a
            if labels is not None:
                o.labels = list(labels)
            if description:
                o.description = description
            if viewer_a is not None:
                o.viewer_xyz = viewer_a
            self._bump_obs_revision(oid)
            self._record_world_view_for_obs(oid)
            return True
        return False

    def add_observation(
        self,
        rgb: np.ndarray | Image.Image,
        xyz: np.ndarray,
        labels: list[str],
        description: str | None = None,
        *,
        viewer_xyz: np.ndarray | None = None,
        bbox_xyxy: tuple[int, int, int, int] | None = None,
        extent_half: np.ndarray | None = None,
        identity_key: str | None = None,
        countable_instance: bool = False,
    ) -> int:
        """
        Add one observation to the graph: create a node and update edges.

        Args:
            rgb: RGB image (H, W, 3) or PIL Image
            xyz: (3,) world position for this observation (e.g. camera or centroid)
            labels: list of object/region labels (e.g. from a VLM)
            description: optional text description of the scene (e.g. from VLM)
            viewer_xyz: optional (3,) robot base or head-camera position in world frame when captured
            bbox_xyxy: optional (x0, y0, x1, y1) crop in ``rgb`` for this object (instance mask bbox)
            identity_key: optional stable identity for this detected instance
            countable_instance: whether this observation is a FIND candidate for count questions

        Returns:
            obs_id: 1-based observation id (used as image id in EQA).
        """
        if isinstance(rgb, Image.Image):
            rgb = np.array(rgb)
        step = self._effective_timestep()
        xyz_a = np.asarray(xyz, dtype=float).reshape(-1)[:3]
        viewer_a: np.ndarray | None = None
        if viewer_xyz is not None:
            viewer_a = np.asarray(viewer_xyz, dtype=float).reshape(-1)[:3].copy()
        labels_norm = [str(l).strip() for l in labels if str(l).strip()]
        if not labels_norm:
            labels_norm = ["object"]
        primary = labels_norm[0].lower()
        identity_key_norm = str(identity_key).strip() if identity_key is not None else ""
        identity_key_norm = identity_key_norm or None

        bbox_i: tuple[int, int, int, int] | None = None
        if bbox_xyxy is not None:
            b = tuple(int(x) for x in bbox_xyxy)
            if len(b) == 4:
                bbox_i = (b[0], b[1], b[2], b[3])

        if self.spatial_merge_m > 0 or identity_key_norm is not None:
            from emet.memory.graph_eqa.graph_stats import labels_compatible_for_dedup

            for idx, existing in enumerate(self._nodes):
                if existing.is_viewpoint or existing.is_frontier or is_ground_truth_node(existing):
                    continue
                existing_key = str(existing.identity_key).strip() if existing.identity_key else None
                same_identity = identity_key_norm is not None and existing_key == identity_key_norm
                el = [str(x).strip() for x in existing.labels if str(x).strip()]
                if not same_identity and (not el or not labels_compatible_for_dedup(primary, el[0])):
                    continue
                if (
                    identity_key_norm is not None
                    and existing.countable_instance
                    and existing_key is not None
                    and not same_identity
                ):
                    continue
                ex = np.asarray(existing.xyz, dtype=float).reshape(-1)[:3]
                spatial_match = (
                    self.spatial_merge_m > 0
                    and float(np.linalg.norm(ex[:2] - xyz_a[:2])) <= self.spatial_merge_m
                )
                if not same_identity and not spatial_match:
                    continue
                sc = int(existing.support_count) + 1
                new_xyz, covariance, history, changes, belief_confidence = self._position_update(
                    existing, xyz_a, step=step
                )
                merged_labels = sorted({*(str(x).strip() for x in existing.labels if str(x).strip()), *labels_norm})
                new_desc = description if description else existing.description
                merged_bbox = bbox_i if bbox_i is not None else existing.bbox_xyxy
                self._nodes[idx] = replace(
                    existing,
                    xyz=new_xyz,
                    labels=merged_labels,
                    last_seen=step,
                    support_count=sc,
                    description=new_desc,
                    bbox_xyxy=merged_bbox,
                    position_covariance=covariance,
                    position_history=history,
                    change_events=changes,
                    belief_confidence=belief_confidence,
                    identity_key=(
                        identity_key_norm
                        if identity_key_norm is not None and countable_instance
                        else existing.identity_key or identity_key_norm
                    ),
                    countable_instance=bool(existing.countable_instance or countable_instance),
                    close_look_label=existing.close_look_label,
                )
                # Keep the graph node's candidate image in sync with this revisit.
                self.refresh_observation_candidate(
                    int(existing.obs_id),
                    rgb,
                    xyz=new_xyz,
                    labels=merged_labels,
                    description=new_desc if new_desc else None,
                    viewer_xyz=viewer_a,
                )
                if viewer_a is not None:
                    self._ensure_viewpoint_node(int(existing.obs_id), viewer_a)
                self._update_edges()
                return int(existing.obs_id)

        obs_id = self._next_obs_id
        self._next_obs_id += 1
        node_id = len(self._nodes) + 1
        ext = None
        if extent_half is not None:
            ext = np.asarray(extent_half, dtype=float).reshape(-1)[:3].copy()
        node = GraphNode(
            node_id=node_id,
            labels=labels_norm,
            xyz=xyz_a.copy(),
            obs_id=obs_id,
            description=description,
            last_seen=step,
            support_count=1,
            extent_half=ext,
            bbox_xyxy=bbox_i,
            belief_confidence=0.55,
            position_covariance=np.zeros((3, 3), dtype=float),
            position_history=[
                {
                    "step": int(step),
                    "xyz": xyz_a.tolist(),
                    "confidence": 0.55,
                }
            ],
            identity_key=(
                identity_key_norm
                or (
                    description[len(GT_BODY_DESC_PREFIX) :]
                    if isinstance(description, str) and description.startswith(GT_BODY_DESC_PREFIX)
                    else f"{re.sub(r'[^a-z0-9]+', '-', primary).strip('-')}:{obs_id}"
                )
            ),
            countable_instance=bool(countable_instance),
        )
        self._nodes.append(node)
        self._observations.append(
            GraphObservation(
                obs_id=obs_id,
                rgb=rgb,
                xyz=xyz_a.copy(),
                labels=list(labels_norm),
                description=description,
                viewer_xyz=viewer_a,
            )
        )
        self._obs_revisions[int(obs_id)] = 1
        self._last_obs_content_update_id = int(obs_id)
        self._record_world_view_for_obs(obs_id)
        if viewer_a is not None:
            self._ensure_viewpoint_node(obs_id, viewer_a)
        self._update_edges()
        return obs_id

    def merge_object_detection(
        self,
        rgb: np.ndarray | Image.Image,
        candidate: Any,
        *,
        merge_into_node_id: int | None,
        viewer_xyz: np.ndarray | None = None,
    ) -> int:
        """
        Add or merge an instance detection (GraphObjectFusion path).

        ``candidate`` is a :class:`~emet.memory.graph_eqa.graph_object_fusion.fusion.GraphDetectionCandidate`
        or any object with ``label``, ``xyz``, optional ``bbox_xyxy``, ``bounds_3d``, ``embedding``.
        """
        if isinstance(rgb, Image.Image):
            rgb = np.array(rgb)
        label = str(getattr(candidate, "label", "object"))
        xyz_a = np.asarray(candidate.xyz, dtype=float).reshape(-1)[:3]
        bbox_xyxy = getattr(candidate, "bbox_xyxy", None)
        bounds_3d = getattr(candidate, "bounds_3d", None)
        embedding = getattr(candidate, "embedding", None)
        identity_key = getattr(candidate, "identity_key", None)
        identity_key_norm = str(identity_key).strip() if identity_key is not None else ""
        identity_key_norm = identity_key_norm or None
        semantic_only = bool(getattr(candidate, "semantic_only", False))
        countable_instance = bool(
            (getattr(candidate, "countable_instance", False) or identity_key_norm) and not semantic_only
        )
        if embedding is not None:
            embedding = np.asarray(embedding, dtype=np.float32).reshape(-1).copy()

        step = self._effective_timestep()
        viewer_a: np.ndarray | None = None
        if viewer_xyz is not None:
            viewer_a = np.asarray(viewer_xyz, dtype=float).reshape(-1)[:3].copy()

        bbox_i: tuple[int, int, int, int] | None = None
        if bbox_xyxy is not None:
            b = tuple(int(x) for x in bbox_xyxy)
            if len(b) == 4:
                bbox_i = (b[0], b[1], b[2], b[3])

        if merge_into_node_id is not None:
            for idx, existing in enumerate(self._nodes):
                if int(existing.node_id) != int(merge_into_node_id):
                    continue
                if existing.is_viewpoint:
                    break
                if (
                    identity_key_norm is not None
                    and existing.countable_instance
                    and existing.identity_key
                    and str(existing.identity_key) != identity_key_norm
                ):
                    break
                if semantic_only and existing.countable_instance:
                    break
                if existing.countable_instance or countable_instance:
                    from emet.memory.graph_eqa.graph_stats import labels_compatible_for_dedup

                    primary = str(existing.labels[0]).strip() if existing.labels else label
                    if not labels_compatible_for_dedup(label, primary):
                        break
                    merged_labels = sorted(
                        {
                            *[str(x).strip() for x in existing.labels if str(x).strip() and labels_compatible_for_dedup(str(x).strip(), primary)],
                            label,
                        }
                    )
                else:
                    merged_labels = sorted({*(str(x).strip() for x in existing.labels if str(x).strip()), label})
                sc = int(existing.support_count) + 1
                new_xyz, covariance, history, changes, belief_confidence = self._position_update(
                    existing, xyz_a, step=step
                )
                new_emb = embedding if embedding is not None else existing.embedding
                if embedding is not None and existing.embedding is not None:
                    a = float(getattr(candidate, "embedding_blend_alpha", 0.35))
                    new_emb = (1.0 - a) * np.asarray(existing.embedding, dtype=np.float32) + a * embedding
                new_bounds = bounds_3d if bounds_3d is not None else existing.bounds_3d
                if bounds_3d is not None and existing.bounds_3d is not None:
                    mn = np.minimum(
                        np.asarray(existing.bounds_3d["min"], dtype=np.float64),
                        np.asarray(bounds_3d["min"], dtype=np.float64),
                    )
                    mx = np.maximum(
                        np.asarray(existing.bounds_3d["max"], dtype=np.float64),
                        np.asarray(bounds_3d["max"], dtype=np.float64),
                    )
                    c = 0.5 * (mn + mx)
                    new_bounds = {
                        "min": mn.tolist(),
                        "max": mx.tolist(),
                        "center": c.tolist(),
                        "size": (mx - mn).tolist(),
                    }
                self._nodes[idx] = replace(
                    existing,
                    xyz=new_xyz,
                    labels=merged_labels,
                    last_seen=step,
                    support_count=sc,
                    bbox_xyxy=bbox_i if bbox_i is not None else existing.bbox_xyxy,
                    embedding=new_emb,
                    bounds_3d=new_bounds,
                    position_covariance=covariance,
                    position_history=history,
                    change_events=changes,
                    belief_confidence=belief_confidence,
                    identity_key=identity_key_norm or existing.identity_key,
                    countable_instance=bool(existing.countable_instance or countable_instance),
                    close_look_label=existing.close_look_label,
                )
                self.refresh_observation_candidate(
                    int(existing.obs_id),
                    rgb,
                    xyz=new_xyz,
                    labels=merged_labels,
                    viewer_xyz=viewer_a,
                )
                if viewer_a is not None:
                    self._ensure_viewpoint_node(int(existing.obs_id), viewer_a)
                self._update_edges()
                return int(existing.obs_id)

        obs_id = self.add_observation(
            rgb,
            xyz_a,
            [label],
            viewer_xyz=viewer_a,
            bbox_xyxy=bbox_i,
            identity_key=identity_key_norm,
            countable_instance=countable_instance,
        )
        for idx, n in enumerate(self._nodes):
            if int(n.obs_id) == int(obs_id) and not n.is_viewpoint:
                self._nodes[idx] = replace(
                    n,
                    embedding=embedding,
                    bounds_3d=bounds_3d,
                )
                break
        return obs_id

    def absorb_object_node(self, src_node_id: int, dst_node_id: int) -> bool:
        """Fold ``src`` object node into ``dst`` and remove ``src`` from the graph."""
        if int(src_node_id) == int(dst_node_id):
            return False
        src = dst = None
        for n in self._nodes:
            if int(n.node_id) == int(src_node_id):
                src = n
            elif int(n.node_id) == int(dst_node_id):
                dst = n
        if src is None or dst is None:
            return False
        if src.is_viewpoint or dst.is_viewpoint or src.is_frontier or dst.is_frontier:
            return False
        if (
            src.countable_instance
            and dst.countable_instance
            and src.identity_key
            and dst.identity_key
            and str(src.identity_key) != str(dst.identity_key)
        ):
            return False

        sc_src = int(src.support_count)
        sc_dst = int(dst.support_count)
        total = sc_src + sc_dst
        new_xyz, covariance, history, changes, belief_confidence = self._position_update(
            dst,
            np.asarray(src.xyz, dtype=float),
            step=max(int(dst.last_seen), int(src.last_seen)),
        )

        merged_labels = sorted(
            {
                *(str(x).strip() for x in dst.labels if str(x).strip()),
                *(str(x).strip() for x in src.labels if str(x).strip()),
            }
        )
        if not merged_labels:
            merged_labels = ["object"]

        new_emb = dst.embedding
        if src.embedding is not None and dst.embedding is not None:
            a = 0.35
            new_emb = (1.0 - a) * np.asarray(dst.embedding, dtype=np.float32) + a * np.asarray(
                src.embedding, dtype=np.float32
            )
        elif src.embedding is not None:
            new_emb = np.asarray(src.embedding, dtype=np.float32).copy()

        new_bounds = dst.bounds_3d
        if src.bounds_3d is not None and dst.bounds_3d is not None:
            mn = np.minimum(
                np.asarray(dst.bounds_3d["min"], dtype=np.float64),
                np.asarray(src.bounds_3d["min"], dtype=np.float64),
            )
            mx = np.maximum(
                np.asarray(dst.bounds_3d["max"], dtype=np.float64),
                np.asarray(src.bounds_3d["max"], dtype=np.float64),
            )
            c = 0.5 * (mn + mx)
            new_bounds = {
                "min": mn.tolist(),
                "max": mx.tolist(),
                "center": c.tolist(),
                "size": (mx - mn).tolist(),
            }
        elif src.bounds_3d is not None:
            new_bounds = src.bounds_3d

        dst_idx = next(i for i, n in enumerate(self._nodes) if int(n.node_id) == int(dst_node_id))
        self._nodes[dst_idx] = replace(
            dst,
            xyz=new_xyz,
            labels=merged_labels,
            support_count=total,
            embedding=new_emb,
            bounds_3d=new_bounds,
            bbox_xyxy=dst.bbox_xyxy or src.bbox_xyxy,
            last_seen=max(int(dst.last_seen), int(src.last_seen)),
            position_covariance=covariance,
            position_history=history,
            change_events=changes,
            belief_confidence=belief_confidence,
            identity_key=dst.identity_key or src.identity_key,
            countable_instance=bool(dst.countable_instance or src.countable_instance),
            close_look_label=dst.close_look_label or src.close_look_label,
        )
        for o in self._observations:
            if int(o.obs_id) == int(dst.obs_id):
                o.xyz = new_xyz.copy()
                o.labels = list(merged_labels)
                break

        src_obs_id = int(src.obs_id)
        self.world_evidence.absorb_entity(
            src_node_id=int(src_node_id),
            dst_node_id=int(dst_node_id),
        )
        self._nodes = [n for n in self._nodes if int(n.node_id) != int(src_node_id)]
        self._observations = [o for o in self._observations if int(o.obs_id) != src_obs_id]
        for i, n in enumerate(self._nodes, start=1):
            self._nodes[i - 1] = replace(n, node_id=i)
        self.world_evidence.reindex_entities(
            self._nodes,
            step=self._effective_timestep(),
        )
        self._rebuild_viewpoint_index()
        self._update_edges()
        return True

    def upsert_ground_truth_observation(
        self,
        body_key: str,
        rgb: np.ndarray | Image.Image,
        xyz: np.ndarray,
        labels: list[str],
        *,
        extent_half: np.ndarray | None = None,
    ) -> int:
        """
        Insert or refresh one sim GT node keyed by MuJoCo ``body_key``.

        GT nodes use ``description=ground_truth:{body_key}`` so repeated updates
        deduplicate instead of adding duplicate detections over time.
        """
        if isinstance(rgb, Image.Image):
            rgb = np.array(rgb)
        step = self._effective_timestep()
        xyz_a = np.asarray(xyz, dtype=float).reshape(-1)[:3]
        labels_norm = [str(l).strip() for l in labels if str(l).strip()]
        if not labels_norm:
            labels_norm = ["object"]
        desc = f"{GT_BODY_DESC_PREFIX}{body_key}"
        ext = None
        if extent_half is not None:
            ext = np.asarray(extent_half, dtype=float).reshape(-1)[:3].copy()

        for idx, existing in enumerate(self._nodes):
            if existing.description != desc:
                continue
            same_pose = np.allclose(existing.xyz, xyz_a, atol=1e-4, rtol=0.0)
            same_labels = list(existing.labels) == labels_norm
            same_ext = ext is None or (
                existing.extent_half is not None and np.allclose(existing.extent_half, ext, atol=1e-4, rtol=0.0)
            )
            if same_pose and same_labels and same_ext:
                self._nodes[idx] = replace(existing, last_seen=step)
                self._update_edges()
                return int(existing.obs_id)
            sc = int(existing.support_count) + 1
            self._nodes[idx] = replace(
                existing,
                xyz=xyz_a.copy(),
                labels=labels_norm,
                last_seen=step,
                support_count=sc,
                extent_half=ext if ext is not None else existing.extent_half,
            )
            for o in self._observations:
                if o.obs_id == existing.obs_id:
                    o.xyz = xyz_a.copy()
                    o.labels = list(labels_norm)
                    o.description = desc
                    break
            self._update_edges()
            return int(existing.obs_id)

        return self.add_observation(
            rgb,
            xyz_a,
            labels_norm,
            description=desc,
            extent_half=ext,
        )

    def attach_detection_to_ground_truth_node(
        self,
        body_key: str,
        rgb: np.ndarray | Image.Image,
        *,
        detection_label: str | None = None,
    ) -> bool:
        """Refresh a GT node's stored RGB when an instance detector sees it nearby."""
        if isinstance(rgb, Image.Image):
            rgb = np.array(rgb)
        rgb_a = np.asarray(rgb, dtype=np.uint8)
        desc = f"{GT_BODY_DESC_PREFIX}{body_key}"
        det_tag = f"|det:{detection_label.strip()}" if detection_label and detection_label.strip() else ""
        step = self._effective_timestep()
        for idx, existing in enumerate(self._nodes):
            if existing.description is None or not str(existing.description).startswith(desc):
                continue
            new_desc = f"{desc}{det_tag}" if det_tag else desc
            self._nodes[idx] = replace(existing, last_seen=step, description=new_desc)
            self.refresh_observation_candidate(
                int(existing.obs_id),
                rgb_a,
                description=new_desc,
            )
            self._update_edges()
            return True
        return False

    def record_navigation_sample(
        self,
        rgb: np.ndarray | Image.Image,
        xyz: np.ndarray,
        *,
        base_xyz: np.ndarray | None = None,
        link_viewpoint_node: bool = True,
    ) -> None:
        """
        Record a navigation-time viewpoint without adding a scene-graph node.

        Used when perception returns no usable object labels (e.g. generic
        ``object`` fallback) so the trajectory is still available for debugging
        and optional EQA image context.
        """
        if not self._record_navigation:
            return
        if isinstance(rgb, Image.Image):
            rgb = np.array(rgb)
        rgb = np.asarray(rgb)
        xyz = np.asarray(xyz, dtype=float).reshape(-1)[:3]
        bx = None
        if base_xyz is not None:
            bx = np.asarray(base_xyz, dtype=float).reshape(-1)[:3]
        self._nav_samples.append(GraphNavigationSample(rgb=rgb, xyz=xyz, base_xyz=bx))
        if len(self._nav_samples) > self._nav_max:
            drop = len(self._nav_samples) - self._nav_max
            self._nav_samples = self._nav_samples[drop:]
        if bx is not None and link_viewpoint_node:
            nav_obs_id = self._next_obs_id
            self._next_obs_id += 1
            self._ensure_viewpoint_node(nav_obs_id, bx, labels=["viewpoint", "nav"])

    def get_navigation_samples(self) -> list[GraphNavigationSample]:
        return list(self._nav_samples)

    def _frontier_desc(self, cluster_id: str) -> str:
        from emet.memory.graph_eqa.frontier_nodes import FRONTIER_DESC_PREFIX

        return f"{FRONTIER_DESC_PREFIX}{cluster_id}"

    def _find_frontier_node(self, cluster_id: str) -> GraphNode | None:
        desc = self._frontier_desc(cluster_id)
        for n in self._nodes:
            if n.is_frontier and n.description == desc:
                return n
        return None

    def _remove_frontier_nodes(self, keep_cluster_ids: set[str]) -> None:
        from emet.memory.graph_eqa.frontier_nodes import FRONTIER_DESC_PREFIX

        drop_obs: set[int] = set()
        drop_nodes: set[int] = set()
        for n in self._nodes:
            if not n.is_frontier:
                continue
            desc = n.description or ""
            cid = desc[len(FRONTIER_DESC_PREFIX) :] if desc.startswith(FRONTIER_DESC_PREFIX) else ""
            if cid not in keep_cluster_ids:
                drop_obs.add(int(n.obs_id))
                drop_nodes.add(int(n.node_id))
        if not drop_nodes:
            return
        self._nodes = [n for n in self._nodes if int(n.node_id) not in drop_nodes]
        self._observations = [o for o in self._observations if int(o.obs_id) not in drop_obs]
        for i, n in enumerate(self._nodes, start=1):
            self._nodes[i - 1] = replace(n, node_id=i)
        self._reindex_world_entities()
        self._rebuild_viewpoint_index()
        self._update_edges()

    def _frontier_support_attachment(
        self,
        *,
        grid_ij: tuple[int, int],
        xyz: np.ndarray,
        unexplored: np.ndarray,
        explored: np.ndarray,
        reachable: np.ndarray | None,
        step: int,
    ) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
        """Return real nearby RGB or an explicitly namespaced map crop."""
        nearest_view = None
        nearest_distance = float("inf")
        if self.world_evidence.enabled:
            for view in self.world_evidence.views.values():
                if view.rgb is None:
                    continue
                anchor = view.base_pose_world or view.object_xyz
                distance = float(np.linalg.norm(np.asarray(anchor[:2], dtype=float) - np.asarray(xyz[:2], dtype=float)))
                if distance < nearest_distance:
                    nearest_view, nearest_distance = view, distance
        if nearest_view is not None and nearest_distance <= 5.0:
            return (
                np.asarray(nearest_view.rgb, dtype=np.uint8).copy(),
                (nearest_view.view_id,),
                (f"view:{nearest_view.view_id}",),
            )

        row, col = int(grid_ij[0]), int(grid_ij[1])
        radius = 12
        r0, r1 = max(0, row - radius), min(unexplored.shape[0], row + radius + 1)
        c0, c1 = max(0, col - radius), min(unexplored.shape[1], col + radius + 1)
        crop = np.full((r1 - r0, c1 - c0, 3), 32, dtype=np.uint8)
        crop[np.asarray(explored[r0:r1, c0:c1], dtype=bool)] = (90, 90, 90)
        if reachable is not None:
            crop[np.asarray(reachable[r0:r1, c0:c1], dtype=bool)] = (70, 100, 160)
        crop[np.asarray(unexplored[r0:r1, c0:c1], dtype=bool)] = (40, 210, 100)
        if crop.size == 0:
            crop = np.full((8, 8, 3), (40, 210, 100), dtype=np.uint8)
        scale = max(1, int(np.ceil(64 / max(crop.shape[:2]))))
        crop = np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1)[:96, :96]
        attachment_id = f"map:grid-{row}-{col}:step-{int(step)}"
        return crop, (), (attachment_id,)

    def sync_frontier_nodes(
        self,
        voxel_map: Any,
        planner: Any,
        xyt: Any,
        *,
        question_keywords: list[str] | None = None,
    ) -> int:
        """Upsert/remove frontier graph nodes from the voxel unexplored-frontier mask."""
        if not self.frontier_nodes_enabled:
            return sum(1 for n in self._nodes if n.is_frontier)

        from emet.memory.graph_eqa.frontier_nodes import (
            _as_bool_numpy,
            cluster_frontier_mask,
            frontier_components,
            hint_labels_near_grid,
            keyword_overlap_score,
        )

        try:
            outside = voxel_map.get_outside_frontier(xyt, planner)
            _, explored = voxel_map.get_2d_map()
            reachable = None
            if hasattr(voxel_map, "get_reachable_map"):
                reachable = _as_bool_numpy(voxel_map.get_reachable_map(xyt, planner))
        except Exception as e:
            _logger.warning(f"Frontier upsert: could not read map frontiers ({e})")
            return sum(1 for n in self._nodes if n.is_frontier)

        unexplored = _as_bool_numpy(outside) & ~_as_bool_numpy(explored)
        components = (
            frontier_components(
                unexplored,
                min_cells=self._frontier_min_cluster_cells,
                reachable=reachable,
            )
            if self.world_evidence.enabled
            else None
        )
        clusters = (
            [(component.transient_id, component.goal_ij, component.cell_count) for component in components]
            if components is not None
            else cluster_frontier_mask(
                unexplored,
                min_cells=self._frontier_min_cluster_cells,
                reachable=reachable,
            )
        )
        component_by_id = {component.transient_id: component for component in (components or ())}
        image_descriptions = getattr(voxel_map, "image_descriptions", None) or []
        keywords = list(question_keywords or self._relevant_objects or self._enrich_object_hints or [])

        scored: list[tuple[float, str, tuple[int, int], int]] = []
        for cluster_id, grid_ij, cell_count in clusters:
            hints = hint_labels_near_grid(grid_ij, image_descriptions)
            kw_score = keyword_overlap_score(hints, keywords) if keywords else 0.0
            scored.append((kw_score, cluster_id, grid_ij, cell_count))
        scored.sort(key=lambda x: (-x[0], -x[3]))

        step = self._effective_timestep()
        prepared: list[dict[str, Any]] = []
        for kw_score, transient_id, grid_ij, cell_count in scored[: self._frontier_max_nodes]:
            gi, gj = grid_ij
            try:
                xy = voxel_map.grid_coords_to_xy(np.array([gi, gj], dtype=float))
            except Exception:
                continue
            xyz = np.array([float(xy[0]), float(xy[1]), 0.0], dtype=float)
            hints = hint_labels_near_grid(grid_ij, image_descriptions)
            support_rgb, support_view_ids, attachment_ids = self._frontier_support_attachment(
                grid_ij=grid_ij,
                xyz=xyz,
                unexplored=unexplored,
                explored=_as_bool_numpy(explored),
                reachable=reachable,
                step=step,
            )
            component = component_by_id.get(transient_id)
            prepared.append(
                {
                    "transient_id": transient_id,
                    "grid_ij": grid_ij,
                    "cell_count": int(cell_count),
                    "cells": (component.cells if component is not None else ((int(grid_ij[0]), int(grid_ij[1])),)),
                    "centroid_xyz": xyz,
                    "keyword_score": float(kw_score),
                    "hints": hints,
                    "support_rgb": support_rgb,
                    "support_view_ids": support_view_ids,
                    "attachment_ids": attachment_ids,
                }
            )

        tracks = self.world_evidence.update_frontier_tracks(prepared, step=step) if self.world_evidence.enabled else []
        keep_ids: set[str] = set()
        for index, item in enumerate(prepared):
            track = tracks[index] if index < len(tracks) else None
            cluster_id = track.frontier_id if track is not None else str(item["transient_id"])
            keep_ids.add(cluster_id)
            xyz = np.asarray(item["centroid_xyz"], dtype=float)
            hints = list(item["hints"])
            cell_count = int(item["cell_count"])
            kw_score = float(item["keyword_score"])
            labels = ["frontier"] + hints[:3]
            desc = self._frontier_desc(cluster_id)
            obs_desc = (
                f"frontier_id={cluster_id}; attachments="
                + ",".join(item["attachment_ids"])
                + ("; unexplored areas; " + ", ".join(hints) if hints else "; unexplored space")
                if hints
                else f"frontier_id={cluster_id}; attachments=" + ",".join(item["attachment_ids"]) + "; unexplored space"
            )

            existing = self._find_frontier_node(cluster_id)
            if existing is not None:
                idx = next(i for i, n in enumerate(self._nodes) if n.node_id == existing.node_id)
                self._nodes[idx] = replace(
                    existing,
                    xyz=xyz,
                    labels=labels,
                    last_seen=step,
                    description=desc,
                    frontier_cell_count=int(cell_count),
                    frontier_keyword_score=float(kw_score),
                )
                for o in self._observations:
                    if int(o.obs_id) == int(existing.obs_id):
                        o.xyz = xyz.copy()
                        o.labels = list(labels)
                        o.description = obs_desc
                        o.rgb = np.asarray(item["support_rgb"], dtype=np.uint8).copy()
                        break
                if track is not None:
                    self.world_evidence.set_frontier_obs(track.frontier_id, int(existing.obs_id))
            else:
                obs_id = self._next_obs_id
                self._next_obs_id += 1
                node_id = len(self._nodes) + 1
                self._nodes.append(
                    GraphNode(
                        node_id=node_id,
                        labels=labels,
                        xyz=xyz.copy(),
                        obs_id=obs_id,
                        description=desc,
                        last_seen=step,
                        is_frontier=True,
                        frontier_cell_count=int(cell_count),
                        frontier_keyword_score=float(kw_score),
                    )
                )
                self._observations.append(
                    GraphObservation(
                        obs_id=obs_id,
                        rgb=np.asarray(item["support_rgb"], dtype=np.uint8).copy(),
                        xyz=xyz.copy(),
                        labels=list(labels),
                        description=obs_desc,
                    )
                )
                if track is not None:
                    self.world_evidence.set_frontier_obs(track.frontier_id, obs_id)

        self._remove_frontier_nodes(keep_ids)
        self._update_edges()
        return sum(1 for n in self._nodes if n.is_frontier)

    def _rebuild_viewpoint_index(self) -> None:
        self._viewpoint_by_obs_id = {int(n.obs_id): int(n.node_id) for n in self._nodes if n.is_viewpoint}

    def _find_nearby_viewpoint_node(self, viewer_xyz: np.ndarray) -> GraphNode | None:
        """Nearest viewpoint within ``viewpoint_merge_m`` (stationary-stream dedup)."""
        radius = float(self.viewpoint_merge_m)
        if radius <= 0.0:
            return None
        vxyz = np.asarray(viewer_xyz, dtype=float).reshape(-1)[:3]
        best: GraphNode | None = None
        best_d = float("inf")
        for n in self._nodes:
            if not n.is_viewpoint:
                continue
            d = float(np.linalg.norm(np.asarray(n.xyz, dtype=float).reshape(3) - vxyz))
            if d <= radius and d < best_d:
                best_d = d
                best = n
        return best

    def _ensure_viewpoint_node(
        self,
        obs_id: int,
        viewer_xyz: np.ndarray,
        *,
        labels: list[str] | None = None,
    ) -> int:
        """Create or refresh a graph node at the observation vantage (``seen_from`` target)."""
        vxyz = np.asarray(viewer_xyz, dtype=float).reshape(-1)[:3].copy()
        step = self._effective_timestep()
        vp_labels = labels or [f"view img {int(obs_id)}"]
        existing_id = self._viewpoint_by_obs_id.get(int(obs_id))
        if existing_id is not None:
            for idx, n in enumerate(self._nodes):
                if int(n.node_id) == int(existing_id):
                    self._nodes[idx] = replace(
                        n,
                        xyz=vxyz,
                        labels=list(vp_labels),
                        last_seen=step,
                    )
                    return int(existing_id)
        nearby = self._find_nearby_viewpoint_node(vxyz)
        if nearby is not None:
            for idx, n in enumerate(self._nodes):
                if int(n.node_id) == int(nearby.node_id):
                    self._nodes[idx] = replace(
                        n,
                        xyz=vxyz,
                        labels=list(vp_labels),
                        last_seen=step,
                    )
                    self._viewpoint_by_obs_id[int(obs_id)] = int(nearby.node_id)
                    return int(nearby.node_id)
        node_id = len(self._nodes) + 1
        self._nodes.append(
            GraphNode(
                node_id=node_id,
                labels=list(vp_labels),
                xyz=vxyz,
                obs_id=int(obs_id),
                last_seen=step,
                is_viewpoint=True,
            )
        )
        self._viewpoint_by_obs_id[int(obs_id)] = int(node_id)
        return int(node_id)

    def _ensure_seen_from_edge(self, node_id: int, obs_id: int) -> None:
        """Link object *node_id* to the viewpoint graph node for observation *obs_id*."""
        vp_id = self._viewpoint_by_obs_id.get(int(obs_id))
        if vp_id is None:
            return
        edge = (int(node_id), int(vp_id), "seen_from")
        if edge not in self._edges:
            self._edges.append(edge)

    def _observation_by_id(self, obs_id: int) -> GraphObservation | None:
        for o in self._observations:
            if int(o.obs_id) == int(obs_id):
                return o
        return None

    def _update_edges(self) -> None:
        """Compute spatial/context relations and timestamp their uncertain evidence."""
        self._edges.clear()
        objects = [n for n in self._nodes if not n.is_viewpoint and not n.is_frontier]
        viewpoints = [n for n in self._nodes if n.is_viewpoint]
        for i, na in enumerate(objects):
            if _on_floor(na.xyz):
                self._edges.append((na.node_id, -1, "on"))  # -1 = floor
            for j, nb in enumerate(objects):
                if i >= j:
                    continue
                if _near(na.xyz, nb.xyz, self.max_near_distance):
                    if (nb.node_id, na.node_id, "near") not in self._edges:
                        self._edges.append((na.node_id, nb.node_id, "near"))
                if _on(na.xyz, nb.xyz):
                    self._edges.append((na.node_id, nb.node_id, "on"))
                    self._edges.append((nb.node_id, na.node_id, "supports"))
                elif _on(nb.xyz, na.xyz):
                    self._edges.append((nb.node_id, na.node_id, "on"))
                    self._edges.append((na.node_id, nb.node_id, "supports"))
                if _node_is_room(na) and _inside_bounds(nb.xyz, na.bounds_3d):
                    self._edges.append((na.node_id, nb.node_id, "contains"))
                elif _node_is_room(nb) and _inside_bounds(na.xyz, nb.bounds_3d):
                    self._edges.append((nb.node_id, na.node_id, "contains"))
        for node in objects:
            self._ensure_seen_from_edge(node.node_id, int(node.obs_id))
            if viewpoints:
                nearest = min(
                    viewpoints,
                    key=lambda view: float(
                        np.linalg.norm(np.asarray(view.xyz, dtype=float)[:2] - np.asarray(node.xyz, dtype=float)[:2])
                    ),
                )
                distance = float(
                    np.linalg.norm(np.asarray(nearest.xyz, dtype=float)[:2] - np.asarray(node.xyz, dtype=float)[:2])
                )
                failure_risk = float(node.nav_failures) / max(1, int(node.nav_attempts))
                if distance <= max(2.0, self.max_near_distance) and failure_risk < 0.8:
                    self._edges.append((node.node_id, nearest.node_id, "accessible_from"))

        step = self._effective_timestep()
        prior = self._relation_beliefs
        current: dict[tuple[int, int, str], RelationBelief] = {}
        confidence_by_relation = {
            "seen_from": 0.95,
            "contains": 0.85,
            "supports": 0.80,
            "on": 0.75,
            "near": 0.65,
            "accessible_from": 0.60,
        }
        for edge in self._edges:
            old = prior.get(edge)
            current[edge] = RelationBelief(
                source_id=edge[0],
                target_id=edge[1],
                relation=edge[2],
                confidence=max(
                    confidence_by_relation.get(edge[2], 0.5),
                    float(old.confidence) if old is not None else 0.0,
                ),
                last_evidence_step=step,
                contradiction_count=old.contradiction_count if old is not None else 0,
            )
        for edge, old in prior.items():
            if edge in current:
                continue
            decayed = float(old.confidence) * 0.5
            if decayed >= 0.1:
                current[edge] = replace(
                    old,
                    confidence=decayed,
                    contradiction_count=int(old.contradiction_count) + 1,
                )
        self._relation_beliefs = current
        self.refresh_room_clusters()

    def _room_link_radius_m(self) -> float:
        env = os.environ.get("EMET_EQA_ROOM_LINK_RADIUS_M", "").strip()
        if env:
            try:
                return float(env)
            except ValueError:
                pass
        try:
            return float(self._eqa_cfg_value("room_link_radius_m", 2.0))
        except Exception:
            return 2.0

    def _room_assign_max_m(self) -> float:
        env = os.environ.get("EMET_EQA_ROOM_ASSIGN_MAX_M", "").strip()
        if env:
            try:
                return float(env)
            except ValueError:
                pass
        try:
            return float(self._eqa_cfg_value("room_assign_max_m", 3.0))
        except Exception:
            return 3.0

    def set_room_connectivity_checker(
        self,
        checker: Callable[[tuple[float, float], tuple[float, float]], bool] | None,
    ) -> None:
        """Set a free-space line/connectivity gate used by room clustering."""
        self._room_connectivity_fn = checker

    def refresh_room_clusters(self) -> list[Any]:
        """Recompute near+planar connected components over object nodes."""
        from emet.memory.graph_eqa.room_clusters import cluster_object_nodes

        clusters = cluster_object_nodes(
            self._nodes,
            self._edges,
            link_radius_m=self._room_link_radius_m(),
            connectivity_fn=self._room_connectivity_fn,
        )
        if self.world_evidence.enabled:
            self._reindex_world_entities()
            node_to_place: dict[int, str] = {}
            for node in self._nodes:
                entity = self.world_evidence.entity_for_node(int(node.node_id))
                if entity is not None:
                    node_to_place[int(node.node_id)] = entity.place_id
            hypotheses = self.world_evidence.update_room_hypotheses(
                list(clusters),
                node_to_place=node_to_place,
                step=self._effective_timestep(),
            )
            clusters = [
                replace(
                    cluster,
                    room_id=hypothesis.room_id,
                    room_name=(hypothesis.room_name if hypothesis.room_name != "unknown" else cluster.room_name),
                )
                for cluster, hypothesis in zip(clusters, hypotheses, strict=True)
            ]
        self._room_clusters = list(clusters)
        self.last_room_clusters = list(clusters)
        return self._room_clusters

    def graph_room_at_robot(self, robot_xy: Any) -> str:
        """Nearest graph room-cluster label at ``robot_xy``, or ``unknown``."""
        from emet.memory.graph_eqa.room_clusters import estimate_room_at_xy

        if not self._room_clusters:
            self.refresh_room_clusters()
        if robot_xy is None:
            return "unknown"
        try:
            xy = (float(robot_xy[0]), float(robot_xy[1]))
        except Exception:
            return "unknown"
        return estimate_room_at_xy(
            self._room_clusters,
            xy,
            max_dist_m=self._room_assign_max_m(),
        )

    def format_rooms_line(self, *, max_chars: int = 200) -> str:
        """Compact ``Rooms: …`` summary for router / memory prompts."""
        from emet.memory.graph_eqa.room_clusters import format_rooms_compact

        if not self._room_clusters:
            self.refresh_room_clusters()
        return format_rooms_compact(self._room_clusters, max_chars=max_chars)

    def stamp_vlm_room_at_robot(
        self,
        robot_xy: Any,
        room: str | None,
        *,
        protect_indoor_from_outdoor: bool = True,
        corroborating_labels: list[str] | tuple[str, ...] | None = None,
        source: str = "router_vlm",
        source_view_id: str | None = None,
        agent_round: int | None = None,
        pose_round: int | None = None,
    ) -> str:
        """Stamp VLM ``current_room`` onto the nearest cluster; return stamped name or unknown.

        ``room`` should already be policy-coerced (canonical bucket or free-text phrase).
        By default refuses outdoor overwriting a named indoor cluster without corroboration.
        Returns ``unknown`` when the stamp is skipped/blocked.
        """
        from emet.memory.graph_eqa.agentic_tools import sanitize_room_phrase
        from emet.memory.graph_eqa.room_clusters import estimate_room_at_xy, stamp_room_at_xy

        name = sanitize_room_phrase(room)
        if name == "unknown" or robot_xy is None:
            return "unknown"
        if not self._room_clusters:
            self.refresh_room_clusters()
        try:
            xy = (float(robot_xy[0]), float(robot_xy[1]))
        except Exception:
            return "unknown"
        prev = estimate_room_at_xy(
            self._room_clusters,
            xy,
            max_dist_m=self._room_assign_max_m(),
        )
        labs = [str(x) for x in (corroborating_labels or ()) if str(x).strip()] or None
        self._room_clusters = stamp_room_at_xy(
            self._room_clusters,
            xy,
            name,
            max_dist_m=self._room_assign_max_m(),
            protect_indoor_from_outdoor=bool(protect_indoor_from_outdoor),
            corroborating_labels=labs,
        )
        self.last_room_clusters = list(self._room_clusters)
        after = estimate_room_at_xy(
            self._room_clusters,
            xy,
            max_dist_m=self._room_assign_max_m(),
        )
        after_s = sanitize_room_phrase(after)
        if after_s != name:
            # Protection blocked the write (or nearest cluster out of range).
            return "unknown" if sanitize_room_phrase(prev) != name else after_s
        if self.world_evidence.enabled:
            nearest = min(
                self._room_clusters,
                key=lambda cluster: float(
                    (cluster.centroid_xy[0] - xy[0]) ** 2 + (cluster.centroid_xy[1] - xy[1]) ** 2
                ),
                default=None,
            )
            if nearest is not None and nearest.room_id:
                confidence = 0.75 if source == "investigate_vlm" else 0.65
                hypothesis = self.world_evidence.stamp_room(
                    nearest.room_id,
                    name,
                    source=source,
                    confidence=confidence,
                    step=self._effective_timestep(),
                    view_id=source_view_id,
                    agent_round=agent_round,
                    pose_round=pose_round,
                )
                if hypothesis is not None:
                    self._room_clusters = [
                        replace(cluster, room_name=hypothesis.room_name)
                        if cluster.room_id == hypothesis.room_id
                        else cluster
                        for cluster in self._room_clusters
                    ]
                    self.last_room_clusters = list(self._room_clusters)
                    if hypothesis.room_name != name:
                        return "unknown"
        return name

    def nearby_object_observations(
        self,
        robot_xy: Any,
        *,
        k: int = 3,
        max_dist_m: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Nearest object observations with RGB for multimodal router room context."""
        if robot_xy is None or k <= 0:
            return []
        try:
            rxy = np.asarray(robot_xy, dtype=float).reshape(-1)[:2]
        except Exception:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for obs in list(getattr(self, "_observations", None) or []):
            rgb = getattr(obs, "rgb", None)
            if not isinstance(rgb, np.ndarray) or rgb.ndim != 3:
                continue
            labels = [str(x).strip() for x in list(getattr(obs, "labels", None) or []) if str(x).strip()]
            if any(lab.lower() == "frontier" for lab in labels) and len(labels) <= 1:
                continue
            xyz = getattr(obs, "xyz", None)
            if xyz is None:
                continue
            try:
                oxy = np.asarray(xyz, dtype=float).reshape(-1)[:2]
                dist = float(np.linalg.norm(oxy - rxy))
            except Exception:
                continue
            if dist > float(max_dist_m):
                continue
            phrase = labels[0] if labels else f"obs_{int(obs.obs_id)}"
            scored.append(
                (
                    dist,
                    {
                        "obs_id": int(obs.obs_id),
                        "dist_m": round(dist, 2),
                        "labels": labels[:6],
                        "phrase": phrase,
                        "rgb": np.asarray(rgb),
                    },
                )
            )
        scored.sort(key=lambda t: t[0])
        return [item for _, item in scored[: int(k)]]

    def _node_nav_status_suffix(self, node: GraphNode) -> str:
        failures = int(getattr(node, "nav_failures", 0) or 0)
        if failures <= 0:
            return ""
        note = (getattr(node, "last_nav_note", None) or "").strip()
        tail = f", last: {note}" if note else ""
        return f"; unreachable ({failures} nav failure(s){tail})"

    def _attempt_ledger_enabled(self) -> bool:
        """True when ``eqa.attempt_ledger`` / ``EMET_EQA_ATTEMPT_LEDGER`` is on (default off)."""
        env = os.environ.get("EMET_EQA_ATTEMPT_LEDGER", "").strip().lower()
        if env in ("1", "true", "yes", "on"):
            return True
        if env in ("0", "false", "no", "off"):
            return False
        raw = self._eqa_cfg_value("attempt_ledger", False)
        if isinstance(raw, dict):
            raw = raw.get("enabled", False)
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)

    def set_attempt_ledger_question_id(self, question_id: str | None) -> None:
        """Tag subsequent ledger rows with a question/episode id (does not clear the store)."""
        self._attempt_ledger_question_id = str(question_id) if question_id else None
        self.world_evidence.set_question_id(self._attempt_ledger_question_id)

    def record_attempt(
        self,
        *,
        action_kind: str,
        outcome: str,
        status_code: str,
        note: str = "",
        step: int | None = None,
        target_node_id: int | None = None,
        obs_id: int | None = None,
        xyz: tuple[float, float, float] | None = None,
        source: AttemptSource | str = "unknown",
        question_id: str | None = None,
        phrase: str = "",
        room: str = "",
        target_kind: str = "",
        target_id: str = "",
        view_id: str = "",
        force: bool = False,
    ) -> AttemptRecord | None:
        """Append one :class:`AttemptRecord` when the ledger is enabled (or ``force``).

        Returns the stored record, or ``None`` when the ledger is off.
        """
        st = int(step if step is not None else self._effective_timestep())
        qid = question_id if question_id is not None else self._attempt_ledger_question_id
        stable_kind = str(target_kind or "")
        stable_id = str(target_id or "")
        stable_view = str(view_id or "")
        if target_node_id is not None and not stable_id:
            entity = self.world_evidence.entity_for_node(int(target_node_id))
            if entity is not None:
                stable_kind = "place"
                stable_id = entity.place_id
        if obs_id is not None and not stable_view:
            stable_view = self.view_id_for_obs(int(obs_id))
        if not stable_id and stable_view:
            view = self.world_evidence.views.get(stable_view)
            if view is not None and view.place_id:
                stable_kind = "place"
                stable_id = view.place_id
        ledger_enabled = bool(force or self._attempt_ledger_enabled())
        if self.world_evidence.enabled:
            subject_kind = stable_kind or ("observation" if obs_id is not None else "action")
            subject_id = stable_id or (str(obs_id) if obs_id is not None else str(action_kind))
            self.world_evidence.record_event(
                subject_kind=subject_kind,
                subject_id=subject_id,
                predicate=str(action_kind),
                polarity="positive" if str(outcome) in {"ok", "present"} else "negative",
                source=str(source or "unknown"),
                confidence=1.0,
                step=st,
                view_id=stable_view or None,
                place_id=stable_id if stable_kind == "place" else None,
                payload={
                    "outcome": str(outcome),
                    "status_code": str(status_code),
                    "note": str(note)[:240],
                    "legacy_obs_id": obs_id,
                    "legacy_node_id": target_node_id,
                },
            )
        if not ledger_enabled:
            return None
        src = str(source or "unknown")
        if src not in ("chat", "eqa", "unknown"):
            src = "unknown"
        rec = AttemptRecord.from_dict(
            {
                "action_kind": action_kind,
                "outcome": outcome,
                "status_code": status_code,
                "note": note,
                "step": st,
                "target_node_id": target_node_id,
                "obs_id": obs_id,
                "xyz": list(xyz) if xyz is not None else None,
                "source": src,
                "question_id": qid,
                "phrase": phrase,
                "room": room,
                "target_kind": stable_kind,
                "target_id": stable_id,
                "view_id": stable_view,
            }
        )
        self._attempt_records.append(rec)
        max_n = max(1, int(self._attempt_ledger_max))
        if len(self._attempt_records) > max_n:
            self._attempt_records = self._attempt_records[-max_n:]
        return rec

    def clear_room_events(self) -> None:
        """Drop the room timeline (new episode / world-change)."""
        self._room_events = []

    def record_room_event(
        self,
        *,
        room: str | None,
        kind: str,
        step: int | None = None,
        phrase: str = "",
        obs_id: int | None = None,
        note: str = "",
        agent_round: int | None = None,
    ) -> dict[str, Any] | None:
        """Append a room-scoped timeline event when ``room`` is a known label.

        Does **not** invent ``unknown``. Survives room-cluster rebuilds. Independent
        of the attempt-ledger opt-in so agentic state can still show history.
        """
        from emet.memory.graph_eqa.room_clusters import normalize_current_room

        room_n = normalize_current_room(room)
        if room_n == "unknown":
            return None
        kind_s = str(kind or "").strip().lower()
        if not kind_s:
            return None
        st = int(step if step is not None else self._effective_timestep())
        event = {
            "step": st,
            "room": room_n,
            "kind": kind_s[:40],
            "phrase": str(phrase or "").strip().lower()[:80],
            "obs_id": int(obs_id) if obs_id is not None else None,
            "note": str(note or "").strip()[:120],
            "world_step": st,
            "agent_round": int(agent_round) if agent_round is not None else None,
        }
        world_event = self.world_evidence.record_event(
            subject_kind="room",
            subject_id=room_n,
            predicate=kind_s,
            polarity="negative" if "absent" in kind_s else "positive",
            source="room_timeline",
            confidence=0.7,
            step=st,
            view_id=self.view_id_for_obs(int(obs_id)) if obs_id is not None else None,
            payload={
                "phrase": str(phrase or "").strip()[:120],
                "legacy_obs_id": int(obs_id) if obs_id is not None else None,
                "note": str(note or "").strip()[:120],
                "agent_round": int(agent_round) if agent_round is not None else None,
            },
        )
        if world_event is not None:
            event["event_id"] = world_event.event_id
        self._room_events.append(event)
        max_n = max(8, int(self._room_events_max))
        if len(self._room_events) > max_n:
            self._room_events = self._room_events[-max_n:]
        return dict(event)

    def get_room_events(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return room timeline rows (oldest first)."""
        rows = [dict(e) for e in self._room_events]
        if limit is not None and int(limit) >= 0:
            rows = rows[-int(limit) :]
        return rows

    def format_room_history(
        self,
        *,
        max_chars: int = 220,
        target_rooms: list[str] | set[str] | None = None,
    ) -> str:
        """Newest-first compact room timeline for the agentic state card."""
        if not self._room_events:
            targets = sorted({str(t).strip().lower() for t in (target_rooms or []) if str(t).strip()})
            if not targets:
                return ""
            return "Room history: (none) | targets=" + ",".join(targets)
        bits: list[str] = []
        for ev in reversed(self._room_events):
            step = int(ev.get("step") or 0)
            room = str(ev.get("room") or "").strip() or "?"
            kind = str(ev.get("kind") or "").strip() or "?"
            phrase = str(ev.get("phrase") or "").strip()
            bit = f"r{step} {room} {kind}"
            if phrase and kind.startswith("verify"):
                bit += f"({phrase})"
            bits.append(bit)
        targets = sorted({str(t).strip().lower() for t in (target_rooms or []) if str(t).strip()})
        prefix = "Room history: "
        tail = (" | targets=" + ",".join(targets)) if targets else ""
        # Drop oldest (end of newest-first list) until under budget.
        while bits:
            body = "; ".join(bits)
            text = prefix + body + tail
            if len(text) <= int(max_chars):
                return text
            bits = bits[:-1]
        return (prefix + "(truncated)" + tail)[: int(max_chars)]

    def get_attempt_records(
        self,
        *,
        obs_id: int | None = None,
        action_kind: str | None = None,
        target_node_id: int | None = None,
        question_id: str | None = None,
        limit: int | None = None,
    ) -> list[AttemptRecord]:
        """Return ledger rows matching optional filters (oldest first)."""
        rows = list(self._attempt_records)
        if obs_id is not None:
            oid = int(obs_id)
            rows = [r for r in rows if r.obs_id is not None and int(r.obs_id) == oid]
        if action_kind is not None:
            kind = str(action_kind)
            rows = [r for r in rows if r.action_kind == kind]
        if target_node_id is not None:
            nid = int(target_node_id)
            rows = [r for r in rows if r.target_node_id is not None and int(r.target_node_id) == nid]
        if question_id is not None:
            qid = str(question_id)
            rows = [r for r in rows if r.question_id == qid]
        if limit is not None and int(limit) >= 0:
            rows = rows[-int(limit) :]
        return rows

    def export_attempt_ledger(self) -> list[dict[str, Any]]:
        """JSON-serializable snapshot of the attempt ledger."""
        return records_to_dicts(self._attempt_records)

    def import_attempt_ledger(self, items: list[Any], *, replace: bool = True) -> int:
        """Load ledger rows from dicts (or :class:`AttemptRecord`). Returns count loaded."""
        loaded = records_from_dicts(list(items or []))
        if replace:
            self._attempt_records = loaded
        else:
            self._attempt_records.extend(loaded)
        max_n = max(1, int(self._attempt_ledger_max))
        if len(self._attempt_records) > max_n:
            self._attempt_records = self._attempt_records[-max_n:]
        return len(loaded)

    def clear_attempt_ledger(self) -> None:
        self._attempt_records.clear()

    def derive_nav_counters_from_ledger(self, obs_id: int) -> tuple[int, int, str | None, int]:
        """Compute ``(attempts, failures, last_note, last_step)`` for ``obs_id`` from the ledger."""
        rows = self.get_attempt_records(obs_id=obs_id, action_kind="navigate")
        if not rows:
            return 0, 0, None, 0
        failures = sum(1 for r in rows if r.outcome != "ok")
        last = rows[-1]
        note = (last.note or last.status_code or "").strip() or None
        return len(rows), failures, note, int(last.step)

    def record_nav_attempt(
        self,
        obs_id: int | None,
        *,
        success: bool,
        note: str,
        dist_m: float = 0.0,
        step: int | None = None,
        status_code: str | None = None,
        source: AttemptSource | str = "eqa",
        question_id: str | None = None,
        target_node_id: int | None = None,
    ) -> None:
        """Update graph node(s) tied to ``obs_id`` after an EQA navigation attempt.

        When the attempt ledger is enabled, also append a ``navigate``
        :class:`AttemptRecord`. Node ``nav_attempts`` / ``nav_failures`` counters
        remain dual-written for compatibility.
        """
        if obs_id is None:
            self.last_nav_result_note = note
            return
        oid = int(obs_id)
        self._obs_nav_dists.setdefault(oid, []).append(float(dist_m))
        st = int(step if step is not None else self._effective_timestep())
        moved = float(dist_m) >= 0.12
        ok = bool(success) and moved
        matched_node_id = target_node_id
        xyz_t: tuple[float, float, float] | None = None
        for idx, node in enumerate(self._nodes):
            if int(node.obs_id) != oid:
                continue
            if matched_node_id is None:
                matched_node_id = int(node.node_id)
            if xyz_t is None:
                try:
                    arr = np.asarray(node.xyz, dtype=float).reshape(-1)
                    if arr.size >= 3:
                        xyz_t = (float(arr[0]), float(arr[1]), float(arr[2]))
                except Exception:
                    xyz_t = None
            failures = int(getattr(node, "nav_failures", 0)) + (0 if ok else 1)
            self._nodes[idx] = replace(
                node,
                nav_attempts=int(getattr(node, "nav_attempts", 0)) + 1,
                nav_failures=failures,
                last_nav_note=str(note or "")[:120] or None,
                last_nav_at_step=st,
            )
        code = status_code or infer_nav_status_code(success=ok, note=str(note or ""))
        outcome = infer_nav_outcome(success=ok, status_code=code)
        self.record_attempt(
            action_kind="navigate",
            outcome=outcome,
            status_code=code,
            note=str(note or "")[:240],
            step=st,
            target_node_id=matched_node_id,
            obs_id=oid,
            xyz=xyz_t,
            source=source,
            question_id=question_id,
        )
        self.last_nav_result_note = note

    @staticmethod
    def strip_caption_block_from_history(text: str) -> str:
        """Drop a leading ``Caption:`` block so HISTORY cannot reinforce caption loops."""
        if not text:
            return text
        return re.sub(
            r"(?is)^\s*Caption:\s*.*?(?=\n\s*(?:Reasoning|Answer)\s*:|\Z)",
            "",
            text,
            count=1,
        ).lstrip()

    def _append_eqa_history(self, text: str) -> None:
        self._history_outputs.append(self.strip_caption_block_from_history(text))

    def append_nav_outcome_to_last_history(self, *, dist_m: float, success: bool, note: str) -> None:
        if not self._history_outputs:
            return
        status = "ok" if success else "failed"
        self._history_outputs[-1] += f"\nNav_result: moved {float(dist_m):.2f}m ({status}; {note})"

    def eqa_obs_look_spent(self, obs_id: int | None) -> bool:
        """True when this observation has already been inspected (do not re-nav there)."""
        if obs_id is None:
            return False
        oid = int(obs_id)
        attempts = 0
        for node in self._nodes:
            if int(node.obs_id) != oid:
                continue
            attempts = max(attempts, int(getattr(node, "nav_attempts", 0) or 0))
        return eqa_look_is_spent(self._obs_nav_dists.get(oid, ()), nav_attempts=attempts)

    def next_unspent_eqa_obs_id(
        self,
        candidates: list[int] | None,
        *,
        skip: set[int] | None = None,
    ) -> int | None:
        """First usable observation in ``candidates`` that is not look-spent."""
        blocked = {int(x) for x in (skip or set())}
        for raw in candidates or []:
            oid = int(raw)
            if oid in blocked or not self._obs_usable_for_eqa_image(oid):
                continue
            if self.eqa_obs_look_spent(oid):
                continue
            return oid
        return None

    def _obs_xy(self, obs_id: int) -> np.ndarray | None:
        obs = self._observation_by_id(int(obs_id))
        if obs is None or getattr(obs, "xyz", None) is None:
            return None
        return np.asarray(obs.xyz, dtype=float)[:2]

    def _spread_obs_xy(self, obs_ids: list[int], *, max_n: int) -> list[int]:
        return spread_obs_ids_xy(obs_ids, self._obs_xy, max_n=max_n)

    def _eqa_find_phrases(self) -> list[str]:
        phrases: list[str] = []
        for raw in (
            list(self._relevant_phrases or [])
            + list(self._relevant_objects or [])
            + list(self._confirmed_memory_phrases())
        ):
            text = str(raw or "").strip()
            if text and text not in phrases:
                phrases.append(text)
        return phrases

    def _phrases_need_rgb_highlight(self, visual_pins: list[int], phrases: list[str]) -> bool:
        """True when SigLIP retrieve is empty or the question phrases are weak (e.g. ``time``)."""
        if len(visual_pins) >= 2:
            return False
        tokens: set[str] = set()
        for phrase in phrases:
            tokens |= _object_match_tokens(phrase)
        concrete = {tok for tok in tokens if tok not in _WEAK_SIGLIP_FIND_TOKENS}
        if visual_pins and concrete:
            return False
        return True

    def _merge_highlight_phrases(self, extra: list[str]) -> None:
        objects = list(self._relevant_objects or [])
        phrases = list(self._relevant_phrases or [])
        for name in extra:
            key = str(name).strip().lower()
            if not key:
                continue
            if key not in objects:
                objects.append(key)
            if key not in phrases:
                phrases.append(key)
        self._relevant_objects = objects
        # Keep an empty phrase list empty so coverage still uses the full object list.
        if self._relevant_phrases:
            self._relevant_phrases = phrases

    def _highlight_relevant_from_latest_rgb(self, question: str) -> list[str]:
        """One pass on the live/latest full frame: names that could answer the question.

        Not detector boxes. Skip when the client is missing, the latest view was already
        highlighted for this question, or the reply looks like an EQA field dump.
        """
        client = self.image_description_client
        if client is None:
            return []
        obs = None
        for candidate in reversed(self._observations):
            if self._obs_usable_for_eqa_image(int(candidate.obs_id)):
                obs = candidate
                break
        if obs is None:
            return []
        key = (str(question), int(obs.obs_id))
        if self._eqa_highlight_key == key:
            return []
        prompt = (
            f"Question: {question}\n"
            "Name 1-3 objects visible in this image that could help answer the question. "
            "Comma-separated short nouns only (for example: wall clock, stool). "
            "If nothing relevant is visible, reply none."
        )
        try:
            image = Image.fromarray(rgb_uint8(obs.rgb), mode="RGB")
            raw = client([prompt, image])
        except Exception:
            return []
        self._eqa_highlight_key = key
        text = raw if isinstance(raw, str) else str(raw or "")
        low = text.lower()
        if "reasoning:" in low or "answer:" in low or "confidence:" in low:
            return []
        out: list[str] = []
        for part in re.split(r"[,;\n]", text):
            name = part.strip(" .").lower()
            if not name or name in {"none", "n/a", "nothing", "no", "unknown"}:
                continue
            if len(name) < 3 or len(name) > 40:
                continue
            if len(name.split()) > 4:
                continue
            if name not in out:
                out.append(name)
            if len(out) >= 3:
                break
        return out

    def resolve_voxel_frame_to_graph_obs_id(self, voxel_obs_count: int, voxel_map: Any | None = None) -> int | None:
        """Map a DynaMem voxel ``obs_count`` (1-based frame index) to a graph observation id.

        Voxel frames and graph obs ids diverge after instance merges. Prefer a graph view
        whose capture pose matches the voxel camera; fall back to the raw id when it is
        already a usable graph observation.
        """
        voc = int(voxel_obs_count)
        cam = None
        frame_rgb = None
        obs_list = getattr(voxel_map, "observations", None) if voxel_map is not None else None
        if obs_list:
            idx = voc - 1
            if 0 <= idx < len(obs_list):
                frame = obs_list[idx]
                rgb = getattr(frame, "rgb", None)
                if rgb is not None:
                    frame_rgb = np.asarray(rgb)
                pose = getattr(frame, "camera_pose", None)
                if pose is not None:
                    try:
                        cam = np.asarray(pose, dtype=float).reshape(4, 4)[:3, 3]
                    except Exception:
                        cam = None
        best_oid: int | None = None
        best_score = -1.0
        for gobs in self._observations:
            oid = int(gobs.obs_id)
            if not self._obs_usable_for_eqa_image(oid):
                continue
            score = 0.0
            if cam is not None:
                view = gobs.viewer_xyz if gobs.viewer_xyz is not None else gobs.xyz
                if view is not None:
                    dist = float(
                        np.linalg.norm(np.asarray(view, dtype=float).reshape(-1)[:3] - cam[:3])
                    )
                    if dist < 0.75:
                        score += 10.0 - dist
            if frame_rgb is not None:
                graph_rgb = np.asarray(gobs.rgb)
                if graph_rgb.shape == frame_rgb.shape:
                    score += 0.5
                    if np.shares_memory(graph_rgb, frame_rgb) or graph_rgb is frame_rgb:
                        score += 5.0
            if score > best_score:
                best_score = score
                best_oid = oid
        if best_oid is not None and best_score > 0:
            return best_oid
        if self._obs_usable_for_eqa_image(voc):
            return voc
        return None

    def nearest_graph_obs_to_xyz(self, xyz: Any, *, max_dist_m: float = 2.0) -> int | None:
        """Nearest usable graph observation to a SigLIP voxel point (planar XY)."""
        try:
            target = np.asarray(xyz, dtype=float).reshape(-1)[:2]
        except Exception:
            return None
        if target.size < 2:
            return None
        best_oid: int | None = None
        best_dist = float("inf")
        for gobs in self._observations:
            oid = int(gobs.obs_id)
            if not self._obs_usable_for_eqa_image(oid) or gobs.xyz is None:
                continue
            xy = np.asarray(gobs.xyz, dtype=float).reshape(-1)[:2]
            dist = float(np.linalg.norm(xy - target))
            if dist < best_dist:
                best_dist = dist
                best_oid = oid
        if best_oid is None or best_dist > float(max_dist_m):
            return None
        return best_oid

    def _visual_find_obs_ids(self, phrases: list[str], *, max_n: int) -> list[int]:
        """Rank stored RGB by SigLIP retrieve vs question phrases (not YoloE names)."""
        if max_n <= 0:
            return []
        scored: list[tuple[float, int]] = []
        seen: set[int] = set()

        def _add(sim: float, oid: int | None) -> None:
            if oid is None:
                return
            oi = int(oid)
            if not self._obs_usable_for_eqa_image(oi):
                return
            if oi in seen:
                for i, (old_sim, old_oid) in enumerate(scored):
                    if old_oid == oi and float(sim) > old_sim:
                        scored[i] = (float(sim), oi)
                return
            seen.add(oi)
            scored.append((float(sim), oi))

        visual_fn = getattr(self, "_visual_find_fn", None)
        if visual_fn is not None:
            for phrase in phrases:
                hits: Any = []
                try:
                    hits = visual_fn(str(phrase), max(int(max_n) * 3, 8))
                except TypeError:
                    try:
                        hits = visual_fn(str(phrase))
                    except Exception:
                        hits = []
                except Exception:
                    hits = []
                for item in hits or []:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        _add(float(item[0]), int(item[1]))
                    else:
                        try:
                            _add(max(SIGLIP_PRESENT_THRESHOLD, 0.25), int(item))
                        except (TypeError, ValueError):
                            continue
        rank_cache = getattr(self, "_visual_find_rank_cache", None) or {}
        for phrase in phrases:
            key = str(phrase or "").strip().lower()
            if not key:
                continue
            for sim, oid in rank_cache.get(key, []):
                _add(float(sim), int(oid))
        enc = getattr(self, "_confirmed_memory_siglip_encoder", None)
        feats = getattr(self, "_obs_siglip_features", None) or {}
        if enc is not None and feats:
            from emet.memory.graph_eqa.graph_eqa_siglip import rank_observations_for_phrase

            for phrase in phrases:
                for sim, oid in rank_observations_for_phrase(str(phrase), enc, feats):
                    if sim < SIGLIP_PRESENT_THRESHOLD:
                        break
                    _add(sim, oid)
        for phrase in phrases:
            key = str(phrase or "").strip().lower()
            if not key:
                continue
            cached = self._siglip_phrase_cache.get(key)
            if cached is None or cached[2] is None:
                continue
            if float(cached[0]) >= SIGLIP_PRESENT_THRESHOLD:
                _add(float(cached[0]), int(cached[2]))
        grounder = getattr(self, "_obs_id_grounder", None)
        if grounder is not None:
            for phrase in phrases:
                try:
                    oid = grounder(str(phrase))
                except Exception:
                    oid = None
                if oid is not None:
                    _add(max(SIGLIP_PRESENT_THRESHOLD, 0.25), int(oid))
        scored.sort(key=lambda t: -t[0])
        ordered = [oid for _sim, oid in scored]
        return self._spread_obs_xy(ordered, max_n=max_n)

    def eqa_attached_target_obs_id(self) -> int | None:
        """Image 1 obs id when that view is already the visual FIND target."""
        ids = list(self.last_eqa_obs_ids or [])
        if not ids:
            return None
        oid = int(ids[0])
        if not self._obs_usable_for_eqa_image(oid):
            return None
        phrases = self._eqa_find_phrases()
        visual = self._visual_find_obs_ids(phrases, max_n=8)
        if visual:
            return oid if oid in set(visual) else None
        if self._target_visible_in_obs_ids([oid]):
            return oid
        return None

    def eqa_stay_on_attached_view(self) -> bool:
        """True when Image 1 is already a FIND view — do not frontier-chase away from it."""
        return self.eqa_attached_target_obs_id() is not None

    def eqa_approach_attached_find(self, robot_xyt: Any | None = None) -> np.ndarray | None:
        """Waypoint toward Image-1 FIND when the robot is still far; None to stay put."""
        oid = self.eqa_attached_target_obs_id()
        if oid is None or self.eqa_obs_look_spent(oid):
            return None
        waypoint = self._navigation_waypoint_for_obs(int(oid), robot_xyt)
        if waypoint is None:
            return None
        robot_xy = self._robot_planar_xy(robot_xyt)
        if robot_xy is None:
            return waypoint
        dist = float(math.hypot(float(waypoint[0]) - robot_xy[0], float(waypoint[1]) - robot_xy[1]))
        if dist < EQA_SAME_OBS_PROGRESS_M:
            return None
        return waypoint

    def _eqa_rgb_for_obs(self, obs_id: int) -> np.ndarray | None:
        """Full camera frame for an observation (scene context for counting)."""
        obs = self._observation_by_id(int(obs_id))
        if obs is None or not self._obs_usable_for_eqa_image(int(obs_id)):
            return None
        return rgb_uint8(obs.rgb)

    def _eqa_crop_for_obs(self, obs_id: int) -> np.ndarray | None:
        """Tight detector crop, or None when the bbox is missing or covers the frame."""
        obs = self._observation_by_id(int(obs_id))
        if obs is None or not self._obs_usable_for_eqa_image(int(obs_id)):
            return None
        nodes = [
            n
            for n in self._nodes
            if int(n.obs_id) == int(obs_id) and not n.is_frontier and not n.is_viewpoint
        ]
        return tightest_node_crop(nodes, rgb_uint8(obs.rgb))

    def _eqa_pick_closeup_obs_id(
        self,
        obs_ids: list[int],
        look_obs_id: int | None,
        pin_obs: list[int],
    ) -> int | None:
        """Choose an already-attached scene whose detector crop is a useful extra view."""
        attached = {int(oid) for oid in obs_ids}
        ordered: list[int] = []
        if look_obs_id is not None and int(look_obs_id) in attached:
            ordered.append(int(look_obs_id))
        ordered.extend(int(oid) for oid in pin_obs if int(oid) in attached)
        ordered.extend(int(oid) for oid in obs_ids)
        seen: set[int] = set()
        for oid in ordered:
            if oid in seen:
                continue
            seen.add(oid)
            if self._eqa_crop_for_obs(oid) is not None:
                return oid
        return None

    def _eqa_reserve_closeup_slot(
        self,
        obs_ids: list[int],
        pin_obs: list[int],
        look_obs_id: int | None,
    ) -> list[int]:
        """Drop one extra FIND view so a labeled close-up can occupy the last slot."""
        if len(obs_ids) < 2:
            return list(obs_ids)
        pin_set = {int(x) for x in pin_obs}
        look = int(look_obs_id) if look_obs_id is not None else None
        ids = list(obs_ids)
        for i in range(len(ids) - 1, 0, -1):
            oid = int(ids[i])
            if oid in pin_set and oid != int(ids[0]) and oid != look:
                return ids[:i] + ids[i + 1 :]
        return ids[:-1]

    def alternate_nav_target_for_failed_action(
        self,
        question: str,
        blocked_obs_id: int,
        planner: Any,
        base_xyt: Any,
    ) -> np.ndarray | None:
        """Pick a different frontier/fluid goal when the VLM re-picks a failed image action."""
        frontier_nodes = [
            n for n in self._nodes if getattr(n, "is_frontier", False) and int(n.obs_id) != int(blocked_obs_id)
        ]
        if frontier_nodes:
            frontier_nodes.sort(key=lambda n: (int(getattr(n, "nav_failures", 0)), -int(n.last_seen)))
            pick = frontier_nodes[0]
            return np.array([float(pick.xyz[0]), float(pick.xyz[1]), 1.0], dtype=float)
        return None

    def _rank_nodes_for_eqa_prompt(
        self,
        *,
        keywords: list[str] | None = None,
        prefer_obs_ids: list[int] | None = None,
    ) -> list[GraphNode]:
        """Rank object/frontier nodes for a bounded EQA SCENE_GRAPH block.

        Viewpoints are omitted from the ranked list (they bloat prompts); edges still
        reference kept object ids. Frontiers are included and ranked after objects.
        """
        from emet.memory.graph_eqa.frontier_nodes import keyword_overlap_score

        kws = list(keywords or self._relevant_objects or [])
        prefer = {int(x) for x in (prefer_obs_ids or self.last_eqa_obs_ids or [])}
        objects: list[tuple[float, GraphNode]] = []
        frontiers: list[tuple[float, GraphNode]] = []
        for n in self._nodes:
            if n.is_viewpoint:
                continue
            kw = keyword_overlap_score(list(n.labels or []), kws) if kws else 0.0
            support = float(getattr(n, "support_count", 1) or 1)
            prefer_bonus = 2.0 if int(n.obs_id) in prefer else 0.0
            score = 10.0 * kw + prefer_bonus + 0.1 * support
            if n.is_frontier:
                frontiers.append((score, n))
            else:
                objects.append((score, n))
        objects.sort(key=lambda t: (-t[0], int(t[1].node_id)))
        frontiers.sort(key=lambda t: (-t[0], int(t[1].node_id)))
        return [n for _, n in objects] + [n for _, n in frontiers]

    def _eqa_cfg_value(self, key: str, default: Any = None) -> Any:
        """Read ``eqa.<key>`` from Parameters or a nested dict."""
        params = self.parameters
        if params is None:
            return default
        if isinstance(params, dict):
            eqa = params.get("eqa")
            if isinstance(eqa, dict) and key in eqa:
                return eqa.get(key, default)
            return params.get(f"eqa/{key}", params.get(key, default))
        if hasattr(params, "get"):
            return params.get(f"eqa/{key}", default)
        return default

    def _eqa_override_gate(self, key: str, default: bool) -> bool:
        """Boolean eqa flag with an env escape hatch: ``EMET_EQA_<KEY>=0|1``.

        Lets the Habitat HM-EQA runner / overnight jobs toggle location-letter
        override behavior (equip/image gates) without editing a config file.
        """
        env_name = "EMET_EQA_" + key.upper()
        env = os.environ.get(env_name, "").strip().lower()
        if env in ("1", "true", "yes", "on"):
            return True
        if env in ("0", "false", "no", "off"):
            return False
        return bool(self._eqa_cfg_value(key, default))

    def _spatial_rag_enabled(self) -> bool:
        """True when eqa.spatial_rag or EMET_EQA_SPATIAL_RAG requests REGION prompts."""
        env = os.environ.get("EMET_EQA_SPATIAL_RAG", "").strip().lower()
        if env in ("1", "true", "yes", "on"):
            return True
        if env in ("0", "false", "no", "off"):
            return False
        raw = self._eqa_cfg_value("spatial_rag", False)
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)

    def _spatial_rag_float(self, key: str, default: float) -> float:
        try:
            return float(self._eqa_cfg_value(key, default))
        except Exception:
            return default

    def _spatial_rag_int(self, key: str, default: int) -> int:
        try:
            return int(self._eqa_cfg_value(key, default))
        except Exception:
            return default

    def _merged_memory_enabled(self) -> bool:
        """True when eqa.merged_memory / EMET_EQA_MERGED_MEMORY folds CONFIRMED_MEMORY into SCENE_GRAPH.

        Default on. The HM-EQA paper row pins ``merged_memory: false`` via
        ``configs/benchmarks/dynagraph.yaml`` (harness ``habitat_eqa.dynagraph``) so its
        numbers stay on the standalone summary block; every other path gets the folded
        format. When on, the main EQA prompt tags SCENE_GRAPH nodes with status
        (inspect / candidate) and room names, and emits a short CONFIRMED_MEMORY tail
        only for phrases with no tagged node, instead of a separate summary block
        (one fact, one line — no duplicate object mentions).
        """
        env = os.environ.get("EMET_EQA_MERGED_MEMORY", "").strip().lower()
        if env in ("1", "true", "yes", "on"):
            return True
        if env in ("0", "false", "no", "off"):
            return False
        raw = self._eqa_cfg_value("merged_memory", True)
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)

    def to_string(
        self,
        *,
        max_object_nodes: int | None = None,
        question_keywords: list[str] | None = None,
        prefer_obs_ids: list[int] | None = None,
        record_prompt_count: bool = False,
        merge_confirmed: bool = False,
    ) -> str:
        """Serialize the scene graph to a string for mLLM prompts.

        When ``max_object_nodes`` is set, keep the top-K ranked object/frontier nodes
        (keyword + support + Image-N preference) so blowups cannot starve the VLM.
        With ``eqa.spatial_rag`` / ``EMET_EQA_SPATIAL_RAG``, emit compact REGION blocks
        around keyword / preferred-obs neighborhoods instead of a flat node dump.
        Full untruncated serialization is the default for exports / debugging.

        When ``merge_confirmed`` is set (prompt path only), fold CONFIRMED_MEMORY status
        into the serialization instead of emitting a separate summary block: matching
        nodes get an ``inspect`` / ``candidate`` tag, object nodes are tagged with their
        room-cluster name, and a short CONFIRMED_MEMORY tail lists only phrases with no
        tagged node (SigLIP-only sightings, weak matches, unobserved objects) plus a
        compact ``Rooms:`` line. The spatial-RAG branch is left untouched.
        """
        lines = []

        def _prompt_labels(node: GraphNode, max_len: int = 120) -> str:
            return node_display_name(node, max_len=max_len)

        if max_object_nodes is not None and max_object_nodes > 0 and self._spatial_rag_enabled():
            from emet.memory.graph_eqa.spatial_rag import (
                format_regions_for_prompt,
                select_spatial_regions,
            )

            radius = self._spatial_rag_float("spatial_rag_radius_m", 2.5)
            max_regions = self._spatial_rag_int("spatial_rag_max_regions", 6)
            max_nodes = self._spatial_rag_int(
                "spatial_rag_max_nodes",
                int(max_object_nodes) if max_object_nodes else 48,
            )
            frontier_budget = max(4, int(max_object_nodes) // 4)
            rag = select_spatial_regions(
                list(self._nodes),
                keywords=question_keywords or list(self._relevant_objects or []),
                prefer_obs_ids=prefer_obs_ids or self.last_eqa_obs_ids,
                radius_m=radius,
                max_regions=max_regions,
                max_nodes=max_nodes,
                max_frontiers=frontier_budget,
            )
            if rag.regions:
                text = format_regions_for_prompt(rag)
                keep_ids = set(rag.kept_node_ids)
                for n in rag.frontier_nodes:
                    keep_ids.add(int(n.node_id))
                edge_lines: list[str] = []
                for a, b, rel in self._edges:
                    if int(a) not in keep_ids:
                        continue
                    if b != -1 and int(b) not in keep_ids:
                        continue
                    b_str = "floor" if b == -1 else str(b)
                    edge_lines.append(f"  {rel}({a}, {b_str})")
                if edge_lines:
                    text = text + "\n" + "\n".join(edge_lines)
                if record_prompt_count:
                    self.last_eqa_prompt_node_count = len(keep_ids)
                    self.last_eqa_prompt_regions = len(rag.regions)
                    self.last_eqa_spatial_rag = {
                        "n_regions": len(rag.regions),
                        "n_nodes": len(keep_ids),
                        "seed_node_ids": list(rag.seed_node_ids),
                        "radius_m": radius,
                    }
                return text

        if max_object_nodes is not None and max_object_nodes > 0:
            ranked = self._rank_nodes_for_eqa_prompt(
                keywords=question_keywords,
                prefer_obs_ids=prefer_obs_ids,
            )
            # Always keep at least a few frontiers if present.
            objects = [n for n in ranked if not n.is_frontier]
            frontiers = [n for n in ranked if n.is_frontier]
            keep_obj = objects[: max(0, int(max_object_nodes))]
            frontier_budget = max(0, min(len(frontiers), max(4, int(max_object_nodes) // 4)))
            keep = keep_obj + frontiers[:frontier_budget]
            keep_ids = {int(n.node_id) for n in keep}
            nodes_for_prompt = keep
        else:
            nodes_for_prompt = list(self._nodes)
            keep_ids = {int(n.node_id) for n in nodes_for_prompt}

        if record_prompt_count:
            self.last_eqa_prompt_node_count = len(nodes_for_prompt)
            self.last_eqa_prompt_regions = 0
            self.last_eqa_spatial_rag = None

        # Merged-memory mode: fold CONFIRMED_MEMORY status into node lines so each
        # confirmed object appears once (one fact, one line). Only in prompt path,
        # and only when confirmed-memory is enabled at all.
        merge_active = (
            merge_confirmed and self.memory_summary_enabled and max_object_nodes is not None and max_object_nodes > 0
        )
        statuses: dict[str, tuple[str, list[int], float | None, np.ndarray | None, int | None]] = {}
        node_rooms: dict[int, str] = {}
        tagged: set[int] = set()
        tail_lines: list[str] = []
        nearest_by_phrase: dict[str, tuple[int, str]] = {}
        if merge_active:
            statuses = self._confirmed_phrase_statuses()
            node_rooms = self._node_room_by_id()
            for phrase, (status, ids, sim, xyz, obs_id) in statuses.items():
                if status == "present" and ids:
                    kept = [nid for nid in ids if nid in keep_ids]
                    if kept:
                        tagged.update(kept)
                        # Preserve nearest-furniture context (old CONFIRMED_MEMORY).
                        anchor = next(
                            (n for n in self._nodes if int(n.node_id) == int(kept[0])),
                            None,
                        )
                        if anchor is not None:
                            neighbors = self._nearest_object_neighbors(
                                np.asarray(anchor.xyz, dtype=np.float64),
                                exclude_node_ids=set(kept),
                                max_neighbors=2,
                                max_dist_m=3.0,
                            )
                            if neighbors:
                                near_bits = []
                                for n, dist in neighbors:
                                    lab = node_display_name(n)
                                    near_bits.append(f"{lab} at ({n.xyz[0]:.1f}, {n.xyz[1]:.1f}) {dist:.1f}m")
                                nearest_by_phrase[phrase] = (
                                    int(kept[0]),
                                    "nearest: " + "; ".join(near_bits),
                                )
                    else:
                        # All matches fell outside the shown node budget: keep the
                        # legacy-style facts (count, coordinates, nearest furniture)
                        # instead of dangling "Node 9" ids the model cannot see.
                        matches_nodes = sorted(
                            (n for n in self._nodes if int(n.node_id) in set(ids)),
                            key=lambda n: int(n.node_id),
                        )
                        parts = [
                            "candidate views: " + format_graph_node_candidates(matches_nodes, max_nodes=4),
                            _GRAPH_CANDIDATE_COUNT_DISCLAIMER,
                        ]
                        anchor = matches_nodes[0] if matches_nodes else None
                        if anchor is not None:
                            neighbors = self._nearest_object_neighbors(
                                np.asarray(anchor.xyz, dtype=np.float64),
                                exclude_node_ids=set(ids),
                                max_neighbors=2,
                                max_dist_m=3.0,
                            )
                            if neighbors:
                                near_bits = []
                                for n, dist in neighbors:
                                    lab = node_display_name(n)
                                    near_bits.append(f"{lab} at ({n.xyz[0]:.1f}, {n.xyz[1]:.1f}) {dist:.1f}m")
                                parts.append("nearest: " + "; ".join(near_bits))
                        tail_lines.append(
                            f"- {phrase}: LOOK — " + "; ".join(parts) + " (nodes not shown in graph above)"
                        )
                elif status == "candidate":
                    pos = f" near ({xyz[0]:.1f}, {xyz[1]:.1f})" if xyz is not None else ""
                    obs_note = f", obs_id={obs_id}" if obs_id is not None else ""
                    sim_s = f"{sim:.2f}" if sim is not None else "?"
                    tail_lines.append(
                        f"- {phrase}: CANDIDATE (SigLIP-only sim={sim_s}{pos}{obs_note}) "
                        "- verify in attached images before finalizing; "
                        "do not treat as confirmed present or absent"
                    )
                elif status == "weak_siglip":
                    # Do not assert absence — detector miss ≠ not in scene.
                    sim_s = f"{sim:.2f}" if sim is not None else "?"
                    tail_lines.append(
                        f"- {phrase}: weak SigLIP only (sim={sim_s}) — not evidence of absence; trust attached images"
                    )
                elif status == "not_observed":
                    tail_lines.append(f"- {phrase}: not observed during exploration")

        for n in nodes_for_prompt:
            lbl = _prompt_labels(n)
            sup = f" n={n.support_count}" if getattr(n, "support_count", 1) != 1 else ""
            if n.is_frontier:
                kind = "Frontier"
            elif n.is_viewpoint:
                kind = "View"
            else:
                kind = "Node"
            nid = int(n.node_id)
            room_tag = f" ({node_rooms[nid]})" if nid in node_rooms else ""
            status_tag = ""
            nearest_tag = ""
            if merge_active and not n.is_frontier and not n.is_viewpoint:
                if nid in tagged:
                    status_tag = " inspect"
                    # Attach nearest on the in-budget anchor node for each phrase.
                    for _phrase, (anchor_nid, near_txt) in nearest_by_phrase.items():
                        if int(anchor_nid) == nid:
                            nearest_tag = f" ({near_txt})"
                            break
                elif any(
                    status == "candidate" and obs_id is not None and int(n.obs_id) == obs_id
                    for status, _ids, _sim, _xyz, obs_id in statuses.values()
                ):
                    status_tag = " candidate"
            lines.append(
                f"{kind} {n.node_id}{room_tag}: {lbl} at ({n.xyz[0]:.2f}, {n.xyz[1]:.2f}, {n.xyz[2]:.2f}) "
                f"[Image {n.obs_id}]{sup}{self._node_nav_status_suffix(n)}{status_tag}{nearest_tag}"
            )
        for a, b, rel in self._edges:
            if int(a) not in keep_ids:
                continue
            if b != -1 and int(b) not in keep_ids:
                continue
            b_str = "floor" if b == -1 else str(b)
            lines.append(f"  {rel}({a}, {b_str})")
        if tail_lines:
            lines.append(
                "CONFIRMED_MEMORY (index of views to look at, not the answer; "
                "LOOK = candidate Image N to look at; CANDIDATE/weak SigLIP are "
                "navigation hints; if images contradict memory, trust the images):"
            )
            lines.extend(tail_lines)
        if merge_active and self._room_clusters:
            rooms_line = self.format_rooms_line(max_chars=200)
            if rooms_line.strip() and rooms_line.strip() != "Rooms:":
                lines.append(rooms_line)
        return (
            "SCENE_GRAPH (views to look at, not the answer; labels are proposals for WHERE to look):\n"
            + "\n".join(lines)
            if lines
            else "SCENE_GRAPH: (empty)"
        )

    def to_tree_string(self, indent: str = "  ") -> str:
        """
        Format the 3D spatial scene graph as an indented tree (text).

        Root = Scene; Floor is a virtual node; objects on floor are children of Floor;
        objects on other objects are nested. "Near" relations are listed at the end.
        Includes object labels, (x,y,z), and optional descriptions.
        """
        edge_set = set(self._edges)
        node_by_id = {n.node_id: n for n in self._nodes}
        object_nodes = [n for n in self._nodes if not n.is_viewpoint]

        def on_floor(nid: int) -> bool:
            return (nid, -1, "on") in edge_set

        def has_on_parent(nid: int) -> int | None:
            """Return node_id that this node is 'on', or None if on floor or no 'on' edge."""
            for a, b, rel in edge_set:
                if rel == "on" and a == nid and b != -1:
                    return b
            return None

        def children_of(nid: int | None) -> list[GraphNode]:
            if nid is None:
                # Floor children: explicitly on floor, or no "on" relation (in-scene)
                out = [
                    node_by_id[n.node_id]
                    for n in object_nodes
                    if on_floor(n.node_id) or has_on_parent(n.node_id) is None
                ]
            else:
                out = [node_by_id[a] for a, b, rel in edge_set if rel == "on" and b == nid and a in node_by_id]
            return sorted(out, key=lambda n: n.node_id)

        near_pairs = [(a, b) for a, b, rel in self._edges if rel == "near" and a < b]

        lines: list[str] = []
        lines.append("Scene (3D spatial graph)")
        lines.append(f"{indent}Floor")

        def visit(node: GraphNode, depth: int) -> None:
            pref = indent * (depth + 1)
            x, y, z = float(node.xyz[0]), float(node.xyz[1]), float(node.xyz[2])
            lbl = node_display_name(node)
            line = f"{pref}[{node.node_id}] {lbl}  at ({x:.2f}, {y:.2f}, {z:.2f})"
            if node.description:
                d = node.description
                if len(d) > 160:
                    d = d[:157] + "..."
                line += f"  — {d}"
            lines.append(line)
            for c in children_of(node.node_id):
                visit(c, depth + 1)

        for node in children_of(None):
            visit(node, 1)

        if near_pairs:
            lines.append("")
            lines.append("Near relations:")
            for a, b in near_pairs:
                na, nb = node_by_id.get(a), node_by_id.get(b)
                la = ", ".join(na.labels) if na and na.labels else str(a)
                lb = ", ".join(nb.labels) if nb and nb.labels else str(b)
                lines.append(f"{indent}{la} — {lb}")

        seen_from_edges = [(a, b) for a, b, rel in self._edges if rel == "seen_from"]
        if seen_from_edges:
            lines.append("")
            lines.append("Seen from (viewpoint node → object):")
            for a, b in seen_from_edges:
                na = node_by_id.get(a)
                nb = node_by_id.get(b)
                la = ", ".join(na.labels) if na and na.labels else str(a)
                if nb is not None:
                    vx, vy, vz = (float(nb.xyz[i]) for i in range(3))
                    lb = ", ".join(nb.labels) if nb.labels else str(b)
                    lines.append(f"{indent}{la} ← {lb} [{b}] at ({vx:.2f}, {vy:.2f}, {vz:.2f})")
                else:
                    lines.append(f"{indent}{la} ← node {b}")

        return "\n".join(lines) if lines else "Scene (3D spatial graph): (empty)"

    def seed_object_hints(self, labels: str) -> None:
        """GraphEQA HM-EQA enrich labels (per-question object hints for planning)."""
        from emet.habitat.hmeqa_enrich_labels import parse_enrich_label_text

        self._enrich_object_hints = parse_enrich_label_text(labels)

    def extract_relevant_objects(self, question: str) -> None:
        """Extract keywords from the question for image selection (same idea as DynaMem)."""
        if self._question == question:
            return
        self._obs_nav_dists.clear()
        self._question = question
        prompt = (
            "Assume there is an agent doing Question Answering in an environment. "
            "When it receives a question, tell the agent few objects (preferably 1-3) to pay attention to. "
            "Example: Where is the pen? -> pen. Is there grey cloth on cloth hanger? -> grey cloth, cloth hanger"
        )
        out = self.image_description_client([prompt, question])
        enrich_hints = getattr(self, "_enrich_object_hints", None) or []
        llm_parts = [s.strip() for s in out.split(",") if s.strip()]
        mcq_landmarks = location_mcq_landmark_phrases(question)
        phrase_seed = list(enrich_hints) + heuristic_relevant_phrases(question) + list(mcq_landmarks)
        if enrich_hints:
            for hint in enrich_hints:
                h = hint.strip().lower()
                if h and " " in h and h not in phrase_seed:
                    phrase_seed.insert(0, h)
        extra_seed = llm_parts + heuristic_relevant_objects(question) + list(mcq_landmarks)
        # Location MCQs need stem object + option landmarks in the same recall set.
        max_items = 8 if mcq_landmarks else 4
        phrases, objects = consolidate_relevant_keywords(phrase_seed, extra_seed, max_items=max_items)
        self._relevant_phrases = phrases
        self._relevant_objects = objects

    def set_confirmed_memory_siglip_encoder(self, encoder: Any | None) -> None:
        """Attach a SigLIP encoder used only for CONFIRMED_MEMORY (survives voxel encoder drop)."""
        self._confirmed_memory_siglip_encoder = encoder

    def refresh_siglip_confirmed_memory(self) -> None:
        """Encode new graph observation RGBs and refresh phrase→best-view alignments."""
        if not self.memory_summary_enabled:
            return
        enc = self._confirmed_memory_siglip_encoder
        if enc is None:
            return
        from emet.memory.graph_eqa.graph_eqa_siglip import (
            align_phrase_to_observation_features,
            encode_observation_rgb,
        )

        for obs in self._observations:
            oid = int(obs.obs_id)
            if oid in self._obs_siglip_features:
                continue
            feat = encode_observation_rgb(enc, obs.rgb)
            if feat is not None:
                self._obs_siglip_features[oid] = feat
        phrases = self._confirmed_memory_phrases()
        for phrase in phrases:
            match = align_phrase_to_observation_features(
                phrase,
                enc,
                self._observations,
                self._obs_siglip_features,
            )
            if match is not None:
                self._siglip_phrase_cache[phrase.strip().lower()] = match

    def _node_for_obs(self, obs_id: int) -> GraphNode | None:
        return next(
            (node for node in self._nodes if int(node.obs_id) == int(obs_id) and not node.is_viewpoint),
            None,
        )

    def _answerability_gain_for_obs(self, question: str, obs_id: int, phrase: str) -> float:
        obs = self._observation_by_id(int(obs_id))
        labels = list(obs.labels or []) if obs is not None else []
        target_hit = any(label_matches_relevant_object(phrase, label) for label in labels)
        try:
            from emet.habitat.metrics import parse_mcq_choices_from_question

            choices = parse_mcq_choices_from_question(question)
        except Exception:
            choices = []
        if not choices:
            return 1.0 if target_hit else 0.25
        landmark_hit = any(label_matches_relevant_object(choice, label) for choice in choices for label in labels)
        if target_hit and landmark_hit:
            return 1.0
        if target_hit or landmark_hit:
            return 0.55
        return 0.15

    def _recall_rank_score(
        self,
        hypothesis: NavHypothesis,
        question: str,
        robot_xyt: np.ndarray | None,
    ) -> NavHypothesis:
        """Cheap recall key for top-K packing (not a VLM decision policy)."""
        answerability = self._answerability_gain_for_obs(
            question,
            hypothesis.obs_id,
            hypothesis.phrase,
        )
        # Map answerability to a small recall boost (landmark/target label hits).
        if answerability >= 1.0:
            hit_boost = 20.0
        elif answerability >= 0.55:
            hit_boost = 10.0
        else:
            hit_boost = 0.0
        path_cost = 0.0
        if robot_xyt is not None and np.asarray(robot_xyt).size >= 2:
            path_cost = float(
                np.linalg.norm(
                    np.asarray(hypothesis.xyz, dtype=float)[:2] - np.asarray(robot_xyt, dtype=float).reshape(-1)[:2]
                )
            )
        tier = float(_RECALL_SOURCE_TIER.get(str(hypothesis.source), 0.0))
        # Path is a weak tiebreak only (cm-scale in the key).
        total = tier + hit_boost - 0.01 * path_cost
        return replace(
            hypothesis,
            score=total,
            answerability_gain=answerability,
            belief_reduction=0.0,
            revisit_change_value=0.0,
            path_cost=path_cost,
            failure_risk=0.0,
        )

    @staticmethod
    def _pack_diversified_hypotheses(
        scored: list[NavHypothesis],
        max_k: int,
    ) -> list[NavHypothesis]:
        """Pack top-K with source diversity (graph + frontier when both exist)."""
        k = max(1, int(max_k))
        if not scored:
            return []
        picked: list[NavHypothesis] = []
        seen: set[int] = set()

        def _take_from(sources: tuple[str, ...]) -> None:
            for h in scored:
                oid = int(h.obs_id)
                if oid in seen:
                    continue
                if str(h.source) in sources:
                    picked.append(h)
                    seen.add(oid)
                    return

        # Seed diversity: one graph, one siglip/confirmed, one frontier when present.
        _take_from(("graph",))
        _take_from(("confirmed", "siglip"))
        _take_from(("frontier",))
        for h in scored:
            if len(picked) >= k:
                break
            oid = int(h.obs_id)
            if oid not in seen:
                picked.append(h)
                seen.add(oid)
        return picked[:k]

    def hypothesize_nav_targets(
        self,
        question: str,
        max_k: int = 6,
        robot_xyt: np.ndarray | None = None,
    ) -> list[NavHypothesis]:
        """Retrieve a small diversified set of nav evidence cards for the router/fallback.

        Ranking is **recall only** (source tier + keyword/landmark hit + distance
        tiebreak). The VLM router decides where to go among the returned cards.
        """
        if not self._observations and not any(getattr(n, "is_frontier", False) for n in self._nodes):
            return []
        phrases = list(self._confirmed_memory_phrases()) + list(self._relevant_objects or [])
        if not phrases and question:
            self.extract_relevant_objects(question)
            phrases = list(self._confirmed_memory_phrases()) + list(self._relevant_objects or [])
        # Always merge location-MCQ landmarks (even if extract already ran thin).
        for landmark in location_mcq_landmark_phrases(question):
            if landmark not in phrases:
                phrases.append(landmark)
        if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
            cand_labels = [
                (
                    int(n.obs_id),
                    str(n.labels)[:40],
                    bool(getattr(n, "is_viewpoint", False)),
                    bool(getattr(n, "is_frontier", False)),
                )
                for n in self._nodes
            ]
            _logger.info(
                "[recall] q=%r phrases=%r n_obs=%d n_nodes=%d all_nodes=%s",
                question[:40],
                phrases[:6],
                len(self._observations),
                len(self._nodes),
                cand_labels,
            )
        scored: list[NavHypothesis] = []
        seen: set[int] = set()
        retracted = getattr(self, "_retracted_nav_claims", None) or set()

        def _claim_blocked(oid: int, phrase: str) -> bool:
            key = (int(oid), str(phrase or "").strip().lower())
            return key in retracted

        # Object / SigLIP cards need phrases; frontiers are still valid cold-start evidence.
        for phrase in phrases:
            for o in self._observations:
                oid = int(o.obs_id)
                if oid in seen or _claim_blocked(oid, phrase):
                    continue
                if self._obs_is_frontier(oid):
                    continue
                # Viewpoint-only / camera-station obs are not place cards.
                if not self._obs_is_object_place(oid):
                    continue
                if any(label_matches_relevant_object(phrase, lab) for lab in (o.labels or [])):
                    seen.add(oid)
                    scored.append(
                        NavHypothesis(
                            phrase=phrase,
                            obs_id=oid,
                            xyz=np.asarray(o.xyz, dtype=float).reshape(-1)[:3].copy(),
                            score=0.0,
                            source="graph",
                        )
                    )
            # Also match graph nodes (centroid) when observations lack the label string.
            for node in self._nodes:
                if getattr(node, "is_frontier", False) or getattr(node, "is_viewpoint", False):
                    continue
                oid = int(node.obs_id)
                if oid in seen or _claim_blocked(oid, phrase):
                    continue
                if any(label_matches_relevant_object(phrase, lab) for lab in (node.labels or [])):
                    seen.add(oid)
                    scored.append(
                        NavHypothesis(
                            phrase=phrase,
                            obs_id=oid,
                            xyz=np.asarray(node.xyz, dtype=float).reshape(-1)[:3].copy(),
                            score=0.0,
                            source="graph",
                        )
                    )
        for phrase in phrases:
            sig = self._siglip_match_for_phrase(phrase)
            if sig is None:
                continue
            sim, xyz, oid = float(sig[0]), np.asarray(sig[1], dtype=float), sig[2]
            if oid is None:
                continue
            oid = int(oid)
            if oid in seen or self._obs_is_frontier(oid) or _claim_blocked(oid, phrase):
                continue
            if not self._obs_is_object_place(oid):
                continue
            if sim >= SIGLIP_CONFIRM_THRESHOLD:
                source = "confirmed"
            elif sim >= SIGLIP_PRESENT_THRESHOLD:
                source = "siglip"
            else:
                continue
            seen.add(oid)
            scored.append(
                NavHypothesis(
                    phrase=phrase,
                    obs_id=oid,
                    xyz=xyz.reshape(-1)[:3].copy(),
                    score=0.0,
                    source=source,
                    siglip_sim=float(sim),
                )
            )
        for node in self._nodes:
            if not node.is_frontier or int(node.obs_id) in seen:
                continue
            # Do NOT attach the question object as the frontier phrase — that made
            # every frontier look like a "fruit bowl" hit and drowned graph places.
            scored.append(
                NavHypothesis(
                    phrase="unexplored frontier",
                    obs_id=int(node.obs_id),
                    xyz=np.asarray(node.xyz, dtype=float).copy(),
                    score=0.0,
                    source="frontier",
                )
            )
            seen.add(int(node.obs_id))
        if not scored:
            return []
        scored = [self._recall_rank_score(hypothesis, question, robot_xyt) for hypothesis in scored]
        scored.sort(key=lambda h: (-h.score, h.path_cost, -h.obs_id))
        return self._pack_diversified_hypotheses(scored, max_k)

    def retire_frontier_obs(self, obs_id: int) -> bool:
        """Drop a frontier node after visit — visited space is not a frontier."""
        oid = int(obs_id)
        drop_nodes: set[int] = set()
        frontier_ids: set[str] = set()
        from emet.memory.graph_eqa.frontier_nodes import FRONTIER_DESC_PREFIX

        for n in self._nodes:
            if n.is_frontier and int(n.obs_id) == oid:
                drop_nodes.add(int(n.node_id))
                desc = str(n.description or "")
                if desc.startswith(FRONTIER_DESC_PREFIX):
                    frontier_ids.add(desc[len(FRONTIER_DESC_PREFIX) :])
        if not drop_nodes:
            return False
        for frontier_id in frontier_ids:
            self.world_evidence.set_frontier_status(frontier_id, "visited")
        self._nodes = [n for n in self._nodes if int(n.node_id) not in drop_nodes]
        self._observations = [o for o in self._observations if int(o.obs_id) != oid]
        for i, n in enumerate(self._nodes, start=1):
            self._nodes[i - 1] = replace(n, node_id=i)
        self._reindex_world_entities()
        self._rebuild_viewpoint_index()
        self._update_edges()
        return True

    def frontier_id_near_xy(self, xyz: Any, *, max_dist_m: float = 1.5) -> str:
        track = self.world_evidence.frontier_near_xyz(
            xyz,
            max_dist_m=max_dist_m,
        )
        return track.frontier_id if track is not None else ""

    def retract_phrase_claim_at_obs(
        self,
        obs_id: int,
        phrase: str,
        *,
        strip_matching_labels: bool = True,
        strip_across_obs: bool = False,
        apply_blacklist: bool = True,
        evidence_obs_id: int | None = None,
        evidence_source: str = "vlm",
        room: str | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Stop offering a disproved stem-object claim without deleting the place.

        After a close look verifies ABSENT for ``phrase`` at ``obs_id``, blacklist
        that (obs, phrase) for hyp recall and optionally strip matching labels from
        the observation / node. Location-MCQ *place* landmarks should not call this
        for the place name itself (the island is real; only the object was missing).

        ``strip_across_obs`` (default False): cross-view stripping is opt-in and
        corroborated-only — a closer look at one view disproving the object should NOT
        strip the label from nodes at OTHER views unless ABSENT is corroborated at 2+
        distinct views (callers that prove corroboration pass True). One weak glance
        must not delete a node seen elsewhere (exp1 regression fix).

        ``evidence_obs_id`` identifies the fresh view that produced ABSENT. It may
        differ from ``obs_id``, which remains the navigation/place claim being
        retracted.
        """
        oid = int(obs_id)
        key_phrase = str(phrase or "").strip().lower()
        if not key_phrase:
            return {"ok": False, "error": "empty phrase", "obs_id": oid}
        if not hasattr(self, "_retracted_nav_claims"):
            self._retracted_nav_claims = set()
        if apply_blacklist:
            self._retracted_nav_claims.add((oid, key_phrase))
        if not hasattr(self, "_retraction_evidence_views"):
            self._retraction_evidence_views = set()
        evidence_oid = int(evidence_obs_id) if evidence_obs_id is not None else oid
        self._retraction_evidence_views.add((evidence_oid, key_phrase))
        source = str(evidence_source or "unknown").strip().lower()[:40]
        stripped_obs = 0
        stripped_nodes = 0
        if strip_matching_labels:
            for o in self._observations:
                if not strip_across_obs and int(o.obs_id) != oid:
                    continue
                before = list(o.labels or [])
                kept = [lab for lab in before if not label_matches_relevant_object(key_phrase, lab)]
                if len(kept) != len(before):
                    o.labels = kept if kept else ["object"]
                    stripped_obs += 1
            for i, n in enumerate(self._nodes):
                if getattr(n, "is_frontier", False) or getattr(n, "is_viewpoint", False):
                    continue
                if not strip_across_obs and int(n.obs_id) != oid:
                    continue
                before = list(n.labels or [])
                kept = [lab for lab in before if not label_matches_relevant_object(key_phrase, lab)]
                if len(kept) != len(before):
                    self._nodes[i] = replace(
                        n,
                        labels=kept if kept else ["object"],
                    )
                    stripped_nodes += 1
        room_ev = self.record_room_event(
            room=room,
            kind="verify_absent",
            step=step,
            phrase=key_phrase,
            obs_id=evidence_oid,
            note=f"{source} absent; claim_obs={oid}",
        )
        room_label = str((room_ev or {}).get("room") or "")
        # Persist ABSENT as a verify attempt when the ledger is on (does not change
        # per-view semantics — ABSENT is not scene-wide proof of absence).
        self.record_attempt(
            action_kind="verify",
            outcome="absent",
            status_code=f"{source}_absent",
            note=f"negative evidence {key_phrase!r}: claim_obs={oid} evidence_obs={evidence_oid}",
            obs_id=evidence_oid,
            phrase=key_phrase,
            source="eqa",
            room=room_label,
            step=step,
        )
        return {
            "ok": True,
            "obs_id": oid,
            "claim_obs_id": oid,
            "evidence_obs_id": evidence_oid,
            "evidence_source": source,
            "phrase": key_phrase,
            "stripped_obs": stripped_obs,
            "stripped_nodes": stripped_nodes,
            "n_retracted": len(self._retracted_nav_claims),
            "recorded_only": not apply_blacklist and not strip_matching_labels,
            "room": room_label or None,
        }

    def has_absent_retraction_at_other_view(self, phrase: str, evidence_obs_id: int) -> bool:
        """Return whether the phrase was already ABSENT at a different evidence view."""
        key_phrase = str(phrase or "").strip().lower()
        evidence_oid = int(evidence_obs_id)
        views = getattr(self, "_retraction_evidence_views", None) or set()
        return any(
            int(view_id) != evidence_oid and str(view_phrase).strip().lower() == key_phrase
            for view_id, view_phrase in views
        )

    def clear_retracted_nav_claims(self) -> None:
        """Drop claim blacklist (e.g. new question).

        When ``persist_absent_claims`` is on (``eqa.attempt_ledger.persist_absent_claims``
        / ``EMET_ATTEMPT_LEDGER_PERSIST_ABSENT``), keep the blacklist across questions.
        Ledger rows always persist for the graph lifetime regardless.
        """
        if self.persist_absent_claims:
            return
        self._retracted_nav_claims = set()
        self._retraction_evidence_views = set()

    def retire_frontier_near_xy(
        self,
        xy: Any,
        *,
        radius_m: float = 1.25,
    ) -> int:
        """Retire frontier nodes within ``radius_m`` of a visited explore goal."""
        try:
            pt = np.asarray(xy, dtype=float).reshape(-1)[:2]
        except Exception:
            return 0
        if pt.size < 2:
            return 0
        r2 = float(radius_m) ** 2
        drop_obs: list[int] = []
        for n in self._nodes:
            if not n.is_frontier:
                continue
            nxy = np.asarray(n.xyz, dtype=float).reshape(-1)[:2]
            if nxy.size < 2:
                continue
            if float(np.sum((nxy - pt) ** 2)) <= r2:
                drop_obs.append(int(n.obs_id))
        n_dropped = 0
        for oid in drop_obs:
            if self.retire_frontier_obs(oid):
                n_dropped += 1
        return n_dropped

    def verify_phrase_at_obs(
        self,
        phrase: str,
        obs_id: int,
        rgb: np.ndarray | None = None,
        *,
        min_sim: float | None = None,
    ) -> VerifyResult:
        """SigLIP-verify *phrase* against observation *obs_id* (optional live *rgb*)."""
        thresh = float(min_sim if min_sim is not None else SIGLIP_CONFIRM_THRESHOLD)
        oid = int(obs_id)
        text = (phrase or "").strip()
        obs = self._observation_by_id(oid)
        if obs is None and rgb is None:
            return VerifyResult(status="ABSENT", sim=0.0, obs_id=oid, phrase=text, ok=False)

        label_hit = False
        if obs is not None and text:
            label_hit = any(label_matches_relevant_object(text, lab) for lab in (obs.labels or []))

        enc = self._confirmed_memory_siglip_encoder
        text_feat: np.ndarray | None = None
        img_feat: np.ndarray | None = None
        sim = 0.0
        if enc is not None and text:
            try:
                from emet.memory.graph_eqa.graph_eqa_siglip import (
                    _feature_vector,
                    encode_observation_rgb,
                )

                text_feat = _feature_vector(enc.encode_text(text))
                if rgb is not None:
                    img_feat = encode_observation_rgb(enc, np.asarray(rgb, dtype=np.uint8))
                elif oid in self._obs_siglip_features:
                    img_feat = np.asarray(self._obs_siglip_features[oid], dtype=np.float32)
                elif obs is not None:
                    img_feat = encode_observation_rgb(enc, obs.rgb)
                    if img_feat is not None:
                        self._obs_siglip_features[oid] = img_feat
                if text_feat is not None and img_feat is not None:
                    sim = float(np.dot(text_feat, img_feat))
            except Exception as e:
                _logger.warning(f"verify_phrase_at_obs SigLIP failed: {e}")

        if sim >= thresh:
            status, ok = "PRESENT", True
        elif sim >= SIGLIP_PRESENT_THRESHOLD or label_hit:
            status, ok = "CANDIDATE", False
        elif text and img_feat is None and text_feat is None:
            # SigLIP is released before submit_answer to free VRAM for the VLM, so any
            # verify after the first submit computes no features. Reporting that as
            # ABSENT looks like real negative evidence in traces and to the loop.
            status, ok = "UNAVAILABLE", False
        else:
            status, ok = "ABSENT", False
        return VerifyResult(
            status=status,
            sim=float(sim),
            obs_id=oid,
            phrase=text,
            ok=ok,
            text_feat=text_feat,
            img_feat=img_feat,
        )

    def select_obs_ids_for_verified_answer(
        self,
        verified_obs_id: int,
        max_images: int = 1,
    ) -> list[int]:
        """Prefer the verified observation; cap at *max_images*."""
        if max_images <= 0:
            return []
        oid = int(verified_obs_id)
        if self._observation_by_id(oid) is None:
            return []
        return [oid][:max_images]

    def _confirmed_memory_phrases(self) -> list[str]:
        if self._relevant_phrases:
            return list(self._relevant_phrases)
        return list(self._relevant_objects or [])

    def _siglip_match_for_phrase(self, phrase: str) -> tuple[float, np.ndarray, int | None] | None:
        key = (phrase or "").strip().lower()
        if not key:
            return None
        cached = self._siglip_phrase_cache.get(key)
        if cached is not None:
            return cached
        grounder = self._text_grounder
        if grounder is None:
            return None
        try:
            sig = grounder(phrase)
        except Exception as e:
            _logger.warning(f"SigLIP grounder failed for {phrase!r}: {e}")
            return None
        if sig is None:
            return None
        sim, xyz = float(sig[0]), np.asarray(sig[1], dtype=float)
        return sim, xyz, None

    def _object_present_in_graph_or_siglip(self, obj: str) -> bool:
        if any(label_matches_relevant_object(obj, lab) for o in self._observations for lab in o.labels):
            return True
        sig = self._siglip_match_for_phrase(obj)
        return sig is not None and float(sig[0]) >= SIGLIP_PRESENT_THRESHOLD

    def _obs_usable_for_eqa_image(self, obs_id: int) -> bool:
        """True when ``obs_id`` may be attached as a VLM answer image.

        Frontier sync stores black 8×8 placeholders — never answer off those.
        Frontiers remain in the SCENE_GRAPH text for Action navigation targets.
        """
        if self._obs_is_frontier(int(obs_id)):
            return False
        obs = self._observation_by_id(int(obs_id))
        if obs is None:
            return False
        rgb = np.asarray(obs.rgb)
        return bool(rgb.ndim == 3 and rgb.shape[0] >= 2 and rgb.shape[1] >= 2)

    def _select_relevant_obs_ids(
        self,
        max_images: int = 6,
        choices: list[str] | None = None,
        attribute_question: bool = False,
    ) -> list[int]:
        """Select a diverse set of observation IDs for the EQA prompt (1-based).

        P2 diversification: instead of "all keyword matches then fill", build a
        prioritized pool so the VLM sees question-relevant views *and* a recent
        view *and* spatially spread context, capped at ``max_images``. Falls back
        to the most recent non-frontier observations when there are no keyword
        objects. Frontier placeholder RGB is never selected for answering.

        When ``choices`` are location MCQ options, prefer views whose labels match
        option landmarks (refrigerator, treadmill, …) *before* SigLIP nearest —
        false CONFIRMED_MEMORY coords must not steal Image 1. Count / clock / other
        questions rank stored RGB by visual FIND (DynaMem ``find_all_images`` /
        SigLIP top-k) first; YoloE class strings are leftover recall only, spread
        in XY so one cluster cannot fill the budget.

        For attribute/state questions, prefer views with lamp/light/curtain labels
        over frontiers before answering on/off or up/down.
        """
        if not self._observations:
            return []
        if max_images <= 0:
            return []
        if not self._relevant_objects:
            recent = [int(o.obs_id) for o in self._observations if self._obs_usable_for_eqa_image(o.obs_id)]
            return recent[-max_images:]

        by_id = {int(o.obs_id): o for o in self._observations}
        selected: list[int] = []

        def take(oid: int) -> bool:
            oid = int(oid)
            if oid in selected or oid not in by_id:
                return False
            if not self._obs_usable_for_eqa_image(oid):
                return False
            selected.append(oid)
            return len(selected) >= max_images

        reserved = 0
        if max_images >= 3:
            reserved = min(2, max_images - 1)
        keyword_budget = max(1, max_images - reserved)

        boost: set[str] = set()
        if choices and not attribute_question:
            for phrase in list(self._confirmed_memory_phrases()) + list(self._relevant_objects or []):
                for tok in _object_match_tokens(phrase):
                    boost |= set(_QUESTION_LANDMARK_BOOST.get(tok, frozenset()))

        def _obs_blob(o: GraphObservation) -> str:
            return " ".join(str(lab) for lab in (o.labels or [])).lower()

        def _direct_target_match(o: GraphObservation) -> bool:
            """True when a label matches the question object without relying only on aliases
            that are absent from the MCQ options (recycle bin vs refrigerator)."""
            labels = [str(lab) for lab in (o.labels or []) if lab]
            if not labels:
                return False
            phrases = list(self._relevant_objects or []) + list(self._confirmed_memory_phrases())
            choice_blob = " ".join(choices or []).lower()
            for lab in labels:
                for phrase in phrases:
                    if not label_matches_relevant_object(phrase, lab):
                        continue
                    # Direct token overlap with the question phrase/object.
                    if _object_match_tokens(phrase) & _object_match_tokens(lab):
                        return True
                    # Alias match (trash↔recycle): keep only if the label appears in options.
                    lab_toks = _object_match_tokens(lab)
                    if any(t in choice_blob for t in lab_toks):
                        return True
            return False

        from emet.habitat.metrics import choices_are_location_mcq

        location_q = bool(choices) and not attribute_question and choices_are_location_mcq(list(choices))

        # Unified Image-1 ranking for location MCQs:
        # boosted choice landmarks (fridge) > direct target (ladder) > weak aliases / generics.
        if location_q:
            scored: list[tuple[float, int]] = []
            for o in self._observations:
                oid = int(o.obs_id)
                blob = _obs_blob(o)
                if not blob.strip():
                    continue
                score = 0.0
                if _direct_target_match(o):
                    score += 10.0
                for ch in choices[:4]:
                    for tok in distinctive_choice_tokens(ch):
                        hit = tok in blob or any(lab.startswith(tok) or tok.startswith(lab) for lab in blob.split())
                        if not hit:
                            continue
                        if tok in _LANDMARK_GENERIC_TOKENS:
                            score += 0.25
                        elif tok in boost:
                            score += 12.0  # fridge for trash beats recycle-alias (+10)
                        else:
                            score += 1.0
                for tok in boost:
                    if tok in blob:
                        score += 0.5  # recycle/bin mild, not enough to beat fridge
                if score > 0:
                    scored.append((score, oid))
            scored.sort(key=lambda t: (-t[0], -t[1]))
            for _score, oid in scored[:keyword_budget]:
                if take(oid):
                    return selected

        # Visual FIND (DynaMem SigLIP top-k) before YoloE/caption labels.
        phrases = self._eqa_find_phrases()
        for oid in self._visual_find_obs_ids(phrases, max_n=keyword_budget):
            if take(oid):
                return selected

        # Target keyword / confirmed-memory label matches (extra recall).
        # Round-robin across phrases; within a noun, pick views far apart in XY.
        seen_kw: set[int] = set()
        buckets: list[list[int]] = []
        for obj in phrases:
            bucket: list[int] = []
            for o in reversed(self._observations):
                oid = int(o.obs_id)
                if oid in seen_kw:
                    continue
                if any(label_matches_relevant_object(obj, lab) for lab in o.labels):
                    bucket.append(oid)
                    seen_kw.add(oid)
            if bucket:
                buckets.append(bucket)
        keyword_hits: list[int] = []
        while buckets and len(keyword_hits) < keyword_budget:
            bucket = buckets.pop(0)
            pick = self._spread_obs_xy(keyword_hits + bucket, max_n=len(keyword_hits) + 1)
            chosen = pick[-1] if pick else bucket[0]
            if chosen not in bucket:
                chosen = bucket[0]
            keyword_hits.append(chosen)
            rest = [oid for oid in bucket if oid != chosen]
            if rest:
                buckets.append(rest)
        for oid in keyword_hits[:keyword_budget]:
            if take(oid):
                return selected

        # Attribute/state: prefer lamp/light/curtain views over frontiers for Image 1.
        if attribute_question:
            attr_tokens = (
                "lamp",
                "light",
                "lights",
                "ceiling",
                "curtain",
                "curtains",
                "window",
                "fixture",
            )
            attr_hits: list[int] = []
            for o in reversed(self._observations):
                oid = int(o.obs_id)
                if oid in selected or self._obs_is_frontier(oid):
                    continue
                blob = _obs_blob(o)
                if any(t in blob for t in attr_tokens):
                    attr_hits.append(oid)
            for oid in attr_hits:
                if take(oid):
                    return selected

        # SigLIP phrase cache (caption-independent) for targets not already selected.
        for phrase in self._confirmed_memory_phrases():
            cached = self._siglip_phrase_cache.get(phrase.strip().lower())
            if cached is None or cached[2] is None:
                continue
            if float(cached[0]) >= SIGLIP_PRESENT_THRESHOLD and take(int(cached[2])):
                return selected

        # SigLIP obs grounder per relevant object.
        obs_grounder = getattr(self, "_obs_id_grounder", None)
        if obs_grounder is not None:
            for obj in self._relevant_objects:
                try:
                    oid = obs_grounder(obj)
                except Exception:
                    oid = None
                if oid is not None and take(int(oid)):
                    return selected

        # Most recent non-frontier observation (fresh context).
        for o in reversed(self._observations):
            if take(int(o.obs_id)):
                return selected
            break

        # Spatial spread: greedily add observations farthest from those chosen.
        remaining = [
            int(o.obs_id)
            for o in self._observations
            if int(o.obs_id) not in selected and self._obs_usable_for_eqa_image(o.obs_id)
        ]
        while remaining and len(selected) < max_images:
            best_oid = None
            best_dist = -1.0
            for oid in remaining:
                cand = by_id[oid].xyz[:2]
                if selected:
                    d = min(float(np.linalg.norm(cand - by_id[s].xyz[:2])) for s in selected if s in by_id)
                else:
                    d = 0.0
                if d > best_dist:
                    best_dist = d
                    best_oid = oid
            if best_oid is None:
                break
            remaining.remove(best_oid)
            if take(best_oid):
                return selected

        return selected

    def set_text_grounder(self, grounder: Callable[[str], tuple[float, np.ndarray] | None] | None) -> None:
        """Register an open-vocab visual grounder: ``text -> (similarity, xyz) | None``.

        Backed by the voxel map's SigLIP features so existence/location can be grounded in
        pixels rather than the VLM's caption-derived node labels.
        """
        self._text_grounder = grounder

    def set_obs_id_grounder(self, grounder: Callable[[str], int | None] | None) -> None:
        """Register an open-vocab ``text -> obs_id`` selector (SigLIP-backed).

        Used by image selection to force the best-aligned observation of each relevant object
        into the VLM prompt regardless of its caption label.
        """
        self._obs_id_grounder = grounder

    def set_visual_find_fn(self, fn: Callable[..., list[Any]] | None) -> None:
        """Register DynaMem retrieve: ``phrase, max_n -> [(similarity, graph_obs_id), ...]``.

        Backed by voxel ``find_all_images`` (top-k, not argmax) mapped onto graph
        observation ids. SigLIP only proposes RGB; the VLM still reads the frames.
        """
        self._visual_find_fn = fn

    def snapshot_visual_find_ranks(self, *, question: str = "") -> None:
        """Cache top-k FIND ranks while SigLIP can still ``encode_text``.

        Habitat dynagraph calls ``prepare_dynagraph_vram_for_eqa`` after
        ``extract_relevant_objects`` and then drops GPU SigLIP so Qwen can load.
        ``query_answer`` reads this cache instead of calling ``find_all_images``.
        """
        q = str(question or self._question or "").strip()
        phrases: list[str] = []
        for raw in self._eqa_find_phrases():
            text = str(raw or "").strip()
            if text and text not in phrases:
                phrases.append(text)
        if q:
            for raw in heuristic_relevant_phrases(q) + heuristic_relevant_objects(q):
                text = str(raw or "").strip()
                if text and text not in phrases:
                    phrases.append(text)
            q_low = q.lower()
            clockish = any(
                s in q_low for s in ("what time", "time is it", "o'clock", "o’clock", "wall clock", "clock")
            )
            if clockish:
                for extra in ("clock", "wall clock"):
                    if extra not in phrases:
                        phrases.append(extra)
        if not phrases:
            return
        cache = dict(self._visual_find_rank_cache)
        visual_fn = self._visual_find_fn
        enc = self._confirmed_memory_siglip_encoder
        feats = self._obs_siglip_features or {}

        def _merge(rows: list[tuple[float, int]], sim: float, oid: int | None) -> None:
            if oid is None:
                return
            oi = int(oid)
            for i, (old_sim, old_oid) in enumerate(rows):
                if old_oid == oi:
                    if float(sim) > old_sim:
                        rows[i] = (float(sim), oi)
                    return
            rows.append((float(sim), oi))

        for phrase in phrases:
            key = phrase.strip().lower()
            if not key:
                continue
            rows: list[tuple[float, int]] = list(cache.get(key, []))
            if visual_fn is not None:
                hits: Any = []
                try:
                    hits = visual_fn(str(phrase), 12)
                except TypeError:
                    try:
                        hits = visual_fn(str(phrase))
                    except Exception:
                        hits = []
                except Exception:
                    hits = []
                for item in hits or []:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        _merge(rows, float(item[0]), int(item[1]))
                    else:
                        try:
                            _merge(rows, max(SIGLIP_PRESENT_THRESHOLD, 0.25), int(item))
                        except (TypeError, ValueError):
                            continue
            if enc is not None and feats:
                from emet.memory.graph_eqa.graph_eqa_siglip import rank_observations_for_phrase

                for sim, oid in rank_observations_for_phrase(str(phrase), enc, feats):
                    if float(sim) < SIGLIP_PRESENT_THRESHOLD:
                        break
                    _merge(rows, float(sim), int(oid))
            if rows:
                rows.sort(key=lambda t: -t[0])
                cache[key] = rows
        self._visual_find_rank_cache = cache

    def _nearest_object_neighbors(
        self,
        xyz: np.ndarray,
        *,
        exclude_node_ids: set[int] | None = None,
        max_neighbors: int = 2,
        max_dist_m: float = 3.0,
    ) -> list[tuple[Any, float]]:
        """Nearest non-frontier/viewpoint object nodes to ``xyz`` (planar XY)."""
        exclude = exclude_node_ids or set()
        anchor = np.asarray(xyz, dtype=np.float64).reshape(-1)[:2]
        scored: list[tuple[Any, float]] = []
        for n in self._nodes:
            if getattr(n, "is_frontier", False) or getattr(n, "is_viewpoint", False):
                continue
            if int(n.node_id) in exclude:
                continue
            other = np.asarray(n.xyz, dtype=np.float64).reshape(-1)[:2]
            dist = float(np.linalg.norm(anchor - other))
            if dist <= max_dist_m:
                scored.append((n, dist))
        scored.sort(key=lambda t: t[1])
        return scored[:max_neighbors]

    def _confirmed_phrase_statuses(
        self,
    ) -> dict[str, tuple[str, list[int], float | None, np.ndarray | None, int | None]]:
        """Map each confirmed-memory phrase -> (status, node_ids, sig_sim, sig_xyz, sig_obs_id).

        status is one of:
          * present       — graph label match (grounded in SCENE_GRAPH nodes)
          * candidate     — SigLIP >= PRESENT threshold, no graph match (sighted only)
          * weak_siglip   — SigLIP below PRESENT (do not treat as absence)
          * not_observed  — no graph match and no SigLIP signal
        Used by merged ``to_string(merge_confirmed=True)``. Stricter than the legacy
        summary: only graph matches are ``present``; SigLIP never asserts presence/absence.
        """
        phrases = self._confirmed_memory_phrases()
        if not phrases:
            return {}
        object_nodes = [n for n in self._nodes if not n.is_frontier and not n.is_viewpoint]
        out: dict[str, tuple[str, list[int], float | None, np.ndarray | None, int | None]] = {}
        for obj in phrases:
            matches = [
                int(n.node_id) for n in object_nodes if any(label_matches_relevant_object(obj, lab) for lab in n.labels)
            ]
            sig = self._siglip_match_for_phrase(obj)
            sim: float | None = None
            xyz: np.ndarray | None = None
            obs_id: int | None = None
            if sig is not None:
                sim = float(sig[0])
                xyz = np.asarray(sig[1], dtype=float)
                if sig[2] is not None:
                    obs_id = int(sig[2])
            # Graph label match is the only path to "present" (grounded in SCENE_GRAPH).
            # SigLIP-only stays candidate even above CONFIRM — never assert presence/absence
            # from detector scores in the answer prompt (ABSENT coloring / false presents).
            if matches:
                status = "present"
            elif sim is not None and sim >= SIGLIP_PRESENT_THRESHOLD:
                status = "candidate"
            elif sim is not None:
                status = "weak_siglip"
            else:
                status = "not_observed"
            out[obj] = (status, matches, sim, xyz, obs_id)
        return out

    def _node_room_by_id(self) -> dict[int, str]:
        """Map node_id -> stamped room-cluster name for object nodes (unknown rooms skipped)."""
        if not self._room_clusters:
            self.refresh_room_clusters()
        out: dict[int, str] = {}
        for c in self._room_clusters:
            name = str(getattr(c, "room_name", "") or "")
            if not name or name == "unknown":
                continue
            for nid in getattr(c, "node_ids", ()) or ():
                out[int(nid)] = name
        return out

    def _relevant_memory_summary(self) -> str:
        """Surface question-relevant objects as 'confirmed memory' for the VLM.

        Graph label matches are LOOK (candidate views to inspect). SigLIP matches over
        observed points are CANDIDATE / weak-SigLIP navigation hints — they catch
        mislabeled sightings but must not assert presence, absence, or a count.
        """
        if not self._confirmed_memory_phrases():
            return ""
        object_nodes = [n for n in self._nodes if not n.is_frontier and not n.is_viewpoint]
        present_thresh = SIGLIP_PRESENT_THRESHOLD
        lines: list[str] = []
        for obj in self._confirmed_memory_phrases():
            matches = [n for n in object_nodes if countable_primary_label_matches(obj, n)]
            if not matches:
                matches = [
                    n
                    for n in object_nodes
                    if any(label_matches_relevant_object(obj, lab) for lab in (n.labels or []))
                ]
            sig = self._siglip_match_for_phrase(obj)
            parts: list[str] = []
            if matches:
                parts.append("candidate views: " + format_graph_node_candidates(matches, max_nodes=6))
                parts.append(_GRAPH_CANDIDATE_COUNT_DISCLAIMER)
            sig_present = sig is not None and float(sig[0]) >= present_thresh
            if sig is not None:
                sim, xyz = float(sig[0]), sig[1]
                obs_note = f", obs_id={int(sig[2])}" if sig[2] is not None else ""
                if sig_present:
                    parts.append(f"SigLIP phrase match sim={sim:.2f} near ({xyz[0]:.1f}, {xyz[1]:.1f}){obs_note}")
                else:
                    parts.append(f"no strong SigLIP match (sim={sim:.2f})")
            # Graph label match is a FIND pointer (LOOK), not a presence/count answer.
            if matches:
                anchor_xyz = np.asarray(matches[0].xyz, dtype=np.float64)
                exclude_ids = {int(n.node_id) for n in matches}
                status = "LOOK"
            elif sig_present:
                anchor_xyz = np.asarray(sig[1], dtype=np.float64) if sig is not None else None
                exclude_ids = set()
                status = (
                    "CANDIDATE (SigLIP-only — verify in attached images before finalizing; "
                    "do not treat as confirmed present or absent)"
                )
            elif sig is not None:
                lines.append(
                    f"- {obj}: weak SigLIP only — "
                    + "; ".join(parts)
                    + " — not evidence of absence; trust attached images"
                )
                continue
            else:
                lines.append(f"- {obj}: not observed during exploration")
                continue
            if anchor_xyz is not None:
                neighbors = self._nearest_object_neighbors(
                    anchor_xyz, exclude_node_ids=exclude_ids, max_neighbors=2, max_dist_m=3.0
                )
                if neighbors:
                    near_bits = []
                    for n, dist in neighbors:
                        lab = node_display_name(n)
                        near_bits.append(f"{lab} at ({n.xyz[0]:.1f}, {n.xyz[1]:.1f}) {dist:.1f}m")
                    parts.append("nearest: " + "; ".join(near_bits))
            # Compact attempt-ledger tags for matched obs ids (opt-in; empty when off).
            attempt_bits: list[str] = []
            for n in matches[:3]:
                bit = self.attempt_summary_for_obs(int(n.obs_id), max_bits=2)
                if bit:
                    attempt_bits.append(bit)
            if sig is not None and sig[2] is not None:
                bit = self.attempt_summary_for_obs(int(sig[2]), max_bits=2)
                if bit and bit not in attempt_bits:
                    attempt_bits.append(bit)
            if attempt_bits:
                parts.append("attempts: " + " | ".join(attempt_bits[:2]))
            lines.append(f"- {obj}: {status} — " + "; ".join(parts))
        if not lines:
            return ""
        header = (
            "CONFIRMED_MEMORY (index of views to look at, not the answer. LOOK = "
            "candidate Image N to look at; CANDIDATE/weak SigLIP are navigation hints. "
            "Detector class names are proposals for WHERE to look. Identify and count "
            "from attached RGB; if images contradict memory, trust the images and keep "
            "exploring; for location MCQs, prefer option landmarks visible in Image 1 "
            "over nearest-furniture guesses):"
        )
        return header + "\n" + "\n".join(lines)

    def _graph_covers_relevant_objects(self) -> bool:
        """True when every keyword object appears in at least one graph node label."""
        eqa_cfg = self.parameters.get("eqa", {}) if hasattr(self.parameters, "get") else {}
        if isinstance(eqa_cfg, dict) and eqa_cfg.get("sqa3d_allow_partial_graph"):
            return True
        if not self._confirmed_memory_phrases() or not self._observations:
            return True
        for obj in self._confirmed_memory_phrases():
            if not self._object_present_in_graph_or_siglip(obj):
                return False
        return True

    def _target_visible_in_obs_ids(self, obs_ids: list[int]) -> bool:
        """True when a question target label appears on an attached Image 1..N view."""
        if not obs_ids:
            return False
        by_id = {int(o.obs_id): o for o in self._observations}
        phrases = list(self._confirmed_memory_phrases()) + list(self._relevant_objects or [])
        for oid in obs_ids:
            o = by_id.get(int(oid))
            if o is None:
                continue
            for phrase in phrases:
                if any(label_matches_relevant_object(phrase, lab) for lab in (o.labels or [])):
                    return True
        return False

    def _obs_is_frontier(self, obs_id: int) -> bool:
        for n in self._nodes:
            if int(n.obs_id) == int(obs_id) and n.is_frontier:
                return True
        return False

    def _obs_is_object_place(self, obs_id: int) -> bool:
        """True when ``obs_id`` anchors a real object node (not frontier/viewpoint-only)."""
        for n in self._nodes:
            if int(n.obs_id) != int(obs_id):
                continue
            if n.is_frontier or n.is_viewpoint:
                continue
            return True
        return False

    def _get_image_descriptions_str(
        self,
        obs_ids: list[int],
        *,
        omit_labels_for_obs: set[int] | None = None,
    ) -> str:
        """Build IMAGE_DESCRIPTIONS for attached EQA images only (Image 1..N).

        When ``omit_labels_for_obs`` contains an obs id (already tagged on SCENE_GRAPH
        Image-N lines), emit coords/nav suffix only so labels are not restated.
        """
        if not obs_ids:
            return "IMAGE_DESCRIPTIONS: (none)"
        skip_labels = {int(x) for x in (omit_labels_for_obs or set())}
        id_to_obs = {int(o.obs_id): o for o in self._observations}
        options: list[str] = []
        for img_idx, oid in enumerate(obs_ids, start=1):
            obs = id_to_obs.get(int(oid))
            if obs is None:
                continue
            if int(obs.obs_id) in skip_labels and self._obs_is_object_place(int(obs.obs_id)):
                line = f"Image {img_idx}. at ({obs.xyz[0]:.2f}, {obs.xyz[1]:.2f});"
            else:
                lbl = ", ".join(obs.labels) if obs.labels else "object"
                line = f"Image {img_idx}. {lbl} at ({obs.xyz[0]:.2f}, {obs.xyz[1]:.2f});"
            node = next((n for n in self._nodes if int(n.obs_id) == int(obs.obs_id)), None)
            if node is not None:
                line += self._node_nav_status_suffix(node)
            if self._obs_is_frontier(obs.obs_id):
                line += " unexplored frontier;"
            elif obs.description and "unexplored" in obs.description.lower():
                line += f" {obs.description.strip()};"
            options.append(line)
        return "IMAGE_DESCRIPTIONS: " + "\n".join(options) if options else "IMAGE_DESCRIPTIONS: (none)"

    @staticmethod
    def _coerce_eqa_confidence(raw: Any) -> bool:
        if isinstance(raw, bool):
            return raw
        if raw is None:
            return False
        s = str(raw).strip().lower().replace(" ", "")
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
        return "true" in s

    @staticmethod
    def _normalize_eqa_answer_field(raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            s = str(int(raw)) if float(raw) == int(raw) else str(raw)
        else:
            s = str(raw).strip()
        # JSON / labeled MCQ often wraps the letter in quotes or trailing punctuation.
        m = re.search(r"\b([A-Da-d])\b", s)
        if m and len(s) <= 4:
            return m.group(1).upper()
        return s.replace("\n", " ").replace("\t", " ").strip()

    @classmethod
    def _parse_answer_from_json_dict(cls, data: dict[str, Any]) -> tuple[str, str, bool, str, str] | None:
        """Map a JSON answer object onto the labeled-field tuple, or None if unusable."""
        if not isinstance(data, dict):
            return None
        # Accept common key aliases from chatty VLMs.
        key_map = {str(k).strip().lower(): k for k in data}

        def _get(*names: str) -> Any:
            for n in names:
                k = key_map.get(n)
                if k is not None:
                    return data[k]
            return None

        if _get("answer", "ans") is None and _get("reasoning", "reason") is None:
            return None
        reasoning = str(_get("reasoning", "reason") or "").strip().replace("\n", " ").replace("\t", " ")
        answer = cls._normalize_eqa_answer_field(_get("answer", "ans"))
        confidence = cls._coerce_eqa_confidence(_get("confidence", "confident"))
        action_raw = _get("action", "next_action")
        if action_raw is None:
            action = ""
        else:
            action = str(action_raw).strip().replace("\n", " ").replace("\t", " ")
        confidence_reasoning = (
            str(_get("confidence_reasoning", "confidence_reason", "conf_reasoning") or "")
            .strip()
            .replace("\n", " ")
            .replace("\t", " ")
        )
        return reasoning, answer, confidence, action, confidence_reasoning

    def parse_answer(
        self,
        answer_outputs: str,
        *,
        prefer_json: bool = True,
        json_prefill: str | None = None,
    ) -> tuple[str, str, bool, str, str]:
        """Parse mLLM output into reasoning, answer, confidence, action, confidence_reasoning.

        Tries a JSON object first (HM-EQA / chat-style contract), then the legacy labeled
        ``Reasoning:/Answer:/…`` scrape so old HISTORY / DualMem traces still work.
        """
        text = answer_outputs or ""
        if prefer_json:
            from emet.utils.json_parse import first_json_dict_lenient

            data = first_json_dict_lenient(text, prefill=json_prefill)
            if data is not None:
                parsed = self._parse_answer_from_json_dict(data)
                if parsed is not None:
                    return parsed

        # Labeled scrape is case-insensitive; strip light markdown noise.
        lowered = text.replace("*", "").replace("#", "").lower()

        def extract_between(src: str, start: str, end: str) -> str:
            pattern = re.compile(
                rf"{re.escape(start)}\s*(.*?)\s*{re.escape(end)}",
                flags=re.IGNORECASE | re.DOTALL,
            )
            m = pattern.search(src)
            if not m:
                return ""
            return m.group(1).strip().replace("\n", " ").replace("\t", " ")

        def extract_after(src: str, start: str) -> str:
            pattern = re.compile(rf"{re.escape(start)}\s*(.*)", flags=re.IGNORECASE | re.DOTALL)
            m = pattern.search(src)
            if not m:
                return ""
            return m.group(1).strip().replace("\n", " ").replace("\t", " ")

        reasoning = extract_between(lowered, "reasoning:", "answer:")
        answer = extract_between(lowered, "answer:", "confidence:")
        confidence_text = extract_between(lowered, "confidence:", "action:")
        confidence = "true" in confidence_text.replace(" ", "").lower()
        action = extract_between(lowered, "action:", "confidence_reasoning:")
        confidence_reasoning = extract_after(lowered, "confidence_reasoning:")
        if not answer.strip():
            m = re.search(r"answer\s*:\s*([a-d])\b", lowered)
            if m:
                answer = m.group(1).upper()
        if not answer.strip():
            m = re.search(r"(?:^|\n)\s*([a-d])\s*(?:\n|$)", lowered)
            if m:
                answer = m.group(1).upper()
        # Terse letter-only replies (``A}``, ``A) <choice text>``, ``A.``) that skip
        # both the JSON contract and the labeled scrape — common under a trailing
        # ``Answer:`` cue with remote/weaker VLMs.
        if not answer.strip():
            terse = extract_mcq_letter(text)
            if terse:
                answer = terse
        answer = self._normalize_eqa_answer_field(answer) if answer.strip() else answer
        return reasoning, answer, confidence, action, confidence_reasoning

    @staticmethod
    def format_eqa_history_outcome(
        *,
        answer: str,
        confidence: bool,
        action: str,
        reasoning: str,
        salvage: bool = False,
    ) -> str:
        """One-line HISTORY entry — semantic answer/outcome, not a raw model replay."""
        ans = (answer or "").strip().replace("\n", " ")[:40] or "?"
        act = (action or "").strip().replace("\n", " ")
        act_bit = ""
        if act:
            m = re.search(r"\d+", act)
            act_bit = m.group(0) if m else act[:16]
        reason = (reasoning or "").replace("\n", " ").strip()[:80]
        return (
            f"Iter: answer={ans} conf={str(bool(confidence)).lower()} "
            f"action={act_bit or '-'} salvage={1 if salvage else 0} | {reason}"
        )

    @staticmethod
    def estimate_eqa_prompt_tokens(text: str) -> int:
        from emet.llms.eqa_vl_settings import estimate_eqa_prompt_tokens

        return estimate_eqa_prompt_tokens(text)

    @classmethod
    def _truncate_scene_graph_text(cls, graph_str: str, *, drop_edges: bool, max_node_lines: int | None) -> str:
        """Trim SCENE_GRAPH edges and/or lowest-ranked (trailing) node lines for budget."""
        if not graph_str.startswith("SCENE_GRAPH"):
            return graph_str
        prefix, _, body = graph_str.partition("\n")
        if not body.strip():
            return graph_str
        lines = body.split("\n")
        node_lines: list[str] = []
        edge_lines: list[str] = []
        tail_lines: list[str] = []
        in_tail = False
        for line in lines:
            if in_tail or line.startswith(("CONFIRMED_MEMORY", "Rooms:", "GRAPH_COUNT")):
                in_tail = True
                tail_lines.append(line)
                continue
            if line.startswith("  ") and "(" in line:
                edge_lines.append(line)
            else:
                node_lines.append(line)
        if max_node_lines is not None and max_node_lines >= 0:
            # Keep highest-ranked nodes (to_string emits best-first).
            node_lines = node_lines[:max_node_lines]
        if drop_edges:
            edge_lines = []
        kept = node_lines + edge_lines + tail_lines
        return prefix + ("\n" + "\n".join(kept) if kept else "")

    @classmethod
    def _trim_confirmed_memory_block(cls, block: str, *, max_lines: int) -> str:
        if max_lines < 0 or not block:
            return block
        lines = block.split("\n")
        if len(lines) <= 1:
            return block
        head, rest = lines[0], lines[1:]
        return "\n".join([head] + rest[:max_lines])

    @classmethod
    def build_eqa_prompt_text(
        cls,
        *,
        question_line: str,
        extra_hints: list[str] | None = None,
        memory_summary: str | None = None,
        history_entries: list[str] | None = None,
        history_start_index: int = 0,
        graph_str: str,
        img_desc_str: str,
        max_tokens: int = 2500,
    ) -> list[str]:
        """Assemble EQA text blocks under an approximate token budget.

        Truncation order: oldest HISTORY → CONFIRMED_MEMORY / merged tail lines →
        SCENE_GRAPH edges → lowest-ranked SCENE_GRAPH node labels.
        """
        hints = list(extra_hints or [])
        history = list(history_entries or [])
        mem = memory_summary or ""
        graph = graph_str
        img = img_desc_str
        max_tok = int(max_tokens)

        def _parts(hist: list[str], mem_block: str, graph_block: str) -> list[str]:
            out: list[str] = [question_line]
            out.extend(hints)
            if mem_block:
                out.append(mem_block)
            out.append("HISTORY: ")
            for i, h in enumerate(hist):
                out.append("Iteration_" + str(history_start_index + i) + ":" + h)
            out.append(graph_block)
            out.append(img)
            return out

        def _tok(parts: list[str]) -> int:
            return cls.estimate_eqa_prompt_tokens("\n".join(parts))

        parts = _parts(history, mem, graph)
        if max_tok <= 0 or _tok(parts) <= max_tok:
            return parts

        # 1) Drop oldest HISTORY entries.
        while history and _tok(_parts(history, mem, graph)) > max_tok:
            history = history[1:]
            history_start_index += 1
        parts = _parts(history, mem, graph)
        if _tok(parts) <= max_tok:
            return parts

        # 2) Trim CONFIRMED_MEMORY / merged tail lines.
        if mem:
            for n in (8, 4, 2, 1, 0):
                mem = cls._trim_confirmed_memory_block(mem, max_lines=n) if n else ""
                if _tok(_parts(history, mem, graph)) <= max_tok:
                    return _parts(history, mem, graph)
        # Also trim merged-memory tail inside SCENE_GRAPH.
        if "CONFIRMED_MEMORY" in graph:
            g_lines = graph.split("\n")
            try:
                idx = next(i for i, ln in enumerate(g_lines) if ln.startswith("CONFIRMED_MEMORY"))
            except StopIteration:
                idx = -1
            if idx >= 0:
                for n_tail in (4, 2, 0):
                    head = g_lines[: idx + (0 if n_tail == 0 else 1)]
                    tail = [] if n_tail == 0 else g_lines[idx + 1 : idx + 1 + n_tail]
                    # Keep Rooms: and GRAPH_COUNT lines even when the preceding
                    # merged-memory tail is trimmed.
                    protected = [
                        ln
                        for ln in g_lines[idx + 1 :]
                        if ln.startswith(("Rooms:", "GRAPH_COUNT"))
                    ]
                    tail = [
                        ln
                        for ln in tail
                        if not ln.startswith(("Rooms:", "GRAPH_COUNT"))
                    ]
                    graph_try = "\n".join(head + tail + protected)
                    if _tok(_parts(history, mem, graph_try)) <= max_tok:
                        graph = graph_try
                        return _parts(history, mem, graph)

        # 3) Drop SCENE_GRAPH edges.
        graph = cls._truncate_scene_graph_text(graph, drop_edges=True, max_node_lines=None)
        parts = _parts(history, mem, graph)
        if _tok(parts) <= max_tok:
            return parts

        # 4) Drop lowest-ranked node labels (trailing lines after rank order).
        body_lines = graph.split("\n")[1:] if "\n" in graph else []
        n_nodes = sum(
            1
            for ln in body_lines
            if (
                ln
                and not ln.startswith("  ")
                and not ln.startswith(("CONFIRMED_MEMORY", "Rooms:", "GRAPH_COUNT"))
            )
        )
        for keep in list(range(max(0, n_nodes - 1), -1, -1)):
            graph_try = cls._truncate_scene_graph_text(graph, drop_edges=True, max_node_lines=keep)
            if _tok(_parts(history, mem, graph_try)) <= max_tok:
                return _parts(history, mem, graph_try)
        return _parts(history, mem, cls._truncate_scene_graph_text(graph, drop_edges=True, max_node_lines=0))

    def _any_confirmed_phrase_present(self) -> bool:
        for phrase in self._confirmed_memory_phrases():
            if self._object_present_in_graph_or_siglip(phrase):
                return True
        return False

    def _visibility_location_mcq_hint(self, choices: list[str]) -> str:
        lines = "\n".join(f"  {chr(65 + i)}) {choice}" for i, choice in enumerate(choices[:4]))
        return (
            "LOCATION_MCQ: The options are places, not yes/no. When the question asks "
            "'did you see … anywhere?', answer with the matching location option text for "
            "WHERE the object was observed. Do not output its A/B/C/D label. "
            "Prefer landmarks visible in the attached images of the object; "
            "treat WORKING_MEMORY / CONFIRMED_MEMORY / SCENE_GRAPH as views to look at, "
            "not as the WHERE answer. Do not cite Node N or xyz as the location. "
            "Never answer yes/no on answer:.\n"
            f"{lines}"
        )

    def _salvage_answer_letter(self, question: str, commands: list[Any]) -> str:
        """Terse follow-up when the main EQA output never produced an ``answer:`` field.

        The VLM names the semantic answer; this compatibility helper maps that text to
        the benchmark letter expected by older callers.
        """
        if self.eqa_client is None:
            return ""
        from emet.habitat.metrics import parse_mcq_choices_from_question

        choices = parse_mcq_choices_from_question(question)
        if not choices:
            return ""
        images = [c for c in commands if isinstance(c, Image.Image)]
        directive = (
            "Answer the multiple-choice question with the exact text of the best option. "
            "Do not output its A/B/C/D label. Do not caption images or explain.\n"
            f"Question: {question}"
        )
        try:
            salvage_raw = self.eqa_client([directive, *images])
        except Exception as e:
            _logger.warning(f"EQA answer salvage failed ({e})")
            return ""
        text = (salvage_raw or "").strip()
        idx = match_freeform_to_choice(text, choices)
        if idx is not None and 0 <= idx < min(len(choices), len(LETTERS)):
            return LETTERS[idx]
        # Read-only compatibility for old clients/traces that still return a letter.
        return extract_mcq_letter(text, choices)

    def _neighbor_label_blob_for_present_objects(self) -> str:
        """Concatenate nearest-furniture labels around PRESENT question objects."""
        object_nodes = [n for n in self._nodes if not n.is_frontier and not n.is_viewpoint]
        labels: list[str] = []
        for obj in self._confirmed_memory_phrases():
            matches = [n for n in object_nodes if any(label_matches_relevant_object(obj, lab) for lab in n.labels)]
            sig = self._siglip_match_for_phrase(obj)
            sig_present = sig is not None and float(sig[0]) >= SIGLIP_PRESENT_THRESHOLD
            if not matches and not sig_present:
                continue
            if matches:
                anchor_xyz = np.asarray(matches[0].xyz, dtype=np.float64)
                exclude_ids = {int(n.node_id) for n in matches}
            else:
                assert sig is not None
                anchor_xyz = np.asarray(sig[1], dtype=np.float64)
                exclude_ids = set()
            for n, _dist in self._nearest_object_neighbors(
                anchor_xyz, exclude_node_ids=exclude_ids, max_neighbors=2, max_dist_m=3.0
            ):
                labels.extend(str(lab) for lab in (n.labels or []) if lab)
        return " ".join(labels).lower()

    def _score_choices_against_label_blob(
        self,
        choices: list[str],
        blob: str,
        *,
        ignore_generic: bool = False,
    ) -> list[int]:
        """Per-option token overlap scores against a lowercase label blob."""
        blob_l = (blob or "").lower()
        scores: list[int] = []
        for ch in choices[:4]:
            tokens = distinctive_choice_tokens(ch)
            score = 0
            for t in tokens:
                if ignore_generic and t in _LANDMARK_GENERIC_TOKENS:
                    continue
                if t in blob_l:
                    score += 2
                elif any(lab.startswith(t) or t.startswith(lab) for lab in blob_l.split()):
                    score += 1
            scores.append(score)
        return scores

    def _unique_best_choice_letter(self, scores: list[int]) -> str:
        if not scores or max(scores) < 1:
            return ""
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        if len(winners) != 1:
            return ""
        return chr(65 + winners[0])

    def _any_graph_label_match_for_confirmed(self) -> bool:
        """True when at least one confirmed phrase matches a non-frontier graph/obs label."""
        object_nodes = [n for n in self._nodes if not n.is_frontier and not n.is_viewpoint]
        for obj in self._confirmed_memory_phrases():
            if any(label_matches_relevant_object(obj, lab) for n in object_nodes for lab in (n.labels or [])):
                return True
            if any(label_matches_relevant_object(obj, lab) for o in self._observations for lab in (o.labels or [])):
                return True
        return False

    def _location_letter_from_option_label_hits(self, choices: list[str]) -> str:
        """Map MCQ options onto graph/obs labels (e.g. refrigerator in graph → that letter)."""
        from emet.habitat.metrics import choices_are_location_mcq

        if not choices_are_location_mcq(choices):
            return ""
        parts: list[str] = []
        for n in self._nodes:
            if n.is_frontier or n.is_viewpoint:
                continue
            parts.extend(str(lab) for lab in (n.labels or []) if lab)
        for o in self._observations:
            parts.extend(str(lab) for lab in (o.labels or []) if lab)
        blob = " ".join(parts).lower()
        return self._unique_best_choice_letter(self._score_choices_against_label_blob(choices, blob))

    def _location_letter_from_attached_images(self, choices: list[str], obs_ids: list[int]) -> str:
        """Map MCQ options onto labels of the attached Image 1..N observations.

        Prefer Image 1 landmarks; only fall back to the full attached set when Image 1
        does not uniquely map to a choice (avoids bowl-on-table / kitchen-cabinet noise).
        """
        from emet.habitat.metrics import choices_are_location_mcq

        if not choices_are_location_mcq(choices) or not obs_ids:
            return ""
        by_id = {int(o.obs_id): o for o in self._observations}

        def _blob_for(oids: list[int]) -> str:
            parts: list[str] = []
            for oid in oids:
                o = by_id.get(int(oid))
                if o is None:
                    continue
                parts.extend(str(lab) for lab in (o.labels or []) if lab)
            return " ".join(parts).lower()

        primary = self._unique_best_choice_letter(
            self._score_choices_against_label_blob(choices, _blob_for(obs_ids[:1]), ignore_generic=True)
        )
        if primary:
            return primary
        return self._unique_best_choice_letter(
            self._score_choices_against_label_blob(choices, _blob_for(obs_ids), ignore_generic=True)
        )

    def _equipment_letter_from_target_distances(self, choices: list[str]) -> str:
        """For under-equipment MCQs, pick the option whose equipment label is closest to the target."""
        from emet.habitat.metrics import choices_are_location_mcq

        if not choices_are_location_mcq(choices):
            return ""
        # Need a target object with known xyz (graph or strong SigLIP).
        object_nodes = [n for n in self._nodes if not n.is_frontier and not n.is_viewpoint]
        anchors: list[np.ndarray] = []
        for obj in self._confirmed_memory_phrases():
            matches = [n for n in object_nodes if any(label_matches_relevant_object(obj, lab) for lab in n.labels)]
            for n in matches[:3]:
                anchors.append(np.asarray(n.xyz, dtype=np.float64).reshape(-1)[:2])
            sig = self._siglip_match_for_phrase(obj)
            if sig is not None and float(sig[0]) >= SIGLIP_CONFIRM_THRESHOLD:
                anchors.append(np.asarray(sig[1], dtype=np.float64).reshape(-1)[:2])
        if not anchors:
            return ""
        anchor = anchors[0]

        # Only apply when ≥2 options look like "under <equipment>".
        underish = sum(1 for ch in choices[:4] if "under" in (ch or "").lower())
        if underish < 2:
            return ""

        best_letter = ""
        best_dist = float("inf")
        ties = 0
        matched_options = 0
        for i, ch in enumerate(choices[:4]):
            tokens = distinctive_choice_tokens(ch)
            if not tokens:
                continue
            # Find nearest graph node matching this option's equipment tokens.
            option_hit = False
            for n in object_nodes:
                labs = [str(lab).lower() for lab in (n.labels or []) if lab]
                if not labs:
                    continue
                if not any(any(t in lab or lab.startswith(t) or t.startswith(lab) for lab in labs) for t in tokens):
                    continue
                option_hit = True
                xy = np.asarray(n.xyz, dtype=np.float64).reshape(-1)[:2]
                dist = float(np.linalg.norm(anchor - xy))
                if dist < best_dist - 1e-6:
                    best_dist = dist
                    best_letter = chr(65 + i)
                    ties = 1
                elif abs(dist - best_dist) <= 1e-6 and chr(65 + i) != best_letter:
                    ties += 1
            if option_hit:
                matched_options += 1
        # Need ≥2 equipment options grounded in the graph (bike alone must not win).
        if matched_options < 2 or ties != 1 or best_dist == float("inf"):
            return ""
        return best_letter

    def _location_letter_from_nearest_memory(self, choices: list[str]) -> str:
        """Map PRESENT nearest-furniture labels onto a location MCQ letter (no VLM).

        Used when the model answers yes/no or picks a room that conflicts with
        CONFIRMED_MEMORY neighbors (e.g. woven basket nearest armchair → D).
        Prefer graph-label matches for the question object; SigLIP-only PRESENT is
        weaker and should not override a letter that attached images support.
        """
        from emet.habitat.metrics import choices_are_location_mcq

        if not choices_are_location_mcq(choices) or not self._any_confirmed_phrase_present():
            return ""
        # Prefer direct option↔graph label hits when unique (fridge vs dining table).
        direct = self._location_letter_from_option_label_hits(choices)
        equip = self._equipment_letter_from_target_distances(choices)
        if equip:
            return equip
        blob = self._neighbor_label_blob_for_present_objects()
        nearest = self._unique_best_choice_letter(self._score_choices_against_label_blob(choices, blob))
        if direct and nearest and direct != nearest:
            # Conflict: trust option landmarks in the graph over nearest-furniture of a
            # possibly wrong SigLIP anchor when we lack a graph label on the target.
            if self._any_graph_label_match_for_confirmed():
                return nearest
            return direct
        return nearest or direct

    def _salvage_location_mcq_letter(
        self,
        question: str,
        choices: list[str],
        commands: list[Any],
    ) -> str:
        """Re-ask for semantic location text, then map it for legacy callers."""
        if self.eqa_client is None or len(choices) < 2:
            return ""
        images = [c for c in commands if isinstance(c, Image.Image)]
        memory = self._relevant_memory_summary()
        choice_lines = "\n".join(f"  {chr(65 + i)}) {choice}" for i, choice in enumerate(choices[:4]))
        stem = question.split("Answer:")[0].strip()
        directive = (
            "CONFIRMED_MEMORY lists candidate views to inspect, not the answer. "
            "This is a WHERE-did-you-see-it multiple choice question. "
            "Answer from the attached images with the exact text of the single best "
            "location option, without its A/B/C/D label. Do NOT answer yes/no or explain.\n"
        )
        if memory:
            directive += memory + "\n"
        directive += f"Question: {stem}\nOptions:\n{choice_lines}"
        try:
            salvage_raw = self.eqa_client([directive, *images])
        except Exception as e:
            _logger.warning(f"Location-MCQ salvage failed ({e})")
            return ""
        text = (salvage_raw or "").strip()
        idx = match_freeform_to_choice(text, choices)
        if idx is not None and 0 <= idx < min(len(choices), len(LETTERS)):
            return LETTERS[idx]
        return extract_mcq_letter(text, choices)

    def vote_mcq_letter(
        self,
        question: str,
        choices: list[str],
        *,
        max_votes: int = -1,
    ) -> str:
        """Debiased final MCQ letter (see mcq_debias.py).

        Two stages, both letter-token-free at the selection step:
          1. Free-form ask ("answer in a few words", no choices shown) matched to the
             closest choice by token overlap — immune to MCQ position bias.
          2. Fallback: choice-rotation voting — re-ask with cyclically rotated choice
             orders, map each reply back to the original choice index, majority-vote.

        ``max_votes`` caps stage 2 when the caller is latency-sensitive (e.g. the
        agentic forced-answer ladder at budget exhaustion); ``-1`` = unlimited.
        Returns the winning original letter, or ``""`` when neither stage finds a
        clear signal (caller keeps its main answer). Details in ``self.last_mcq_debias``.
        """
        self.last_mcq_debias = {}
        if self.eqa_client is None or len(choices) < 2:
            return ""
        n = min(4, len(choices))
        images = [
            Image.fromarray(o.rgb.astype(np.uint8), mode="RGB")
            for o in self._observations
            if o.obs_id in set(self.last_eqa_obs_ids)
        ]

        from emet.habitat.metrics import choices_are_location_mcq

        memory = ""
        if self.memory_summary_enabled and choices_are_location_mcq(choices):
            memory = self._relevant_memory_summary()
        freeform_directive = (
            "Look at the images and answer the question in a few words. "
            "Do not use option letters. Do not caption images. Do not explain.\n"
            f"Question: {question}"
        )
        if memory:
            freeform_directive = memory + "\n" + freeform_directive
        try:
            freeform = (self.eqa_client([freeform_directive, *images]) or "").strip()
        except Exception as e:
            _logger.warning(f"MCQ freeform vote failed ({e})")
            freeform = ""
        ff_idx = match_freeform_to_choice(freeform, choices[:n])
        if ff_idx is not None:
            letter = LETTERS[ff_idx]
            self.last_mcq_debias = {
                "letter": letter,
                "freeform": freeform[:300],
                "freeform_match": letter,
                "votes": [],
                "prior": None,
                "replies": [],
            }
            return letter

        prior_text = str(self.last_eqa_parsed[1] or "")
        prior_index = match_freeform_to_choice(prior_text, choices[:n])
        if prior_index is None:
            prior_index = letter_to_original_index(extract_single_letter(prior_text, n), rotation=0, n_choices=n)
        votes: list[int | None] = []
        replies: list[str] = []
        n_votes = int(max_votes) if int(max_votes) >= 0 else n
        question_stem = re.split(r"\s+A\s*[\).]\s*", question, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        for r in range(min(n, n_votes)):
            order = rotated_choice_order(n, r)
            option_lines = "\n".join(f"- {choices[idx]}" for idx in order)
            directive = (
                "Answer the multiple-choice question with the exact text of the best "
                "option. Do not output an option letter, caption images, or explain.\n"
                f"Question: {question_stem}\nOptions:\n{option_lines}"
            )
            try:
                reply = self.eqa_client([directive, *images])
            except Exception as e:
                _logger.warning(f"MCQ letter vote failed ({e})")
                reply = ""
            replies.append((reply or "").strip()[:200])
            votes.append(match_freeform_to_choice(reply, choices[:n]))
        win = tally_choice_votes(votes, choices[:n], prior_index=prior_index)
        letter = LETTERS[win] if win is not None else ""
        self.last_mcq_debias = {
            "letter": letter,
            "freeform": freeform[:300],
            "freeform_match": None,
            "votes": [None if v is None else LETTERS[v] for v in votes],
            "prior": None if prior_index is None else LETTERS[prior_index],
            "replies": replies,
        }
        return letter

    def _node_for_obs_id(self, obs_id: int) -> GraphNode | None:
        for n in self._nodes:
            if int(n.obs_id) == int(obs_id):
                return n
        return None

    def _robot_planar_xy(self, robot_xyt: Any | None) -> tuple[float, float] | None:
        if robot_xyt is None:
            return None
        r = np.asarray(robot_xyt, dtype=float).reshape(-1)
        if r.size < 2:
            return None
        return float(r[0]), float(r[1])

    def _viewpoint_xyz_for_obs(self, obs_id: int, obs: GraphObservation | None = None) -> np.ndarray | None:
        if obs is None:
            obs = self._observation_by_id(obs_id)
        if obs is not None and obs.viewer_xyz is not None:
            return np.asarray(obs.viewer_xyz, dtype=float).reshape(-1)[:3]
        vp_id = self._viewpoint_by_obs_id.get(int(obs_id))
        if vp_id is None:
            return None
        for n in self._nodes:
            if int(n.node_id) == int(vp_id) and n.is_viewpoint:
                return np.asarray(n.xyz, dtype=float).reshape(-1)[:3]
        return None

    def _standoff_waypoint_toward(
        self,
        robot_xy: tuple[float, float],
        anchor: np.ndarray,
        *,
        min_approach_m: float | None = None,
    ) -> np.ndarray:
        """Planar goal: move toward ``anchor``, stopping ``min_approach_m`` short of it.

        Habitat/navmesh snaps the goal to the nearest navigable cell; we only pick the
        geometric approach point (closest sensible XY to the object / frontier).
        """
        min_m = float(min_approach_m if min_approach_m is not None else self.image_nav_min_approach_m)
        rx, ry = robot_xy
        ax, ay = float(anchor[0]), float(anchor[1])
        dx, dy = ax - rx, ay - ry
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return np.array([rx + min_m, ry, 1.0], dtype=float)
        travel = dist if dist <= min_m else max(min_m, dist - min_m)
        ux, uy = dx / dist, dy / dist
        return np.array([rx + ux * travel, ry + uy * travel, 1.0], dtype=float)

    def _obs_nav_anchor(self, obs_id: int) -> np.ndarray | None:
        obs = self._observation_by_id(obs_id)
        if obs is None:
            return None
        node = self._node_for_obs_id(obs_id)
        if node is not None:
            return np.asarray(node.xyz, dtype=float).reshape(-1)[:3]
        return np.asarray(obs.xyz, dtype=float).reshape(-1)[:3]

    def place_footprint_for_obs(self, obs_id: int) -> Any:
        """Planar footprint for coverage / annulus sampling (``PlaceFootprint`` or None)."""
        from emet.memory.graph_eqa.place_approaches import (
            footprint_from_node,
            footprint_from_xyz,
        )

        node = self._node_for_obs_id(int(obs_id))
        fp = footprint_from_node(node)
        if fp is not None:
            return fp
        anchor = self._obs_nav_anchor(int(obs_id))
        return footprint_from_xyz(anchor) if anchor is not None else None

    def place_coverage_for_obs(
        self,
        obs_id: int,
        *,
        voxel_map: Any | None = None,
        planner: Any | None = None,
        robot_xyt: Any | None = None,
    ) -> Any:
        """Local frontier completeness for a place card (``PlaceCoverage``)."""
        from emet.memory.graph_eqa.place_approaches import (
            count_frontier_in_footprint,
            coverage_from_frontier_count,
            make_grid_converters,
        )

        fp = self.place_footprint_for_obs(int(obs_id))
        if fp is None or voxel_map is None or not hasattr(voxel_map, "get_2d_map"):
            return coverage_from_frontier_count(None)
        converters = make_grid_converters(voxel_map)
        if converters is None:
            return coverage_from_frontier_count(None)
        xy_to_ij, _ij_to_xy, res = converters
        try:
            from emet.memory.graph_eqa.frontier_nodes import _as_bool_numpy

            xyt = robot_xyt
            if xyt is None:
                return coverage_from_frontier_count(None)
            if planner is not None and hasattr(voxel_map, "get_outside_frontier"):
                outside = voxel_map.get_outside_frontier(xyt, planner)
                _, explored = voxel_map.get_2d_map()
                frontier = _as_bool_numpy(outside) & ~_as_bool_numpy(explored)
            else:
                obstacles, explored = voxel_map.get_2d_map()
                exp = _as_bool_numpy(explored)
                obs = _as_bool_numpy(obstacles)
                from scipy.ndimage import binary_dilation

                frontier = binary_dilation(exp) & ~exp & ~obs
            n = count_frontier_in_footprint(fp, frontier, xy_to_ij=xy_to_ij, resolution_m=res)
            return coverage_from_frontier_count(n)
        except Exception as e:
            _logger.warning(f"place_coverage_for_obs({obs_id}) failed: {e}")
            return coverage_from_frontier_count(None)

    def _orbit_approach_samples(
        self,
        anchor: np.ndarray,
        robot_xy: tuple[float, float] | None,
        *,
        n: int = 4,
        radius_m: float | None = None,
    ) -> list[np.ndarray]:
        """Legacy evenly spaced bearings (fallback when voxel sampling is unavailable)."""
        n_ap = max(1, int(n))
        radius = float(radius_m if radius_m is not None else max(0.85, float(self.image_nav_min_approach_m) + 0.5))
        ax, ay = float(anchor[0]), float(anchor[1])
        if robot_xy is not None:
            rx, ry = robot_xy
            base = math.atan2(ry - ay, rx - ax)
            first = self._standoff_waypoint_toward(robot_xy, anchor)
        else:
            base = 0.0
            first = np.array([ax + radius, ay, 1.0], dtype=float)
        samples: list[np.ndarray] = [np.asarray(first, dtype=float).reshape(-1)[:3].copy()]
        for k in range(1, n_ap):
            bearing = base + (2.0 * math.pi * float(k) / float(n_ap))
            samples.append(
                np.array(
                    [ax + radius * math.cos(bearing), ay + radius * math.sin(bearing), 1.0],
                    dtype=float,
                )
            )
        return samples

    def _navigation_approach_waypoint_for_obs(
        self,
        obs_id: int,
        robot_xyt: Any | None = None,
        *,
        approach_index: int = 0,
        n_approaches: int = 4,
        avoid_xy: list[tuple[float, float]] | None = None,
        voxel_map: Any | None = None,
        planner: Any | None = None,
    ) -> np.ndarray | None:
        """Sample a planar approach around the observation (annulus when map available)."""
        from emet.memory.graph_eqa.place_approaches import (
            make_grid_converters,
            sample_annulus_approach_xy,
        )

        anchor = self._obs_nav_anchor(int(obs_id))
        if anchor is None:
            return None
        robot_xy = self._robot_planar_xy(robot_xyt)
        if voxel_map is not None and hasattr(voxel_map, "get_2d_map"):
            converters = make_grid_converters(voxel_map)
            if converters is not None:
                xy_to_ij, ij_to_xy, _res = converters
                try:
                    from emet.memory.graph_eqa.frontier_nodes import _as_bool_numpy

                    obstacles, explored = voxel_map.get_2d_map()
                    obstacles_b = _as_bool_numpy(obstacles)
                    reachable = None
                    frontier = None
                    if robot_xyt is not None and planner is not None:
                        if hasattr(voxel_map, "get_reachable_map"):
                            reachable = _as_bool_numpy(voxel_map.get_reachable_map(robot_xyt, planner))
                        if hasattr(voxel_map, "get_outside_frontier"):
                            outside = voxel_map.get_outside_frontier(robot_xyt, planner)
                            frontier = _as_bool_numpy(outside) & ~_as_bool_numpy(explored)
                    xy = sample_annulus_approach_xy(
                        anchor_xy=(float(anchor[0]), float(anchor[1])),
                        robot_xy=robot_xy,
                        obstacles=obstacles_b,
                        reachable=reachable,
                        frontier=frontier,
                        footprint=self.place_footprint_for_obs(int(obs_id)),
                        xy_to_ij=xy_to_ij,
                        ij_to_xy=ij_to_xy,
                        avoid_xy=avoid_xy,
                        radius_inner_m=max(0.35, float(self.image_nav_min_approach_m)),
                        approach_index=int(approach_index),
                    )
                    if xy is not None:
                        return np.array([float(xy[0]), float(xy[1]), 1.0], dtype=float)
                except Exception as e:
                    _logger.warning(f"annulus approach sample for obs_id={obs_id} failed: {e}")
        samples = self._orbit_approach_samples(anchor, robot_xy, n=max(1, int(n_approaches)))
        idx = int(approach_index) % len(samples)
        return samples[idx]

    def _navigation_waypoint_for_obs(
        self,
        obs_id: int,
        robot_xyt: Any | None = None,
    ) -> np.ndarray | None:
        """Closest approachable planar goal for this observation.

        Always aim at the object/frontier/node anchor (node centroid when present).
        With a robot pose, stop a short standoff short of the anchor; navmesh snapping
        happens downstream. Capture ``viewer_xyz`` is evidence provenance, not a goal.
        """
        anchor = self._obs_nav_anchor(int(obs_id))
        if anchor is None:
            return None
        robot_xy = self._robot_planar_xy(robot_xyt)
        if robot_xy is not None:
            return self._standoff_waypoint_toward(robot_xy, anchor)
        return np.array([float(anchor[0]), float(anchor[1]), 1.0], dtype=float)

    def _target_point_from_image_id(
        self,
        image_id: int,
        robot_xyt: Any | None = None,
    ) -> np.ndarray | None:
        """Return ``(x, y, 1)`` nav waypoint for observation ``image_id``."""
        return self._navigation_waypoint_for_obs(int(image_id), robot_xyt)

    def _resolve_eqa_action_image_ref(self, display_index: int, obs_ids: list[int] | None) -> int | None:
        """Map ``Action: Image N`` to a graph observation id.

        Prompt-attached images are renumbered 1..K (``obs_ids`` order). SCENE_GRAPH
        lines use the raw ``[Image {obs_id}]``, so models often emit that id (e.g.
        ``Navigate to Image 19``). Accept both when a real ``GraphObservation``
        exists — not viewpoint-only nav-sample anchors without RGB history.
        """
        idx = int(display_index)
        if idx < 1:
            return None
        ids = [int(x) for x in (obs_ids or [])]
        if ids and 1 <= idx <= len(ids):
            return ids[idx - 1]
        if self._observation_by_id(idx) is not None:
            return idx
        return None

    def _target_point_from_display_image_index(
        self,
        display_index: int,
        *,
        obs_ids: list[int],
        nav_fallback_tail: list[GraphNavigationSample],
        robot_xyt: Any | None = None,
    ) -> np.ndarray | None:
        """Map 1-based ``Image N`` from the EQA prompt (or graph obs_id) to a waypoint."""
        if display_index < 1:
            return None
        oid = self._resolve_eqa_action_image_ref(display_index, obs_ids)
        if oid is not None:
            pt = self._navigation_waypoint_for_obs(oid, robot_xyt)
            if pt is not None:
                return pt
            return self._target_point_from_image_id(oid, robot_xyt)
        if nav_fallback_tail and display_index <= len(nav_fallback_tail):
            nv = nav_fallback_tail[display_index - 1]
            anchor = np.asarray(nv.xyz, dtype=float).reshape(-1)[:3]
            robot_xy = self._robot_planar_xy(robot_xyt)
            if robot_xy is not None:
                return self._standoff_waypoint_toward(robot_xy, anchor)
            return np.array([float(anchor[0]), float(anchor[1]), 1.0], dtype=float)
        return None

    def _count_candidate_nodes(self, question: str) -> tuple[list[GraphNode], CountTarget | None]:
        """Instance nodes whose primary label matches a count-MCQ target (FIND pointers)."""
        try:
            from emet.habitat.metrics import choices_are_count_mcq, parse_mcq_choices_from_question

            choices = parse_mcq_choices_from_question(question)
            if not choices or not choices_are_count_mcq(choices):
                return [], None
        except Exception:
            return [], None

        target = _count_target_from_stem(question_stem_for_keywords(question or ""))
        if target is None:
            # Anaphoric questions such as "I saw several lamps. How many are there?"
            # need the already extracted phrase rather than a bare stem token.
            for phrase in list(self._relevant_phrases) + list(self._relevant_objects or []):
                tokens = _strip_count_wrappers(_count_tokens(phrase))
                tokens = [token for token in tokens if token not in _QUESTION_STOPWORDS]
                if tokens:
                    target = CountTarget(tokens=tuple(tokens))
                    break
        if target is None:
            return [], None

        matches: list[GraphNode] = []
        target_phrases = [target.tokens, *_COUNT_PHRASE_ALIASES.get(target.tokens, ())]
        # FIND recall, not a new alias table: "table lamps" must still point at
        # detector "lamp" views so the VLM can look (q86 without close-look).
        head = target.tokens[-1] if target.tokens else ""
        if (
            len(target.tokens) > 1
            and head
            and (head in _COUNT_WORD_ALIASES or head.rstrip("s") in _COUNT_WORD_ALIASES)
        ):
            head_phrase = (head,)
            if head_phrase not in target_phrases:
                target_phrases = [*target_phrases, head_phrase]
        for node in self._nodes:
            if (
                node.is_frontier
                or node.is_viewpoint
                or is_ground_truth_node(node)
                or not node.countable_instance
            ):
                continue
            texts = finder_label_texts(node)
            if not texts:
                continue
            if any(_count_phrase_matches(phrase, text) for phrase in target_phrases for text in texts):
                matches.append(node)
        if not matches:
            return [], target

        if target.scope_tokens:
            room_by_id = self._node_room_by_id()
            if room_by_id and all(int(node.node_id) in room_by_id for node in matches):
                scoped = [
                    node
                    for node in matches
                    if _count_phrase_matches(target.scope_tokens, room_by_id[int(node.node_id)])
                ]
                if scoped:
                    matches = scoped
                # Scope miss (q93: dining stools vs "kitchen counter"): keep
                # scene-wide FIND views. List length is still not a count.
        unique: dict[str, GraphNode] = {}
        for node in matches:
            key = str(node.identity_key).strip() if node.identity_key else f"node:{int(node.node_id)}"
            unique.setdefault(key, node)
        return _collapse_count_nodes_spatially(list(unique.values())), target

    def _graph_count_hint(self, question: str) -> str:
        """List views to inspect for count MCQs; never assert an exact integer."""
        matches, target = self._count_candidate_nodes(question)
        phrase = target.phrase if target is not None else "target"
        visual = self._visual_find_obs_ids(self._eqa_find_phrases(), max_n=6)
        if visual:
            labeled = "; ".join(f"[Image {int(oid)}]" for oid in visual)
            return (
                f"GRAPH_COUNT: views to look at for '{phrase}' "
                f"(go look at these Image ids; not an exact count): {labeled}. "
                "Count from attached images after looking; "
                "do not use this list length as the answer."
            )
        if not matches:
            return ""
        scope_note = ""
        if target is not None and target.scope_tokens:
            room_by_id = self._node_room_by_id()
            if not (room_by_id and all(int(node.node_id) in room_by_id for node in matches)):
                scope_note = (
                    f" Scope '{target.scope_phrase}' is not grounded in the graph; "
                    "look at these scene-wide candidate views."
                )
        labeled = format_graph_node_candidates(matches, max_nodes=6)
        return (
            f"GRAPH_COUNT: views to look at for '{phrase}' "
            f"(go look at these Image ids; not an exact count): {labeled}."
            f"{scope_note} Count from attached images after looking; "
            "do not use this list length as the answer."
        )

    def _location_finder_nodes(self) -> list[GraphNode]:
        """Object nodes matching the question phrase (close-look name or detector primary)."""
        phrases = [p for p in self._confirmed_memory_phrases() if str(p).strip()]
        for obj in list(self._relevant_objects or []):
            if obj and obj not in phrases:
                phrases.append(str(obj))
        if not phrases:
            return []
        matches: list[GraphNode] = []
        seen_obs: set[int] = set()
        for node in self._nodes:
            if node.is_frontier or node.is_viewpoint or is_ground_truth_node(node):
                continue
            oid = int(node.obs_id)
            if oid in seen_obs:
                continue
            texts = finder_label_texts(node)
            if not texts:
                continue
            hit = False
            for phrase in phrases:
                tokens = tuple(_strip_count_wrappers(_count_tokens(phrase)))
                for text in texts:
                    if label_matches_relevant_object(phrase, text):
                        hit = True
                        break
                    if tokens and _count_phrase_matches(tokens, text):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                seen_obs.add(oid)
                matches.append(node)
        return matches

    def _compose_eqa_answer_obs_ids(
        self,
        *,
        forced: list[int],
        pin_obs: list[int],
        selected: list[int],
        max_images: int,
        count_question: bool,
        look_obs_id: int | None,
    ) -> list[int]:
        """Build Image 1..K from stored views: question-relevant RGB first.

        Image 1 is a pin / FIND / keyword view, not a leftover hallway look.
        Remaining slots mix other recalled views so one FIND noun cannot occupy
        the whole ``max_images`` budget. ``count_question`` keeps FIND pins ahead
        of unverified force_obs when both exist.
        """
        if max_images <= 0:
            return []

        def _take(dst: list[int], src: list[int | None]) -> None:
            seen = set(dst)
            if len(dst) >= max_images:
                return
            for raw in src:
                if raw is None:
                    continue
                oid = int(raw)
                if oid in seen:
                    continue
                if not self._obs_usable_for_eqa_image(oid):
                    continue
                dst.append(oid)
                seen.add(oid)
                if len(dst) >= max_images:
                    return

        look: list[int] = []
        _take(look, [look_obs_id])
        pins: list[int] = []
        _take(pins, pin_obs)
        extra: list[int] = []
        _take(extra, selected)
        out: list[int] = []
        pin_set = set(pins)
        # Verified evidence on a non-count question is Image 1 (agentic verified submit):
        # a stale Action look or location-FIND pin must not displace the confirmed evidence.
        # Count MCQs keep FIND pins ahead of force_obs so a single verified frame cannot
        # occupy the whole budget.
        if not count_question:
            _take(out, forced)
        look_relevant = bool(look and look[0] in pin_set)
        # Image 1 is a question-relevant stored view, not the last nav frame.
        # Hallway look after frontier chase must not displace clock / FIND RGB.
        if look and look_relevant:
            _take(out, look)
        elif pins:
            _take(out, pins[:1])
        elif look:
            _take(out, look)
        elif count_question:
            _take(out, extra[:1])
        for oid in extra:
            if oid not in out and oid not in pin_set:
                _take(out, [oid])
                break
        _take(out, pins)
        _take(out, extra)
        _take(out, look)
        _take(out, forced)
        return out

    def query_answer(
        self,
        question: str,
        xyt: Any | np.ndarray | list | None = None,
        planner: Any = None,
        *,
        force_obs_ids: list[int] | None = None,
    ) -> tuple[str, str, bool, str, np.ndarray | None, list[Image.Image]]:
        """
        Answer the question using the scene graph and task-relevant images.
        Same return contract as voxel_dynamem.SparseVoxelMap.query_answer.

        Args:
            force_obs_ids: When set (agentic verified submit), prefer these observation
                ids as Image 1..K instead of a pure diversified pick. Remaining slots
                fill from FIND pins and ``_select_relevant_obs_ids``. Count MCQs attach
                FIND / pending Action views first so a single verified frame cannot
                occupy the whole budget (``max_images=1`` used to drop lamp/stool RGB).

        Returns:
            reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images
        """
        import time as _time

        from emet.habitat.metrics import (
            answer_is_visibility_abstain,
            choices_are_attribute_state,
            choices_are_count_mcq,
            choices_are_location_mcq,
            choices_are_time_of_day,
            parse_mcq_choices_from_question,
            question_is_attribute_state,
            question_is_visibility_location,
        )
        from emet.llms.eqa_vl_settings import (
            get_eqa_vl_int,
            resolve_eqa_answer_format,
            resolve_eqa_answer_max_new_tokens,
            resolve_eqa_answer_prefill,
            resolve_eqa_include_image_descriptions,
            resolve_eqa_prompt_max_tokens,
            resolve_eqa_prompt_variant,
        )

        _t0 = _time.monotonic()
        _logger.info("query_answer: ensure_llm_clients…")
        self._ensure_llm_clients()
        _logger.info("query_answer: extract_relevant_objects…")
        self.extract_relevant_objects(question)
        if self.memory_summary_enabled:
            # Encoder may already be dropped by prepare_dynagraph_vram_for_eqa; refresh
            # is a no-op without it (uses cached phrase features when present).
            _logger.info("query_answer: refresh_siglip_confirmed_memory…")
            self.refresh_siglip_confirmed_memory()
        max_images = get_eqa_vl_int(self.parameters, "eqa_max_images", 4)
        include_image_descriptions = resolve_eqa_include_image_descriptions(self.parameters)
        parsed_choices = parse_mcq_choices_from_question(question)
        attribute_q = question_is_attribute_state(question) or choices_are_attribute_state(parsed_choices)
        count_q = bool(parsed_choices and choices_are_count_mcq(parsed_choices) and not attribute_q)
        time_q = bool(parsed_choices and choices_are_time_of_day(parsed_choices) and not attribute_q)
        location_q = bool(
            parsed_choices
            and choices_are_location_mcq(parsed_choices)
            and not count_q
            and not attribute_q
            and not time_q
        )
        # Honor Action:N from the previous call / post-nav look even when this submit
        # forces a different verified obs as Image 1. Do not drop the pin until
        # this call actually attaches it as Image 1 (q86 Action 163).
        look_obs_id = self.last_eqa_look_obs_id
        if look_obs_id is None:
            look_obs_id = self.last_eqa_action_obs_id
        count_nodes, _count_target = ([], None) if attribute_q else self._count_candidate_nodes(question)
        pin_obs: list[int] = []
        phrases = self._eqa_find_phrases()
        visual_pins = self._visual_find_obs_ids(phrases, max_n=max_images)
        q_low = str(question or "").lower()
        clockish = any(s in q_low for s in ("what time", "time is it", "o'clock", "o’clock", "wall clock"))
        if (
            (count_q or time_q or clockish)
            and not location_q
            and self._phrases_need_rgb_highlight(visual_pins, phrases)
        ):
            extra_hl = self._highlight_relevant_from_latest_rgb(question)
            if extra_hl:
                self._merge_highlight_phrases(extra_hl)
                phrases = self._eqa_find_phrases()
                visual_pins = self._visual_find_obs_ids(phrases, max_n=max_images)
        # Location MCQs keep landmark Image 1; visual FIND is leftover recall there.
        if not location_q:
            for oid in visual_pins:
                if oid not in pin_obs and self._obs_usable_for_eqa_image(oid):
                    pin_obs.append(int(oid))
        # YoloE instance nodes are FIND only when SigLIP retrieved nothing.
        if not pin_obs:
            for node in count_nodes:
                oid = int(node.obs_id)
                if oid in pin_obs:
                    continue
                if self._obs_usable_for_eqa_image(oid):
                    pin_obs.append(oid)
        if not pin_obs:
            for node in self._location_finder_nodes()[:4]:
                oid = int(node.obs_id)
                if oid in pin_obs:
                    continue
                if self._obs_usable_for_eqa_image(oid):
                    pin_obs.append(oid)
        pin_obs = self._spread_obs_xy(pin_obs, max_n=max_images)
        forced: list[int] = []
        if force_obs_ids:
            for oid in force_obs_ids:
                oi = int(oid)
                if oi in forced:
                    continue
                if self._obs_usable_for_eqa_image(oi):
                    forced.append(oi)
        selected = [
            int(oid)
            for oid in self._select_relevant_obs_ids(
                max_images=max_images,
                choices=parsed_choices if parsed_choices else None,
                attribute_question=attribute_q,
            )
            if self._obs_usable_for_eqa_image(oid)
        ]
        obs_ids = self._compose_eqa_answer_obs_ids(
            forced=forced,
            pin_obs=pin_obs,
            selected=selected,
            max_images=max_images,
            count_question=count_q,
            look_obs_id=look_obs_id,
        )
        crop_oid: int | None = None
        crop_rgb: np.ndarray | None = None
        if max_images >= 2:
            crop_oid = self._eqa_pick_closeup_obs_id(obs_ids, look_obs_id, pin_obs)
            if crop_oid is not None:
                crop_rgb = self._eqa_crop_for_obs(crop_oid)
            if crop_rgb is None:
                crop_oid = None
            elif len(obs_ids) >= max_images:
                obs_ids = self._eqa_reserve_closeup_slot(obs_ids, pin_obs, look_obs_id)
                if crop_oid not in obs_ids:
                    crop_oid = self._eqa_pick_closeup_obs_id(obs_ids, look_obs_id, pin_obs)
                    crop_rgb = self._eqa_crop_for_obs(crop_oid) if crop_oid is not None else None
                    if crop_rgb is None:
                        crop_oid = None
        self.last_eqa_obs_ids = list(obs_ids)
        action_obs_ids = list(obs_ids)
        if crop_oid is not None:
            action_obs_ids.append(int(crop_oid))
        if look_obs_id is not None and (not obs_ids or int(obs_ids[0]) != int(look_obs_id)):
            self.last_eqa_look_obs_id = int(look_obs_id)
        else:
            self.last_eqa_look_obs_id = None
        max_graph_nodes = get_eqa_vl_int(self.parameters, "eqa_max_graph_nodes", 48)
        merged_memory = self._merged_memory_enabled()
        merge_confirmed = (
            merged_memory and self.memory_summary_enabled and not attribute_q and not self._spatial_rag_enabled()
        )
        graph_str = self.to_string(
            max_object_nodes=max_graph_nodes if max_graph_nodes > 0 else None,
            question_keywords=list(self._relevant_objects or []),
            prefer_obs_ids=obs_ids,
            record_prompt_count=True,
            merge_confirmed=merge_confirmed,
        )
        count_hint = self._graph_count_hint(question)
        # Prefer real RGB. If selection is empty (only frontier placeholders in memory),
        # fall back to navigation viewpoint samples — never attach black 8×8 frontiers.
        nav_fallback_tail: list[GraphNavigationSample] = []
        graph_obs_ids = {int(n.obs_id) for n in self._nodes if not n.is_frontier and not n.is_viewpoint}
        if obs_ids:
            if include_image_descriptions:
                img_desc_str = self._get_image_descriptions_str(
                    obs_ids,
                    omit_labels_for_obs=graph_obs_ids,
                )
            else:
                n = len(obs_ids)
                img_desc_str = (
                    f"Attached images: Image 1..{n} are full camera views; match them to SCENE_GRAPH "
                    "nodes via Image tags on nodes. Do not re-list objects from the images."
                )
        elif self._nav_samples:
            nav_fallback_tail = self._nav_samples[-max_images:]
            if include_image_descriptions:
                lines = [
                    "IMAGE_DESCRIPTIONS (navigation-only views; no object graph nodes yet):",
                ]
                for i, nv in enumerate(nav_fallback_tail, start=1):
                    tail = (
                        f" robot base (~{nv.base_xyz[0]:.2f}, {nv.base_xyz[1]:.2f})." if nv.base_xyz is not None else ""
                    )
                    lines.append(
                        f"Image {i}. viewpoint anchor at ({nv.xyz[0]:.2f}, {nv.xyz[1]:.2f}, {nv.xyz[2]:.2f});{tail}"
                    )
                img_desc_str = "\n".join(lines)
            else:
                n = len(nav_fallback_tail)
                img_desc_str = (
                    f"Attached images: Image 1..{n} are navigation-only RGB views "
                    "(no object graph nodes yet). Do not re-list objects from the images."
                )
        else:
            img_desc_str = (
                "IMAGE_DESCRIPTIONS: (none — explore for a real camera view before answering)"
                if include_image_descriptions
                else "Attached images: (none — explore for a real camera view before answering)"
            )
        if crop_oid is not None and crop_rgb is not None and obs_ids:
            try:
                parent_txt = f"Image {obs_ids.index(int(crop_oid)) + 1}"
            except ValueError:
                parent_txt = f"graph obs {int(crop_oid)}"
            n_scene = len(obs_ids)
            img_desc_str = str(img_desc_str).rstrip() + (
                f" Image {n_scene + 1} is a close-up of the object in {parent_txt} "
                "(the same instance, not another object). Count from the scene views. "
                "If a scene view is too wide to see the object, set action to that Image id "
                "for a close-up next turn."
            )

        extra_hints: list[str] = []
        if count_hint:
            extra_hints.append(count_hint)
        if crop_oid is not None:
            extra_hints.append(
                "A later attached image is a close-up of an object already shown in a scene view; "
                "do not count that close-up as a second object."
            )
        if parsed_choices and choices_are_location_mcq(parsed_choices) and question_is_visibility_location(question):
            extra_hints.append(self._visibility_location_mcq_hint(parsed_choices))
        # Attribute/state questions: answer from images; do not inject memory priors.
        # Merged-memory mode folds status into SCENE_GRAPH, so skip the separate block.
        memory_summary = ""
        if self.memory_summary_enabled and not attribute_q and not merge_confirmed:
            memory_summary = self._relevant_memory_summary() or ""
        max_history = get_eqa_vl_int(self.parameters, "eqa_max_history", 4)
        history = self._history_outputs
        start = max(0, len(history) - max_history) if max_history > 0 else 0
        history_slice = list(history[start:])
        prompt_max_tokens = resolve_eqa_prompt_max_tokens(self.parameters)
        text_blocks = self.build_eqa_prompt_text(
            question_line="Question: " + question,
            extra_hints=extra_hints,
            memory_summary=memory_summary or None,
            history_entries=history_slice,
            history_start_index=start,
            graph_str=graph_str,
            img_desc_str=img_desc_str,
            max_tokens=prompt_max_tokens,
        )
        commands: list[Any] = list(text_blocks)

        relevant_images: list[Image.Image] = []
        for oid in obs_ids:
            rgb = self._eqa_rgb_for_obs(int(oid))
            if rgb is None:
                continue
            im = Image.fromarray(rgb, mode="RGB")
            relevant_images.append(im)
            commands.append(im)
        if crop_rgb is not None:
            im = Image.fromarray(crop_rgb, mode="RGB")
            relevant_images.append(im)
            commands.append(im)
        for nv in nav_fallback_tail:
            im = Image.fromarray(nv.rgb.astype(np.uint8), mode="RGB")
            relevant_images.append(im)
            commands.append(im)
        self.last_eqa_nav_fallback_count = len(nav_fallback_tail)
        # Keep the attached frames reachable after query_answer returns so the salvage
        # counterfactual can re-ask on the same images instead of silently no-op'ing.
        self.last_relevant_images = list(relevant_images)

        _logger.info(
            f"query_answer: calling eqa_client (n_images={len(relevant_images)} "
            f"n_cmd={len(commands)} prep_s={_time.monotonic() - _t0:.1f} "
            f"include_image_descriptions={include_image_descriptions} "
            f"history_n={len(self._history_outputs)})…"
        )
        assistant_prefill: str | None = None
        answer_format = resolve_eqa_answer_format(self.parameters)
        _variant = resolve_eqa_prompt_variant(self.parameters)
        try:
            t_vl = _time.monotonic()
            ans_cap = resolve_eqa_answer_max_new_tokens(self.parameters)
            eqa_kw: dict[str, Any] = {}
            if ans_cap > 0:
                eqa_kw["max_new_tokens"] = ans_cap
            # Force the first output field so Qwen cannot open with Caption: — prompt edits
            # alone still left a 26% caption share on the 2026-07-30 q2 probe.
            assistant_prefill = resolve_eqa_answer_prefill(self.parameters)
            if assistant_prefill:
                eqa_kw["assistant_prefill"] = assistant_prefill
            _logger.info(
                f"query_answer: eqa_kw max_new_tokens={eqa_kw.get('max_new_tokens')} "
                f"assistant_prefill={assistant_prefill!r} prompt_variant={_variant!r} "
                f"answer_format={answer_format!r}"
            )
            try:
                raw = self.eqa_client(commands, **eqa_kw)
            except TypeError:
                # Older / test doubles that only accept the command list.
                raw = self.eqa_client(commands)
            _logger.info(
                f"query_answer: eqa_client done wall_s={_time.monotonic() - t_vl:.1f} out_chars={len(raw or '')}"
            )
        except Exception as exc:
            raw = f"Error: {exc}"
            self.last_eqa_raw = raw
            self.last_eqa_parsed = ("", "Unknown", False, "", str(exc))
            self.last_eqa_model_raw = raw
            self.last_eqa_model_parsed = self.last_eqa_parsed
            self._append_eqa_history(
                self.format_eqa_history_outcome(
                    answer="Unknown",
                    confidence=False,
                    action="",
                    reasoning=str(exc),
                    salvage=False,
                )
            )
            return (
                str(exc),
                "Unknown",
                False,
                str(exc),
                None,
                relevant_images,
            )
        self.last_eqa_raw = raw
        prefer_json = answer_format == "json"
        reasoning, answer, confidence, action, confidence_reasoning = self.parse_answer(
            raw or "",
            prefer_json=prefer_json,
            json_prefill=assistant_prefill if prefer_json else None,
        )
        # Also accept labeled scrape when JSON was preferred but incomplete.
        if prefer_json and not (answer or "").strip():
            reasoning, answer, confidence, action, confidence_reasoning = self.parse_answer(
                raw or "",
                prefer_json=False,
            )
        self.last_eqa_model_raw = str(raw or "")
        self.last_eqa_model_parsed = (
            reasoning,
            answer,
            bool(confidence),
            action,
            confidence_reasoning,
        )
        answer_outputs = (raw or "").replace("*", "").replace("#", "").lower()
        # Salvage: small VLMs sometimes run away captioning and never emit ``answer:``.
        # Re-ask tersely for semantic option text.
        # - Empty answer → always salvage (64-token truncation / runaway caption).
        # - ``Unknown`` on attribute/yes-no → salvage (holdout q65).
        # - Empty/``Unknown`` on location MCQ → do NOT invent a letter (holdout q104/q105);
        #   agentic should follow Action:/explore instead of memory/salvage A–D.
        # - ``Unknown`` on count MCQ → do NOT invent One/Two (q86 pin: bathroom RGB
        #   salvaged to One while GRAPH_COUNT pointed at a lamp view).
        _ans_stripped = (answer or "").strip()
        _ans_unknownish = answer_is_unknownish(_ans_stripped, parsed_choices)
        _loc_mcq = bool(parsed_choices and choices_are_location_mcq(parsed_choices) and not attribute_q)
        _count_mcq = bool(parsed_choices and choices_are_count_mcq(parsed_choices) and not attribute_q)
        # A stream that never reached ``answer:`` / ``"answer"`` was cut off mid-caption.
        _answer_field_emitted = bool(
            re.search(r"answer\s*:", answer_outputs)
            or re.search(r'["\']answer["\']\s*:', answer_outputs)
            or (prefer_json and not _ans_unknownish)
        )
        _truncated_before_answer = _ans_unknownish and not _answer_field_emitted
        # Surfaced per episode so a decode-budget regression is visible in the results
        # table instead of only showing up as a mysterious accuracy drop.
        self.last_eqa_answer_field_emitted = _answer_field_emitted
        self.last_eqa_salvage_used = False
        # Location / count MCQ: never invent A–D from a second guess, but do recover
        # a truncated stream that never emitted ``answer:``.
        _should_salvage = _ans_unknownish and (not (_loc_mcq or _count_mcq) or _truncated_before_answer)
        if _loc_mcq and _ans_unknownish and not _ans_stripped:
            # Truncated streams often omit ``answer:`` entirely (failfix5); normalize so
            # human_answer / agentic follow-up treat this as Unknown, not memory-B.
            answer = "Unknown"
            _ans_stripped = "Unknown"
            _ans_unknownish = True
        if _should_salvage:
            salvage_letter = self._salvage_answer_letter(question, commands)
            if salvage_letter:
                salvage_idx = ord(salvage_letter) - ord("A")
                salvage_text = parsed_choices[salvage_idx] if 0 <= salvage_idx < len(parsed_choices) else salvage_letter
                answer = salvage_text
                raw = (raw or "") + f"\n[salvage]\nanswer:\n{salvage_text}\n"
                self.last_eqa_raw = raw
                self.last_eqa_salvage_used = True
        elif (
            parsed_choices
            and choices_are_location_mcq(parsed_choices)
            and self._any_confirmed_phrase_present()
            and not attribute_q
        ):
            # Location letter overrides (equip → image → abstaining memory) are intentional
            # Dynagraph eval levers. Accuracy can move vs GE-only / no-override ablations;
            # always report HM-EQA deltas with the harness fingerprint + git commit.
            # Geometric under-equipment (mat under treadmill) may correct VLM guesses.
            # Image landmarks may correct memory-steered letters. Nearest-furniture memory
            # alone must NOT override a clear VLM A–D (Q6: VLM B correct, memory A) **or**
            # free-text that uniquely matches a choice ("the room with the blue curtains").
            img_letter = self._location_letter_from_attached_images(parsed_choices, obs_ids)
            equip_letter = self._equipment_letter_from_target_distances(parsed_choices)
            memory_letter = self._location_letter_from_nearest_memory(parsed_choices)
            parsed_letter = extract_mcq_letter(answer, parsed_choices)
            abstain = answer_is_visibility_abstain(answer) or not parsed_letter
            # The geometric equipment-distance guess must not override a confident VLM
            # letter: on the full-113 (2026-08-13) it flipped correct image-grounded
            # answers to wrong (q44/q47/q94/q101: json_answer == gold, scored via
            # [memory-location]). Config-gated so the legacy (always-override) behavior
            # can be reproduced for A/B (eqa.location_override_equip_gate; env
            # EMET_EQA_LOCATION_OVERRIDE_EQUIP_GATE=0 restores legacy).
            _gate_equip = self._eqa_override_gate("location_override_equip_gate", True)
            _gate_img = self._eqa_override_gate("location_override_image_gate", True)
            vlm_clear = bool(confidence) and bool(parsed_letter)
            preferred = ""
            if not (vlm_clear and _gate_equip) and equip_letter and (abstain or parsed_letter != equip_letter):
                preferred = equip_letter
            elif not (vlm_clear and _gate_img) and img_letter and (abstain or parsed_letter != img_letter):
                preferred = img_letter
            elif abstain and memory_letter and (self._any_graph_label_match_for_confirmed() or not img_letter):
                preferred = memory_letter
            # Empty / Unknown location MCQ: keep Unknown so agentic can follow Action:N.
            # Inventing A–D via memory/salvage-location caused failfix5 wrong B letters.
            if _ans_unknownish:
                pass
            elif preferred and (abstain or parsed_letter != preferred):
                preferred_idx = ord(preferred) - ord("A")
                preferred_text = (
                    parsed_choices[preferred_idx] if 0 <= preferred_idx < len(parsed_choices) else preferred
                )
                answer = preferred_text
                raw = (raw or "") + f"\n[memory-location]\nanswer:\n{preferred_text}\n"
                self.last_eqa_raw = raw
            elif abstain:
                # Visibility-style Yes/No on a WHERE question may still salvage; empty/Unknown
                # already handled above. Do not salvage bare abstains without a letter.
                if answer_is_visibility_abstain(answer) and not _ans_unknownish:
                    salvage_letter = self._salvage_location_mcq_letter(question, parsed_choices, commands)
                    if salvage_letter:
                        salvage_idx = ord(salvage_letter) - ord("A")
                        salvage_text = (
                            parsed_choices[salvage_idx] if 0 <= salvage_idx < len(parsed_choices) else salvage_letter
                        )
                        answer = salvage_text
                        raw = (raw or "") + f"\n[salvage-location]\nanswer:\n{salvage_text}\n"
                        self.last_eqa_raw = raw
        self.last_eqa_model_confident = bool(confidence)
        covered = self._graph_covers_relevant_objects()
        if confidence and not covered:
            confidence = False
            confidence_reasoning = (
                confidence_reasoning
                + " The scene graph does not yet include all question-relevant objects; explore further."
            ).strip()
        # Do not finalize Yes/No from absence while relevant objects are still uncovered.
        if (
            not covered
            and answer_is_visibility_abstain(answer)
            and not (parsed_choices and choices_are_location_mcq(parsed_choices))
        ):
            confidence = False
            confidence_reasoning = (
                confidence_reasoning
                + " Yes/No from missing evidence is not final; keep exploring until objects are observed."
            ).strip()
        # Abstentions are never confirmed MCQ answers. A count option whose
        # semantic text is ``None`` is a real answer, not an abstention.
        if answer_is_unknownish(str(answer or ""), parsed_choices):
            confidence = False
            confidence_reasoning = (
                confidence_reasoning + " No clear letter yet; explore and refresh memory before confirming."
            ).strip()
        # Require a clear picture: don't confirm location letters unsupported by attached
        # image labels when memory is only SigLIP-candidate (no graph label on the target).
        if (
            confidence
            and parsed_choices
            and choices_are_location_mcq(parsed_choices)
            and not attribute_q
            and not self._any_graph_label_match_for_confirmed()
        ):
            parsed_letter = extract_mcq_letter(str(answer or ""), parsed_choices)
            img_letter = self._location_letter_from_attached_images(parsed_choices, obs_ids)
            if parsed_letter and img_letter and parsed_letter != img_letter:
                confidence = False
                confidence_reasoning = (
                    confidence_reasoning
                    + " Answer conflicts with landmarks in attached images; update memory / views before confirming."
                ).strip()
            elif parsed_letter and not img_letter:
                confidence = False
                confidence_reasoning = (
                    confidence_reasoning + " Location not yet verified in attached images; explore for a clearer view."
                ).strip()
        # Never finalize a WHERE answer if the target object is not in attached views
        # (guessing "dining table" / "side table" without seeing towel/fruit bowl).
        if (
            confidence
            and parsed_choices
            and choices_are_location_mcq(parsed_choices)
            and not attribute_q
            and not self._target_visible_in_obs_ids(obs_ids)
        ):
            confidence = False
            confidence_reasoning = (
                confidence_reasoning + " Target object not visible in attached images; explore before confirming."
            ).strip()
        # Under-equipment MCQs: do not finalize until geometric equipment letter is known
        # (otherwise bike vs treadmill is a coin flip from a partial gym view).
        if confidence and parsed_choices and choices_are_location_mcq(parsed_choices) and not attribute_q:
            underish = sum(1 for ch in parsed_choices[:4] if "under" in (ch or "").lower())
            if underish >= 2:
                equip = self._equipment_letter_from_target_distances(parsed_choices)
                if not equip:
                    confidence = False
                    confidence_reasoning = (
                        confidence_reasoning
                        + " Under-equipment location needs a clearer mat↔equipment distance before confirming."
                    ).strip()
                else:
                    parsed_letter = extract_mcq_letter(str(answer or ""), parsed_choices)
                    if parsed_letter and parsed_letter != equip:
                        equip_idx = ord(equip) - ord("A")
                        equip_text = parsed_choices[equip_idx] if 0 <= equip_idx < len(parsed_choices) else equip
                        answer = equip_text
                        raw = (raw or "") + f"\n[equipment-location]\nanswer:\n{equip_text}\n"
                        self.last_eqa_raw = raw
        # Attribute/state: never finalize from memory priors (images only).
        if confidence and attribute_q and self.memory_summary_enabled:
            # Soft gate: if Image 1 is a frontier-only view, keep exploring.
            if obs_ids and self._obs_is_frontier(int(obs_ids[0])):
                confidence = False
                confidence_reasoning = (
                    confidence_reasoning + " Attribute/state needs a non-frontier view of the object before confirming."
                ).strip()
        missing_find: list[int] = []
        if _count_mcq and count_nodes:
            attached = {int(oid) for oid in obs_ids}
            attached_find = False
            for node in count_nodes:
                oid = int(node.obs_id)
                if not self._obs_usable_for_eqa_image(oid):
                    continue
                if oid in attached:
                    attached_find = True
                elif oid not in missing_find:
                    missing_find.append(oid)
            if (
                missing_find
                and not attached_find
                and count_answer_is_none_or_zero(str(answer or ""), parsed_choices)
            ):
                if confidence:
                    confidence = False
                    confidence_reasoning = (
                        confidence_reasoning
                        + " FIND views were not attached; look at those RGB frames before answering None."
                    ).strip()
                if self.last_eqa_look_obs_id is None:
                    self.last_eqa_look_obs_id = missing_find[0]
        raw_answer = answer
        self.last_eqa_parsed = (reasoning, raw_answer, confidence, action, confidence_reasoning)
        human = format_human_eqa_answer(
            question,
            answer,
            reasoning,
            self,
            confidence=confidence,
            confidence_reasoning=confidence_reasoning,
            selected_obs_ids=obs_ids,
        )
        answer = human.user_answer
        reasoning = human.debug_reasoning

        target_point = None
        self.last_eqa_action_obs_id = None
        hist_action = ""
        display_index: int | None = None
        if action.strip():
            match = re.search(r"\d+", action.strip())
            if match:
                display_index = int(match.group())
                hist_action = action.strip()
        if not confidence and display_index is not None:
            resolved_oid = self._resolve_eqa_action_image_ref(display_index, action_obs_ids)
            if resolved_oid is not None and self.eqa_obs_look_spent(resolved_oid):
                # Still attach this RGB next turn; do not orbit the same obs.
                if self.last_eqa_look_obs_id is None:
                    self.last_eqa_look_obs_id = int(resolved_oid)
                nxt = self.next_unspent_eqa_obs_id(
                    list(pin_obs) + list(missing_find),
                    skip={int(resolved_oid)},
                )
                self.last_eqa_action_obs_id = nxt
                if nxt is not None:
                    target_point = self._navigation_waypoint_for_obs(int(nxt), xyt)
            else:
                self.last_eqa_action_obs_id = resolved_oid
                target_point = self._target_point_from_display_image_index(
                    display_index,
                    obs_ids=action_obs_ids,
                    nav_fallback_tail=nav_fallback_tail,
                    robot_xyt=xyt,
                )
        if not confidence and self.last_eqa_action_obs_id is None and missing_find:
            nxt = self.next_unspent_eqa_obs_id(missing_find)
            self.last_eqa_action_obs_id = nxt
            if nxt is not None and target_point is None:
                target_point = self._navigation_waypoint_for_obs(int(nxt), xyt)
        pending_look = self.last_eqa_action_obs_id
        if pending_look is not None and (not obs_ids or int(obs_ids[0]) != int(pending_look)):
            self.last_eqa_look_obs_id = int(pending_look)
        self._append_eqa_history(
            self.format_eqa_history_outcome(
                answer=raw_answer,
                confidence=confidence,
                action=hist_action,
                reasoning=reasoning,
                salvage=bool(self.last_eqa_salvage_used),
            )
        )

        return (
            reasoning,
            answer,
            confidence,
            confidence_reasoning,
            target_point,
            relevant_images,
        )

    def fill_descriptions_from_vlm(
        self,
        prompt: str | None = None,
        max_tokens: int = 80,
    ) -> None:
        """
        Fill missing node/observation descriptions using the VLM (e.g. Qwen 2.5-VL / 3.5).
        Skips observations that already have a description. Can be slow for many images.
        """
        if self.image_description_client is None:
            self._init_clients()
        default_prompt = (
            "In one short sentence, describe what is visible in this image: "
            "main objects, their arrangement, and any notable spatial relationships. "
            "Be concise."
        )
        prompt = prompt or default_prompt
        for obs in self._observations:
            if obs.description:
                continue
            try:
                # VLM accepts list of text + image(s)
                out = self.image_description_client(
                    [prompt, Image.fromarray(obs.rgb.astype(np.uint8), mode="RGB")],
                    verbose=False,
                )
                if isinstance(out, str) and out.strip():
                    desc = out.strip()
                    # Update observation (same object as stored)
                    obs.description = desc
                    # Update corresponding node
                    for n in self._nodes:
                        if n.obs_id == obs.obs_id:
                            n.description = desc
                            break
            except Exception as e:
                _logger.warning(f"fill_descriptions_from_vlm failed for obs {obs.obs_id}: {e}")
                continue

    def get_observations(self) -> list[GraphObservation]:
        return list(self._observations)

    def get_nodes(self) -> list[GraphNode]:
        return list(self._nodes)

    def get_edges(self) -> list[tuple[int, int, str]]:
        return list(self._edges)

    def print_memory(self) -> str:
        """
        Return the 3D scene graph as a human-readable tree (same as to_tree_string).
        Use this as the canonical "print" output for the graph memory.
        """
        return self.to_tree_string()

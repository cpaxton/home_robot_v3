# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Question keywords, label matching, and MCQ landmark helpers."""
from __future__ import annotations

import re

import numpy as np

from emet.memory.graph_eqa.types import GraphNode
from emet.utils.logger import Logger

_logger = Logger(__name__)

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

_ACTION_READ_RE = re.compile(r"^read\s*(?:image\s+)?(\d+)$")

_ACTION_LOOK_RE = re.compile(r"^(?:image\s+)?(\d+)$")

_GRAPH_CANDIDATE_COUNT_DISCLAIMER = "list length is not a count; verify in attached images"

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

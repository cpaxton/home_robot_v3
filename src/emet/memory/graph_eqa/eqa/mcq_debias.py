# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""MCQ choice-rotation debiasing for EQA answers.

Small VLMs (e.g. Qwen2.5-VL-3B) have a strong position bias against letter ``A``
(<7% of answers vs the expected 25% on HM-EQA). Re-asking the final question with
cyclically rotated choice orders and voting over the *underlying choices* (not the
letters) cancels the positional bias: each choice appears at each letter position
exactly once across the rotation set.
"""

from __future__ import annotations

import re
from collections import Counter

LETTERS = "ABCD"
PLACEHOLDER_RE = re.compile(r"do not choose", re.IGNORECASE)


def is_placeholder_choice(choice: str) -> bool:
    """Return whether a benchmark option is an explicit non-answer placeholder."""
    return bool(PLACEHOLDER_RE.search(str(choice or "")))


def valid_choice_indices(choices: list[str]) -> list[int]:
    """Indices that may be selected by a fallback policy."""
    return [idx for idx, choice in enumerate(choices) if not is_placeholder_choice(choice)]


def rotated_choice_order(n_choices: int, rotation: int) -> list[int]:
    """Original-choice index shown at each letter position for a cyclic rotation."""
    return [(i + rotation) % n_choices for i in range(n_choices)]


def format_rotated_question(question: str, choices: list[str], rotation: int) -> str:
    """Format ``question`` with choices cyclically rotated by ``rotation`` positions."""
    order = rotated_choice_order(len(choices), rotation)
    parts = [f"{LETTERS[i]}) {choices[j]}" for i, j in enumerate(order)]
    return f"{question} " + " ".join(parts) + ". Answer:"


def letter_to_original_index(letter: str, rotation: int, n_choices: int) -> int | None:
    """Map a predicted letter from a rotated presentation back to the original choice index."""
    letter = (letter or "").strip().upper()
    if len(letter) != 1 or letter not in LETTERS[:n_choices]:
        return None
    return (LETTERS.index(letter) + rotation) % n_choices


def extract_single_letter(text: str, n_choices: int = 4) -> str:
    """Pull a single A–D letter out of a terse VLM reply."""
    t = (text or "").strip()
    valid = LETTERS[:n_choices]
    m = re.search(rf"\b([{valid}])\b", t, flags=re.IGNORECASE)
    return m.group(1).upper() if m else ""


def tally_choice_votes(
    votes: list[int | None],
    choices: list[str],
    prior_index: int | None = None,
) -> int | None:
    """Majority-vote over original choice indices.

    Votes for placeholder choices ("(Do not choose this option)") are discarded.
    Ties prefer ``prior_index`` (the un-rotated main answer) when it is among the
    leaders; otherwise a tie carries no signal (a position-locked model un-rotates
    to a uniform split) and ``None`` is returned so the caller keeps its main answer.
    """
    counted = Counter(
        v for v in votes if v is not None and 0 <= v < len(choices) and not is_placeholder_choice(choices[v])
    )
    if not counted:
        return None
    top = max(counted.values())
    leaders = sorted(idx for idx, c in counted.items() if c == top)
    if prior_index in leaders:
        return prior_index
    if len(leaders) > 1:
        return None
    return leaders[0]


_STOPWORDS = frozenset("a an the it is are was were in at on of to and or i my your did you leave left".split())


def _norm_tokens(text: str) -> set[str]:
    return {t for t in re.sub(r"[^a-z0-9\- ]", " ", (text or "").lower()).split() if t and t not in _STOPWORDS}


def match_freeform_to_choice(answer: str, choices: list[str]) -> int | None:
    """Map a free-form answer (no letters involved) to the closest MCQ choice.

    Scoring: token Jaccard over stopword-stripped tokens, with a containment bonus
    when all of a choice's tokens appear in the answer. Requires a clear winner
    (score >= 0.34 and margin >= 0.15 over the runner-up); placeholder choices are
    never matched. Returns the original choice index or ``None`` when ambiguous.
    """
    ans = _norm_tokens(answer)
    if not ans:
        return None
    scores: list[tuple[float, int]] = []
    for idx, choice in enumerate(choices[:4]):
        if is_placeholder_choice(choice):
            continue
        ct = _norm_tokens(choice)
        if not ct:
            continue
        inter = len(ans & ct)
        score = inter / len(ans | ct)
        if ct <= ans:
            score = max(score, 0.8)
        scores.append((score, idx))
    if not scores:
        return None
    scores.sort(reverse=True)
    best, best_idx = scores[0]
    second = scores[1][0] if len(scores) > 1 else 0.0
    if best >= 0.34 and best - second >= 0.15:
        return best_idx
    return None


def answer_is_unknownish(answer: str, choices: list[str] | None = None) -> bool:
    """Treat ``None`` as a count answer when valid; keep abstention sentinels unknown.

    HM-EQA count questions legitimately use option text ``None``. By contrast,
    ``Unknown`` and ``N/A`` remain abstentions even when a benchmark includes
    them as fallback choices, so they should still trigger semantic salvage.
    """
    text = str(answer or "").strip()
    if not text:
        return True
    normalized = text.lower()
    if normalized == "none" and choices and match_freeform_to_choice(text, choices) is not None:
        return False
    return normalized in {"unknown", "none", "n/a", "na"} or "frontier" in normalized


def count_answer_is_none_or_zero(answer: str, choices: list[str] | None = None) -> bool:
    """True when the scored count letter is ``None`` / ``Zero`` / ``0``."""
    text = str(answer or "").strip()
    if not text:
        return False
    if choices:
        matched = match_freeform_to_choice(text, choices)
        if matched is not None:
            text = str(choices[matched] or "")
    return str(text).strip().lower() in {"none", "zero", "0"}

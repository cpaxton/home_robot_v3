# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for the 2026-08 agentic submit-guard trio.

Trace audit (12q merged-memory probe, 2026-08-01) findings these fix:

* q2  early unverified auto-submit (2 rounds, budget left) → no-early-unverified hold.
* q84 time question explored 7 frontiers with zero close looks → close-look
  classifier + station look_around redirect.
* Wrong unverified forced letters were overwhelmingly the last MCQ option (D)
  → letter-free debias (``vote_mcq_letter``) on the forced-answer ladder.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

Q_WHERE = (
    "Where did you see the soap dispenser? "
    "A) Above the sink B) On the toilet tank C) By the bathtub D) On the windowsill"
)
Q_TIME = "What time is it now? A) 10am-12pm B) 2pm-4pm C) 6pm-8pm D) 8am-10am"
Q_COUNT = "How many red pillows are on the sofa? A) One B) Two C) Three D) None"


def _executor(question: str, *, query_answer: str = "", raw: str = "", **kwargs):
    """Build an executor over a mocked graph memory with a canned EQA response."""
    pytest.importorskip("emet.memory.graph_eqa.agentic_eqa")
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    gm = MagicMock()
    agent.graph_memory = gm
    agent.planner = None
    agent.robot = None
    gm.last_eqa_raw = raw
    gm.query_answer.return_value = ("", query_answer, False, "", None, [])
    gm.select_obs_ids_for_verified_answer = MagicMock(return_value=[1])
    gm.vote_mcq_letter = MagicMock(return_value="")
    gm.last_mcq_debias = {}
    ex = AgenticEQAExecutor(agent, question, router=False, collect_trace=True, **kwargs)
    return ex, gm


# --- (1) mcq debias on the forced-answer ladder -------------------------------


def test_forced_debias_replaces_lettered_eq_answer():
    """A last-option EQA letter must not survive the forced path when debias wins."""
    ex, gm = _executor(Q_WHERE, raw="\n".join(["Answer:", "D) On the windowsill", ""]), mcq_debias=True)
    gm.vote_mcq_letter = MagicMock(return_value="B")

    out = ex._forced_answer_fallback(reason="budget exhausted without VLM answerable")

    assert out["answer"] == "B"
    assert out["answer_provenance"] == "mcq_debias"
    gm.vote_mcq_letter.assert_called_once()
    row = [r for r in ex._trace_rows if r.get("tool") == "forced_answer"][0]
    assert row["raw_eqa_letter"] == "D"


def test_forced_debias_fills_uniform_prior_hole():
    """Unknown from the EQA: the debias letter replaces the SHA-1 hash prior."""
    ex, gm = _executor(Q_WHERE, raw="Answer:\nUnknown\n", mcq_debias=True)
    gm.vote_mcq_letter = MagicMock(return_value="A")

    out = ex._forced_answer_fallback(reason="budget exhausted without VLM answerable")

    assert out["answer"] == "A"
    assert out["answer_provenance"] == "mcq_debias"


def test_forced_debias_off_keeps_eq_letter():
    """``mcq_debias=False`` preserves the raw EQA letter (A/B parity)."""
    ex, gm = _executor(Q_WHERE, raw="\n".join(["Answer:", "D) On the windowsill", ""]), mcq_debias=False)

    out = ex._forced_answer_fallback(reason="budget exhausted without VLM answerable")

    assert out["answer"] == "D"
    assert out["answer_provenance"] == "eqa_answer"
    gm.vote_mcq_letter.assert_not_called()


def test_forced_debias_failure_falls_back_to_ladder():
    """A debias exception must never crash the forced answer."""
    ex, gm = _executor(Q_WHERE, raw="Answer:\nUnknown\n", mcq_debias=True)
    gm.vote_mcq_letter = MagicMock(side_effect=RuntimeError("boom"))

    out = ex._forced_answer_fallback(reason="budget exhausted without VLM answerable")

    assert out["answer"] in "ABCD"
    assert out["answer_provenance"] == "uniform_prior"


# --- (3) no-early-unverified auto-submit hold ---------------------------------


def test_auto_submit_allowed_verified_anytime():
    """Corroborated (verified) ANSWER may submit early — q28's good path."""
    ex, _gm = _executor(Q_WHERE, mcq_debias=False)
    ex._verified = True
    assert ex._auto_submit_allowed(round_idx=0)
    assert ex._auto_submit_allowed(round_idx=6)


def test_auto_submit_allowed_unverified_only_last_round():
    """Unverified ANSWER with budget left must NOT auto-submit (q2 fix)."""
    ex, _gm = _executor(Q_WHERE, max_rounds=8, mcq_debias=False)
    ex._verified = False
    assert not ex._auto_submit_allowed(round_idx=2)
    assert not ex._auto_submit_allowed(round_idx=6)
    assert ex._auto_submit_allowed(round_idx=7)


def test_auto_submit_allowed_guard_off():
    """``no_early_unverified=False`` restores the old early submit."""
    ex, _gm = _executor(Q_WHERE, no_early_unverified=False, mcq_debias=False)
    ex._verified = False
    assert ex._auto_submit_allowed(round_idx=2)


# --- (2) close-look question classifier ---------------------------------------


def test_close_look_keywords():
    from emet.memory.graph_eqa.agentic_eqa import question_requires_close_look_keywords

    assert question_requires_close_look_keywords(Q_TIME)
    assert question_requires_close_look_keywords(Q_COUNT)
    assert question_requires_close_look_keywords("Is the microwave on or off? A) On B) Off")
    assert not question_requires_close_look_keywords(Q_WHERE)


def test_close_look_fallback_keyword():
    """No VLM available: the keyword heuristic still flags a time question."""
    ex, _gm = _executor(Q_TIME, close_look=True, mcq_debias=False)
    ex._apply_close_look_fallback()
    assert ex._close_look_required
    assert ex._close_look_source == "keyword"


def test_close_look_fallback_none():
    ex, _gm = _executor(Q_WHERE, close_look=True, mcq_debias=False)
    ex._apply_close_look_fallback()
    assert not ex._close_look_required
    assert ex._close_look_source == "none"


def test_close_look_disabled():
    ex, _gm = _executor(Q_TIME, close_look=False, mcq_debias=False)
    ex._apply_close_look_fallback()
    assert not ex._close_look_required
    assert ex._close_look_source == "disabled"


def test_extract_vlm_target_captures_close_look():
    """The per-episode VLM extract also answers the close-look question."""
    ex, gm = _executor(Q_TIME, close_look=True, mcq_debias=False)
    gm.eqa_client = MagicMock(
        return_value='{"target_phrase": "wall clock", "question_type": "state", "requires_close_look": true}'
    )
    ex._extract_vlm_target()
    assert ex._close_look_required
    assert ex._close_look_source == "vlm"
    assert ex._target_phrase == "wall clock"


def test_extract_vlm_target_close_look_default_false():
    ex, gm = _executor(Q_WHERE, close_look=True, mcq_debias=False)
    gm.eqa_client = MagicMock(return_value='{"target_phrase": "soap dispenser", "question_type": "location"}')
    ex._extract_vlm_target()
    assert not ex._close_look_required
    assert ex._close_look_source == "vlm"

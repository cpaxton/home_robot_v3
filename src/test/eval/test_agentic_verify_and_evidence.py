# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for verification-rate fixes (2026-08 bal-32 follow-up).

Field data (61 episodes) showed verified answers score ~86% but fire only ~13% of
the time; everything else is a position-anchored forced guess (~35%). These fix the
verification gate and ground unverified final answers:

* single-view present-confirm: ``answerable + present + letter`` on ONE view
  confirms (was: phrase-token hit or two-view agreement required).
* evidence-grounded Image 1: the final EQA pins the best VLM-assessed view as
  Image 1 when nothing was corroborated.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

Q_WHERE = (
    "Where did you see the soap dispenser? "
    "A) Above the sink B) On the toilet tank C) By the bathtub D) On the windowsill"
)


def _executor(question: str = Q_WHERE, *, query_answer: str = "", raw: str = "", **kwargs):
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
    gm.last_eqa_obs_ids = None
    gm.vote_mcq_letter = MagicMock(return_value="")
    gm.last_mcq_debias = {}
    ex = AgenticEQAExecutor(agent, question, router=False, collect_trace=True, **kwargs)
    return ex, gm


# --- single-view present-confirm ----------------------------------------------


def test_single_view_present_confirms():
    """One view that saw the target and offered a letter is enough to verify."""
    ex, _gm = _executor(single_view_confirm=True)
    ex._answerable_phrase_hit = MagicMock(return_value=False)

    confirmed, reason = ex._maybe_confirm_answerable(
        obs_id=1,
        present=True,
        answerable=True,
        need_more_views=False,
        suggested_answer="B",
        phrase="soap dispenser",
    )

    assert (confirmed, reason) == (True, "single_view_present")


def test_single_view_confirm_absent_never_confirms():
    """The present guard from q28/q39 stays: absence is not an answer."""
    ex, _gm = _executor(single_view_confirm=True)
    ex._answerable_phrase_hit = MagicMock(return_value=True)

    confirmed, reason = ex._maybe_confirm_answerable(
        obs_id=1,
        present=False,
        answerable=True,
        need_more_views=False,
        suggested_answer="B",
        phrase="soap dispenser",
    )

    assert not confirmed
    assert reason != "single_view_present"


def test_single_view_confirm_needs_letter():
    """Answerable+present without a suggested letter stays deferred."""
    ex, _gm = _executor(single_view_confirm=True)
    ex._answerable_phrase_hit = MagicMock(return_value=False)

    confirmed, reason = ex._maybe_confirm_answerable(
        obs_id=1,
        present=True,
        answerable=True,
        need_more_views=False,
        suggested_answer=None,
        phrase="soap dispenser",
    )

    assert (confirmed, reason) == (False, "deferred")


def test_single_view_confirm_off_keeps_phrase_gate():
    """Flag off restores the corroboration gate (phrase hit still confirms)."""
    ex, _gm = _executor(single_view_confirm=False)
    ex._answerable_phrase_hit = MagicMock(return_value=True)

    confirmed, reason = ex._maybe_confirm_answerable(
        obs_id=1,
        present=True,
        answerable=True,
        need_more_views=False,
        suggested_answer="B",
        phrase="soap dispenser",
    )

    assert (confirmed, reason) == (True, "phrase_corroborated")


def test_single_view_confirm_off_phrase_miss_defers():
    """Flag off + no phrase hit + one view = deferred (old behavior)."""
    ex, _gm = _executor(single_view_confirm=False)
    ex._answerable_phrase_hit = MagicMock(return_value=False)

    confirmed, reason = ex._maybe_confirm_answerable(
        obs_id=1,
        present=True,
        answerable=True,
        need_more_views=False,
        suggested_answer="B",
        phrase="soap dispenser",
    )

    assert (confirmed, reason) == (False, "deferred")


# --- evidence-grounded Image 1 -------------------------------------------------


def test_best_evidence_obs_ranks_present_answerable_first():
    ex, _gm = _executor()
    ex._assess_history = {
        1: {"present": False, "answerable": False, "need_more_views": True},
        2: {"present": True, "answerable": True, "need_more_views": False},
        3: {"present": True, "answerable": False, "need_more_views": True},
    }
    assert ex._best_evidence_obs_id() == 2


def test_best_evidence_obs_prefers_present_over_nothing():
    ex, _gm = _executor()
    ex._assess_history = {
        1: {"present": False, "answerable": False, "need_more_views": True},
        3: {"present": True, "answerable": False, "need_more_views": True},
    }
    assert ex._best_evidence_obs_id() == 3


def test_best_evidence_obs_none_when_nothing_seen():
    ex, _gm = _executor()
    ex._assess_history = {
        1: {"present": False, "answerable": False, "need_more_views": True},
        2: {"present": False, "answerable": False, "need_more_views": False},
    }
    assert ex._best_evidence_obs_id() is None


def test_best_evidence_obs_none_when_empty():
    ex, _gm = _executor()
    assert ex._best_evidence_obs_id() is None


def test_unverified_submit_pins_evidence_image():
    """Unverified final EQA pins the best assessed view as Image 1."""
    ex, gm = _executor(query_answer="", raw="Answer:\nA) Above the sink\n")
    ex._assess_history = {
        1: {"present": False, "answerable": False, "need_more_views": True},
        5: {"present": True, "answerable": True, "need_more_views": False},
    }

    ex._do_submit_answer()

    gm.select_obs_ids_for_verified_answer.assert_called_once_with(5, max_images=1)
    assert gm.last_eqa_obs_ids == [1]


def test_unverified_submit_without_evidence_keeps_diversified_pick():
    """No assessed view saw anything: no force, diversified selection unchanged."""
    ex, gm = _executor(query_answer="", raw="Answer:\nA) Above the sink\n")
    ex._assess_history = {
        1: {"present": False, "answerable": False, "need_more_views": True},
    }

    ex._do_submit_answer()

    gm.select_obs_ids_for_verified_answer.assert_not_called()
    assert gm.last_eqa_obs_ids is None


def test_evidence_image_off_keeps_diversified_pick():
    ex, gm = _executor(query_answer="", raw="Answer:\nA) Above the sink\n", evidence_image=False)
    ex._assess_history = {
        5: {"present": True, "answerable": True, "need_more_views": False},
    }

    ex._do_submit_answer()

    gm.select_obs_ids_for_verified_answer.assert_not_called()

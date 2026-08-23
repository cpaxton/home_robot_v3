# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Replay tests for agentic EQA answer resolution.

Each test reproduces a concrete failure from the 2026-07 bal-32 trace audit
(``docs/experiments/agentic_eqa_trace_audit.md``) as a deterministic fixture, so the
Tier 0 fixes can be iterated without spending GPU time:

* q28  override destroyed a correct answer (coordinate-dump ``query_answer`` handed
  control to a ``present: false`` single-view letter).
* q39  absence became an answer (two non-detections "corroborated" each other).
* q47  two-stage override (``[memory-location]`` block replaced the EQA's own letter).
* q43  prose answer scored blank because no ``A-D`` letter parsed.
* q80  budget exhaustion abstained without ever invoking the four-image EQA.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from emet.habitat.metrics import choices_are_location_mcq, choices_are_time_of_day

Q28 = "How many fans are in the living room? A) One B) Three C) None D) Two"
Q39 = (
    "Is there a fan in the bedroom? A) Yes, on the ceiling B) No, there is none C) Yes, next to a bed D) Yes, on a desk"
)
Q47 = (
    "Where did you see the soap dispenser? "
    "A) Above the sink B) On the toilet tank C) By the bathtub D) On the windowsill"
)
Q43 = "What time is it now? A) 10am-12pm B) 2pm-4pm C) 6pm-8pm D) 8am-10am"
Q12 = "How many bedside tables are there? A) Three B) One C) None D) Two"
Q11 = (
    "Where did I leave the silver trash can at? "
    "A) Next to the dining table B) Next to the TV "
    "C) Next to the kitchen sink D) Next to the refrigerator"
)


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


def _set_confirmed_vlm_answer(ex, letter: str, *, obs_id: int = 7, answer_text: str = ""):
    record = ex._record_answer_evidence(
        letter=letter,
        source="vlm_suggested",
        answer_text=answer_text,
        obs_id=obs_id,
        present=True,
        answerable=True,
        need_more_views=False,
        confidence=0.5,
    )
    assert record is not None
    ex._confirmed_answer_evidence = record
    return record


def test_q28_eqa_letter_beats_absence_and_coordinate_dump():
    """Gold ``D) Two`` from the EQA must survive a coordinate dump plus an absent view."""
    ex, _gm = _executor(
        Q28,
        query_answer="The fan, lamp is at approximately (1.77, -1.71, 0.63) m.",
        raw="\n".join(["Answer:", "D) Two", "Confidence: TRUE", ""]),
    )
    ex._verified = True
    ex._verified_obs_id = 1
    # The recliner view that historically supplied the scored "C" (option C is "None").
    ex._last_vlm_assess = {"present": False, "suggested_answer": "C"}

    out = ex._do_submit_answer()

    assert out["answer"] == "Two"
    assert out["answer_source"] == "eqa_answer"


def test_confirmed_q12_vlm_letter_beats_conflicting_four_image_eqa():
    """The answer that opened ANSWER must survive a conflicting final EQA call."""
    ex, _gm = _executor(
        Q12,
        query_answer="The nightstand is at approximately (1.36, 2.64, 0.25) m.",
        raw="Answer:\nB) One\n",
    )
    _set_confirmed_vlm_answer(ex, "D")
    ex._verified = True
    ex._verified_obs_id = 7

    out = ex._do_submit_answer()

    assert out["answer"] == "Two"
    assert out["answer_source"] == "vlm_suggested"
    assert out["final_decision"]["evidence"]["obs_id"] == 7


def test_forced_debias_cannot_replace_confirmed_vlm_answer():
    """A later budget path may not overwrite a previously confirmed view."""
    ex, gm = _executor(Q12, raw="Answer:\nB) One\n", mcq_debias=True)
    _set_confirmed_vlm_answer(ex, "D")
    # Reproduce the old coverage-motion bug: current-hypothesis verification was
    # reset even though episode-level answer evidence already existed.
    ex._verified = False
    ex._verified_obs_id = None
    gm.vote_mcq_letter = MagicMock(return_value="A")

    out = ex._forced_answer_fallback(reason="budget exhausted without VLM answerable")

    assert out["answer"] == "Two"
    assert out["answer_provenance"] == "vlm_suggested"
    assert out["answer_confidence"] == pytest.approx(0.65)
    assert out["confidence"] is True
    assert out["verified"] is True
    assert out["final_decision"]["evidence"]["obs_id"] == 7
    gm.vote_mcq_letter.assert_not_called()
    forced = next(row for row in ex._trace_rows if row.get("tool") == "forced_answer")
    assert forced["raw_eqa_answer"] == "B) One"
    assert forced["raw_eqa_choice_index"] == 1
    assert forced["resolved_choice_index"] == 3
    assert forced["agentic_mcq_debias_enabled"] is True
    assert forced["evidence_backed_decision"] is True


def test_coverage_look_around_preserves_confirmed_answer_state():
    """A map-coverage sweep after stalled capture must not clear ANSWER."""
    from emet.memory.graph_eqa.agentic_policy import AgenticState

    ex, _gm = _executor(Q12)
    ex._verified = True
    ex._verified_obs_id = 7
    ex._evidence_policy.state = AgenticState.ANSWER
    ex._tool_capture_and_update = MagicMock(return_value={"ok": True, "obs_id": 45, "status": "CONTENT_REFRESHED"})

    out = ex._tool_look_around(verify=False)

    assert out["ok"] is True
    assert ex._evidence_policy.state is AgenticState.ANSWER
    assert ex._verified is True
    assert ex._verified_obs_id == 7


def test_grounded_v2_confirmed_view_beats_conflicting_eqa(monkeypatch):
    """Image that opened ANSWER outranks graph-steered multi-image EQA JSON."""
    monkeypatch.setenv("EMET_EQA_AGENTIC_DECISION_POLICY", "grounded_v2")
    ex, gm = _executor(
        Q12,
        query_answer="The nightstand is at approximately (1.36, 2.64, 0.25) m.",
        raw='{"reasoning":"one nightstand in SCENE_GRAPH","answer":"B","confidence":true}',
    )
    gm.last_eqa_model_raw = gm.last_eqa_raw
    gm.last_eqa_model_parsed = ("one nightstand in SCENE_GRAPH", "B", True, "", "")
    _set_confirmed_vlm_answer(ex, "D")
    ex._verified = True
    ex._verified_obs_id = 7

    out = ex._do_submit_answer(prefer_answer="B")

    assert out["answer"] == "Two"
    assert out["answer_source"] == "vlm_suggested"
    assert out["final_decision"]["evidence"]["obs_id"] == 7


def test_grounded_v2_eqa_used_only_without_image_evidence(monkeypatch):
    """Native EQA parse is still the fallback when no present+answerable view exists."""
    monkeypatch.setenv("EMET_EQA_AGENTIC_DECISION_POLICY", "grounded_v2")
    ex, gm = _executor(
        Q28,
        query_answer="",
        raw='{"reasoning":"two fans are visible","answer":"B","confidence":true}',
    )
    gm.last_eqa_model_raw = gm.last_eqa_raw
    gm.last_eqa_model_parsed = ("two fans are visible", "B", True, "", "")

    out = ex._do_submit_answer()

    assert out["answer"] == "Three"
    assert out["answer_source"] == "eqa_answer"


def test_q11_semantic_eqa_answer_maps_to_d_without_fabricated_view_evidence(monkeypatch):
    monkeypatch.setenv("EMET_EQA_AGENTIC_DECISION_POLICY", "grounded_v2")
    semantic = "Next to the refrigerator"
    ex, gm = _executor(
        Q11,
        raw=(f'{{"reasoning":"The trash can is beside the refrigerator.","answer":"{semantic}","confidence":true}}'),
    )
    gm.last_eqa_model_raw = gm.last_eqa_raw
    gm.last_eqa_model_parsed = (
        "The trash can is beside the refrigerator.",
        semantic,
        True,
        "",
        "visible",
    )
    ex._assess_history[15] = {
        "present": False,
        "answerable": False,
        "need_more_views": True,
    }

    out = ex._forced_answer_fallback(reason="budget exhausted after absent views")

    assert out["answer"] == semantic
    assert out["answer_source"] == "eqa_answer"
    assert out["verified"] is False
    assert out["final_decision"]["answer_text"] == semantic
    assert out["final_decision"]["choice_index"] == 3
    assert out["final_decision"]["evidence"] is None


def test_q11_semantic_view_answer_survives_channel_selection(monkeypatch):
    monkeypatch.setenv("EMET_EQA_AGENTIC_DECISION_POLICY", "grounded_v2")
    semantic = "Next to the refrigerator"
    ex, gm = _executor(Q11, raw="")
    gm.last_eqa_model_raw = ""
    gm.last_eqa_model_parsed = ("", "", False, "", "")
    record = _set_confirmed_vlm_answer(ex, semantic, obs_id=11, answer_text=semantic)
    ex._verified = True
    ex._verified_obs_id = 11

    out = ex._do_submit_answer(prefer_answer=semantic)

    assert record.letter == "D"
    assert out["answer"] == semantic
    assert out["final_decision"]["answer_text"] == semantic
    assert out["final_decision"]["choice_index"] == 3
    assert out["final_decision"]["evidence"]["answer_text"] == semantic
    assert out["final_decision"]["evidence"]["choice_index"] == 3
    assert "letter" not in out["final_decision"]["evidence"]


def test_grounded_v2_rejects_unaligned_router_answer(monkeypatch):
    monkeypatch.setenv("EMET_EQA_AGENTIC_DECISION_POLICY", "grounded_v2")
    ex, gm = _executor(Q28, query_answer="", raw="")
    gm.last_eqa_model_raw = ""
    gm.last_eqa_model_parsed = ("", "", False, "", "")

    out = ex._do_submit_answer(prefer_answer="C")

    assert out["answer"] == "Unknown"
    assert any(row.get("event") == "answer_proposal_rejected" for row in ex._trace_rows)


def test_grounded_v2_cheap_absent_cannot_retract_vlm_positive(monkeypatch):
    monkeypatch.setenv("EMET_EQA_AGENTIC_DECISION_POLICY", "grounded_v2")
    ex, gm = _executor(Q39)

    out = ex._maybe_retract_claim_after_station(
        4,
        closest_m=0.5,
        verify_out={
            "status": "ABSENT",
            "obs_id": 9,
            "phrase": "fan",
            "vlm_assess": {"ok": True, "obs_id": 9, "present": True, "answerable": True},
        },
    )

    assert out is None
    gm.retract_phrase_claim_at_obs.assert_not_called()


def test_absent_view_never_supplies_the_scored_letter():
    """A ``present: false`` assess is a non-detection, not an MCQ answer."""
    ex, _gm = _executor(Q39, query_answer="")
    ex._last_vlm_assess = {"present": False, "suggested_answer": "B"}

    assert ex._trusted_vlm_letter() == ""


def test_q39_two_absent_views_do_not_confirm_answerable():
    """Two frames that both failed to see the fan are not two-view corroboration."""
    ex, _gm = _executor(Q39)
    ex._answerable_phrase_hit = MagicMock(return_value=False)

    first, reason_first = ex._maybe_confirm_answerable(
        obs_id=1,
        present=False,
        answerable=True,
        need_more_views=False,
        suggested_answer="B",
        phrase="fan",
    )
    second, reason_second = ex._maybe_confirm_answerable(
        obs_id=2,
        present=False,
        answerable=True,
        need_more_views=False,
        suggested_answer="B",
        phrase="fan",
    )

    assert (first, reason_first) == (False, "deferred")
    assert (second, reason_second) == (False, "deferred")


def test_two_present_views_still_confirm_answerable():
    """The two-view unlock must keep working when both frames saw the target."""
    ex, _gm = _executor(Q39, single_view_confirm=False)
    ex._answerable_phrase_hit = MagicMock(return_value=False)

    ex._maybe_confirm_answerable(
        obs_id=1,
        present=True,
        answerable=True,
        need_more_views=False,
        suggested_answer="C",
        phrase="fan",
    )
    confirmed, reason = ex._maybe_confirm_answerable(
        obs_id=2,
        present=True,
        answerable=True,
        need_more_views=False,
        suggested_answer="C",
        phrase="fan",
    )

    assert (confirmed, reason) == (True, "two_view_agree")


def test_q47_memory_location_block_does_not_replace_eqa_letter():
    """Read the EQA's own ``Answer:`` block, not a later override block."""
    ex, _gm = _executor(
        Q47,
        query_answer="",
        raw="Answer:\nA) Above the sink\n[memory-location]\nanswer:\nB\n",
    )

    assert ex._eqa_self_answer_letter() == "A"


def test_q43_prose_answer_still_scores_a_letter():
    """A prose EQA answer must not fall through to a blank scored prediction."""
    ex, _gm = _executor(
        Q43,
        query_answer="The time is 2:30 PM based on the wall clock.",
        raw="Answer:\nThe time is 2:30 PM\n",
    )
    ex._round = 7

    out = ex._forced_answer_fallback(reason="budget exhausted without VLM answerable")

    assert out["answer"] in {"10am-12pm", "2pm-4pm", "6pm-8pm", "8am-10am"}
    assert out["answer_provenance"] in {"eqa_answer", "uniform_prior"}


def test_q80_budget_exhaustion_invokes_the_four_image_eqa():
    """The old abstain path returned Unknown without ever calling ``query_answer``."""
    ex, gm = _executor(
        Q47,
        query_answer="",
        raw="Answer:\nC) By the bathtub\n",
        require_verified=True,
        max_rounds=2,
        max_nav_steps=0,
    )
    ex._round = 1

    out = ex._tool_submit_answer("")

    assert gm.query_answer.called
    assert out["answer"] == "By the bathtub"
    assert out["answer_provenance"] == "eqa_answer"
    assert any(row.get("tool") == "forced_answer" for row in ex._trace_rows)


def test_forced_guess_is_deterministic_and_low_confidence():
    """With no usable channel we still commit to option text, flagged as a guess."""
    ex_a, _ = _executor(Q28, query_answer="")
    ex_b, _ = _executor(Q28, query_answer="")
    ex_a._round = 7
    ex_b._round = 7

    out_a = ex_a._forced_answer_fallback(reason="budget exhausted without VLM answerable")
    out_b = ex_b._forced_answer_fallback(reason="budget exhausted without VLM answerable")

    assert out_a["answer"] == out_b["answer"]
    assert out_a["answer"] in {"One", "Three", "None", "Two"}
    assert out_a["answer_provenance"] == "uniform_prior"
    assert out_a["answer_confidence"] == pytest.approx(0.25)
    assert out_a["confidence"] is False


def test_forced_ladder_tags_trusted_view_as_vlm_suggested_not_pending():
    """Calibration must not conflate a view that saw the target with a deferred letter."""
    ex, _ = _executor(Q28, query_answer="")
    ex._round = 7
    ex._last_vlm_assess = {"present": True, "suggested_answer": "D"}
    ex._pending_answerable = {"present": True, "letter": "B"}

    out = ex._forced_answer_fallback(reason="budget exhausted without VLM answerable")

    assert out["answer"] == "Two"
    assert out["answer_provenance"] == "vlm_suggested"
    assert out["answer_confidence"] == pytest.approx(0.50)


def test_forced_ladder_tags_deferred_assess_as_pending_letter():
    ex, _ = _executor(Q28, query_answer="")
    ex._round = 7
    ex._last_vlm_assess = {"present": False, "suggested_answer": "C"}
    ex._last_positive_letter = ""
    ex._pending_answerable = {"present": True, "letter": "B"}

    out = ex._forced_answer_fallback(reason="budget exhausted without VLM answerable")

    assert out["answer"] == "Three"
    assert out["answer_provenance"] == "pending_letter"
    assert out["answer_confidence"] == pytest.approx(0.35)


def test_uniform_prior_spreads_across_choices():
    """A fixed guess letter would bias a whole benchmark; the prior must vary."""
    letters = set()
    for i in range(24):
        ex, _ = _executor(f"Question {i}? A) a B) b C) c D) d")
        letters.add(ex._uniform_prior_letter(4))
    assert len(letters) >= 3


def test_force_answer_env_flag_restores_abstain(monkeypatch):
    monkeypatch.setenv("EMET_EQA_FORCE_ANSWER", "0")
    ex, gm = _executor(Q28, query_answer="")

    out = ex._forced_answer_fallback(reason="budget exhausted without VLM answerable")

    assert out["answer"] == "Unknown"
    assert not gm.query_answer.called
    assert any(row.get("tool") == "abstain_unverified" for row in ex._trace_rows)


def test_force_answer_off_does_not_discard_confirmed_evidence(monkeypatch):
    monkeypatch.setenv("EMET_EQA_FORCE_ANSWER", "0")
    ex, gm = _executor(Q12, raw="Answer:\nB) One\n")
    _set_confirmed_vlm_answer(ex, "D")

    out = ex._forced_answer_fallback(reason="later coverage exhausted the budget")

    assert out["answer"] == "Two"
    assert out["answer_provenance"] == "vlm_suggested"
    assert out["verified"] is True
    assert gm.query_answer.called


def test_time_of_day_choices_are_not_a_location_mcq():
    """Time questions were routed through the location-visibility gate and salvage."""
    choices = ["Morning", "Afternoon", "Evening", "Night"]
    assert choices_are_time_of_day(choices) is True
    assert choices_are_location_mcq(choices) is False
    assert choices_are_location_mcq(["Above the sink", "On the toilet tank", "By the bathtub"]) is True

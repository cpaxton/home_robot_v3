# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from emet.memory.graph_eqa.agentic.place import _maybe_retract_claim_after_station, _place_close_and_absent
from emet.memory.graph_eqa.agentic.types import PlaceInspectRecord
from emet.memory.graph_eqa.agentic.verify import visual_presence


@pytest.mark.parametrize("retrieval", ["ABSENT", "PRESENT", "CANDIDATE"])
@pytest.mark.parametrize("present", [True, False, None])
def test_retraction_follows_visual_evidence_not_retrieval(retrieval, present):
    graph = SimpleNamespace(retract_phrase_claim_at_obs=Mock(return_value={"ok": True}))
    executor = SimpleNamespace(
        graph_memory=graph,
        decision_policy="legacy",
        _target_phrase="mug",
        _observation_room=lambda oid: None,
        _known_room_for_event=lambda: None,
        _graph_world_step=lambda: 0,
        _append_trace=Mock(),
    )
    out = {
        "status": "VISUAL_ABSENT" if present is False else "VISUAL_PRESENT",
        "retrieval": {"status": retrieval},
        "visual": {"present": present},
        "obs_id": 7,
        "phrase": "mug",
        "answerable": False,
    }
    _maybe_retract_claim_after_station(executor, 7, closest_m=0.5, verify_out=out)
    assert graph.retract_phrase_claim_at_obs.called is (present is False)


@pytest.mark.parametrize("status", ["ABSENT", "SKIPPED_SAME_VIEW", "NOT_ASSESSED"])
def test_no_visual_assessment_does_not_exhaust_close_place(status):
    record = PlaceInspectRecord(investigate_count=1, closest_m=0.5, last_verify=status)
    assert not _place_close_and_absent(SimpleNamespace(_place_inspect={7: record}), 7)


def test_explicit_visual_absence_resolves_close_place():
    record = PlaceInspectRecord(investigate_count=1, closest_m=0.5, last_verify="VISUAL_ABSENT")
    assert _place_close_and_absent(SimpleNamespace(_place_inspect={7: record}), 7)
    record.last_assess_present = True
    assert not _place_close_and_absent(SimpleNamespace(_place_inspect={7: record}), 7)


def test_missing_pixels_cannot_become_legacy_visual_absence():
    assert (
        visual_presence(
            {"status": "ABSENT", "vlm_assess": {"ok": True, "present": False, "visual_observation_available": False}}
        )
        is None
    )


@pytest.mark.parametrize(
    "status,expected", [("VISUAL_ABSENT", "absent"), ("VISUAL_PRESENT", "present"), ("NOT_ASSESSED", "candidate")]
)
def test_attempt_ledger_keeps_separated_visual_status(status, expected):
    from emet.agent.tool_outcome import ToolOutcome, maybe_record_tool_attempt

    graph = SimpleNamespace(record_attempt=Mock())
    maybe_record_tool_attempt(graph, ToolOutcome(ok=True, tool="verify_siglip", status=status))
    assert graph.record_attempt.call_args.kwargs["outcome"] == expected

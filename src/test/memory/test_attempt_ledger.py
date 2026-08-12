# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for GraphEQA action-outcome attempt ledger (Phase 1a)."""

from __future__ import annotations

import numpy as np
import pytest

from emet.memory.graph_eqa.attempt_ledger import (
    AttemptRecord,
    infer_nav_outcome,
    infer_nav_status_code,
    records_from_dicts,
    records_to_dicts,
)
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, GraphNode


def test_attempt_record_roundtrip():
    rec = AttemptRecord(
        action_kind="navigate",
        outcome="unreachable",
        status_code="navmesh_no_path",
        note="navmesh_no_path",
        step=7,
        target_node_id=1,
        obs_id=3,
        xyz=(1.0, 2.0, 0.5),
        source="eqa",
        question_id="q42",
        room="kitchen",
    )
    d = rec.to_dict()
    assert d["xyz"] == [1.0, 2.0, 0.5]
    assert d["schema_version"] == 2
    assert d["room"] == "kitchen"
    back = AttemptRecord.from_dict(d)
    assert back == rec
    assert records_from_dicts(records_to_dicts([rec])) == [rec]


def test_attempt_record_v1_import_missing_room():
    """Legacy exports without room still load (room defaults to empty)."""
    back = AttemptRecord.from_dict(
        {
            "schema_version": 1,
            "action_kind": "verify",
            "outcome": "absent",
            "status_code": "vlm_absent",
            "step": 2,
            "obs_id": 9,
            "phrase": "clock",
            "source": "eqa",
        }
    )
    assert back.room == ""
    assert back.phrase == "clock"


def test_attempt_record_rejects_bad_kind():
    with pytest.raises(ValueError, match="action_kind"):
        AttemptRecord.from_dict({"action_kind": "teleport", "outcome": "ok", "status_code": "ok"})


def test_infer_nav_status_and_outcome():
    assert infer_nav_status_code(success=True, note="arrived") == "ok"
    assert infer_nav_status_code(success=False, note="navmesh_no_path") == "navmesh_no_path"
    assert infer_nav_status_code(success=False, note="rejected low clearance") == "rejected_low_clearance"
    assert infer_nav_outcome(success=True, status_code="ok") == "ok"
    assert infer_nav_outcome(success=False, status_code="no_path") == "unreachable"
    assert infer_nav_outcome(success=False, status_code="timeout") == "aborted"


def test_ledger_default_off(monkeypatch):
    monkeypatch.delenv("EMET_EQA_ATTEMPT_LEDGER", raising=False)
    mem = GraphEQAMemory(defer_llm_clients=True)
    assert mem._attempt_ledger_enabled() is False
    mem._nodes = [
        GraphNode(
            node_id=1,
            obs_id=3,
            xyz=np.array([1.0, 2.0, 0.0]),
            labels=["frontier"],
            is_frontier=True,
        )
    ]
    mem.record_nav_attempt(3, success=False, note="navmesh_no_path", dist_m=0.0, step=7)
    node = mem.get_nodes()[0]
    assert node.nav_attempts == 1
    assert node.nav_failures == 1
    assert mem.get_attempt_records() == []


def test_ledger_enabled_via_config_records_nav():
    mem = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={"eqa": {"attempt_ledger": True}},
    )
    assert mem._attempt_ledger_enabled() is True
    mem.set_attempt_ledger_question_id("q7")
    mem._nodes = [
        GraphNode(
            node_id=9,
            obs_id=3,
            xyz=np.array([1.0, 2.0, 0.25]),
            labels=["frontier"],
            is_frontier=True,
        )
    ]
    mem.record_nav_attempt(3, success=False, note="navmesh_no_path", dist_m=0.0, step=7)
    rows = mem.get_attempt_records()
    assert len(rows) == 1
    rec = rows[0]
    assert rec.action_kind == "navigate"
    assert rec.outcome == "unreachable"
    assert rec.status_code == "navmesh_no_path"
    assert rec.obs_id == 3
    assert rec.target_node_id == 9
    assert rec.xyz == (1.0, 2.0, 0.25)
    assert rec.question_id == "q7"
    assert rec.source == "eqa"
    # Dual-write: node counters still updated.
    node = mem.get_nodes()[0]
    assert node.nav_attempts == 1
    assert node.nav_failures == 1
    attempts, failures, note, last_step = mem.derive_nav_counters_from_ledger(3)
    assert (attempts, failures, last_step) == (1, 1, 7)
    assert note == "navmesh_no_path"


def test_ledger_env_override(monkeypatch):
    mem = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={"eqa": {"attempt_ledger": False}},
    )
    monkeypatch.setenv("EMET_EQA_ATTEMPT_LEDGER", "1")
    assert mem._attempt_ledger_enabled() is True
    monkeypatch.setenv("EMET_EQA_ATTEMPT_LEDGER", "0")
    mem2 = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={"eqa": {"attempt_ledger": True}},
    )
    assert mem2._attempt_ledger_enabled() is False


def test_ledger_export_import_and_filter():
    mem = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={"eqa": {"attempt_ledger": {"enabled": True}}},
    )
    mem.record_attempt(
        action_kind="navigate",
        outcome="failed",
        status_code="timeout",
        obs_id=1,
        step=1,
        source="eqa",
        question_id="q1",
    )
    mem.record_attempt(
        action_kind="verify",
        outcome="absent",
        status_code="vlm_absent",
        obs_id=2,
        step=2,
        source="eqa",
        question_id="q1",
    )
    blob = mem.export_attempt_ledger()
    assert len(blob) == 2
    other = GraphEQAMemory(defer_llm_clients=True, parameters={"eqa": {"attempt_ledger": True}})
    n = other.import_attempt_ledger(blob)
    assert n == 2
    assert len(other.get_attempt_records(action_kind="navigate")) == 1
    assert len(other.get_attempt_records(obs_id=2)) == 1
    assert other.get_attempt_records(question_id="q1", limit=1)[0].obs_id == 2


def test_ledger_cap():
    mem = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={"eqa": {"attempt_ledger": True}},
    )
    mem._attempt_ledger_max = 3
    for i in range(5):
        mem.record_attempt(
            action_kind="navigate",
            outcome="failed",
            status_code="failed_move",
            obs_id=i,
            step=i,
            force=True,
        )
    rows = mem.get_attempt_records()
    assert len(rows) == 3
    assert [r.obs_id for r in rows] == [2, 3, 4]


def test_retract_records_verify_absent_and_persist_flag():
    mem = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={"eqa": {"attempt_ledger": {"enabled": True, "persist_absent_claims": True}}},
    )
    assert mem.persist_absent_claims is True
    mem._nodes = [
        GraphNode(
            node_id=1,
            obs_id=5,
            xyz=np.array([0.0, 0.0, 0.5]),
            labels=["fruit bowl", "table"],
        )
    ]
    out = mem.retract_phrase_claim_at_obs(5, "fruit bowl", room="kitchen", step=3)
    assert out["ok"] is True
    assert out.get("room") == "kitchen"
    rows = mem.get_attempt_records(action_kind="verify")
    assert len(rows) == 1
    assert rows[0].outcome == "absent"
    assert rows[0].phrase == "fruit bowl"
    assert rows[0].room == "kitchen"
    hist = mem.format_room_history(target_rooms=["bathroom"])
    assert "verify_absent" in hist
    assert "kitchen" in hist
    assert "targets=bathroom" in hist
    # Persist flag keeps blacklist across clear.
    assert (5, "fruit bowl") in mem._retracted_nav_claims
    mem.clear_retracted_nav_claims()
    assert (5, "fruit bowl") in mem._retracted_nav_claims


def test_attempt_summary_for_obs():
    mem = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={"eqa": {"attempt_ledger": True}},
    )
    mem.record_attempt(
        action_kind="navigate",
        outcome="unreachable",
        status_code="navmesh_no_path",
        obs_id=2,
        step=1,
    )
    mem.record_attempt(
        action_kind="verify",
        outcome="absent",
        status_code="vlm_absent",
        obs_id=2,
        step=2,
        phrase="cup",
    )
    summary = mem.attempt_summary_for_obs(2)
    assert "verify:absent" in summary
    assert "navigate:unreachable" in summary


def test_room_timeline_ignores_unknown_and_formats_newest_first():
    mem = GraphEQAMemory(defer_llm_clients=True)
    assert mem.record_room_event(room="unknown", kind="stamp", step=1) is None
    assert mem.record_room_event(room="", kind="stamp", step=1) is None
    assert mem.record_room_event(room="kitchen", kind="stamp", step=1) is not None
    assert mem.record_room_event(
        room="kitchen", kind="verify_absent", step=3, phrase="clock"
    ) is not None
    assert mem.record_room_event(room="bathroom", kind="coverage_closed", step=5) is not None
    hist = mem.format_room_history(max_chars=220, target_rooms=["bathroom", "bedroom"])
    assert hist.startswith("Room history:")
    # Newest-first: bathroom coverage before kitchen absent.
    assert hist.index("bathroom") < hist.index("verify_absent")
    assert "targets=bathroom,bedroom" in hist
    mem.clear_room_events()
    assert mem.get_room_events() == []
    assert "Room history: (none)" in mem.format_room_history(target_rooms=["kitchen"])


def test_room_history_does_not_set_escape_floor():
    """Room timeline is agent-facing memory — not a nav latch."""
    from unittest.mock import MagicMock

    from emet.memory.graph_eqa.agentic_eqa import ESCAPE_MIN_TRAVEL_M, AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {"eqa": {"room_policy": "canonical"}}
    agent.graph_memory = GraphEQAMemory(defer_llm_clients=True)
    agent.voxel_map = None
    ex = AgenticEQAExecutor(
        agent,
        "Where are the towels in the bathroom?",
        router=False,
        collect_trace=False,
    )
    agent.graph_memory.record_room_event(room="kitchen", kind="verify_absent", step=2, phrase="towels")
    ex._last_room_estimate = "kitchen"
    # Escape floor still requires the existing streak / policy path — history alone is inert.
    assert ex._not_present_streak == 0
    # Without the sticky-mismatch latch from the abandoned PR, streak-only floor stays 0.
    assert ex._escape_min_travel_m() == 0.0
    assert ESCAPE_MIN_TRAVEL_M == 3.0

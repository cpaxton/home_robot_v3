# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared ToolOutcome schema + ledger write helper (Phase 2)."""

from __future__ import annotations

import numpy as np

from emet.agent.tool_outcome import ToolOutcome, maybe_record_tool_attempt
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, GraphNode


def test_tool_outcome_coerce_and_render():
    ok = ToolOutcome.coerce("explore", "Done.")
    assert ok.ok is True
    assert "[explore]" in ok.render()

    bad = ToolOutcome.coerce("explore", "Tool explore failed: boom")
    assert bad.ok is False

    wrapped = ToolOutcome.from_eqa_dict("verify_siglip", {"ok": True, "status": "ABSENT", "obs_id": 3})
    assert wrapped.ok is True
    assert wrapped.status == "ABSENT"


def test_maybe_record_tool_attempt_writes_when_ledger_on():
    mem = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={"eqa": {"attempt_ledger": True}},
    )
    mem._nodes = [
        GraphNode(
            node_id=1,
            obs_id=3,
            xyz=np.array([0.0, 0.0, 0.0]),
            labels=["cup"],
        )
    ]
    maybe_record_tool_attempt(
        mem,
        ToolOutcome(
            ok=False,
            status="ABSENT",
            note="not visible",
            tool="verify_siglip",
            payload={
                "obs_id": 3,
                "phrase": "cup",
                "target_kind": "place",
                "target_id": "place_cup",
                "view_id": "view_cup_1",
                "room": "kitchen",
            },
        ),
        source="eqa",
    )
    rows = mem.get_attempt_records(action_kind="verify")
    assert len(rows) == 1
    assert rows[0].outcome == "absent"
    assert rows[0].phrase == "cup"
    assert rows[0].target_kind == "place"
    assert rows[0].target_id == "place_cup"
    assert rows[0].view_id == "view_cup_1"
    assert rows[0].room == "kitchen"


def test_from_refusal_carries_margin_and_threshold():
    out = ToolOutcome.from_refusal(
        "move_forward",
        refusal_code="clearance",
        margin_m=-0.02,
        threshold_m=0.22,
        relax_allowed=True,
        note="obstacle ahead",
    )
    assert out.ok is False
    assert out.status == "refused"
    assert out.refused is True
    assert out.refusal_code == "clearance"
    assert out.margin_m == -0.02
    assert out.threshold_m == 0.22
    assert out.relax_allowed is True


def test_to_dict_additive_fields_are_absent_when_default():
    plain = ToolOutcome.coerce("explore", "Done.")
    d = plain.to_dict()
    assert "refused" not in d
    assert "margin_m" not in d
    assert "threshold_m" not in d
    assert "refusal_code" not in d
    assert "relax_allowed" not in d


def test_to_dict_refusal_fields_override_payload():
    out = ToolOutcome(
        ok=False,
        status="refused",
        tool="move_forward",
        refused=True,
        refusal_code="clearance",
        margin_m=-0.02,
        threshold_m=0.22,
        relax_allowed=True,
        payload={"refused": False, "margin_m": 999.0},
    )
    d = out.to_dict()
    assert d["refused"] is True
    assert d["margin_m"] == -0.02
    assert d["refusal_code"] == "clearance"
    assert d["threshold_m"] == 0.22
    assert d["relax_allowed"] is True


def test_render_surfaces_margin_for_refusal():
    out = ToolOutcome.from_refusal(
        "move_forward",
        refusal_code="clearance",
        margin_m=-0.02,
        threshold_m=0.22,
        relax_allowed=True,
    )
    text = out.render()
    assert "refused=clearance" in text
    assert "margin=-0.020m vs 0.220m" in text
    assert "(relaxable)" in text


def test_refusal_without_margin_does_not_render_margin():
    out = ToolOutcome.from_refusal("move_forward", refusal_code="collision")
    assert "margin=" not in out.render()
    assert "refused=collision" in out.render()


def test_aim_arm_stub_records_closer_look(monkeypatch):
    mem = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={"eqa": {"attempt_ledger": True}},
    )
    out = ToolOutcome(
        ok=False,
        status="not_implemented",
        note="stub",
        tool="aim_arm_at",
        payload={"phrase": "cables"},
    )
    maybe_record_tool_attempt(mem, out, source="chat")
    rows = mem.get_attempt_records(action_kind="closer_look")
    assert len(rows) == 1
    assert rows[0].outcome == "failed"
    assert rows[0].status_code == "not_implemented"

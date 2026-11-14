# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared ToolOutcome schema + ledger write helper (Phase 2)."""

from __future__ import annotations

import json

import numpy as np

from emet.agent.tool_outcome import ToolOutcome, maybe_record_tool_attempt
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, GraphNode


def test_chat_render_preserves_capture_and_grounding_failure_separately():
    verification = {
        "ok": True,
        "status": "VISUAL_PRESENT",
        "obs_id": 8,
        "visual": {"present": True},
        "geometry": {
            "admitted": False,
            "reason": "outside_depth_range",
            "depth_support": {"view": {"source_obs_id": 8}, "median_depth_m": 3.34},
        },
        "raw": "auxiliary prompt and embedding details",
    }
    outcome = ToolOutcome.coerce(
        "explore_frontier",
        {
            "ok": False,
            "status": "navigation_incomplete",
            "capture": {"ok": True, "obs_id": 8},
            "verify": verification,
        },
    )
    text = outcome.render()
    details = json.loads(text[text.index("{") :])
    assert details["capture"] == {"ok": True, "obs_id": 8}
    assert details["verify"]["visual"]["present"] is True
    assert details["verify"]["geometry"] == verification["geometry"]
    assert "auxiliary prompt" not in text
    assert outcome.ok is False


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

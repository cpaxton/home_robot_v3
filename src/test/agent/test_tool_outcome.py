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
            payload={"obs_id": 3, "phrase": "cup"},
        ),
        source="eqa",
    )
    rows = mem.get_attempt_records(action_kind="verify")
    assert len(rows) == 1
    assert rows[0].outcome == "absent"
    assert rows[0].phrase == "cup"


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

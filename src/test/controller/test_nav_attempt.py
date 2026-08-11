# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Structured NavAttemptResult → ledger / plan meta sync."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from emet.controller.habitat_nav import NavAttemptResult
from emet.controller.nav_attempt import (
    nav_outcome,
    nav_status_code,
    sync_nav_attempt_to_ledger,
    sync_nav_plan_meta,
)
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, GraphNode


def test_nav_status_code_from_note():
    res = NavAttemptResult(
        success=False,
        finished=False,
        dist_m=0.0,
        method="habitat_navmesh",
        note="navmesh_no_path",
    )
    assert nav_status_code(res) == "navmesh_no_path"
    assert nav_outcome(res) == "unreachable"

    ok = NavAttemptResult(True, True, 0.5, "voxel_astar", "ok")
    assert nav_status_code(ok) == "ok"
    assert nav_outcome(ok) == "ok"


def test_sync_nav_plan_meta_stamps_status():
    agent = SimpleNamespace(_last_nav_plan={"mode": "navigation"})
    res = NavAttemptResult(
        success=False,
        finished=False,
        dist_m=0.0,
        method="voxel_astar",
        note="rejected_low_clearance",
    )
    sync_nav_plan_meta(agent, res)
    assert agent._last_nav_plan["status_code"] == "rejected_low_clearance"
    assert agent._last_nav_plan["nav_success"] is False


def test_sync_nav_attempt_to_ledger_when_enabled():
    mem = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={"eqa": {"attempt_ledger": True}},
    )
    mem._nodes = [
        GraphNode(
            node_id=1,
            obs_id=4,
            xyz=np.array([1.0, 2.0, 0.0]),
            labels=["frontier"],
            is_frontier=True,
        )
    ]
    agent = SimpleNamespace(graph_memory=mem, _last_nav_plan={})
    res = NavAttemptResult(
        success=False,
        finished=False,
        dist_m=0.0,
        method="habitat_navmesh",
        note="navmesh_no_path",
        target_obs_id=4,
        goal_xy=(1.0, 2.0),
    )
    sync_nav_attempt_to_ledger(agent, res, source="eqa")
    rows = mem.get_attempt_records(action_kind="navigate")
    assert len(rows) == 1
    assert rows[0].status_code == "navmesh_no_path"
    assert rows[0].outcome == "unreachable"
    assert rows[0].obs_id == 4
    assert agent._last_nav_plan["status_code"] == "navmesh_no_path"


def test_sync_nav_attempt_updates_counters_once_even_when_ledger_off():
    """Classic EQA must not double-call record_nav_attempt after _log_nav_attempt."""
    mem = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={"eqa": {"attempt_ledger": False}},
    )
    mem._nodes = [
        GraphNode(
            node_id=1,
            obs_id=9,
            xyz=np.array([0.5, 0.0, 0.0]),
            labels=["chair"],
        )
    ]
    agent = SimpleNamespace(graph_memory=mem, _last_nav_plan={})
    res = NavAttemptResult(
        success=False,
        finished=False,
        dist_m=0.0,
        method="voxel_astar",
        note="timeout",
        target_obs_id=9,
        goal_xy=(0.5, 0.0),
    )
    sync_nav_attempt_to_ledger(agent, res, source="eqa")
    # Simulate classic path: only fallback when nav_res is None (no second write).
    nav_res = res
    if nav_res is None:
        mem.record_nav_attempt(9, success=False, note="no_nav", dist_m=0.0)
    node = mem._nodes[0]
    assert int(node.nav_attempts) == 1
    assert int(node.nav_failures) == 1
    assert mem.get_attempt_records() == []

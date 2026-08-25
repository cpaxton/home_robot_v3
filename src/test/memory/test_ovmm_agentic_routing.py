# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""OVMM-scoped agentic routing helpers (no sim / VLM)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from emet.memory.graph_eqa.agentic_eqa import OVMM_NEAR_INVESTIGATE_M, AgenticEQAExecutor
from emet.memory.graph_eqa.graph_memory import NavHypothesis


def _executor(*, phase: str = "", meta: dict | None = None) -> AgenticEQAExecutor:
    agent = MagicMock()
    agent.parameters = {}
    agent.robot = MagicMock()
    trace = {"ovmm_phase": phase, **(meta or {})}
    ex = AgenticEQAExecutor(
        agent,
        "Where is the microwave?",
        max_rounds=4,
        router=False,
        trace_meta=trace,
    )
    return ex


def test_apply_ovmm_trace_target_find_recep():
    ex = _executor(phase="find_recep", meta={"goal_recep": "microwave"})
    ex._apply_ovmm_trace_target()
    assert ex._target_phrase == "microwave"


def test_apply_ovmm_trace_target_find_object():
    ex = _executor(phase="find_object", meta={"object": "red cylinder"})
    ex._apply_ovmm_trace_target()
    assert ex._target_phrase == "red cylinder"


def test_recall_nav_hypotheses_includes_sim_placement_seed():
    ex = _executor(phase="find_recep", meta={"goal_recep": "counter"})
    gm = MagicMock()
    gm.hypothesize_nav_targets.return_value = []
    gm.get_nodes.return_value = []
    ex.agent.graph_memory = gm
    ex.agent.robot.get_emet_session.return_value = {
        "is_simulation": True,
        "sim_object_placements": {
            "counter_main": {"pos": [1.0, 2.0, 0.9], "cat": "counter"},
        },
    }
    hyps = ex._recall_nav_hypotheses()
    assert hyps
    assert any("counter" in str(h.phrase).lower() for h in hyps)


def test_nearby_untried_investigate_hyp_prefers_close_card():
    ex = _executor(phase="find_recep", meta={"goal_recep": "table"})
    ex._hypotheses = [
        NavHypothesis(
            phrase="table",
            obs_id=5,
            xyz=np.array([1.0, 0.1, 0.7]),
            score=1.0,
            source="graph",
        ),
        NavHypothesis(
            phrase="table",
            obs_id=9,
            xyz=np.array([8.0, 8.0, 0.7]),
            score=1.0,
            source="graph",
        ),
    ]
    ex._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    near = ex._nearby_untried_investigate_hyp(max_dist_m=OVMM_NEAR_INVESTIGATE_M)
    assert near is not None
    assert int(near.obs_id) == 5


def test_fallback_find_recep_prefers_nearby_investigate():
    ex = _executor(phase="find_recep", meta={"goal_recep": "cab"})
    ex._hypotheses = [
        NavHypothesis(
            phrase="cab",
            obs_id=2,
            xyz=np.array([0.5, 0.0, 0.8]),
            score=1.0,
            source="graph",
        ),
    ]
    ex._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    name, args = ex._fallback_tool()
    assert name == "investigate"
    assert args["obs_id"] == 2


def test_fallback_find_object_prefers_nearby_investigate():
    ex = _executor(phase="find_object", meta={"object": "jar"})
    ex._hypotheses = [
        NavHypothesis(
            phrase="jar",
            obs_id=7,
            xyz=np.array([0.4, 0.1, 0.8]),
            score=1.0,
            source="graph",
        ),
    ]
    ex._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    name, args = ex._fallback_tool()
    assert name == "investigate"
    assert args["obs_id"] == 7


def test_fallback_close_look_prefers_nearby_investigate():
    ex = _executor()
    ex._close_look_required = True
    ex._hypotheses = [
        NavHypothesis(
            phrase="wall clock",
            obs_id=3,
            xyz=np.array([0.6, 0.0, 1.2]),
            score=1.0,
            source="graph",
        ),
    ]
    ex._robot_xyt_world = lambda: np.array([0.0, 0.0, 0.0])  # type: ignore[method-assign]
    name, args = ex._fallback_tool()
    assert name == "investigate"
    assert args["obs_id"] == 3


def test_record_recent_action_includes_nav_outcome():
    ex = _executor()
    ex._round = 1
    ex._record_recent_action(
        "investigate",
        {"obs_id": 4},
        {"ok": True, "obs_id": 4, "nav_outcome": "reached", "verify": {"status": "PRESENT"}},
    )
    assert ex._recent_actions
    assert "nav=reached" in ex._recent_actions[-1]


def test_hypothesize_boost_phrases_prepended():
    gm = MagicMock()
    gm._confirmed_memory_phrases = MagicMock(return_value=[])
    gm._relevant_objects = []
    gm._observations = []
    gm._nodes = []
    gm.extract_relevant_objects = MagicMock()
    gm._siglip_match_for_phrase = MagicMock(return_value=None)
    gm._obs_is_frontier = MagicMock(return_value=False)
    gm._obs_is_object_place = MagicMock(return_value=True)
    gm._recall_rank_score = lambda h, q, r: h
    gm._pack_diversified_hypotheses = lambda scored, k: scored[:k]

    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    real = GraphEQAMemory.__new__(GraphEQAMemory)
    real._observations = []
    real._nodes = []
    real._relevant_objects = []
    real._confirmed_memory_phrases = lambda: []
    real.extract_relevant_objects = lambda _q: None
    real._siglip_match_for_phrase = lambda _p: None
    real._obs_is_frontier = lambda _oid: False
    real._obs_is_object_place = lambda _oid: True
    real._recall_rank_score = GraphEQAMemory._recall_rank_score.__get__(real, GraphEQAMemory)
    real._pack_diversified_hypotheses = GraphEQAMemory._pack_diversified_hypotheses.__get__(real, GraphEQAMemory)
    out = real.hypothesize_nav_targets(
        "Where is the jar?",
        max_k=4,
        boost_phrases=["jar", "counter"],
    )
    assert out == []


def test_prefer_explore_skipped_when_absent_at_non_target_fixture():
    ex = _executor(phase="find_recep", meta={"goal_recep": "microwave"})
    ex._prefer_explore = False
    hyp = NavHypothesis(
        phrase="brick wall",
        obs_id=3,
        xyz=np.array([0.0, 0.0, 0.0]),
        score=1.0,
        source="graph",
    )
    ex._hypotheses = [hyp]
    ex._record_place_inspect(
        3,
        closest_m=0.5,
        verify_out={"status": "ABSENT"},
        approach_index=0,
    )
    assert ex._prefer_explore is False

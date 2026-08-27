# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for OVMM → AgenticEQA question phrasing (no sim / VLM)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np

from emet.eval.ovmm_agentic_find import (
    ovmm_find_object_question,
    ovmm_find_recep_question,
    run_ovmm_agentic_localize,
    should_use_agentic_find,
    xyz_from_verified_obs,
)
from emet.mapping.voxel_localize import localize_text_xyz
from emet.memory.graph_eqa.agentic_eqa import AgenticEQAResult


def _add_claim_and_return(claims: set, claim) -> AgenticEQAResult:
    def _run(*_args, **_kwargs) -> AgenticEQAResult:
        claims.add(claim)
        return AgenticEQAResult(
            discord_text="found",
            answer="here",
            confidence=True,
            verified=True,
            verified_obs_id=7,
            n_rounds=3,
            n_nav=1,
            n_explore=1,
        )

    return _run


def test_ovmm_find_questions():
    assert ovmm_find_object_question("jar", "counter") == "Where is the jar on the counter?"
    assert ovmm_find_object_question("bowl") == "Where is the bowl?"
    assert ovmm_find_recep_question("cab") == "Where is the cab?"


def test_should_use_agentic_find_defaults():
    assert should_use_agentic_find("dynagraph", agentic_find=None) is True
    assert should_use_agentic_find("static_graph", agentic_find=None) is True
    assert should_use_agentic_find("dynamem", agentic_find=None) is False
    assert should_use_agentic_find("ground_truth", agentic_find=None) is False
    assert should_use_agentic_find("dynagraph", agentic_find=False) is False
    assert should_use_agentic_find("dynamem", agentic_find=True) is True


@dataclass
class _Obs:
    obs_id: int
    xyz: np.ndarray
    labels: list[str]


@dataclass
class _Node:
    obs_id: int
    xyz: np.ndarray
    labels: list[str]
    is_frontier: bool = False
    is_viewpoint: bool = False


def test_xyz_from_verified_obs_prefers_matching_object_node():
    """Object graph node centroid beats observation/camera XYZ for localize scoring."""
    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.graph_memory._observations = [_Obs(3, np.array([0.0, 0.0, 0.05]), ["table"])]
    agent.graph_memory.get_nodes.return_value = [
        _Node(3, np.array([0.08, -0.55, 0.6]), ["red cylinder"]),
        _Node(3, np.array([-0.02, -0.55, 0.6]), ["blue cube"]),
    ]
    xyz = xyz_from_verified_obs(agent, 3, phrases=["blue cube"])
    assert xyz is not None
    assert np.allclose(xyz, [-0.02, -0.55, 0.6])


def test_localize_phrases_from_question_and_meta():
    from emet.eval.ovmm_agentic_find import _localize_phrases

    obj_phrases = _localize_phrases(
        "Where is the red cylinder on the table?",
        {
            "ovmm_phase": "find_object",
            "object": "red cylinder",
            "start_recep": "table",
            "goal_recep": "blue cube",
        },
    )
    assert obj_phrases[0] == "red cylinder"
    assert "blue cube" not in obj_phrases
    assert "table" not in obj_phrases

    recep_phrases = _localize_phrases(
        "Where is the blue cube?",
        {
            "ovmm_phase": "find_recep",
            "object": "red cylinder",
            "start_recep": "table",
            "goal_recep": "blue cube",
        },
    )
    assert recep_phrases[0] == "blue cube"
    assert "red cylinder" not in recep_phrases


def test_xyz_from_verified_obs_ignores_camera_pose():
    """Camera XYZ is a viewpoint, not an object — do not score it as FindRec."""
    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.graph_memory._observations = [_Obs(3, np.array([-0.78, 0.47, 0.0]), ["table"])]
    agent.graph_memory.get_nodes.return_value = []
    assert xyz_from_verified_obs(agent, 3, phrases=["blue cube"]) is None


@patch("emet.memory.graph_eqa.agentic_eqa.run_agentic_eqa_result")
def test_run_ovmm_agentic_localize_maps_verified_obs(mock_run):
    mock_run.return_value = AgenticEQAResult(
        discord_text="found",
        answer="here",
        confidence=True,
        verified=True,
        verified_obs_id=7,
        n_rounds=3,
        n_nav=1,
        n_explore=1,
    )
    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.graph_memory._observations = [_Obs(7, np.array([0.5, -0.2, 1.0]), ["cab"])]
    agent.graph_memory._retracted_nav_claims = {("1", "jar")}
    agent.graph_memory.get_nodes.return_value = [
        _Node(7, np.array([0.08, -0.55, 0.6]), ["cab"]),
    ]

    out = run_ovmm_agentic_localize(agent, "Where is the cab?")
    assert out.verified is True
    assert out.verified_obs_id == 7
    assert out.xyz is not None
    assert np.allclose(out.xyz, [0.08, -0.55, 0.6])
    assert out.extra.get("xyz_source") == "graph_node"
    assert out.n_retracted_claims == 0  # pre-existing claim is not this run's delta
    mock_run.assert_called_once()
    assert mock_run.call_args[0][1] == "Where is the cab?"
    assert mock_run.call_args.kwargs.get("trace_path") is None


@patch("emet.memory.graph_eqa.agentic_eqa.run_agentic_eqa_result")
def test_run_ovmm_agentic_localize_voxel_fallback_when_unverified(mock_run):
    mock_run.return_value = AgenticEQAResult(
        discord_text="unknown",
        answer="unknown",
        confidence=False,
        verified=False,
        verified_obs_id=None,
        n_rounds=6,
        n_nav=1,
        n_explore=4,
        budget_hit=True,
    )

    class _Voxel:
        _last_localize_stats = {"query": "blue cube", "max_cosine": 0.17, "yoloe_hit": True}

        def localize_text(self, text, debug=False, return_debug=False):
            if "blue" in str(text).lower():
                return np.array([-0.02, -0.55, 0.6])
            return None

    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.graph_memory._observations = []
    agent.graph_memory._retracted_nav_claims = set()
    agent.graph_memory.get_nodes.return_value = []
    agent.voxel_map = _Voxel()

    out = run_ovmm_agentic_localize(agent, "Where is the blue cube?")
    assert out.verified is False
    assert out.xyz is not None
    assert np.allclose(out.xyz, [-0.02, -0.55, 0.6])
    assert out.extra.get("xyz_source") == "voxel"


@patch("emet.memory.graph_eqa.agentic_eqa.run_agentic_eqa_result")
def test_run_ovmm_agentic_localize_reuses_mapping_pin_when_live_misses(mock_run):
    """Scoring uses the map pin; no OVMM-harness oneshot rescue."""
    mock_run.return_value = AgenticEQAResult(
        discord_text="unknown",
        answer="unknown",
        confidence=False,
        verified=False,
        verified_obs_id=None,
        n_rounds=6,
        n_nav=1,
        n_explore=4,
        budget_hit=True,
    )

    class _Voxel:
        def __init__(self) -> None:
            self.hits = {"blue cube": np.array([-0.02, -0.55, 0.6])}
            self._last_localize_stats: dict[str, object] = {}

        def localize_text(self, text, debug=False, return_debug=False):
            q = str(text or "").strip().lower()
            pt = self.hits.get(q)
            self._last_localize_stats = {
                "query": text,
                "max_cosine": 0.17 if pt is not None else 0.05,
                "yoloe_hit": pt is not None,
            }
            return None if pt is None else pt

    vm = _Voxel()
    localize_text_xyz(vm, "blue cube")
    vm.hits.clear()

    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.graph_memory._observations = []
    agent.graph_memory._retracted_nav_claims = set()
    agent.graph_memory.get_nodes.return_value = []
    agent.voxel_map = vm
    agent.get_voxel_map = lambda: vm

    out = run_ovmm_agentic_localize(agent, "Where is the blue cube?")
    assert out.verified is False
    assert out.xyz is not None
    assert np.allclose(out.xyz, [-0.02, -0.55, 0.6])
    assert out.extra.get("xyz_source") == "voxel"
    assert out.extra.get("from_pin") is True


@patch("emet.memory.graph_eqa.agentic_eqa.run_agentic_eqa_result")
def test_run_ovmm_agentic_localize_find_object_ignores_recep_pin(mock_run):
    """FindObj must not score a mapping-time cube pin just because meta lists both."""
    mock_run.return_value = AgenticEQAResult(
        discord_text="unknown",
        answer="unknown",
        confidence=False,
        verified=False,
        verified_obs_id=None,
        n_rounds=6,
        n_nav=1,
        n_explore=4,
        budget_hit=True,
    )

    class _Voxel:
        def __init__(self) -> None:
            self.hits = {
                "red cylinder": np.array([0.08, -0.55, 0.6]),
                "blue cube": np.array([-0.02, -0.55, 0.6]),
            }
            self._last_localize_stats: dict[str, object] = {}

        def localize_text(self, text, debug=False, return_debug=False):
            q = str(text or "").strip().lower()
            pt = self.hits.get(q)
            self._last_localize_stats = {
                "query": text,
                "max_cosine": 0.22 if pt is not None else 0.05,
                "yoloe_hit": pt is not None,
            }
            return None if pt is None else pt

    vm = _Voxel()
    localize_text_xyz(vm, "red cylinder")
    localize_text_xyz(vm, "blue cube")
    vm.hits.clear()

    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.graph_memory._observations = []
    agent.graph_memory._retracted_nav_claims = set()
    agent.graph_memory.get_nodes.return_value = []
    agent.voxel_map = vm
    agent.get_voxel_map = lambda: vm

    meta = {
        "ovmm_phase": "find_object",
        "object": "red cylinder",
        "start_recep": "table",
        "goal_recep": "blue cube",
    }
    out = run_ovmm_agentic_localize(
        agent,
        "Where is the red cylinder on the table?",
        trace_meta=meta,
    )
    assert out.xyz is not None
    assert np.allclose(out.xyz, [0.08, -0.55, 0.6])
    assert out.extra.get("voxel_query_used") == "red cylinder"

    recep = run_ovmm_agentic_localize(
        agent,
        "Where is the blue cube?",
        trace_meta={**meta, "ovmm_phase": "find_recep"},
    )
    assert recep.xyz is not None
    assert np.allclose(recep.xyz, [-0.02, -0.55, 0.6])
    assert recep.extra.get("voxel_query_used") == "blue cube"


@patch("emet.memory.graph_eqa.agentic_eqa.run_agentic_eqa_result")
def test_run_ovmm_agentic_localize_uses_episode_dir_trace(mock_run, tmp_path, monkeypatch):
    monkeypatch.setenv("EMET_EQA_EPISODE_DIR", str(tmp_path))
    mock_run.return_value = AgenticEQAResult(
        discord_text="found",
        answer="here",
        confidence=True,
        verified=True,
        verified_obs_id=7,
        n_rounds=1,
    )
    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.graph_memory._observations = [_Obs(7, np.array([0.5, -0.2, 1.0]), ["cab"])]
    agent.graph_memory._retracted_nav_claims = set()
    agent.graph_memory.get_nodes.return_value = []
    agent.voxel_map = None

    run_ovmm_agentic_localize(
        agent,
        "Where is the cab?",
        trace_meta={"ovmm_phase": "find_object"},
    )
    trace = mock_run.call_args.kwargs["trace_path"]
    assert trace is not None
    assert trace.name == "find_object_agentic_trace.jsonl"
    assert trace.parent == tmp_path


@patch("emet.memory.graph_eqa.agentic_eqa.run_agentic_eqa_result")
def test_run_ovmm_agentic_localize_counts_new_retractions(mock_run):
    """Retraction count must be this run's delta, not the shared set's total."""
    claims = {("1", "jar")}
    mock_run.side_effect = _add_claim_and_return(claims, ("2", "jar"))
    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.graph_memory._observations = [_Obs(7, np.array([0.5, -0.2, 1.0]), ["cab"])]
    agent.graph_memory._retracted_nav_claims = claims
    agent.graph_memory.get_nodes.return_value = []
    agent.voxel_map = None

    out = run_ovmm_agentic_localize(agent, "Where is the cab?")
    assert out.verified is True
    assert out.n_retracted_claims == 1  # one claim added during this run

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

    phrases = _localize_phrases(
        "Where is the red cylinder on the table?",
        {"object": "red cylinder", "start_recep": "table"},
    )
    assert "red cylinder" in phrases
    assert "table" in phrases


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
    agent.graph_memory.get_nodes.return_value = []

    out = run_ovmm_agentic_localize(agent, "Where is the cab?")
    assert out.verified is True
    assert out.verified_obs_id == 7
    assert out.xyz is not None
    assert np.allclose(out.xyz, [0.5, -0.2, 1.0])
    assert out.n_retracted_claims == 0  # pre-existing claim is not this run's delta
    mock_run.assert_called_once()
    assert mock_run.call_args[0][1] == "Where is the cab?"
    assert mock_run.call_args.kwargs.get("trace_path") is None


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

    out = run_ovmm_agentic_localize(agent, "Where is the cab?")
    assert out.verified is True
    assert out.n_retracted_claims == 1  # one claim added during this run

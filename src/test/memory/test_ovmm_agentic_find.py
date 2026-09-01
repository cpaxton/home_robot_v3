# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for OVMM → AgenticEQA question phrasing (no sim / VLM)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np

from emet.eval.ovmm_agentic_find import (
    OvmmAgenticLocalizeResult,
    OvmmFindQueryOutcome,
    ovmm_find_object_question,
    ovmm_find_recep_question,
    record_ovmm_agentic_result,
    run_ovmm_agentic_find_pair,
    run_ovmm_agentic_localize,
    run_ovmm_find_queries,
    should_use_agentic_find,
    xyz_from_verified_obs,
)
from emet.mapping.voxel_localize import localize_text_xyz
from emet.memory.graph_eqa import AgenticEQAResult


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


def test_record_ovmm_agentic_result_copies_nav_and_voxel_query():
    res = OvmmAgenticLocalizeResult(
        question="Where is the lamp on the bed?",
        verified=True,
        verified_obs_id=3,
        xyz=np.asarray([1.0, 1.2, 2.0]),
        n_rounds=4,
        n_nav=2,
        n_explore=1,
        n_retracted_claims=1,
        extra={"xyz_source": "voxel", "voxel_query_used": "lamp", "from_pin": False},
    )
    meta: dict = {}
    xyz, ok, q_used, source = record_ovmm_agentic_result(
        res, meta=meta, prefix="obj", default_query="lamp"
    )
    assert ok is True
    assert np.allclose(xyz, [1.0, 1.2, 2.0])
    assert q_used == "lamp"
    assert source == "voxel"
    assert meta["obj_agentic_rounds"] == 4
    assert meta["obj_n_nav"] == 2
    assert meta["obj_n_explore"] == 1
    assert meta["obj_n_retracted_claims"] == 1
    assert meta["obj_xyz_source"] == "voxel"


@patch("emet.eval.ovmm_agentic_find.run_ovmm_agentic_localize")
def test_run_ovmm_agentic_find_pair_phrases_both_phases(mock_loc):

    def _loc(_agent, question, **kwargs):
        phase = (kwargs.get("trace_meta") or {}).get("ovmm_phase")
        xyz = np.array([1.0, 0.0, 2.0]) if phase == "find_object" else np.array([4.0, 0.0, 4.0])
        used = "lamp" if phase == "find_object" else "table"
        return OvmmAgenticLocalizeResult(
            question=question,
            verified=True,
            verified_obs_id=1,
            xyz=xyz,
            n_rounds=2,
            n_nav=1,
            extra={"xyz_source": "voxel", "voxel_query_used": used},
        )

    mock_loc.side_effect = _loc
    out = run_ovmm_agentic_find_pair(
        MagicMock(),
        object_query="lamp",
        start_recep="bed",
        goal_recep="table",
        episode_id="ep1",
        object_gt_body="lamp_body",
        extra_trace_meta={"scene": "00006"},
    )
    assert mock_loc.call_count == 2
    assert mock_loc.call_args_list[0].args[1] == "Where is the lamp on the bed?"
    assert mock_loc.call_args_list[1].args[1] == "Where is the table?"
    obj_meta = mock_loc.call_args_list[0].kwargs["trace_meta"]
    recep_meta = mock_loc.call_args_list[1].kwargs["trace_meta"]
    assert obj_meta["ovmm_phase"] == "find_object"
    assert obj_meta["gt_body_key"] == "lamp_body"
    assert obj_meta["scene"] == "00006"
    assert recep_meta["ovmm_phase"] == "find_recep"
    assert "gt_body_key" not in recep_meta
    assert out.meta["agentic_find"] is True
    assert out.obj_ok is True
    assert out.recep_ok is True
    assert out.meta["obj_n_nav"] == 1
    assert out.meta["recep_n_nav"] == 1


@patch("emet.eval.ovmm_find_phase.run_ovmm_oneshot_find_pair")
@patch("emet.eval.ovmm_agentic_find.run_ovmm_agentic_find_pair")
def test_run_ovmm_find_queries_dispatches_agentic_vs_oneshot(mock_agentic, mock_oneshot):
    canned = OvmmFindQueryOutcome(
        obj_xyz=np.array([1.0, 0.0, 2.0]),
        obj_ok=True,
        obj_query_used="lamp",
        obj_source="voxel",
        recep_xyz=np.array([4.0, 0.0, 4.0]),
        recep_ok=True,
        recep_query_used="table",
        recep_source="voxel",
        meta={"agentic_find": True},
    )
    mock_agentic.return_value = canned
    mock_oneshot.return_value = canned
    kw = {
        "agent": MagicMock(),
        "memory": MagicMock(),
        "object_query": "lamp",
        "start_recep": "bed",
        "goal_recep": "table",
        "episode_id": "ep1",
    }
    run_ovmm_find_queries(use_agentic=True, **kw)
    mock_agentic.assert_called_once()
    mock_oneshot.assert_not_called()
    run_ovmm_find_queries(use_agentic=False, **kw)
    mock_oneshot.assert_called_once()
    oneshot_kw = mock_oneshot.call_args.kwargs
    assert oneshot_kw["capture_voxel_stats"] is True
    assert oneshot_kw["planar_frame"] == "mujoco_xy"


def test_ovmm_find_query_row_includes_localize_pred_and_detect():
    from emet.eval.ovmm_find_phase import ovmm_find_query_row

    out = OvmmFindQueryOutcome(
        obj_xyz=np.array([1.0, 0.0, 2.0]),
        obj_ok=True,
        obj_query_used="lamp",
        obj_source="voxel",
        recep_xyz=np.array([4.0, 0.0, 4.0]),
        recep_ok=True,
        recep_query_used="table",
        recep_source="voxel",
        meta={"agentic_find": False, "obj_n_nav": 0},
        obj_detect_stats={"max_cosine": 0.4, "yoloe_hit": True},
        recep_detect_stats={"max_cosine": 0.2, "yoloe_hit": False},
    )
    row = ovmm_find_query_row(out)
    assert row["obj_localize_success"] is True
    assert row["recep_localize_source"] == "voxel"
    assert row["pred_obj_xyz"] == [1.0, 0.0, 2.0]
    assert row["obj_max_cosine"] == 0.4
    assert row["obj_yoloe_hit"] is True
    assert row["recep_yoloe_hit"] is False
    assert row["agentic_find"] is False


def test_run_ovmm_gt_oracle_find_pair_uses_placements():
    from emet.eval.ovmm_find_phase import run_ovmm_gt_oracle_find_pair

    placements = {
        "lamp_body": {"cat": "lamp", "pos": [1.0, 0.5, 2.0]},
        "table_1": {"cat": "table", "pos": [4.0, 0.0, 4.0]},
    }
    out = run_ovmm_gt_oracle_find_pair(
        memory=MagicMock(),
        object_query="lamp",
        start_recep="bed",
        goal_recep="table",
        object_gt_body="lamp_body",
        placements=placements,
    )
    assert out.obj_ok is True
    assert out.obj_source == "gt_placement"
    assert np.allclose(out.obj_xyz, [1.0, 0.5, 2.0])
    assert out.recep_ok is True
    assert out.recep_source == "gt_placement"
    assert np.allclose(out.recep_xyz, [4.0, 0.0, 4.0])
    assert out.meta["agentic_find"] is False


@patch("emet.eval.ovmm_find_phase.query_find_phase_localization")
def test_run_ovmm_gt_oracle_find_pair_falls_back_when_body_missing(mock_q):
    from emet.eval.ovmm_find_phase import run_ovmm_gt_oracle_find_pair

    mock_q.side_effect = [
        (np.array([1.0, 0.0, 2.0]), True, "lamp", "voxel"),
        (np.array([4.0, 0.0, 4.0]), True, "table", "graph_near_recep"),
    ]
    out = run_ovmm_gt_oracle_find_pair(
        memory=MagicMock(),
        object_query="lamp",
        start_recep="bed",
        goal_recep="table",
        object_gt_body="missing_body",
        placements={"other": {"cat": "chair", "pos": [0.0, 0.0, 0.0]}},
    )
    assert mock_q.call_count == 2
    assert out.obj_source == "voxel"
    assert out.recep_source == "graph_near_recep"


def test_score_ovmm_find_query_habitat_frame_matches_adapter():
    from emet.eval.habitat_ovmm_find import score_habitat_find_phase
    from emet.eval.ovmm_find_phase import score_ovmm_find_query

    placements = {
        "hm3d_lamp_1": {
            "cat": "lamp",
            "pos": [1.0, 1.2, 2.0],
            "bounds": [[0.8, 1.0, 1.8], [1.2, 1.4, 2.2]],
            "frame": "habitat_yup",
        },
        "hm3d_bed_2": {
            "cat": "bed",
            "pos": [0.5, 0.0, 1.0],
            "bounds": [[0.0, 0.0, 0.5], [1.0, 0.5, 1.5]],
            "frame": "habitat_yup",
        },
    }
    query = OvmmFindQueryOutcome(
        obj_xyz=np.array([1.0, 1.2, 2.0]),
        obj_ok=True,
        obj_query_used="lamp",
        obj_source="voxel",
        recep_xyz=np.array([0.5, 0.0, 1.0]),
        recep_ok=True,
        recep_query_used="bed",
        recep_source="voxel",
        meta={"agentic_find": False},
    )
    via_query = score_ovmm_find_query(
        query,
        placements=placements,
        object_query="lamp",
        start_recep="bed",
        goal_recep="bed",
        radius_m=0.75,
        frame="habitat_xz",
    )
    via_adapter = score_habitat_find_phase(
        obj_pred_xyz=query.obj_xyz,
        recep_pred_xyz=query.recep_xyz,
        placements=placements,
        object_query="lamp",
        start_recep="bed",
        goal_recep="bed",
        radius_m=0.75,
    )
    assert via_query == via_adapter
    assert via_query["find_object_success"] is True
    assert via_query["find_recep_success"] is True


def test_attach_ovmm_episode_debug_dir(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from emet.eval.ovmm_agentic_find import attach_ovmm_episode_debug_dir

    agent = SimpleNamespace()
    monkeypatch.delenv("EMET_EQA_EPISODE_DIR", raising=False)
    attach_ovmm_episode_debug_dir(agent)
    assert not hasattr(agent, "_episode_debug_dir")
    monkeypatch.setenv("EMET_EQA_EPISODE_DIR", str(tmp_path))
    attach_ovmm_episode_debug_dir(agent)
    assert agent._episode_debug_dir == str(tmp_path)


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
        voxel_xyz=(-0.02, -0.55, 0.6),
        voxel_phrase="blue cube",
        voxel_from_pin=False,
    )

    class _Voxel:
        def localize_text(self, *_a, **_k):
            raise AssertionError("scoring must not re-query localize_text")

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
    assert out.extra.get("voxel_query_used") == "blue cube"
    assert out.extra.get("from_pin") is False


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

    def _boom(*_a, **_k):
        raise AssertionError("pin lookup must not live-query")

    vm.localize_text = _boom

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
def test_run_ovmm_agentic_localize_uses_loop_voxel_when_live_map_dead(mock_run):
    """Submit releases SigLIP; score the loop's object-phrase XYZ, do not re-query."""
    mock_run.return_value = AgenticEQAResult(
        discord_text="unknown",
        answer="unknown",
        confidence=False,
        verified=False,
        verified_obs_id=None,
        n_rounds=6,
        n_nav=4,
        n_explore=1,
        budget_hit=True,
        voxel_xyz=(0.08, -0.55, 0.6),
        voxel_phrase="red cylinder",
        voxel_from_pin=True,
    )

    class _DeadVoxel:
        def localize_text(self, text, debug=False, return_debug=False):
            raise AssertionError("scoring must not live-query after SigLIP release")

    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.graph_memory._observations = []
    agent.graph_memory._retracted_nav_claims = set()
    agent.graph_memory.get_nodes.return_value = []
    agent.voxel_map = _DeadVoxel()

    out = run_ovmm_agentic_localize(
        agent,
        "Where is the red cylinder on the table?",
        trace_meta={"ovmm_phase": "find_object", "object": "red cylinder"},
    )
    assert out.verified is False
    assert out.xyz is not None
    assert np.allclose(out.xyz, [0.08, -0.55, 0.6])
    assert out.extra.get("xyz_source") == "voxel"
    assert out.extra.get("from_pin") is True
    assert out.extra.get("voxel_query_used") == "red cylinder"


@patch("emet.memory.graph_eqa.agentic_eqa.run_agentic_eqa_result")
def test_run_ovmm_agentic_localize_rejects_fixture_wrap_loop_xyz(mock_run):
    """A table 3-gram snapshot is not FindObj even if the loop stashed it."""
    mock_run.return_value = AgenticEQAResult(
        discord_text="unknown",
        answer="unknown",
        confidence=False,
        verified=False,
        verified_obs_id=None,
        n_rounds=6,
        budget_hit=True,
        voxel_xyz=(0.04, -0.55, 0.6),
        voxel_phrase="red cylinder table",
        voxel_from_pin=False,
    )
    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.graph_memory._observations = []
    agent.graph_memory._retracted_nav_claims = set()
    agent.graph_memory.get_nodes.return_value = []
    agent.voxel_map = None

    out = run_ovmm_agentic_localize(
        agent,
        "Where is the red cylinder on the table?",
        trace_meta={"ovmm_phase": "find_object", "object": "red cylinder"},
    )
    assert out.xyz is None
    assert out.extra.get("xyz_source") is None


@patch("emet.memory.graph_eqa.agentic_eqa.run_agentic_eqa_result")
def test_run_ovmm_agentic_localize_prefers_loop_voxel_over_graph_node(mock_run):
    mock_run.return_value = AgenticEQAResult(
        discord_text="found",
        answer="here",
        confidence=True,
        verified=True,
        verified_obs_id=7,
        n_rounds=3,
        voxel_xyz=(0.08, -0.55, 0.6),
        voxel_phrase="red cylinder",
        voxel_from_pin=False,
    )
    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.graph_memory._observations = []
    agent.graph_memory._retracted_nav_claims = set()
    agent.graph_memory.get_nodes.return_value = [
        _Node(7, np.array([9.0, 9.0, 9.0]), ["red cylinder"]),
    ]
    agent.voxel_map = None

    out = run_ovmm_agentic_localize(
        agent,
        "Where is the red cylinder on the table?",
        trace_meta={"ovmm_phase": "find_object", "object": "red cylinder"},
    )
    assert out.xyz is not None
    assert np.allclose(out.xyz, [0.08, -0.55, 0.6])
    assert out.extra.get("xyz_source") == "voxel"


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

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from emet.habitat.episode_debug import (
    run_tag_from_output_jsonl,
    save_error_episode_bundle,
    write_run_manifest,
)
from emet.habitat.metrics import EpisodeMetrics


def test_run_tag_from_output_jsonl():
    assert run_tag_from_output_jsonl(Path("/tmp/frontier_v2_gemma4_q0-19.jsonl")) == "frontier_v2_gemma4_q0-19"


def test_write_run_manifest(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    questions = tmp_path / "questions.csv"
    questions.write_text("scene,floor,question,choices,answer\ns,0,q,\"['a','b','c','d']\",A\n")
    init_poses = tmp_path / "scene_init_poses.csv"
    init_poses.write_text("scene_floor,init_x,init_y,init_z,init_angle\ns_0,0,0,0,0\n")
    hm3d_root = tmp_path / "hm3d"
    hm3d_root.mkdir()
    manifest = write_run_manifest(
        output_jsonl=out,
        method="graph_eqa",
        question_ids=[0, 1],
        mock_llm=False,
        max_planning_steps=20,
        max_movement_step=10,
        eqa_vl_family="gemma4",
        eqa_hf_model_id="google/gemma-3-4b-it",
        eqa_vl_quantization="int4",
        device="cuda",
        resume=True,
        parameters={"graph_eqa_frontier_nodes": {"enabled": True}},
        hm3d_root=hm3d_root,
        questions_path=questions,
        init_poses_path=init_poses,
    )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["run_tag"] == "run"
    assert data["method"] == "graph_eqa"
    assert data["question_ids"] == [0, 1]
    assert data["schema_version"] == 2
    assert "harness" in data
    assert data["external_inputs"]["questions"]["sha256"].startswith("sha256:")


def test_write_run_manifest_refuses_resume_configuration_or_dataset_drift(
    tmp_path: Path,
    monkeypatch,
):
    out = tmp_path / "run.jsonl"
    questions = tmp_path / "questions.csv"
    questions.write_text("scene,floor,question,choices,answer\ns,0,q,\"['a','b','c','d']\",A\n")
    init_poses = tmp_path / "scene_init_poses.csv"
    init_poses.write_text("scene_floor,init_x,init_y,init_z,init_angle\ns_0,0,0,0,0\n")
    hm3d_root = tmp_path / "hm3d"
    hm3d_root.mkdir()
    kwargs = {
        "output_jsonl": out,
        "method": "static_graph",
        "question_ids": [0],
        "mock_llm": False,
        "max_planning_steps": 20,
        "max_movement_step": 10,
        "eqa_vl_family": "qwen3_vl",
        "eqa_hf_model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "eqa_vl_quantization": "int4",
        "device": "cuda",
        "resume": False,
        "parameters": {"eqa": {"merged_memory": False}},
        "hm3d_root": hm3d_root,
        "questions_path": questions,
        "init_poses_path": init_poses,
    }
    write_run_manifest(**kwargs)
    out.touch()

    with pytest.raises(ValueError, match="refusing to append"):
        write_run_manifest(**kwargs)

    monkeypatch.setenv("EMET_EQA_FORCE_ANSWER", "0")
    ambient = dict(kwargs)
    ambient["resume"] = True
    with pytest.raises(ValueError, match="behavior_environment"):
        write_run_manifest(**ambient)
    monkeypatch.delenv("EMET_EQA_FORCE_ANSWER")

    changed = dict(kwargs)
    changed["resume"] = True
    changed["max_planning_steps"] = 19
    with pytest.raises(ValueError, match="max_planning_steps"):
        write_run_manifest(**changed)

    questions.write_text("scene,floor,question,choices,answer\ns,0,changed,\"['a','b','c','d']\",A\n")
    resumed = dict(kwargs)
    resumed["resume"] = True
    with pytest.raises(ValueError, match="external_inputs"):
        write_run_manifest(**resumed)


def test_write_run_manifest_refuses_resume_when_output_is_missing(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    questions = tmp_path / "questions.csv"
    questions.write_text("scene,floor,question,choices,answer\ns,0,q,\"['a','b','c','d']\",A\n")
    init_poses = tmp_path / "scene_init_poses.csv"
    init_poses.write_text("scene_floor,init_x,init_y,init_z,init_angle\ns_0,0,0,0,0\n")
    hm3d_root = tmp_path / "hm3d"
    hm3d_root.mkdir()
    kwargs = {
        "output_jsonl": out,
        "method": "static_graph",
        "question_ids": [0],
        "mock_llm": False,
        "max_planning_steps": 20,
        "max_movement_step": 10,
        "eqa_vl_family": "qwen3_vl",
        "eqa_hf_model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "eqa_vl_quantization": "int4",
        "device": "cuda",
        "resume": False,
        "parameters": {},
        "hm3d_root": hm3d_root,
        "questions_path": questions,
        "init_poses_path": init_poses,
    }
    write_run_manifest(**kwargs)
    out.touch()
    out.unlink()
    with pytest.raises(ValueError, match="results JSONL is missing"):
        write_run_manifest(**{**kwargs, "resume": True})


def test_write_run_manifest_recovers_empty_output_without_manifest(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    questions = tmp_path / "questions.csv"
    questions.write_text("scene,floor,question,choices,answer\ns,0,q,\"['a','b','c','d']\",A\n")
    init_poses = tmp_path / "scene_init_poses.csv"
    init_poses.write_text("scene_floor,init_x,init_y,init_z,init_angle\ns_0,0,0,0,0\n")
    hm3d_root = tmp_path / "hm3d"
    hm3d_root.mkdir()
    out.touch()

    manifest = write_run_manifest(
        output_jsonl=out,
        method="static_graph",
        question_ids=[0],
        mock_llm=False,
        max_planning_steps=20,
        max_movement_step=10,
        eqa_vl_family="qwen3_vl",
        eqa_hf_model_id="Qwen/Qwen3-VL-8B-Instruct",
        eqa_vl_quantization="int4",
        device="cuda",
        resume=True,
        parameters={},
        hm3d_root=hm3d_root,
        questions_path=questions,
        init_poses_path=init_poses,
    )

    assert manifest == tmp_path / "run_manifest.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["last_invocation_resume"] is True
    assert out.read_bytes() == b""


def test_hmeqa_batch_does_not_delete_existing_output_before_manifest_guard(tmp_path: Path, monkeypatch):
    project_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(project_root / "packages" / "emet_habitat"))
    from emet_habitat import runner

    output = tmp_path / "run.jsonl"
    output.write_text("preserve this row\n", encoding="utf-8")
    monkeypatch.setattr(runner, "load_hmeqa_questions", lambda _path: [])
    monkeypatch.setattr(runner, "_validate_requested_hm3d_semantics", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_configure_frontier_parameters", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_configure_habitat_nav", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_configure_habitat_mapping", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_apply_method_parameters", lambda parameters, _method: parameters)
    monkeypatch.setattr(runner, "apply_dynagraph_harness_overrides", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_configure_eqa_parameters", lambda *_args, **_kwargs: None)

    def _guard(**_kwargs):
        raise RuntimeError("guard")

    monkeypatch.setattr(runner, "write_run_manifest", _guard)

    with pytest.raises(RuntimeError, match="guard"):
        runner.run_hmeqa_batch(question_ids=[], output_jsonl=output, resume=False)
    assert output.read_text(encoding="utf-8") == "preserve this row\n"


def test_hmeqa_runner_binds_numeric_context_before_agent_start(monkeypatch):
    project_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(project_root / "packages" / "emet_habitat"))
    from emet_habitat.runner import _start_hmeqa_agent_with_context

    calls: list[str] = []
    graph_memory = MagicMock()
    graph_memory.bind_episode_context.side_effect = lambda **_kwargs: calls.append("bind")
    agent = SimpleNamespace(
        graph_memory=graph_memory,
        start=MagicMock(side_effect=lambda: calls.append("start")),
    )

    trace_meta = _start_hmeqa_agent_with_context(
        agent,
        question_id=11,
        scene="yogvKWUrdnw",
        method="dynagraph",
        debug_run_tag="gre-q11-integrity",
    )

    assert calls == ["bind", "start"]
    graph_memory.bind_episode_context.assert_called_once_with(
        question_id=11,
        session_id="gre-q11-integrity",
    )
    assert trace_meta["question_id"] == trace_meta["qid"] == 11
    assert trace_meta["session_id"] == "gre-q11-integrity"
    assert agent._eqa_trace_meta == trace_meta


def test_enrich_episode_metrics_harness_fingerprint_merge_on():
    from emet.core.parameters import get_parameters
    from emet.eval.benchmark_dynagraph import apply_habitat_eqa_method_parameters
    from emet.habitat.episode_debug import enrich_episode_metrics

    params = apply_habitat_eqa_method_parameters(get_parameters("dynav_config.yaml"), "dynagraph")
    agent = SimpleNamespace(parameters=params, graph_memory=None)
    metrics = EpisodeMetrics(
        dataset="hmeqa",
        method="dynagraph",
        question_id=17,
        scene="s",
        floor=0,
        question="q",
        gold_answer_letter="D",
        predicted_answer="D",
        correct=True,
        confident=True,
        planning_steps=1,
        success=True,
    )
    enrich_episode_metrics(metrics, agent=agent, choices=["a", "b", "c", "d"])
    assert float(metrics.harness.get("dynagraph_merge_xy_m")) == 0.45
    assert float(metrics.harness.get("fallback_spatial_merge_xy_m")) == 0.45
    assert metrics.harness.get("profile") == "unified_eqa"
    assert metrics.harness.get("explore_when_uncovered") == "conservative"


def test_save_episode_debug_bundle_writes_graph_report(tmp_path: Path, monkeypatch):
    from emet.habitat.episode_debug import save_episode_debug_bundle
    from emet.habitat.metrics import EpisodeMetrics

    monkeypatch.setattr(
        "emet.habitat.episode_debug.default_episodes_root",
        lambda: tmp_path / "episodes",
    )

    class _Node:
        is_frontier = True
        node_id = 1
        labels = ["frontier", "bed"]
        xyz = __import__("numpy").array([1.0, 2.0, 0.0])
        description = "frontier:c1"
        obs_id = 1

    class _GraphMem:
        def get_nodes(self):
            return [_Node()]

        def get_edges(self):
            return []

        def get_navigation_samples(self):
            return []

    class _Agent:
        graph_memory = _GraphMem()
        voxel_map = None

    metrics = EpisodeMetrics(
        dataset="hmeqa",
        method="graph_eqa",
        question_id=0,
        scene="s",
        floor=0,
        question="q",
        gold_answer_letter="B",
        predicted_answer="B",
        correct=True,
        confident=False,
        planning_steps=10,
        success=True,
    )
    bundle = save_episode_debug_bundle(
        run_tag="test_run",
        metrics=metrics,
        agent=_Agent(),
        raw_eqa_full="Answer:\nB\n",
    )
    assert (bundle / "scene_graph_report.txt").is_file()
    assert (bundle / "raw_eqa.txt").read_text(encoding="utf-8") == "Answer:\nB\n"
    assert json.loads((bundle / "frontier_nodes.json").read_text(encoding="utf-8"))[0]["labels"] == [
        "frontier",
        "bed",
    ]


def test_save_episode_debug_bundle_writes_agentic_evidence(tmp_path: Path, monkeypatch):
    import numpy as np

    from emet.habitat.episode_debug import save_episode_debug_bundle
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    monkeypatch.setattr(
        "emet.habitat.episode_debug.default_episodes_root",
        lambda: tmp_path / "episodes",
    )

    graph_memory = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={
            "eqa": {
                "graph_evidence_mode": "shadow",
                "attempt_ledger": True,
            }
        },
    )
    graph_memory.world_evidence.session_id = "session-1"
    graph_memory.set_attempt_ledger_question_id("12")
    obs_id = graph_memory.add_observation(
        np.full((4, 4, 3), 17, dtype=np.uint8),
        np.array([1.0, 2.0, 0.5]),
        ["clock"],
        viewer_xyz=np.array([0.0, 2.0, 0.0]),
    )
    graph_memory.record_attempt(
        action_kind="investigate",
        outcome="absent",
        status_code="vlm_absent",
        obs_id=obs_id,
        step=3,
        source="eqa",
    )
    graph_memory.record_room_event(
        room="kitchen",
        kind="verify_absent",
        obs_id=obs_id,
        step=3,
    )

    class _Agent:
        voxel_map = None
        obs_count = 42

    agent = _Agent()
    agent.graph_memory = graph_memory
    metrics = EpisodeMetrics(
        dataset="hmeqa",
        method="dynagraph",
        question_id=12,
        scene="s",
        floor=0,
        question="q",
        gold_answer_letter="D",
        predicted_answer="D",
        correct=True,
        confident=True,
        planning_steps=3,
        success=True,
    )
    bundle = save_episode_debug_bundle(
        run_tag="shadow_run",
        metrics=metrics,
        agent=agent,
    )

    world = json.loads((bundle / "world_evidence.json").read_text(encoding="utf-8"))
    attempts = json.loads((bundle / "attempt_ledger.json").read_text(encoding="utf-8"))
    rooms = json.loads((bundle / "room_events.json").read_text(encoding="utf-8"))
    assert world["mode"] == "shadow"
    assert world["question_id"] == "12"
    assert world["events"][0]["event_id"] == "event_00000001"
    assert attempts[0]["action_kind"] == "investigate"
    assert attempts[0]["outcome"] == "absent"
    assert rooms[0]["kind"] == "verify_absent"
    assert rooms[0]["room"] == "kitchen"
    assert (bundle / world["views"][0]["rgb_file"]).is_file()


def test_save_episode_debug_bundle_writes_reloadable_compact_memory(tmp_path: Path, monkeypatch):
    import numpy as np

    from emet.eval.episode_diagnostics import EpisodeDiagnosticsConfig
    from emet.habitat.episode_debug import save_episode_debug_bundle
    from emet.memory.adapters import GraphEQABackend
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    monkeypatch.setattr(
        "emet.habitat.episode_debug.default_episodes_root",
        lambda: tmp_path / "episodes",
    )
    graph_memory = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={"eqa": {"graph_evidence_mode": "shadow"}},
    )
    graph_memory.add_observation(
        np.full((32, 32, 3), 17, dtype=np.uint8),
        np.array([1.0, 2.0, 0.5]),
        ["clock"],
        viewer_xyz=np.array([0.0, 2.0, 0.0]),
    )

    class _Agent:
        voxel_map = None
        obs_count = 42

    agent = _Agent()
    agent.graph_memory = graph_memory
    metrics = EpisodeMetrics(
        dataset="hmeqa",
        method="dynagraph",
        question_id=12,
        scene="s",
        floor=0,
        question="q",
        gold_answer_letter="D",
        predicted_answer="D",
        correct=True,
        confident=True,
        planning_steps=3,
        success=True,
    )
    cfg = EpisodeDiagnosticsConfig(
        export_map=False,
        export_obstacle_grids=False,
        export_trajectory=False,
        export_rgb_frames=False,
        export_video=False,
        export_object_crops=False,
        export_compact_memory=True,
        export_world_evidence_rgb=False,
        export_gt_navmesh_map=False,
        export_map_overlay=False,
        export_map_video=False,
        export_video_substeps=False,
    )

    bundle = save_episode_debug_bundle(
        run_tag="compact_run",
        metrics=metrics,
        agent=agent,
        diagnostics_cfg=cfg,
    )
    compact = bundle / "compact_memory"
    restored = GraphEQAMemory(defer_llm_clients=True)
    GraphEQABackend(restored).load(str(compact))

    assert (compact / "manifest.json").is_file()
    assert (compact / "graph.json").is_file()
    assert not (compact / "frames").exists()
    assert not (compact / "world_evidence_views").exists()
    assert not (bundle / "world_evidence_views").exists()
    assert restored.get_nodes()[0].labels == ["clock"]
    manifest = json.loads((compact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["final_step"] == 42
    assert manifest["checkpoint_profile"] == "graph_only"
    diagnostics = json.loads((bundle / "diagnostics_manifest.json").read_text(encoding="utf-8"))
    assert diagnostics["compact_memory"] == str(compact)


def test_grapheqa_baseline_defaults_to_compact_memories():
    script = Path(__file__).resolve().parents[3] / "scripts" / "run_hmeqa_grapheqa_baseline.sh"
    text = script.read_text(encoding="utf-8")

    assert 'EMET_EVAL_EXPORT_COMPACT_MEMORY="${EMET_EVAL_EXPORT_COMPACT_MEMORY:-1}"' in text
    assert 'EMET_EVAL_EXPORT_WORLD_EVIDENCE_RGB="${EMET_EVAL_EXPORT_WORLD_EVIDENCE_RGB:-0}"' in text
    assert 'EMET_EVAL_EXPORT_FRAMES="${EMET_EVAL_EXPORT_FRAMES:-0}"' in text
    assert 'EMET_EVAL_EXPORT_VIDEO="${EMET_EVAL_EXPORT_VIDEO:-0}"' in text
    assert 'cp -a "$checkpoint/." "$compact_out/$episode_name/"' in text


def test_save_error_episode_bundle(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "emet.habitat.episode_debug.default_episodes_root",
        lambda: tmp_path / "episodes",
    )
    metrics = EpisodeMetrics(
        dataset="hmeqa",
        method="graph_eqa",
        question_id=3,
        scene="s",
        floor=0,
        question="q",
        gold_answer_letter="B",
        predicted_answer="ERROR: boom",
        correct=False,
        confident=False,
        planning_steps=0,
        success=False,
        error="boom",
    )
    prior = tmp_path / "episodes" / "test_run" / "q0003_graph_eqa"
    prior.mkdir(parents=True)
    (prior / "stale_success_artifact.json").write_text("stale", encoding="utf-8")
    bundle = save_error_episode_bundle(run_tag="test_run", metrics=metrics)
    assert (bundle / "error.txt").read_text(encoding="utf-8") == "boom"
    assert json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))["question_id"] == 3
    assert not (bundle / "stale_success_artifact.json").exists()

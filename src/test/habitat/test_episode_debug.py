# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import json
from pathlib import Path

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
    manifest = write_run_manifest(
        output_jsonl=out,
        method="graph_eqa",
        question_ids=[0, 1],
        mock_llm=False,
        max_planning_steps=20,
        max_movement_step=10,
        eqa_vl_family="gemma4",
        eqa_hf_model_id="google/gemma-3-4b-it",
        device="cuda",
        resume=True,
        parameters={"graph_eqa_frontier_nodes": {"enabled": True}},
    )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["run_tag"] == "run"
    assert data["method"] == "graph_eqa"
    assert data["question_ids"] == [0, 1]
    assert "harness" in data


def test_enrich_episode_metrics_harness_fingerprint_merge_on():
    from types import SimpleNamespace

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
    bundle = save_error_episode_bundle(run_tag="test_run", metrics=metrics)
    assert (bundle / "error.txt").read_text(encoding="utf-8") == "boom"
    assert json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))["question_id"] == 3

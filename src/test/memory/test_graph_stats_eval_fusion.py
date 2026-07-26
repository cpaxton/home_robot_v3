# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for Dynagraph graph-health metrics, label dedup, and EQA prompt top-K."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from emet.eval.benchmark_dynagraph import apply_eval_graph_fusion_parameters
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.graph_eqa.graph_stats import (
    classify_graph_failure,
    format_graph_node_breakdown,
    format_graph_size_report,
    graph_health_from_checkpoint_nodes,
    graph_health_metrics,
    graph_node_breakdown,
    labels_compatible_for_dedup,
)


class _FakeNode:
    def __init__(self, *, viewpoint=False, frontier=False, support=1, labels=None):
        self.is_viewpoint = viewpoint
        self.is_frontier = frontier
        self.support_count = support
        self.labels = labels or ["object"]


def test_graph_node_breakdown_counts():
    gm = SimpleNamespace(
        get_nodes=lambda: [
            _FakeNode(),
            _FakeNode(viewpoint=True),
            _FakeNode(frontier=True),
        ]
    )
    b = graph_node_breakdown(gm)
    assert b == {"total": 3, "object": 1, "viewpoint": 1, "frontier": 1}
    assert "1 obj" in format_graph_node_breakdown(gm)
    report = format_graph_size_report(gm, verbose=False)
    assert "graph size:" in report
    assert "1 obj" in report
    assert "1 vp" in report
    assert "1 fr" in report


def test_graph_eqa_list_objects_skips_viewpoint_and_frontier():
    from emet.memory.adapters import GraphEQABackend

    gm = SimpleNamespace(
        get_nodes=lambda: [
            _FakeNode(labels=["mug"]),
            _FakeNode(viewpoint=True, labels=["view img 3"]),
            _FakeNode(frontier=True, labels=["frontier"]),
        ]
    )
    assert GraphEQABackend(gm).list_objects() == ["mug"]


def test_graph_health_metrics_singletons_and_prompt():
    gm = SimpleNamespace(
        get_nodes=lambda: [
            _FakeNode(support=1, labels=["mug"]),
            _FakeNode(support=3, labels=["chair"]),
            _FakeNode(frontier=True, labels=["frontier:c0"]),
        ],
        _observations=[1, 2],
        last_eqa_prompt_node_count=2,
        last_eqa_obs_ids=[1],
    )
    h = graph_health_metrics(gm)
    assert h["n_object"] == 2
    assert h["n_frontier"] == 1
    assert h["n_obs"] == 2
    assert h["n_singletons"] == 1
    assert 0.4 < h["singleton_frac"] < 0.6
    assert h["prompt_node_count"] == 2
    assert h["prompt_obs_count"] == 1
    assert classify_graph_failure(h) == "ok"


def test_classify_graph_failure_blowup_and_empty():
    assert classify_graph_failure({"n_object": 0, "n_obs": 0}) == "empty_graph"
    assert classify_graph_failure({"n_object": 250, "singleton_frac": 0.1}) == "blowup"
    assert (
        classify_graph_failure({"n_object": 20, "singleton_frac": 0.9}) == "fragmentation"
    )


def test_graph_health_from_checkpoint_nodes():
    nodes = [
        {"labels": ["mug"], "support_count": 1, "is_viewpoint": False, "is_frontier": False},
        {"labels": ["view"], "support_count": 1, "is_viewpoint": True, "is_frontier": False},
    ]
    h = graph_health_from_checkpoint_nodes(nodes, n_obs=3)
    assert h["n_object"] == 1
    assert h["n_viewpoint"] == 1
    assert h["n_obs"] == 3


def test_labels_compatible_for_dedup_drift_and_rejects():
    assert labels_compatible_for_dedup("mug", "coffee cup")
    assert labels_compatible_for_dedup("fridge", "refrigerator")
    assert labels_compatible_for_dedup("dining table", "table")
    assert not labels_compatible_for_dedup("mug", "chair")


def test_apply_eval_graph_fusion_parameters_enables_fallback():
    from emet.core.parameters import Parameters

    params = Parameters()
    params["dynagraph_merge_xy_m"] = 0.45
    out = apply_eval_graph_fusion_parameters(params)
    fusion = out.get("graph_object_fusion")
    assert fusion["enabled"] is True
    assert fusion["fallback_spatial_merge_xy_m"] == 0.45


def test_apply_eval_graph_fusion_parameters_zeros_fallback_when_merge_disabled():
    from emet.core.parameters import Parameters

    params = Parameters()
    params["dynagraph_merge_xy_m"] = 0.0
    out = apply_eval_graph_fusion_parameters(params, merge_xy_m=0.0)
    fusion = out.get("graph_object_fusion")
    assert fusion["enabled"] is True
    assert fusion["fallback_spatial_merge_xy_m"] == 0.0


def test_add_observation_merges_label_drift():
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.45
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    xyz = np.array([1.0, 2.0, 0.5], dtype=float)
    mem.add_observation(rgb, xyz, ["mug"])
    mem.add_observation(rgb, xyz + np.array([0.05, 0.0, 0.0]), ["coffee cup"])
    objects = [n for n in mem.get_nodes() if not n.is_viewpoint and not n.is_frontier]
    assert len(objects) == 1
    assert objects[0].support_count == 2
    labels_l = {x.lower() for x in objects[0].labels}
    assert "mug" in labels_l
    assert "coffee cup" in labels_l


def test_eqa_prompt_topk_caps_scene_graph_text():
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem._relevant_objects = ["basket"]
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    for i in range(30):
        mem.add_observation(
            rgb,
            np.array([float(i), 0.0, 0.5], dtype=float),
            [f"noise_{i}" if i < 25 else "woven basket"],
        )
    full = mem.to_string()
    capped = mem.to_string(
        max_object_nodes=5,
        question_keywords=["basket"],
        record_prompt_count=True,
    )
    assert full.count("Node ") > capped.count("Node ")
    assert mem.last_eqa_prompt_node_count <= 5 + 4  # objects + frontier budget
    assert "basket" in capped.lower()

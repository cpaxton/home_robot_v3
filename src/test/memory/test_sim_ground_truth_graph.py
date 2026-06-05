# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for sim ground-truth graph upsert and deduplication."""

from __future__ import annotations

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, is_ground_truth_node
from emet.memory.graph_eqa.sim_ground_truth_graph import (
    build_ground_truth_graph_from_session,
    deduplicate_placements,
    ground_truth_alignment_report,
    upsert_graph_memory_from_placements,
)
from emet.simulation.sim_object_placements import DEFAULT_TABLE_SCENE_PLACEMENTS, placements_to_session_dict


def _rgb() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _default_table_placements() -> dict[str, dict]:
    session = placements_to_session_dict(DEFAULT_TABLE_SCENE_PLACEMENTS)
    assert session is not None
    return {k: {"cat": v["cat"], "pos": np.asarray(v["pos"], dtype=np.float64)} for k, v in session.items()}


def test_upsert_gt_deduplicates_by_body_name():
    mem = GraphEQAMemory(defer_llm_clients=True)
    placements = _default_table_placements()
    n1 = upsert_graph_memory_from_placements(mem, _rgb(), placements)
    assert n1 == len(placements)
    n_nodes_1 = len(mem.get_nodes())

    upsert_graph_memory_from_placements(mem, _rgb(), placements)
    assert len(mem.get_nodes()) == n_nodes_1
    gt_nodes = [n for n in mem.get_nodes() if is_ground_truth_node(n)]
    assert len(gt_nodes) == n_nodes_1
    # Second upsert with same poses deduplicates without duplicating nodes (support_count may stay 1).
    assert all(int(n.support_count) == 1 for n in gt_nodes)


def test_maintain_skips_ground_truth_nodes():
    mem = GraphEQAMemory(
        parameters={"dynagraph_staleness_horizon": 1},
        defer_llm_clients=True,
    )
    placements = _default_table_placements()
    upsert_graph_memory_from_placements(mem, _rgb(), placements)
    mem.set_graph_timestep(0)
    mem.maintain(100)
    assert len(mem.get_nodes()) == len(placements)


def test_deduplicate_placements_merges_same_cat_nearby():
    raw = {
        "obj_a": {"cat": "mug", "pos": np.array([0.0, 0.0, 1.0])},
        "obj_b": {"cat": "mug", "pos": np.array([0.01, 0.0, 1.0])},
        "obj_c": {"cat": "bowl", "pos": np.array([0.5, 0.0, 1.0])},
    }
    out = deduplicate_placements(raw, merge_xy_m=0.05)
    assert len(out) == 2
    assert "obj_a" in out
    assert "obj_c" in out


def test_ground_truth_alignment_report_perception_only():
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    mem = GraphEQAMemory(defer_llm_clients=True)
    placements = _default_table_placements()
    upsert_graph_memory_from_placements(mem, _rgb(), placements)
    mem.add_observation(_rgb(), np.array([0.0, -0.5, 0.6]), ["mystery object"])
    all_report = ground_truth_alignment_report(mem, placements)
    perc_report = ground_truth_alignment_report(mem, placements, perception_nodes_only=True)
    assert "node 1" in all_report
    assert "mystery" in perc_report.lower() or "NO GT match" in perc_report


def test_build_ground_truth_graph_from_session_returns_stable_count():
    mem = GraphEQAMemory(defer_llm_clients=True)
    session = {"sim_object_placements": placements_to_session_dict(DEFAULT_TABLE_SCENE_PLACEMENTS)}
    n1, gt = build_ground_truth_graph_from_session(mem, _rgb(), session)
    assert gt is not None
    assert n1 >= 3
    n2, _ = build_ground_truth_graph_from_session(mem, _rgb(), session)
    assert n2 == n1
    assert len(mem.get_nodes()) == n1


def test_gt_graph_stores_extent_half_from_bounds():
    mem = GraphEQAMemory(defer_llm_clients=True)
    placements = {
        "sink": {
            "cat": "sink",
            "pos": np.array([1.0, 0.0, 0.8]),
            "bounds": np.array([[0.8, -0.2, 0.75], [1.2, 0.2, 0.85]]),
        },
    }
    upsert_graph_memory_from_placements(mem, _rgb(), placements)
    node = mem.get_nodes()[0]
    assert node.extent_half is not None
    np.testing.assert_allclose(node.extent_half, [0.2, 0.2, 0.05], atol=1e-6)

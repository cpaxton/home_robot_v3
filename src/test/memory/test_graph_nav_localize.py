# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Graph-memory localization helpers for DynaMem / Dynagraph navigation."""

from __future__ import annotations

import numpy as np

from emet.memory.graph_eqa import GraphEQAMemory
from emet.memory.graph_eqa.nav_benchmark import find_gt_target_xy, score_nav_toward_target


def test_find_gt_target_xy_matches_category():
    placements = {
        "sink_main": {"cat": "sink", "pos": [2.0, 1.0, 0.9]},
        "table_main": {"cat": "table", "pos": [0.5, 0.5, 0.8]},
    }
    match = find_gt_target_xy(placements, "go to the sink", body_key="sink_main")
    assert match is not None
    body, pos = match
    assert body == "sink_main"
    assert float(pos[0]) == 2.0


def test_graph_localize_skips_frontier_nodes():
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
        parameters={"graph_eqa_frontier_nodes": {"enabled": True, "min_cluster_cells": 1}},
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    from emet.memory.graph_eqa.graph_memory import GraphNode

    mem.add_observation(rgb, np.array([5.0, 5.0, 0.0]), ["sofa"])
    mem._nodes.append(
        GraphNode(
            node_id=99,
            labels=["frontier"],
            xyz=np.array([0.0, 0.0, 0.0]),
            obs_id=99,
            description="frontier:c1",
            last_seen=1,
            is_frontier=True,
        )
    )

    from emet.controller.controller_dynamem import DynamemController

    class _Stub:
        graph_memory = mem

    pt = DynamemController._localize_point_from_graph_memory(_Stub(), "sofa")
    assert pt is not None
    assert float(pt[0]) == 5.0


def test_best_frontier_point_from_graph_prefers_keyword_match():
    from emet.controller.controller_dynamem import DynamemController
    from emet.memory.graph_eqa.graph_memory import GraphNode

    class _GraphMem:
        frontier_nodes_enabled = True

        def get_nodes(self):
            return self._nodes

        _nodes = [
            GraphNode(
                node_id=1,
                labels=["kitchen", "counter"],
                xyz=np.array([1.0, 2.0, 0.0]),
                obs_id=1,
                description="frontier:kitchen",
                last_seen=1,
                is_frontier=True,
            ),
            GraphNode(
                node_id=2,
                labels=["bedroom", "bed"],
                xyz=np.array([5.0, 6.0, 0.0]),
                obs_id=2,
                description="frontier:bedroom",
                last_seen=1,
                is_frontier=True,
            ),
        ]

    class _Stub:
        graph_memory = _GraphMem()

    pt = DynamemController._best_frontier_point_from_graph(_Stub(), "blanket on the bed in bedroom")
    assert pt is not None
    assert float(pt[0]) == 5.0
    assert float(pt[1]) == 6.0


def test_score_nav_toward_target_improved():
    s = score_nav_toward_target([0.0, 0.0], [1.5, 0.0], [2.0, 0.0])
    assert s["improved"] is True
    assert s["delta_m"] > 0.0

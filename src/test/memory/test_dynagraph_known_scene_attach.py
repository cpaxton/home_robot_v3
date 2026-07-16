# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Known-scene Dynagraph perception → GraphEQAMemory attach (no GPU / no full sim).

Closes the TESTING.md gap: detections for a default table scene (red cylinder, blue
cube) must become object nodes with allowlisted labels through the fusion / instance
pipeline Dynagraph uses live.
"""

from __future__ import annotations

import numpy as np

from emet.controller.controller_graph_eqa import GraphEQAController
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.graph_eqa.graph_object_fusion.config import GraphObjectFusionConfig
from emet.memory.graph_eqa.graph_object_fusion.fusion import (
    GraphDetectionCandidate,
    GraphObjectFusion,
)
from emet.memory.graph_eqa.graph_observation_pipeline import apply_instance_items_to_graph
from emet.memory.graph_eqa.graph_stats import graph_health_metrics, labels_compatible_for_dedup


def test_known_scene_instance_items_attach_as_object_nodes():
    """Red cylinder + blue cube at distinct XY become ≥2 object nodes with allowlisted labels."""
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.45
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    items = [
        ("red cylinder", np.array([0.08, -0.55, 0.6], dtype=float), (1, 1, 6, 6)),
        ("blue cube", np.array([-0.02, -0.55, 0.6], dtype=float), (8, 1, 14, 6)),
    ]
    apply_instance_items_to_graph(
        mem,
        rgb,
        items,
        dedup_skips=lambda _l, _x: False,
    )
    # Second frame: label drift near the same poses must reinforce, not explode.
    apply_instance_items_to_graph(
        mem,
        rgb,
        [
            ("red cylinder", np.array([0.09, -0.54, 0.6], dtype=float), (1, 1, 6, 6)),
            ("blue box", np.array([-0.01, -0.56, 0.6], dtype=float), (8, 1, 14, 6)),
        ],
        dedup_skips=lambda label, xyz: any(
            labels_compatible_for_dedup(label, str(n.labels[0]))
            and float(np.linalg.norm(n.xyz[:2] - xyz[:2])) < 0.4
            for n in mem.get_nodes()
            if n.labels and not n.is_viewpoint and not n.is_frontier
        ),
    )

    objs = [n for n in mem.get_nodes() if not n.is_viewpoint and not n.is_frontier]
    assert len(objs) >= 2, f"expected ≥2 object nodes, got {len(objs)}: {[n.labels for n in objs]}"
    assert len(objs) <= 3, f"label drift should not explode nodes: {[n.labels for n in objs]}"
    blob = " ".join(" ".join(n.labels) for n in objs).lower()
    assert "red" in blob and "blue" in blob, blob
    health = graph_health_metrics(mem)
    assert health["n_object"] >= 2


def test_known_scene_fusion_attach_distinct_xyz():
    cfg = GraphObjectFusionConfig(
        enabled=True,
        spatial_merge_xy_m=0.35,
        embedding_min_cosine=0.0,
        fallback_spatial_merge_xy_m=0.35,
        require_label_match=False,
    )
    fusion = GraphObjectFusion(cfg)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.0
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    fusion.apply_detection(
        mem,
        rgb,
        GraphDetectionCandidate(label="red cylinder", xyz=np.array([0.1, 0.0, 0.5])),
    )
    fusion.apply_detection(
        mem,
        rgb,
        GraphDetectionCandidate(label="blue cube", xyz=np.array([-0.8, 0.0, 0.5])),
    )
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint and not n.is_frontier]
    assert len(objs) == 2
    blob = " ".join(" ".join(n.labels) for n in objs).lower()
    assert "red" in blob and "blue" in blob


def test_graph_dedup_skips_compatible_labels_near_xy():
    class _Stub(GraphEQAController):
        def __init__(self):
            self._graph_dedup_xy_m = 0.4
            self.graph_memory = GraphEQAMemory(defer_llm_clients=True)

    stub = _Stub()
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    stub.graph_memory.add_observation(rgb, np.array([1.0, 2.0, 0.5]), ["mug"])
    assert stub._graph_dedup_skips("coffee cup", np.array([1.02, 2.01, 0.5]))
    assert not stub._graph_dedup_skips("chair", np.array([1.02, 2.01, 0.5]))

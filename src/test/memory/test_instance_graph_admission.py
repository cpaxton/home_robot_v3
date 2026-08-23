# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, countable_primary_label_matches
from emet.memory.graph_eqa.graph_object_fusion.config import GraphObjectFusionConfig
from emet.memory.graph_eqa.graph_object_fusion.fusion import GraphDetectionCandidate, GraphObjectFusion
from emet.memory.graph_eqa.instance_observations import (
    filter_detections_for_graph_admission,
    frame_instances_to_detections,
)


class _MockYoloEVocab:
    class_list = [f"cls_{i}" for i in range(200)]


def test_frame_instances_propagate_score_and_support():
    h, w = 8, 8
    inst = torch.full((h, w), -1, dtype=torch.long)
    inst[0:4, 0:4] = 0
    fw = torch.zeros(h, w, 3, dtype=torch.float32)
    fw[0:4, 0:4] = torch.tensor([1.0, 2.0, 3.0])
    depth = torch.ones(h, w) * 0.5
    classes = torch.tensor([2], dtype=torch.long)
    scores = torch.tensor([0.42], dtype=torch.float32)
    frame = SimpleNamespace(
        instance=inst,
        full_world_xyz=fw,
        depth=depth,
        instance_classes=classes,
        instance_scores=scores,
    )
    out = frame_instances_to_detections(
        frame,
        min_depth=0.01,
        max_depth=10.0,
        detection_model=_MockYoloEVocab(),
        min_points=4,
    )
    assert len(out) == 1
    assert abs(out[0]["detection_score"] - 0.42) < 1e-5
    assert out[0]["mask_point_count"] == 16


def test_filter_detections_for_graph_admission():
    cfg = GraphObjectFusionConfig(instance_min_confidence=0.2, instance_min_mask_points=10)
    dets = [
        {"detection_score": 0.1, "mask_point_count": 100},
        {"detection_score": 0.5, "mask_point_count": 5},
        {"detection_score": 0.5, "mask_point_count": 50},
    ]
    kept, stats = filter_detections_for_graph_admission(dets, config=cfg)
    assert len(kept) == 1
    assert stats["rejected_confidence"] == 1
    assert stats["rejected_support"] == 1
    assert stats["admitted"] == 1


def test_fusion_keeps_incompatible_instance_labels_distinct():
    cfg = GraphObjectFusionConfig(
        enabled=True,
        spatial_merge_xy_m=0.55,
        fallback_spatial_merge_xy_m=0.55,
        require_label_match=False,
        require_label_match_for_instances=True,
    )
    fusion = GraphObjectFusion(cfg)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.0
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    fusion.apply_detection(
        mem,
        rgb,
        GraphDetectionCandidate(
            label="bathroom stall",
            xyz=np.array([1.0, 0.0, 0.5]),
            countable_instance=True,
        ),
    )
    fusion.apply_detection(
        mem,
        rgb,
        GraphDetectionCandidate(
            label="table lamp",
            xyz=np.array([1.05, 0.02, 0.52]),
            countable_instance=True,
        ),
    )
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint and not n.is_frontier]
    assert len(objs) == 2


def test_semantic_hypothesis_does_not_merge_into_countable_node():
    cfg = GraphObjectFusionConfig(
        enabled=True,
        spatial_merge_xy_m=0.55,
        fallback_spatial_merge_xy_m=0.55,
        require_label_match=False,
        require_label_match_for_instances=True,
    )
    fusion = GraphObjectFusion(cfg)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.0
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    fusion.apply_detection(
        mem,
        rgb,
        GraphDetectionCandidate(label="table lamp", xyz=np.array([1.0, 0.0, 0.5]), countable_instance=True),
    )
    fusion.apply_detection(
        mem,
        rgb,
        GraphDetectionCandidate(label="bed", xyz=np.array([1.02, 0.01, 0.51]), semantic_only=True),
    )
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint and not n.is_frontier]
    assert len(objs) == 2
    lamp = next(n for n in objs if n.labels == ["table lamp"])
    assert lamp.countable_instance is True
    assert not any("bed" in lab for lab in lamp.labels)


def test_countable_primary_label_matches_ignores_secondary_labels():
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.array([1.0, 0.0, 0.5]),
        ["bathroom stall", "table lamp"],
        countable_instance=True,
    )
    node = [n for n in mem.get_nodes() if not n.is_viewpoint][0]
    assert countable_primary_label_matches("table lamp", node) is False
    assert countable_primary_label_matches("bathroom stall", node) is True

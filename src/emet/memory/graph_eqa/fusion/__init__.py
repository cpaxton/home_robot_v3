# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Alias for ``emet.memory.graph_eqa.graph_object_fusion`` (attach + merge)."""

from emet.memory.graph_eqa.graph_object_fusion import (
    GraphDetectionCandidate,
    GraphObjectFusion,
    GraphObjectFusionConfig,
    bounds_3d_from_points,
    bounds_3d_iou,
    cosine_similarity_np,
    grid_search_fusion_config,
    load_calibration_frames_jsonl,
    load_graph_object_fusion_config,
    replay_frames_with_fusion,
    score_fused_nodes_vs_gt,
)
from emet.memory.graph_eqa.graph_object_fusion.attach import attach_graph_object_fusion

__all__ = [
    "GraphDetectionCandidate",
    "GraphObjectFusion",
    "GraphObjectFusionConfig",
    "attach_graph_object_fusion",
    "bounds_3d_from_points",
    "bounds_3d_iou",
    "cosine_similarity_np",
    "grid_search_fusion_config",
    "load_calibration_frames_jsonl",
    "load_graph_object_fusion_config",
    "replay_frames_with_fusion",
    "score_fused_nodes_vs_gt",
]

# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Configuration for GraphObjectFusion (YAML + dataclass)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus
import yaml

from emet.utils.config import resolve_config_yaml_path


@dataclass
class GraphObjectFusionConfig:
    """Tunable fusion thresholds (instance detections → graph nodes).

    Attributes:
        enabled: When true, instance detections use ``GraphObjectFusion`` instead
            of legacy label-only dedup on the instance path.
        spatial_merge_xy_m: Max planar (XY) distance (m) between centroids to
            consider a merge candidate.
        min_centroid_dist_m: Max 3D centroid distance (m) for a merge candidate.
        bounds_3d_iou_min: Minimum 3D AABB IoU when both sides have ``bounds_3d``.
        embedding_min_cosine: Minimum cosine similarity when both sides have
            embeddings; ignored if either side is missing.
        embedding_blend_alpha: EMA weight for blending embeddings on merge.
        require_label_match: When true, only merge detections whose label matches
            the node's primary label (substring match).
        max_candidates: Cap on nodes scanned per detection (reserved).
        match_xy_m: GT association radius used during offline calibration scoring.
    """

    enabled: bool = False
    spatial_merge_xy_m: float = 0.42
    min_centroid_dist_m: float = 0.55
    bounds_3d_iou_min: float = 0.08
    embedding_min_cosine: float = 0.62
    embedding_blend_alpha: float = 0.35
    require_label_match: bool = True
    max_candidates: int = 64
    match_xy_m: float = 0.55


def load_graph_object_fusion_config(path: str | None = None) -> GraphObjectFusionConfig:
    """Load ``graph_object_fusion`` from YAML or return package defaults.

    Args:
        path: YAML path, or ``None`` to use
            ``src/emet/config/agents/default_graph_object_fusion.yaml`` when present.

    Returns:
        Decoded ``GraphObjectFusionConfig``.
    """
    if not path:
        default = Path(__file__).resolve().parents[3] / "config" / "agents" / "default_graph_object_fusion.yaml"
        if default.is_file():
            path = str(default)
        else:
            return GraphObjectFusionConfig()
    full = Path(resolve_config_yaml_path(path))
    with full.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    subset = raw.get("graph_object_fusion", raw)
    if not isinstance(subset, dict):
        return GraphObjectFusionConfig()
    return draccus.decode(GraphObjectFusionConfig, subset)

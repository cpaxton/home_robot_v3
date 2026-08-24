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
    """Tunable fusion thresholds (instance detections → graph nodes)."""

    enabled: bool = False
    spatial_merge_xy_m: float = 0.42
    min_centroid_dist_m: float = 0.55
    bounds_3d_iou_min: float = 0.08
    # When 3D bounds overlap this much, merge even if strict XY/centroid gates fail (0 = off).
    bounds_3d_iou_merge_min: float = 0.0
    embedding_min_cosine: float = 0.62
    embedding_blend_alpha: float = 0.35
    require_label_match: bool = False
    # When either side is a countable YoloE/HM3D instance, require compatible labels to merge.
    require_label_match_for_instances: bool = True
    max_candidates: int = 64

    # When strict gates fail, merge to nearest object node within this XY radius (0 = disabled).
    fallback_spatial_merge_xy_m: float = 0.0

    # Graph admission (mapping may keep low-confidence proposals for recall).
    instance_min_confidence: float = 0.12
    instance_min_mask_points: int = 25

    # Scoring / calibration
    match_xy_m: float = 0.55


def load_graph_object_fusion_config(path: str | None = None) -> GraphObjectFusionConfig:
    """Load standalone YAML or return defaults."""
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

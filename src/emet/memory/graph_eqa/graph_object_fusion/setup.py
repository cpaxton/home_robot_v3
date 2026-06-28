# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Wire GraphObjectFusion from parameters / embodied_agent config."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from emet.config.embodied_agent_config import GraphObjectFusionConfigRef
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.graph_eqa.graph_object_fusion.config import GraphObjectFusionConfig, load_graph_object_fusion_config
from emet.memory.graph_eqa.graph_object_fusion.fusion import GraphObjectFusion


def _parameters_get(parameters: Any | None, key: str, default: Any = None) -> Any:
    if parameters is None:
        return default
    if isinstance(parameters, dict):
        return parameters.get(key, default)
    get = getattr(parameters, "get", None)
    if callable(get):
        return get(key, default)
    return default


def _config_block_to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    items = getattr(value, "items", None)
    if callable(items):
        return {str(k): v for k, v in items()}
    return None


def fusion_config_from_sources(
    *,
    fref: GraphObjectFusionConfigRef | None = None,
    parameters: dict[str, Any] | Any | None = None,
    yaml_path: str | None = None,
) -> GraphObjectFusionConfig:
    if yaml_path:
        return load_graph_object_fusion_config(yaml_path)
    base = GraphObjectFusionConfig()
    if fref is not None:
        base = GraphObjectFusionConfig(**{**asdict(base), **asdict(fref)})
    p_fusion = _config_block_to_dict(_parameters_get(parameters, "graph_object_fusion"))
    if p_fusion is not None:
        base = GraphObjectFusionConfig(**{**asdict(base), **p_fusion})
    return base


def attach_graph_object_fusion(
    graph_memory: GraphEQAMemory,
    parameters: dict[str, Any] | Any | None,
    *,
    fref: GraphObjectFusionConfigRef | None = None,
) -> GraphObjectFusion | None:
    fc = fusion_config_from_sources(fref=fref, parameters=parameters)
    if not fc.enabled and fref is None:
        p_fusion = _config_block_to_dict(_parameters_get(parameters, "graph_object_fusion"))
        if p_fusion is None:
            fc = load_graph_object_fusion_config()
    if not fc.enabled:
        return None
    graph_memory.spatial_merge_m = 0.0
    if fc.fallback_spatial_merge_xy_m <= 0.0:
        merge_xy = _parameters_get(parameters, "dynagraph_merge_xy_m")
        if merge_xy is not None and float(merge_xy) > 0.0:
            fc = GraphObjectFusionConfig(**{**asdict(fc), "fallback_spatial_merge_xy_m": float(merge_xy)})
    return GraphObjectFusion(fc)

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


def fusion_config_from_sources(
    *,
    fref: GraphObjectFusionConfigRef | None = None,
    parameters: dict[str, Any] | None = None,
    yaml_path: str | None = None,
) -> GraphObjectFusionConfig:
    if yaml_path:
        return load_graph_object_fusion_config(yaml_path)
    base = GraphObjectFusionConfig()
    if fref is not None:
        base = GraphObjectFusionConfig(**{**asdict(base), **asdict(fref)})
    p_fusion = (parameters or {}).get("graph_object_fusion")
    if isinstance(p_fusion, dict):
        base = GraphObjectFusionConfig(**{**asdict(base), **p_fusion})
    return base


def attach_graph_object_fusion(
    graph_memory: GraphEQAMemory,
    parameters: dict[str, Any] | None,
    *,
    fref: GraphObjectFusionConfigRef | None = None,
) -> GraphObjectFusion | None:
    fc = fusion_config_from_sources(fref=fref, parameters=parameters)
    if not fc.enabled and fref is None:
        p_fusion = (parameters or {}).get("graph_object_fusion")
        if not isinstance(p_fusion, dict):
            fc = load_graph_object_fusion_config()
    if not fc.enabled:
        return None
    graph_memory.spatial_merge_m = 0.0
    return GraphObjectFusion(fc)

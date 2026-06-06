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
    """Merge fusion settings from YAML, embodied-agent ref, and runtime parameters.

    Priority (later wins): defaults → ``fref`` → ``parameters['graph_object_fusion']``.
    When ``yaml_path`` is set, it replaces all other sources.

    Args:
        fref: Optional structured ref from embodied agent YAML.
        parameters: Dynagraph/dynamem parameter dict from ``get_parameters``.
        yaml_path: Standalone ``graph_object_fusion`` YAML path.

    Returns:
        Resolved ``GraphObjectFusionConfig``.
    """
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
    """Attach fusion to a graph memory instance when enabled in config.

    Disables legacy ``dynagraph_merge_xy_m`` on the instance path by setting
    ``graph_memory.spatial_merge_m = 0``.

    Args:
        graph_memory: Live ``GraphEQAMemory`` on the controller.
        parameters: Runtime parameter dict.
        fref: Optional embodied-agent fusion ref.

    Returns:
        ``GraphObjectFusion`` instance, or ``None`` when ``enabled`` is false.
    """
    fc = fusion_config_from_sources(fref=fref, parameters=parameters)
    if not fc.enabled:
        return None
    graph_memory.spatial_merge_m = 0.0
    return GraphObjectFusion(fc)

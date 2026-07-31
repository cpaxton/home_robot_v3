# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

"""Typed overlay for embodied agent + scene graph (parsed with Draccus from YAML ``embodied_agent``)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from emet.utils.logger import Logger

logger = Logger(__name__)

# Mutually exclusive object-graph plug-ins on the voxel map (``agent.memory_backend``).
# Canonical static baseline is ``static_graph`` (legacy alias: ``graph_eqa``).
MEMORY_BACKENDS = ("dynamem", "open_vocab", "static_graph", "dynagraph")
GRAPH_EQA_FAMILY_BACKENDS = frozenset({"static_graph", "dynagraph"})


@dataclass
class OpenVocabSceneGraphConfig:
    """Attach :class:`SceneGraphProcessor` to the DynaMem voxel map."""

    enabled: bool = False
    config_name: str = "default_scene_graph"
    device: str | None = None


@dataclass
class GraphObjectFusionConfigRef:
    """Nested fusion config (see ``default_graph_object_fusion.yaml``)."""

    enabled: bool = False
    spatial_merge_xy_m: float = 0.42
    min_centroid_dist_m: float = 0.55
    bounds_3d_iou_min: float = 0.08
    embedding_min_cosine: float = 0.62
    embedding_blend_alpha: float = 0.35
    require_label_match: bool = True
    max_candidates: int = 64
    match_xy_m: float = 0.55
    fallback_spatial_merge_xy_m: float = 0.0


@dataclass
class GraphEQAMemoryConfig:
    """Feed :class:`GraphEQAMemory` each controller update (GraphEQA-style EQA)."""

    enabled: bool = False
    use_instance_graph: bool = True
    use_sensor_perception: bool = True
    graph_instance_dedup_xy_m: float | None = None
    graph_object_fusion: GraphObjectFusionConfigRef = field(default_factory=GraphObjectFusionConfigRef)


@dataclass
class EmbodiedAgentConfig:
    """Defaults: open-vocab scene graph + GraphEQA memory off (opt in via YAML ``embodied_agent``).

    Prefer selecting the plug-in via ``agent.memory_backend`` / ``--memory-backend``; nested
    ``enabled`` flags are coerced from that enum so OV and GraphEQA are never both live.
    """

    open_vocab_scene_graph: OpenVocabSceneGraphConfig = field(default_factory=OpenVocabSceneGraphConfig)
    graph_eqa_memory: GraphEQAMemoryConfig = field(default_factory=GraphEQAMemoryConfig)


def legacy_embodied_agent_off() -> EmbodiedAgentConfig:
    """Disable open-vocab + graph memory (default when ``RobotAgent`` is constructed without an overlay)."""

    return EmbodiedAgentConfig(
        open_vocab_scene_graph=OpenVocabSceneGraphConfig(enabled=False),
        graph_eqa_memory=GraphEQAMemoryConfig(enabled=False),
    )


def normalize_memory_backend(memory_backend: str | None) -> str:
    """Return a canonical ``memory_backend`` value (default ``dynagraph``).

    Accepts legacy ``graph_eqa`` → ``static_graph``.
    """
    from emet.eval.memory_backends import normalize_benchmark_backend

    mb = str(memory_backend or "dynagraph").strip().lower().replace("-", "_")
    mb = normalize_benchmark_backend(mb)
    if mb not in MEMORY_BACKENDS:
        raise ValueError(
            f"Unknown memory_backend={memory_backend!r}; expected one of {MEMORY_BACKENDS} "
            f"(legacy alias: graph_eqa → static_graph)"
        )
    return mb


def coerce_embodied_agent_for_memory_backend(
    overlay: EmbodiedAgentConfig | None,
    memory_backend: str | None,
) -> EmbodiedAgentConfig:
    """Derive builder ``enabled`` flags from ``memory_backend`` (at most one graph plug-in).

    Nested YAML still supplies tuning (instance graph, OV config name, fusion). Independent
    dual ``enabled: true`` in Discord presets is coerced away with a warning.
    """
    mb = normalize_memory_backend(memory_backend)
    base = deepcopy(overlay) if overlay is not None else EmbodiedAgentConfig()
    ov_on = bool(base.open_vocab_scene_graph.enabled)
    ge_on = bool(base.graph_eqa_memory.enabled)
    if ov_on and ge_on:
        logger.warning(
            "embodied_agent has both open_vocab_scene_graph and graph_eqa_memory enabled; "
            f"coercing to a single plug-in from memory_backend={mb!r}"
        )

    if mb == "open_vocab":
        base.open_vocab_scene_graph.enabled = True
        base.graph_eqa_memory.enabled = False
    elif mb in GRAPH_EQA_FAMILY_BACKENDS:
        base.open_vocab_scene_graph.enabled = False
        base.graph_eqa_memory.enabled = True
    else:
        # dynamem: voxels only
        base.open_vocab_scene_graph.enabled = False
        base.graph_eqa_memory.enabled = False
    return base


def load_embodied_agent_overlay(config_path: str | None) -> EmbodiedAgentConfig:
    """Load ``embodied_agent`` via the unified config loader (``extends:`` / ``defaults:`` aware)."""
    if not config_path:
        return EmbodiedAgentConfig()
    from emet.config.loader import load_config

    return load_config(config_path).embodied_agent()

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

from dataclasses import dataclass, field


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
    """Defaults: open-vocab scene graph + GraphEQA memory off (opt in via YAML ``embodied_agent``)."""

    open_vocab_scene_graph: OpenVocabSceneGraphConfig = field(default_factory=OpenVocabSceneGraphConfig)
    graph_eqa_memory: GraphEQAMemoryConfig = field(default_factory=GraphEQAMemoryConfig)


def legacy_embodied_agent_off() -> EmbodiedAgentConfig:
    """Disable open-vocab + graph memory (default when ``RobotAgent`` is constructed without an overlay)."""

    return EmbodiedAgentConfig(
        open_vocab_scene_graph=OpenVocabSceneGraphConfig(enabled=False),
        graph_eqa_memory=GraphEQAMemoryConfig(enabled=False),
    )


def load_embodied_agent_overlay(config_path: str | None) -> EmbodiedAgentConfig:
    """Load ``embodied_agent`` via the unified config loader (``extends:`` / ``defaults:`` aware)."""
    if not config_path:
        return EmbodiedAgentConfig()
    from emet.config.loader import load_config

    return load_config(config_path).embodied_agent()

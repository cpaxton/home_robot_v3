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
from pathlib import Path
from typing import Any

import draccus
import yaml

from emet.utils.config import resolve_config_yaml_path


@dataclass
class OpenVocabSceneGraphConfig:
    """Attach :class:`SceneGraphProcessor` to the DynaMem voxel map."""

    enabled: bool = False
    config_name: str = "default_scene_graph"
    device: str | None = None


@dataclass
class GraphEQAMemoryConfig:
    """Feed :class:`GraphEQAMemory` each controller update (GraphEQA-style EQA)."""

    enabled: bool = False
    use_instance_graph: bool = True
    use_sensor_perception: bool = True
    graph_instance_dedup_xy_m: float | None = None


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
    """Load ``embodied_agent`` subtree from a dynav-style YAML.

    Missing ``embodied_agent:`` key → conservative defaults (both features off); enable in YAML when needed.
    """
    if not config_path:
        return EmbodiedAgentConfig()
    full_path = Path(resolve_config_yaml_path(config_path))
    with full_path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    subset = raw.get("embodied_agent")
    if subset is None:
        return EmbodiedAgentConfig()
    if not isinstance(subset, dict):
        return EmbodiedAgentConfig()
    return draccus.decode(EmbodiedAgentConfig, subset)

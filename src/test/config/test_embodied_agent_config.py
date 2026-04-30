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

"""Tests for Draccus-based ``embodied_agent`` YAML overlay."""

from __future__ import annotations

import tempfile
from pathlib import Path

from emet.config.embodied_agent_config import EmbodiedAgentConfig, load_embodied_agent_overlay


def test_load_overlay_missing_key_defaults():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("voxel_size: 0.1\n")
        path = f.name
    try:
        cfg = load_embodied_agent_overlay(path)
        assert cfg.open_vocab_scene_graph.enabled is True
        assert cfg.graph_eqa_memory.enabled is True
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_overlay_partial_override():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("embodied_agent:\n  graph_eqa_memory:\n    enabled: false\n    use_instance_graph: false\n")
        path = f.name
    try:
        cfg = load_embodied_agent_overlay(path)
        assert cfg.graph_eqa_memory.enabled is False
        assert cfg.graph_eqa_memory.use_instance_graph is False
        assert cfg.open_vocab_scene_graph.enabled is True
    finally:
        Path(path).unlink(missing_ok=True)


def test_decode_empty_dict_all_defaults():
    import draccus

    cfg = draccus.decode(EmbodiedAgentConfig, {})
    assert isinstance(cfg, EmbodiedAgentConfig)
    assert cfg.open_vocab_scene_graph.config_name == "default_scene_graph"


def test_legacy_embodied_agent_off():
    from emet.config.embodied_agent_config import legacy_embodied_agent_off

    cfg = legacy_embodied_agent_off()
    assert cfg.open_vocab_scene_graph.enabled is False
    assert cfg.graph_eqa_memory.enabled is False

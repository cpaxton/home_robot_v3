# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for memory_backend ↔ embodied_agent plug-in exclusivity."""

from __future__ import annotations

import pytest

from emet.config.embodied_agent_config import (
    EmbodiedAgentConfig,
    GraphEQAMemoryConfig,
    OpenVocabSceneGraphConfig,
    coerce_embodied_agent_for_memory_backend,
    normalize_memory_backend,
)


def test_normalize_memory_backend_defaults_and_aliases():
    assert normalize_memory_backend(None) == "dynagraph"
    assert normalize_memory_backend("Open-Vocab") == "open_vocab"
    with pytest.raises(ValueError):
        normalize_memory_backend("svm")


def test_coerce_dynagraph_disables_open_vocab():
    overlay = EmbodiedAgentConfig(
        open_vocab_scene_graph=OpenVocabSceneGraphConfig(enabled=True),
        graph_eqa_memory=GraphEQAMemoryConfig(enabled=True, use_instance_graph=False),
    )
    out = coerce_embodied_agent_for_memory_backend(overlay, "dynagraph")
    assert out.open_vocab_scene_graph.enabled is False
    assert out.graph_eqa_memory.enabled is True
    assert out.graph_eqa_memory.use_instance_graph is False


def test_coerce_open_vocab_disables_graph_eqa():
    overlay = EmbodiedAgentConfig(
        open_vocab_scene_graph=OpenVocabSceneGraphConfig(enabled=False, config_name="cpu_scene_graph"),
        graph_eqa_memory=GraphEQAMemoryConfig(enabled=True),
    )
    out = coerce_embodied_agent_for_memory_backend(overlay, "open_vocab")
    assert out.open_vocab_scene_graph.enabled is True
    assert out.open_vocab_scene_graph.config_name == "cpu_scene_graph"
    assert out.graph_eqa_memory.enabled is False


def test_coerce_dynamem_disables_both():
    overlay = EmbodiedAgentConfig(
        open_vocab_scene_graph=OpenVocabSceneGraphConfig(enabled=True),
        graph_eqa_memory=GraphEQAMemoryConfig(enabled=True),
    )
    out = coerce_embodied_agent_for_memory_backend(overlay, "dynamem")
    assert out.open_vocab_scene_graph.enabled is False
    assert out.graph_eqa_memory.enabled is False

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import numpy as np

from emet.eval.dynagraph_vram import prepare_dynagraph_vram_for_eqa
from emet.memory.graph_eqa import GraphEQAMemory


class _FakeEncoder:
    def encode_image(self, rgb):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def encode_text(self, text):
        if "basket" in (text or "").lower():
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


class _FakeAgent:
    def __init__(self):
        self.encoder = _FakeEncoder()
        self.voxel_map = type("VM", (), {"encoder": self.encoder})()
        self.graph_memory = GraphEQAMemory(defer_llm_clients=True)
        self.graph_memory.memory_summary_enabled = True
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        self.graph_memory.add_observation(rgb, np.array([1.0, 2.0, 0.5]), ["plant"])
        self.graph_memory._relevant_phrases = ["woven basket"]


def test_prepare_dynagraph_vram_keeps_siglip_for_confirmed_memory():
    agent = _FakeAgent()
    prepare_dynagraph_vram_for_eqa(agent)
    assert agent.encoder is None
    assert agent.voxel_map.encoder is None
    assert agent.graph_memory._confirmed_memory_siglip_encoder is not None
    assert agent.graph_memory._obs_siglip_features
    assert "woven basket" in agent.graph_memory._siglip_phrase_cache

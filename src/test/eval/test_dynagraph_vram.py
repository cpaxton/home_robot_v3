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


def test_prepare_dynagraph_vram_warms_then_releases_siglip():
    """CONFIRMED_MEMORY features stay cached; encoder must drop before EQA VLM load."""
    agent = _FakeAgent()
    prepare_dynagraph_vram_for_eqa(agent)
    assert agent.encoder is None
    assert agent.voxel_map.encoder is None
    assert agent.graph_memory._confirmed_memory_siglip_encoder is None
    assert agent.graph_memory._obs_siglip_features
    assert "woven basket" in agent.graph_memory._siglip_phrase_cache
    assert "woven basket" in agent.graph_memory._visual_find_rank_cache


def test_warm_keeps_encoder_release_drops():
    from emet.eval.dynagraph_vram import release_siglip_for_vlm, warm_siglip_confirmed_memory

    agent = _FakeAgent()
    warm_siglip_confirmed_memory(agent)
    assert agent.encoder is not None or agent.graph_memory._confirmed_memory_siglip_encoder is not None
    release_siglip_for_vlm(agent)
    assert agent.encoder is None
    assert agent.graph_memory._confirmed_memory_siglip_encoder is None


def _patch_shared_siglip(monkeypatch, sent: dict):
    def fake_get(*, version, device, feature_matching_threshold):
        sent["device"] = device
        sent["version"] = version
        sent["threshold"] = feature_matching_threshold
        return _FakeEncoder()

    monkeypatch.setattr(
        "emet.perception.encoders.siglip_encoder.get_shared_mask_siglip_encoder",
        fake_get,
    )


def test_re_attach_siglip_uses_cpu_when_cuda_unavailable(monkeypatch):
    """FindRec after submit_answer must re-attach even on cpu_only / no-CUDA hosts."""
    from emet.eval.dynagraph_vram import re_attach_siglip_encoder

    sent: dict = {}
    _patch_shared_siglip(monkeypatch, sent)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    agent = _FakeAgent()
    agent.encoder = None
    agent.voxel_map.encoder = None
    enc = re_attach_siglip_encoder(agent)
    assert enc is not None
    assert sent["device"] == "cpu"
    assert agent.encoder is enc
    assert agent.voxel_map.encoder is enc


def test_re_attach_siglip_cpu_only_even_when_cuda_exists(monkeypatch):
    from emet.eval.dynagraph_vram import re_attach_siglip_encoder

    sent: dict = {}
    _patch_shared_siglip(monkeypatch, sent)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    agent = _FakeAgent()
    agent.cpu_only = True
    agent.encoder = None
    enc = re_attach_siglip_encoder(agent)
    assert enc is not None
    assert sent["device"] == "cpu"
    assert agent.encoder is enc

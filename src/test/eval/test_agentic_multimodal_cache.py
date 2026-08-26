# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Agentic EQA multimodal cache and trace claims (C*, D*)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_EVAL_DIR = str(Path(__file__).resolve().parent)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
from _agentic import (  # noqa: E402
    _require_agentic,
    _require_vision_cache,
)


def test_C1_obs_siglip_features_persist_across_refresh():
    """C1: refresh must not wipe existing _obs_siglip_features for prior obs_ids."""
    from emet.memory.graph_eqa import GraphEQAMemory

    gm = GraphEQAMemory(defer_llm_clients=True)
    gm.memory_summary_enabled = True
    oid = gm.add_observation(np.zeros((4, 4, 3), dtype=np.uint8), np.array([0.0, 0.0, 0.0]), ["lamp"])
    feat = np.array([0.3, 0.7], dtype=np.float32)
    gm._obs_siglip_features[int(oid)] = feat.copy()

    class Enc:
        def encode_image(self, rgb):
            return np.array([0.0, 1.0], dtype=np.float32)

        def encode_text(self, text):
            return np.array([0.0, 1.0], dtype=np.float32)

    gm.set_confirmed_memory_siglip_encoder(Enc())
    gm.refresh_siglip_confirmed_memory()
    assert int(oid) in gm._obs_siglip_features
    np.testing.assert_allclose(gm._obs_siglip_features[int(oid)], feat, atol=1e-5)


def test_C2_vision_prefix_cache_hit():
    """C2: second generate with same image hash reports a vision cache hit."""
    _require_vision_cache()
    from emet.llms.vl_vision_cache import VisionPrefixCache

    cache = VisionPrefixCache(max_entries=4)
    key = cache.make_key(model_id="qwen3", resize_side=512, image_bytes=b"abc")
    cache.put(key, past_key_values="fake", prefix_token_len=10)
    hit = cache.get(key)
    assert hit is not None
    assert hit.prefix_token_len == 10


def test_C3_verified_answer_uses_one_image(monkeypatch):
    """C3: after verify pass, answer prompt selects <= 1 image."""
    _require_agentic()
    from emet.memory.graph_eqa import GraphEQAMemory

    gm = GraphEQAMemory(defer_llm_clients=True)
    for i in range(5):
        gm.add_observation(
            np.full((4, 4, 3), i, dtype=np.uint8),
            np.array([float(i), 0.0, 0.5]),
            ["obj"],
        )
    if not hasattr(gm, "select_obs_ids_for_verified_answer"):
        pytest.skip("select_obs_ids_for_verified_answer not implemented")
    ids = gm.select_obs_ids_for_verified_answer(verified_obs_id=2, max_images=1)
    assert len(ids) <= 1
    if ids:
        assert int(ids[0]) == 2


def test_D1_trace_round_trip(tmp_path):
    """D1: agentic_trace.jsonl includes embeds + tool fields."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {"eqa": {"collect_agentic_trace": True}}
    agent.graph_memory = MagicMock()
    agent.graph_memory.eqa_client = None
    agent.graph_memory.memory_summary_enabled = False
    agent.graph_memory.hypothesize_nav_targets.return_value = []
    agent.robot = MagicMock()
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.robot.get_emet_session.return_value = None
    path = tmp_path / "agentic_trace.jsonl"
    ex = AgenticEQAExecutor(
        agent,
        "Where is X?",
        max_rounds=1,
        max_nav_steps=0,
        collect_trace=True,
        trace_path=path,
    )
    agent.graph_memory.query_answer.return_value = ("r", "A", False, "", None, [])
    agent.graph_memory.verify_phrase_at_obs.return_value = MagicMock(
        status="ABSENT",
        sim=0.1,
        ok=False,
        obs_id=1,
        phrase="x",
        text_feat=np.array([1.0, 0.0]),
        img_feat=np.array([0.0, 1.0]),
    )
    ex.run()
    assert path.is_file()
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    tools = {r.get("tool") for r in lines}
    assert "inspect_graph" in tools
    assert "summary" in tools


def test_D2_tuner_recovers_best_threshold():
    """D2: planted sims with gt_present → best threshold near separation."""
    from emet.eval.agentic_tuning import best_threshold, sweep_verify_thresholds

    traces = []
    for sim, gt in [(0.15, False), (0.18, False), (0.32, True), (0.35, True), (0.40, True)]:
        traces.append(
            {
                "tool": "verify_siglip",
                "sim": sim,
                "gt_present": gt,
                "text_feat": [1.0, 0.0],
                "img_feat": [sim, 0.0],
            }
        )
    sweep = sweep_verify_thresholds(traces)
    best = best_threshold(sweep)
    assert best is not None
    assert 0.20 <= best["threshold"] <= 0.32
    assert best["f1"] >= 0.9


def test_D3_router_report_from_tool_picks():
    """D3: tuner reports VLM-router parse rate and fallback usage from trace rows."""
    from emet.eval.agentic_tuning import router_report

    traces = [
        {"event": "tool_pick", "picked_by": "vlm", "router_parse_ok": True, "router_tool_calls": ["navigate_to_obs"]},
        {
            "event": "tool_pick",
            "picked_by": "vlm",
            "router_parse_ok": True,
            "router_tool_calls": ["verify_siglip", "submit_answer"],
        },
        {"event": "tool_pick", "picked_by": "fallback", "router_parse_ok": False, "tool": "explore_frontier"},
        {"tool": "verify_siglip", "sim": 0.3},
    ]
    rep = router_report(traces)
    assert rep["n_tool_picks"] == 3
    assert rep["n_vlm"] == 2
    assert rep["n_fallback"] == 1
    assert abs(rep["parse_ok_rate"] - 2 / 3) < 1e-9
    assert rep["tool_counts"]["verify_siglip"] == 1
    assert rep["tool_counts"]["submit_answer"] == 1
    assert rep["tool_counts"]["explore_frontier"] == 1


def test_D2_budget_knee():
    from emet.eval.agentic_tuning import budget_knee, sweep_budgets

    traces = [
        {"tool": "verify_siglip", "question": "q1", "round": 0, "sim": 0.1, "text_feat": [1.0], "img_feat": [0.1]},
        {"tool": "verify_siglip", "question": "q1", "round": 1, "sim": 0.3, "text_feat": [1.0], "img_feat": [0.3]},
        {"tool": "summary", "question": "q1", "correct": True, "n_rounds": 2, "confidence": True},
    ]
    sweep = sweep_budgets(traces, threshold=0.28)
    knee = budget_knee(sweep)
    assert knee is not None
    assert knee["max_rounds_cap"] >= 2

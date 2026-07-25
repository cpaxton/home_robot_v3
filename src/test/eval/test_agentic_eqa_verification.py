# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE in the repository root.

"""Agentic EQA + multimodal-cache verification matrix.

Run the whole gate (CPU, no GPU)::

    uv run emet test src/test/eval/test_agentic_eqa_verification.py -v

Claims encoded below (each test is one claim):

Hang / reliability (must pass today)
  H1  Port release kills listeners only (not ZMQ clients / self)
  H2  prepare_dynagraph_vram warms SigLIP caches then drops encoders
  H3  Answer-only EQA does not navigate or look_around

Agentic loop (skip until eqa.agentic_verify APIs land)
  A1  hypothesize_nav_targets ranks graph label > SigLIP candidate
  A2  verify_phrase_at_obs returns PRESENT only at/above confirm threshold
  A3  Agentic run navigates + captures + verifies before submit_answer
  A4  submit_answer is blocked until verify passes (or budget exhausted)
  A5  SigLIP stays loaded during verify; released only before VLM answer
  A6  Eval harness uses agentic path when eqa.agentic_verify=true

Multimodal cache (skip until vision-cache APIs land)
  C1  Obs SigLIP features persist across questions in a bank
  C2  Vision-prefix cache hits on second generate with same image hash
  C3  Verified answer path requests at most 1 image

Tool-calling router (agentic_tools + shared JSON contract)
  T1  EQA tools produce valid OpenAI-style schemas; names unique per mode
  T2  Router turn parses {"tool_calls": ...} and dispatches nav
  T3  Multi-tool reply executes in order; submit_answer still gated pre-verify
  T4  EMET_EQA_AGENTIC_ROUTER=0 → VLM never called; fallback-only sequence
  T5  State message annotates tried/failed hypotheses
  T6  Explore mode: finish gated until budget/frontiers done; explores first

GPU smoke (manual / overnight — not in this file)
  G1  scripts/debug_eqa_vlm_hang.py --with-image --n-images 1 finishes < 120s
  G2  run_dynagraph_dynamic_improve_smokes.sh → DONE with non-null eqa_accuracy
"""

from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


def _has_attr_path(mod_name: str, attr: str) -> bool:
    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        return False
    return hasattr(mod, attr)


def _require_agentic():
    if not _has_attr_path("emet.memory.graph_eqa.agentic_eqa", "run_agentic_eqa"):
        pytest.skip("agentic EQA module not present")
    from emet.memory.graph_eqa import GraphEQAMemory

    if not hasattr(GraphEQAMemory, "hypothesize_nav_targets"):
        pytest.skip("GraphEQAMemory.hypothesize_nav_targets not implemented")


def _require_vram_split():
    from emet.eval import dynagraph_vram as dv

    if not hasattr(dv, "warm_siglip_confirmed_memory") or not hasattr(dv, "release_siglip_for_vlm"):
        pytest.skip("VRAM warm/release split not implemented")


def _require_vision_cache():
    if not _has_attr_path("emet.llms.vl_vision_cache", "VisionPrefixCache"):
        pytest.skip("vl_vision_cache not implemented")


# ---------------------------------------------------------------------------
# Hang / reliability — must pass on current branch
# ---------------------------------------------------------------------------


def test_H1_release_zmq_ports_listeners_only():
    """H1: release_zmq_ports must pass listeners_only=True to avoid killing clients."""
    from emet.utils.port_utils import release_zmq_ports

    with patch("emet.utils.port_utils.kill_processes_on_port", return_value=True) as mock_kill:
        freed = release_zmq_ports(0)
    assert freed == [4401, 4402, 4403, 4404]
    assert mock_kill.call_count == 4
    for call in mock_kill.call_args_list:
        assert call.kwargs.get("listeners_only") is True


def test_H2_vram_prep_warms_then_releases_siglip():
    """H2: CONFIRMED_MEMORY features cached; encoders None before VLM."""
    from emet.eval.dynagraph_vram import prepare_dynagraph_vram_for_eqa
    from emet.memory.graph_eqa import GraphEQAMemory

    class Enc:
        def encode_image(self, rgb):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

        def encode_text(self, text):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    enc = Enc()
    agent = MagicMock()
    agent.encoder = enc
    agent.voxel_map = MagicMock()
    agent.voxel_map.encoder = enc
    gm = GraphEQAMemory(defer_llm_clients=True)
    gm.memory_summary_enabled = True
    gm.add_observation(np.zeros((8, 8, 3), dtype=np.uint8), np.array([1.0, 2.0, 0.5]), ["plant"])
    gm._relevant_phrases = ["woven basket"]
    agent.graph_memory = gm

    prepare_dynagraph_vram_for_eqa(agent)
    assert agent.encoder is None
    assert agent.voxel_map.encoder is None
    assert gm._confirmed_memory_siglip_encoder is None
    assert gm._obs_siglip_features


def test_H3_answer_only_skips_nav():
    """H3: allow_navigation=False → no navigate_to_target_pose / look_around."""
    from emet.controller.controller_graph_eqa import GraphEQAController

    agent = MagicMock(spec=GraphEQAController)
    agent._realtime_updates = False
    agent._fast_explore_lookaround = True
    agent._eqa_explore_when_uncovered = True
    agent._vlm_frontier_scoring = False
    agent._habitat_blocked_goals = set()
    agent._habitat_recent_goals = []
    agent.parameters = {"eqa_stall_patience": 0}
    agent.graph_memory = MagicMock()
    agent.graph_memory.query_answer.return_value = (
        "because",
        "near the counter",
        False,
        "still exploring",
        np.array([1.0, 2.0, 0.0]),
        [Image.new("RGB", (8, 8))],
    )
    agent.graph_memory.get_nodes.return_value = []
    agent.graph_memory.last_eqa_action_obs_id = None
    agent.robot = MagicMock()
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.planner = MagicMock()
    agent._planning_base_xyt = lambda xyt: xyt
    agent.navigate_to_target_pose = MagicMock(return_value=True)
    agent.rerun_visualizer = MagicMock()
    agent.space = MagicMock()
    agent.voxel_map = MagicMock()

    with patch.object(GraphEQAController, "_sync_graph_frontier_nodes", lambda self: None):
        with patch.object(GraphEQAController, "_rerun_refresh_monologue_panel", lambda self: None):
            GraphEQAController.run_eqa_one_iter(agent, "Where is the sink?", allow_navigation=False)

    agent.navigate_to_target_pose.assert_not_called()
    agent.look_around.assert_not_called()


# ---------------------------------------------------------------------------
# SigLIP building blocks (usable before full agentic module)
# ---------------------------------------------------------------------------


def test_S1_siglip_align_picks_best_obs():
    """S1: align_phrase_to_observation_features returns highest cosine obs."""
    from emet.memory.graph_eqa.graph_eqa_siglip import align_phrase_to_observation_features

    class Enc:
        def encode_text(self, text):
            return np.array([1.0, 0.0], dtype=np.float32)

    Obs = type("Obs", (), {})
    o1, o2 = Obs(), Obs()
    o1.obs_id, o1.xyz = 1, np.array([0.0, 0.0, 0.0])
    o2.obs_id, o2.xyz = 2, np.array([1.0, 0.0, 0.0])
    feats = {
        1: np.array([0.1, 0.9], dtype=np.float32),
        2: np.array([0.9, 0.1], dtype=np.float32),
    }
    best = align_phrase_to_observation_features("mug", Enc(), [o1, o2], feats)
    assert best is not None
    sim, xyz, oid = best
    assert oid == 2
    assert sim > 0.5
    assert float(xyz[0]) == 1.0


# ---------------------------------------------------------------------------
# Agentic contracts — skip until implemented
# ---------------------------------------------------------------------------


def test_A1_hypothesize_ranks_graph_label_over_siglip_candidate():
    """A1: graph label match outranks SigLIP-only candidate for nav target."""
    _require_agentic()
    from emet.memory.graph_eqa import GraphEQAMemory

    gm = GraphEQAMemory(defer_llm_clients=True)
    gm.memory_summary_enabled = True
    # Graph-labeled sink near (2, 0); unlabeled view near (0, 0) with high fake SigLIP.
    oid_sink = gm.add_observation(
        np.zeros((4, 4, 3), dtype=np.uint8), np.array([2.0, 0.0, 0.5]), ["sink"]
    )
    oid_other = gm.add_observation(
        np.ones((4, 4, 3), dtype=np.uint8) * 200, np.array([0.0, 0.0, 0.5]), ["wall"]
    )
    gm._obs_siglip_features[int(oid_other)] = np.array([1.0, 0.0], dtype=np.float32)
    gm._obs_siglip_features[int(oid_sink)] = np.array([0.2, 0.8], dtype=np.float32)
    gm._relevant_objects = ["sink"]
    gm._relevant_phrases = ["sink"]

    hyps = gm.hypothesize_nav_targets("Where is the sink?", max_k=3)
    assert hyps, "expected at least one hypothesis"
    top = hyps[0]
    top_oid = int(getattr(top, "obs_id", top.get("obs_id") if isinstance(top, dict) else -1))
    assert top_oid == int(oid_sink)


def test_A2_verify_threshold_gates_present():
    """A2: verify_phrase_at_obs PRESENT only when sim >= SIGLIP_CONFIRM_THRESHOLD."""
    _require_agentic()
    from emet.memory.graph_eqa import GraphEQAMemory
    from emet.memory.graph_eqa.graph_memory import SIGLIP_CONFIRM_THRESHOLD

    gm = GraphEQAMemory(defer_llm_clients=True)
    oid = gm.add_observation(np.zeros((4, 4, 3), dtype=np.uint8), np.array([1.0, 1.0, 0.5]), ["cup"])

    class Enc:
        def __init__(self, sim: float):
            self._sim = sim

        def encode_text(self, _t):
            return np.array([1.0, 0.0], dtype=np.float32)

        def encode_image(self, _rgb):
            return np.array([self._sim, 0.0], dtype=np.float32)

    gm.set_confirmed_memory_siglip_encoder(Enc(SIGLIP_CONFIRM_THRESHOLD - 0.05))
    low = gm.verify_phrase_at_obs("cup", int(oid))
    status_low = getattr(low, "status", low.get("status") if isinstance(low, dict) else None)
    assert status_low in ("CANDIDATE", "ABSENT", False) or status_low != "PRESENT"

    gm.set_confirmed_memory_siglip_encoder(Enc(1.0))
    high = gm.verify_phrase_at_obs("cup", int(oid), rgb=np.zeros((4, 4, 3), dtype=np.uint8))
    status_high = getattr(high, "status", high.get("status") if isinstance(high, dict) else None)
    assert status_high == "PRESENT" or getattr(high, "ok", None) is True


def test_A3_agentic_loop_nav_before_answer():
    """A3: run_agentic_eqa must navigate and verify before calling VLM submit_answer."""
    _require_agentic()
    from emet.memory.graph_eqa import agentic_eqa

    order: list[str] = []

    agent = MagicMock()
    agent.parameters = {"eqa": {"agentic_verify": True, "agentic_max_tool_rounds": 4}}
    agent.graph_memory = MagicMock()
    agent.graph_memory.eqa_client = None
    agent.graph_memory.memory_summary_enabled = False
    agent.graph_memory.hypothesize_nav_targets.return_value = [
        MagicMock(obs_id=7, xyz=np.array([1.0, 2.0, 0.0]), phrase="sink", score=0.9, source="graph")
    ]
    agent.graph_memory._navigation_waypoint_for_obs.return_value = np.array([1.0, 2.0, 1.0])
    agent.robot = MagicMock()
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.robot.get_observation.return_value = None

    def _nav(*_a, **_k):
        order.append("nav")
        return True

    def _update(*_a, **_k):
        order.append("update")
        agent.graph_memory._observations = [MagicMock(obs_id=8)]
        agent.graph_memory._obs_usable_for_eqa_image.return_value = True

    def _verify(*_a, **_k):
        order.append("verify")
        return MagicMock(
            status="PRESENT",
            sim=0.9,
            ok=True,
            obs_id=7,
            phrase="sink",
            text_feat=None,
            img_feat=None,
        )

    def _answer(*_a, **_k):
        order.append("answer")
        return ("ok", "A", True, "", None, [])

    agent.navigate_to_target_pose = _nav
    agent.update = _update
    agent.graph_memory.verify_phrase_at_obs = _verify
    agent.graph_memory.query_answer = _answer

    agentic_eqa.run_agentic_eqa(agent, "Where is the sink?")
    assert "nav" in order
    assert "verify" in order
    assert "answer" in order
    assert order.index("nav") < order.index("verify") < order.index("answer")


def test_A4_no_submit_without_verify_or_budget():
    """A4: submit_answer tool rejected until verify PRESENT or rounds exhausted."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    agent.graph_memory = MagicMock()
    ex = AgenticEQAExecutor(
        agent,
        question="Where is the mug?",
        max_rounds=2,
        verify_min_sim=0.28,
    )
    ex._verified = False
    out = ex.handle_tool("submit_answer", {"answer": "A"})
    assert "verify" in str(out).lower() or "not verified" in str(out).lower() or out.get("ok") is False


def test_follow_eqa_action_after_unknown_submit():
    """Unconfident Unknown + Action:obs must navigate instead of accepting the letter."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    order: list[str] = []
    agent = MagicMock()
    agent.parameters = {"eqa": {"agentic_verify": True}}
    agent.graph_memory = MagicMock()
    agent.graph_memory.last_eqa_action_obs_id = 11
    agent.graph_memory._navigation_waypoint_for_obs.return_value = np.array([1.0, 0.0, 1.0])
    agent.robot = MagicMock()
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.robot.get_observation.return_value = None

    def _nav(*_a, **_k):
        order.append("nav")
        return True

    def _update(*_a, **_k):
        order.append("update")

    def _verify(*_a, **_k):
        order.append("verify")
        return MagicMock(
            status="PRESENT",
            sim=0.9,
            ok=True,
            obs_id=11,
            phrase="clock",
            text_feat=None,
            img_feat=None,
        )

    agent.navigate_to_target_pose = _nav
    agent.update = _update
    agent.graph_memory.verify_phrase_at_obs = _verify
    agent.run_exploration = MagicMock(return_value=True)
    agent._siglip_guided_frontier = MagicMock(return_value=None)
    agent._best_frontier_point_from_graph = MagicMock(return_value=None)

    ex = AgenticEQAExecutor(
        agent,
        question="Where is the clock?",
        max_rounds=4,
        max_nav_steps=3,
        router=False,
    )
    # Even on the last round / after explore used the budget, Action:N must still run.
    ex._round = 3
    ex._n_explore = 3
    followed = ex._maybe_follow_eqa_explore_action(
        {"ok": True, "answer": "Unknown", "confidence": False}
    )
    assert followed is True
    assert "nav" in order and "verify" in order
    assert 11 in ex._followed_eqa_actions
    assert agent.graph_memory.last_eqa_action_obs_id is None
    # Same Action obs already followed → soft explore_frontier instead of locking Unknown.
    agent.graph_memory.last_eqa_action_obs_id = 11
    order.clear()
    assert ex._maybe_follow_eqa_explore_action({"ok": True, "answer": "Unknown", "confidence": False}) is True
    assert ex._n_unknown_explore == 1
    assert agent.run_exploration.called


def test_unknown_without_action_obs_explores():
    """Action:N OOB / missing obs id must soft-explore instead of accepting empty Unknown."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {"eqa": {"agentic_verify": True}}
    agent.graph_memory = MagicMock()
    agent.graph_memory.last_eqa_action_obs_id = None
    agent.robot = MagicMock()
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.robot.get_observation.return_value = None
    agent.navigate_to_target_pose = MagicMock(return_value=True)
    agent.update = MagicMock()
    agent.run_exploration = MagicMock(return_value=True)
    agent._siglip_guided_frontier = MagicMock(return_value=None)
    agent._best_frontier_point_from_graph = MagicMock(return_value=None)
    agent.graph_memory.verify_phrase_at_obs = MagicMock(
        return_value=MagicMock(
            status="ABSENT",
            sim=0.1,
            ok=True,
            obs_id=7,
            phrase="bowl",
            text_feat=None,
            img_feat=None,
        )
    )
    ex = AgenticEQAExecutor(agent, question="Where is the bowl?", max_rounds=6, max_nav_steps=4, router=False)
    ex._n_explore = 4
    assert ex._maybe_follow_eqa_explore_action({"ok": True, "answer": "Unknown", "confidence": False}) is True
    assert ex._n_unknown_explore == 1
    assert agent.run_exploration.called


def test_finalize_unknown_location_letter_salvages_after_explore():
    """After Action/explore is exhausted, force a VLM letter — not empty Unknown (q105)."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    agent.graph_memory = MagicMock()
    agent.graph_memory._salvage_location_mcq_letter.return_value = "A"
    img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    ex = AgenticEQAExecutor(
        agent,
        question=(
            "Where is the fruit bowl?\nA) kitchen island\nB) dining table\n"
            "C) coffee table\nD) sunroom"
        ),
        max_rounds=4,
        max_nav_steps=2,
        router=False,
    )
    ex._n_unknown_explore = 2
    out = ex._finalize_unknown_location_letter(
        {"ok": True, "answer": "Unknown", "confidence": False, "relevant_images": [img]}
    )
    assert out["answer"] == "A"
    assert "final-location-salvage" in out["discord_text"]
    agent.graph_memory._salvage_location_mcq_letter.assert_called_once()
    # Non-location / already-letter answers are left alone.
    keep = ex._finalize_unknown_location_letter(
        {"ok": True, "answer": "B", "confidence": False, "relevant_images": [img]}
    )
    assert keep["answer"] == "B"

    """Nav exhausted → submit_answer ok without PRESENT (so Action:N can be followed)."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    agent.graph_memory = MagicMock()
    agent.graph_memory.query_answer.return_value = ("r", "Unknown", False, "cr", None, [])
    agent.graph_memory.last_eqa_action_obs_id = 2
    ex = AgenticEQAExecutor(agent, question="Where?", max_rounds=6, max_nav_steps=2, router=False)
    ex._verified = False
    ex._round = 1
    ex._n_explore = 2
    out = ex.handle_tool("submit_answer", {})
    assert out.get("ok") is True
    assert str(out.get("answer") or "").lower() == "unknown"


def test_fallback_explores_remaining_budget_instead_of_verify_loop():
    """Hypotheses consumed + budget left → explore, not repeat verify (failfix6 burned 5 rounds)."""
    _require_agentic()
    from types import SimpleNamespace

    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    agent.graph_memory = MagicMock()
    agent.graph_memory.get_nodes.return_value = [SimpleNamespace(is_frontier=True, is_viewpoint=False)]
    ex = AgenticEQAExecutor(agent, question="Where is the clock?", max_rounds=8, max_nav_steps=6, router=False)
    ex._hypotheses = [MagicMock(obs_id=1), MagicMock(obs_id=2)]
    ex._hyp_i = 2
    ex._n_nav = 2
    ex._n_explore = 1
    ex._last_verify = MagicMock(status="ABSENT")
    tool, _args = ex._fallback_tool()
    assert tool == "explore_frontier"


def test_fallback_submits_when_budget_exhausted_after_verify():
    """Budget spent + a verify result on record → submit_answer, never re-verify."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    agent.graph_memory = MagicMock()
    ex = AgenticEQAExecutor(agent, question="Where is the clock?", max_rounds=8, max_nav_steps=3, router=False)
    ex._hypotheses = [MagicMock(obs_id=1)]
    ex._hyp_i = 1
    ex._n_nav = 2
    ex._n_explore = 1
    ex._last_verify = MagicMock(status="ABSENT")
    tool, _args = ex._fallback_tool()
    assert tool == "submit_answer"


def test_verify_phrase_prefers_question_stem_over_mcq_option():
    """Default verify phrase must come from the stem (fruit bowl), not an option (kitchen island)."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    gm = MagicMock()
    gm._relevant_phrases = []
    gm._relevant_objects = ["kitchen island", "fruit bowl"]
    seen: list[str] = []

    def _verify(phrase, _oid, rgb=None, min_sim=0.0):
        seen.append(phrase)
        return MagicMock(
            status="ABSENT", sim=0.1, ok=True, obs_id=5, phrase=phrase, text_feat=None, img_feat=None
        )

    gm.verify_phrase_at_obs = _verify
    agent.graph_memory = gm
    agent.robot = None
    ex = AgenticEQAExecutor(
        agent,
        question="I'm looking for the fruit bowl. A) On the kitchen island B) On the dining table. Answer:",
        max_rounds=4,
        router=False,
    )
    out = ex.handle_tool("verify_siglip", {"obs_id": 5})
    assert out.get("ok") is True
    assert seen == ["fruit bowl"]


def test_verify_without_obs_id_uses_latest_observation():
    """Dogfood q104/q105: post-explore verify checked the stale hypothesis every round.

    ``verify_siglip`` with no ``obs_id`` means "what am I looking at now", so it must
    target the frame just captured, not the nav hypothesis seeded at round 0.
    """
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    gm = MagicMock()
    gm._relevant_phrases = ["fruit bowl"]
    gm._relevant_objects = ["fruit bowl"]
    gm._observations = [MagicMock(obs_id=10), MagicMock(obs_id=42)]
    gm._obs_usable_for_eqa_image = lambda _oid: True
    verified_ids: list[int] = []

    def _verify(phrase, oid, rgb=None, min_sim=0.0):
        verified_ids.append(int(oid))
        return MagicMock(
            status="ABSENT", sim=0.05, ok=False, obs_id=int(oid), phrase=phrase,
            text_feat=None, img_feat=None,
        )

    gm.verify_phrase_at_obs = _verify
    agent.graph_memory = gm
    agent.robot = None
    ex = AgenticEQAExecutor(agent, question="I'm looking for the fruit bowl.", max_rounds=4, router=False)
    # Stale hypothesis from round 0; the robot has since explored and captured obs 42.
    ex._hypotheses = [
        MagicMock(obs_id=10, xyz=np.array([1.0, 2.0, 0.0]), phrase="fruit bowl", score=0.9, source="graph")
    ]

    ex.handle_tool("verify_siglip", {})
    assert verified_ids == [42]


def test_A5_siglip_alive_during_verify_released_before_vlm():
    """A5: warm keeps encoder; release_siglip_for_vlm drops it before answer VLM."""
    _require_vram_split()
    from emet.eval.dynagraph_vram import release_siglip_for_vlm, warm_siglip_confirmed_memory
    from emet.memory.graph_eqa import GraphEQAMemory

    class Enc:
        def encode_image(self, rgb):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

        def encode_text(self, text):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    enc = Enc()
    agent = MagicMock()
    agent.encoder = enc
    agent.voxel_map = MagicMock()
    agent.voxel_map.encoder = enc
    gm = GraphEQAMemory(defer_llm_clients=True)
    gm.memory_summary_enabled = True
    gm.add_observation(np.zeros((4, 4, 3), dtype=np.uint8), np.array([0.0, 0.0, 0.0]), ["x"])
    agent.graph_memory = gm

    warm_siglip_confirmed_memory(agent)
    assert agent.encoder is not None or gm._confirmed_memory_siglip_encoder is not None

    release_siglip_for_vlm(agent)
    assert agent.encoder is None
    assert gm._confirmed_memory_siglip_encoder is None


def test_A6_eval_harness_honors_agentic_flag(monkeypatch):
    """A6: _run_eqa_single calls run_agentic_eqa when eqa.agentic_verify is set."""
    _require_agentic()
    from emet.eval import dynamic_exploration_runner as der

    agent = MagicMock()
    agent.parameters = {"eqa": {"agentic_verify": True}}
    agent._fast_explore_lookaround = False
    robot = MagicMock()
    called = {"agentic": False}

    def _fake_run(agent_arg, qtext, **_kw):
        called["agentic"] = True
        return ("Answer: A\n", [])

    monkeypatch.setattr("emet.memory.graph_eqa.agentic_eqa.run_agentic_eqa", _fake_run)
    monkeypatch.setattr("emet.memory.graph_eqa.agentic_eqa.agentic_verify_enabled", lambda _a: True)
    der._run_eqa_single(agent, robot, {"question": "Where is X?", "answer": "A"})
    assert called["agentic"]


def test_A7_explore_when_no_hypothesis():
    """A7: with empty hypotheses, fallback explores before answering."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    order: list[str] = []
    agent = MagicMock()
    agent.parameters = {"eqa": {"agentic_max_nav_steps": 2}}
    agent.graph_memory = MagicMock()
    agent.graph_memory.eqa_client = None
    agent.graph_memory.memory_summary_enabled = False
    agent.graph_memory.hypothesize_nav_targets.return_value = []
    agent.robot = MagicMock()
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.robot.get_observation.return_value = None

    def _explore():
        order.append("explore")
        return True

    def _verify(*_a, **_k):
        order.append("verify")
        return MagicMock(
            status="ABSENT",
            sim=0.0,
            ok=False,
            obs_id=-1,
            phrase="x",
            text_feat=None,
            img_feat=None,
        )

    def _answer(*_a, **_k):
        order.append("answer")
        return ("r", "Unknown", False, "", None, [])

    agent.run_exploration = _explore
    agent.update = MagicMock()
    agent.graph_memory.verify_phrase_at_obs = _verify
    agent.graph_memory.query_answer = _answer

    with patch("emet.controller.habitat_nav.pick_uncovered_explore_target", return_value=None):
        ex = AgenticEQAExecutor(agent, "Where is the sink?", max_rounds=3, max_nav_steps=2)
        ex.run()
    assert "explore" in order
    assert order.index("explore") < order.index("answer")


def test_T1_eqa_tools_valid_schemas():
    """T1: Tool.schema() round-trips as JSON; names unique; mode swaps submit/finish."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_tools import build_agentic_eqa_tools

    ex = MagicMock()
    ex.mode = "answer"
    tools = build_agentic_eqa_tools(ex)
    names = [t.name for t in tools]
    assert len(names) == len(set(names))
    assert "submit_answer" in names
    assert "finish" not in names
    for t in tools:
        s = t.schema()
        assert s["type"] == "function"
        fn = s["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"].get("type") == "object"
        json.loads(json.dumps(s))

    ex.mode = "explore"
    names_x = [t.name for t in build_agentic_eqa_tools(ex)]
    assert "finish" in names_x
    assert "submit_answer" not in names_x


def test_T2_router_parses_and_dispatches_nav():
    """T2: parse_tool_calls_response reply dispatches navigate_to_obs."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    gm = agent.graph_memory
    gm.memory_summary_enabled = False
    reply = '{"tool_calls": [{"name": "navigate_to_obs", "arguments": {"obs_id": 3}}], "message": ""}'
    client = MagicMock(return_value=reply)
    gm.eqa_client = client
    gm._navigation_waypoint_for_obs.return_value = np.array([1.0, 2.0, 0.0])
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.navigate_to_target_pose = MagicMock(return_value=True)

    ex = AgenticEQAExecutor(agent, "Where is the sink?", max_rounds=3, max_nav_steps=2)
    calls, picked_by, meta = ex._route_tool_calls()
    assert picked_by == "vlm"
    assert meta["parse_ok"] is True
    assert calls == [("navigate_to_obs", {"obs_id": 3})]
    # System prompt is fixed and passed on the routing turn (prefix-cache contract).
    assert client.call_args.kwargs.get("system_prompt") == ex._system_prompt
    out = ex.handle_tool(*calls[0])
    assert out["ok"] is True
    agent.navigate_to_target_pose.assert_called_once()


def test_T3_multi_tool_reply_ordered_submit_gated():
    """T3: verify then submit executes in order; submit rejected before PRESENT."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    gm = agent.graph_memory
    gm.memory_summary_enabled = False
    reply = (
        '{"tool_calls": [{"name": "verify_siglip", "arguments": {"phrase": "sink", "obs_id": 7}}, '
        '{"name": "submit_answer", "arguments": {"answer": "B"}}], "message": ""}'
    )
    gm.eqa_client = MagicMock(return_value=reply)
    gm.verify_phrase_at_obs.return_value = MagicMock(
        status="ABSENT", sim=0.1, ok=False, obs_id=7, phrase="sink", text_feat=None, img_feat=None
    )
    agent.robot.get_observation.return_value = None

    ex = AgenticEQAExecutor(agent, "Where is the sink?", max_rounds=4)
    ex._fresh_obs_ids.add(7)
    calls, _picked_by, meta = ex._route_tool_calls()
    assert meta["tool_calls"] == ["verify_siglip", "submit_answer"]
    outs = [ex.handle_tool(n, a) for n, a in calls]
    assert outs[0]["ok"] is True
    assert outs[0]["status"] == "CANDIDATE"
    assert outs[1]["ok"] is False
    assert "verif" in str(outs[1].get("error", "")).lower()
    assert 7 in ex._tried


def test_T4_router_env_off_fallback_only(monkeypatch):
    """T4: EMET_EQA_AGENTIC_ROUTER=0 → no tool-router VLM; fallback still answers.

    Target extract / view assess may still call eqa_client (VLM-first gate).
    """
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    monkeypatch.setenv("EMET_EQA_AGENTIC_ROUTER", "0")
    order: list[str] = []
    agent = MagicMock()
    agent.parameters = {}
    gm = agent.graph_memory
    gm.memory_summary_enabled = False
    client = MagicMock(
        return_value='{"target_phrase":"sink","question_type":"location","notes":""}'
    )
    gm.eqa_client = client
    gm.hypothesize_nav_targets.return_value = [
        MagicMock(obs_id=7, xyz=np.array([1.0, 2.0, 0.0]), phrase="sink", score=0.9, source="graph")
    ]
    gm._navigation_waypoint_for_obs.return_value = np.array([1.0, 2.0, 1.0])
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.robot.get_observation.return_value = None

    def _nav(*_a, **_k):
        order.append("nav")
        return True

    def _verify(*_a, **_k):
        order.append("verify")
        return MagicMock(
            status="PRESENT", sim=0.9, ok=True, obs_id=7, phrase="sink", text_feat=None, img_feat=None
        )

    def _answer(*_a, **_k):
        order.append("answer")
        return ("ok", "A", True, "", None, [])

    agent.navigate_to_target_pose = _nav
    gm.verify_phrase_at_obs = _verify
    gm.query_answer = _answer

    class _Assess:
        target = "sink"
        present = True
        answerable = True
        need_more_views = False
        suggested_answer = "A"
        reason = "sink in view"
        raw = "{}"

    monkeypatch.setattr(
        "emet.eval.agentic_vlm_assess.assess_view_with_vlm",
        lambda *a, **k: _Assess(),
    )

    ex = AgenticEQAExecutor(agent, "Where is the sink?", max_rounds=4, max_nav_steps=2)
    assert ex._router_enabled is False
    result = ex.run()
    # No tool-routing system prompt (identity + Response format block).
    for call in client.call_args_list:
        sp = str((call.kwargs or {}).get("system_prompt") or "")
        assert "Response format" not in sp
        assert "tool_calls" not in sp
    assert order.index("nav") < order.index("verify") < order.index("answer")
    assert result.answer == "A"


def test_T5_state_message_marks_tried_hypotheses():
    """T5: after a failed verify, the state block annotates the hypothesis as tried."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.agentic_tools import build_state_message

    agent = MagicMock()
    agent.parameters = {}
    ex = AgenticEQAExecutor(agent, "Where is the mug?", max_rounds=4)
    hyp = MagicMock(obs_id=7, phrase="mug", score=0.9, source="graph")
    hyp.xyz = np.array([1.0, 2.0, 0.0])
    ex._hypotheses = [hyp]
    ex._tried[7] = "verify ABSENT sim=0.10"

    msg = build_state_message(ex)
    assert "Question: Where is the mug?" in msg
    assert "obs_id=7" in msg
    assert "tried: verify ABSENT" in msg


def test_T6_explore_mode_finish_gated():
    """T6: question=None → explore mode; finish rejected until budget done; explores first."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    gm = agent.graph_memory
    gm.eqa_client = None
    gm.memory_summary_enabled = False
    gm.hypothesize_nav_targets.return_value = []
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.robot.get_emet_session.return_value = None
    explored = {"n": 0}

    def _explore():
        explored["n"] += 1
        return True

    agent.run_exploration = _explore

    with patch("emet.controller.habitat_nav.pick_uncovered_explore_target", return_value=None):
        ex = AgenticEQAExecutor(agent, None, goal="map the apartment", max_rounds=5, max_nav_steps=2)
        assert ex.mode == "explore"
        early = ex.handle_tool("finish", {})
        assert early["ok"] is False
        result = ex.run()

    assert explored["n"] >= 1
    assert "explore_frontier" in ex._tool_log
    assert "finish" in ex._tool_log
    assert result.answer
    assert "Explore finished" in result.discord_text


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
        status="ABSENT", sim=0.1, ok=False, obs_id=1, phrase="x",
        text_feat=np.array([1.0, 0.0]), img_feat=np.array([0.0, 1.0]),
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
        {"event": "tool_pick", "picked_by": "vlm", "router_parse_ok": True,
         "router_tool_calls": ["navigate_to_obs"]},
        {"event": "tool_pick", "picked_by": "vlm", "router_parse_ok": True,
         "router_tool_calls": ["verify_siglip", "submit_answer"]},
        {"event": "tool_pick", "picked_by": "fallback", "router_parse_ok": False,
         "tool": "explore_frontier"},
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


def test_agentic_submit_does_not_clamp_answer_max_tokens(monkeypatch):
    """Bal-32 regression: agentic must not setdefault EMET_EQA_ANSWER_MAX_NEW_TOKENS=64."""
    _require_agentic()
    import os

    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    monkeypatch.delenv("EMET_EQA_ANSWER_MAX_NEW_TOKENS", raising=False)

    agent = MagicMock()
    gm = MagicMock()
    agent.graph_memory = gm
    agent.planner = None
    agent.robot = None

    def _qa(question, xyt, planner):
        # Env must remain unset so graph_memory default (256) applies.
        assert "EMET_EQA_ANSWER_MAX_NEW_TOKENS" not in os.environ
        return ("", "B", True, "", None, [])

    gm.query_answer.side_effect = _qa
    gm.select_obs_ids_for_verified_answer = MagicMock(return_value=[1])

    ex = AgenticEQAExecutor(agent, "Where is the lamp?", router=False, collect_trace=False)
    ex._verified = True
    ex._verified_obs_id = 1
    out = ex._do_submit_answer()
    assert out["ok"] is True
    assert out["answer"] == "B"
    assert "EMET_EQA_ANSWER_MAX_NEW_TOKENS" not in os.environ


def test_require_verified_abstains_when_never_present(monkeypatch):
    """Policy (1): with require_verified, budget exhaust without PRESENT → Unknown."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {"eqa": {"agentic_verify": True}}
    agent.graph_memory = MagicMock()
    agent.graph_memory.eqa_client = None
    agent.graph_memory.memory_summary_enabled = False
    agent.graph_memory.hypothesize_nav_targets.return_value = []
    agent.graph_memory.get_nodes.return_value = []
    agent.voxel_map = None
    agent.robot = MagicMock()
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.robot.get_observation.return_value = None

    agent.graph_memory.verify_phrase_at_obs.return_value = MagicMock(
        status="ABSENT",
        sim=0.05,
        ok=False,
        obs_id=1,
        phrase="towel",
        text_feat=None,
        img_feat=None,
    )
    # Seed one obs so verify has an id.
    agent.graph_memory._observations = [MagicMock(obs_id=1)]
    agent.graph_memory.last_eqa_obs_ids = [1]

    ex = AgenticEQAExecutor(
        agent,
        "Where is the striped towel?",
        router=False,
        require_verified=True,
        max_rounds=3,
        max_nav_steps=0,
        collect_trace=False,
    )
    result = ex.run()
    assert result.verified is False
    assert "Unknown" in result.answer
    assert result.confidence is False


def test_presence_without_answerability_does_not_auto_submit(monkeypatch):
    """Fused presence on a location MCQ must abstain, not force a letter."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor, AgenticState
    from emet.memory.graph_eqa.agentic_policy import EvidenceRecord

    agent = MagicMock()
    agent.parameters = {"eqa": {"agentic_verify": True, "agentic_require_verified": True}}
    agent.graph_memory = MagicMock()
    agent.graph_memory.eqa_client = None
    agent.graph_memory.memory_summary_enabled = False
    agent.graph_memory.hypothesize_nav_targets.return_value = []
    agent.graph_memory.get_nodes.return_value = []
    agent.voxel_map = None
    agent.robot = MagicMock()
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.robot.get_observation.return_value = None
    agent.graph_memory._observations = [MagicMock(obs_id=1, labels=["basket"])]
    agent.graph_memory.last_eqa_obs_ids = [1]
    agent.graph_memory._observation_by_id = MagicMock(
        return_value=MagicMock(obs_id=1, labels=["basket"])
    )

    question = (
        "Did you see the woven basket anywhere? "
        "A) By the kitchen counter B) Between TV and living room sofas "
        "C) Next to the dining table D) Next to the living room armchairs"
    )
    ex = AgenticEQAExecutor(
        agent,
        question,
        router=False,
        require_verified=True,
        max_rounds=2,
        max_nav_steps=0,
        collect_trace=True,
    )
    ex._evidence_policy.register_hypothesis("graph:1", "woven basket", prior_probability=0.5)
    ex._evidence_policy.choose("graph:1")
    ex._evidence_policy.approached(1)
    ex._evidence_policy.add_evidence(
        EvidenceRecord(
            hypothesis_id="graph:1",
            obs_id=1,
            phrase="woven basket",
            detector_score=0.5,
            detector_backend="owlv2",
            graph_label_match=True,
        )
    )
    assessment = ex._evidence_policy.assess(relation_sufficient=False)
    assert assessment.verified is True
    assert assessment.answerable is False
    assert ex._evidence_policy.state == AgenticState.REPLAN
    ex._verified = True
    ex._verified_obs_id = 1
    ex._n_nav = 0
    ex._round = 1

    out = ex._tool_submit_answer("")
    assert out.get("ok") is True
    assert "Unknown" in str(out.get("answer") or out.get("discord_text") or "")
    assert out.get("verified") is False
    assert any(row.get("tool") == "abstain_unverified" for row in ex._trace_rows)
    assert not any(
        row.get("tool") == "submit_answer" and row.get("event") != "tool_pick"
        for row in ex._trace_rows
    )


def test_voxel_sim_upgrades_full_frame_absent_to_present():
    """Policy (2): dense voxel cosine >= 0.21 upgrades PRESENT (DynaMem space)."""
    _require_agentic()
    import torch

    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    gm = MagicMock()
    agent.graph_memory = gm
    gm.verify_phrase_at_obs.return_value = MagicMock(
        status="ABSENT",
        sim=0.04,
        ok=False,
        obs_id=7,
        phrase="bookshelf",
        text_feat=None,
        img_feat=None,
    )
    vm = MagicMock()
    # one point belonging to obs 7 with high alignment
    vm.find_alignment_over_model.return_value = torch.tensor([[0.01, 0.25, 0.02]])
    sm = MagicMock()
    sm._obs_counts = torch.tensor([3, 7, 7])
    vm.semantic_memory = sm
    agent.voxel_map = vm
    agent.robot = MagicMock()
    agent.robot.get_observation.return_value = None
    # Avoid MagicMock rgb / accidental dense encode in unit tests.
    gm._observation_by_id = MagicMock(return_value=None)

    ex = AgenticEQAExecutor(agent, "Where is the large bookshelf?", router=False, collect_trace=True)
    ex._dense_max_sim_for_rgb = lambda *_a, **_k: None  # type: ignore[method-assign]
    out = ex._tool_verify_siglip("large bookshelf", 7)
    assert out["status"] == "PRESENT"
    # Voxel PRESENT is a cheap proposal; submit unlock requires VLM assess.
    assert out["verified"] is False
    assert out["answerable"] is False
    assert out["verify_channel"] == "voxel_obs"
    assert float(out["sim"]) >= 0.21


def test_never_reverify_same_view():
    """Interactive rule: second verify_siglip on the same obs_id is SKIPPED_SAME_VIEW."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    gm = MagicMock()
    agent.graph_memory = gm
    agent.voxel_map = None
    agent.robot = MagicMock()
    agent.robot.get_observation.return_value = None
    gm._observation_by_id = MagicMock(return_value=None)
    gm.verify_phrase_at_obs.return_value = MagicMock(
        status="ABSENT",
        sim=0.02,
        ok=False,
        obs_id=5,
        phrase="towel",
        text_feat=None,
        img_feat=None,
    )
    ex = AgenticEQAExecutor(agent, "Where is the towel?", router=False, collect_trace=True)
    ex._dense_max_sim_for_rgb = lambda *_a, **_k: None  # type: ignore[method-assign]
    first = ex._tool_verify_siglip("towel", 5)
    assert first["ok"] is True
    assert first["status"] == "ABSENT"
    second = ex._tool_verify_siglip("towel", 5)
    assert second["ok"] is False
    assert second["status"] == "SKIPPED_SAME_VIEW"
    assert gm.verify_phrase_at_obs.call_count == 1


def test_fallback_skips_already_tried_hypothesis():
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    agent.graph_memory = MagicMock()
    agent.graph_memory.get_nodes.return_value = []
    agent.voxel_map = None
    ex = AgenticEQAExecutor(
        agent, "Where?", router=False, collect_trace=False, max_nav_steps=3, require_verified=True
    )
    ex._hypotheses = [
        MagicMock(obs_id=1, xyz=np.zeros(3), phrase="a", score=1.0, source="graph"),
        MagicMock(obs_id=2, xyz=np.zeros(3), phrase="b", score=0.9, source="graph"),
    ]
    ex._tried[1] = "verify ABSENT sim=0.01"
    tool, args = ex._fallback_tool()
    assert tool == "navigate_to_obs"
    assert int(args["obs_id"]) == 2


def test_image_verify_three_band_absent_candidate_present():
    """Image SigLIP: <0.10 ABSENT, [0.10,0.12) CANDIDATE, >=0.12 PRESENT."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    def _run(sim: float):
        agent = MagicMock()
        agent.parameters = {"eqa": {}}
        gm = MagicMock()
        agent.graph_memory = gm
        agent.voxel_map = None
        agent.robot = MagicMock()
        agent.robot.get_observation.return_value = None
        gm._observation_by_id = MagicMock(return_value=None)
        gm.verify_phrase_at_obs.return_value = MagicMock(
            status="ABSENT",
            sim=sim,
            ok=False,
            obs_id=1,
            phrase="towel",
            text_feat=None,
            img_feat=None,
        )
        ex = AgenticEQAExecutor(agent, "Where is the towel?", router=False, collect_trace=False)
        ex._dense_max_sim_for_rgb = lambda *_a, **_k: None  # type: ignore[method-assign]
        return ex._tool_verify_siglip("towel", 1)

    assert _run(0.05)["status"] == "ABSENT"
    assert _run(0.05)["verified"] is False
    mid = _run(0.105)
    assert mid["status"] == "CANDIDATE"
    assert mid["verified"] is False
    hi = _run(0.13)
    assert hi["status"] == "PRESENT"
    # Image SigLIP is a proposal channel; it cannot establish fused verification alone.
    assert hi["verified"] is False
    assert hi["fused_verified"] is False


def test_vlm_assess_unlocks_verified_submit_gate(monkeypatch):
    """Multimodal VLM answerable=True is what sets verified / ANSWER — not OWL."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor, AgenticState

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    gm = MagicMock()
    agent.graph_memory = gm
    agent.voxel_map = None
    agent.robot = MagicMock()
    agent.robot.get_observation.return_value = MagicMock(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8)
    )
    gm._observation_by_id = MagicMock(return_value=None)
    gm.verify_phrase_at_obs.return_value = MagicMock(
        status="ABSENT",
        sim=0.04,
        ok=False,
        obs_id=9,
        phrase="utensils",
        text_feat=None,
        img_feat=None,
    )
    gm.eqa_client = MagicMock()

    class _Assess:
        target = "utensils"
        present = True
        answerable = True
        need_more_views = False
        suggested_answer = "A"
        reason = "place settings visible on table"
        raw = "{}"

        def to_dict(self):
            return {}

    monkeypatch.setattr(
        "emet.eval.agentic_vlm_assess.assess_view_with_vlm",
        lambda *a, **k: _Assess(),
    )
    monkeypatch.setattr(
        "emet.eval.agentic_vlm_assess.build_inventory_brief",
        lambda **k: "brief",
    )

    ex = AgenticEQAExecutor(agent, "Where are the utensils?", router=False, collect_trace=True)
    ex._dense_max_sim_for_rgb = lambda *_a, **_k: None  # type: ignore[method-assign]
    ex._target_phrase = "utensils"
    out = ex._tool_verify_siglip("utensils", 9)
    assert out["verified"] is True
    assert out["answerable"] is True
    assert ex._evidence_policy.state == AgenticState.ANSWER
    assert any(r.get("tool") == "vlm_assess" and r.get("answerable") for r in ex._trace_rows)


def test_not_present_streak_sets_escape_floor():
    """q104/q105: repeated 'not visible' views must push the next frontier away."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import (
        ESCAPE_MIN_TRAVEL_M,
        NOT_PRESENT_ESCAPE_STREAK,
        AgenticEQAExecutor,
    )

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    agent.graph_memory = MagicMock()
    agent.voxel_map = None
    ex = AgenticEQAExecutor(agent, "Where is the clock?", router=False, collect_trace=False)

    for _ in range(NOT_PRESENT_ESCAPE_STREAK - 1):
        ex._update_escape_streak(present=False)
    assert ex._escape_min_travel_m() == 0.0

    ex._update_escape_streak(present=False)
    assert ex._escape_min_travel_m() == ESCAPE_MIN_TRAVEL_M
    assert agent._explore_min_travel_m == ESCAPE_MIN_TRAVEL_M

    # Seeing the target again clears the floor so we can close in on it.
    ex._update_escape_streak(present=True)
    assert ex._escape_min_travel_m() == 0.0
    assert agent._explore_min_travel_m == 0.0


def test_capture_rejects_non_advancing_obs():
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    gm = MagicMock()
    agent.graph_memory = gm
    gm.memory_summary_enabled = False
    gm._observations = []

    def _update():
        if not gm._observations:
            gm._observations = [MagicMock(obs_id=1)]

    agent.update = MagicMock(side_effect=_update)

    ex = AgenticEQAExecutor(agent, "Where?", router=False, collect_trace=True)
    first = ex._tool_capture_and_update()
    assert first["ok"] is True
    assert int(first["obs_id"]) == 1
    second = ex._tool_capture_and_update()
    assert second["ok"] is False
    assert second["status"] == "NO_NEW_OBS"

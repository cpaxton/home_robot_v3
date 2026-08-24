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

from emet.controller.habitat_nav import NavOutcome


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


def test_agentic_executor_consumes_manifest_budget_environment(monkeypatch):
    from emet.memory.graph_eqa.agentic_eqa import build_agentic_eqa_executor

    monkeypatch.setenv("EMET_EQA_AGENTIC_MAX_TOOL_ROUNDS", "5")
    monkeypatch.setenv("EMET_EQA_AGENTIC_MAX_NAV_STEPS", "4")
    agent = MagicMock()
    agent.parameters = {"eqa": {"agentic_max_tool_rounds": 8, "agentic_max_nav_steps": 8}}
    with patch("emet.eval.dynagraph_vram.warm_siglip_confirmed_memory"):
        executor = build_agentic_eqa_executor(agent, "Where is the chair?")
    assert executor.max_rounds == 5
    assert executor.max_nav_steps == 4


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
    agent.navigate_to_target_pose = MagicMock(return_value=NavOutcome.REACHED)
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
    oid_sink = gm.add_observation(np.zeros((4, 4, 3), dtype=np.uint8), np.array([2.0, 0.0, 0.5]), ["sink"])
    oid_other = gm.add_observation(np.ones((4, 4, 3), dtype=np.uint8) * 200, np.array([0.0, 0.0, 0.5]), ["wall"])
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
        return NavOutcome.REACHED

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
        return NavOutcome.REACHED

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
    followed = ex._maybe_follow_eqa_explore_action({"ok": True, "answer": "Unknown", "confidence": False})
    assert followed is True
    assert "nav" in order and "verify" in order
    assert 11 in ex._followed_eqa_actions
    assert agent.graph_memory.last_eqa_action_obs_id is None
    assert agent.graph_memory.last_eqa_look_obs_id == 11
    # Same Action obs already followed → soft explore_frontier instead of locking Unknown.
    agent.graph_memory.last_eqa_action_obs_id = 11
    order.clear()
    assert ex._maybe_follow_eqa_explore_action({"ok": True, "answer": "Unknown", "confidence": False}) is True
    assert ex._n_unknown_explore == 1
    assert agent.run_exploration.called


def _follow_action_executor(question: str):
    """Shared nav/verify mocks for Action:N follow tests."""
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    order: list[str] = []
    agent = MagicMock()
    agent.parameters = {"eqa": {"agentic_verify": True}}
    agent.graph_memory = MagicMock()
    agent.graph_memory.last_eqa_action_obs_id = 11
    agent.graph_memory.last_eqa_look_obs_id = None
    agent.graph_memory._navigation_waypoint_for_obs.return_value = np.array([1.0, 0.0, 1.0])
    agent.robot = MagicMock()
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.robot.get_observation.return_value = None

    def _nav(*_a, **_k):
        order.append("nav")
        return NavOutcome.REACHED

    def _update(*_a, **_k):
        order.append("update")

    def _verify(*_a, **_k):
        order.append("verify")
        return MagicMock(
            status="PRESENT",
            sim=0.9,
            ok=True,
            obs_id=11,
            phrase="lamp",
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
        question=question,
        max_rounds=4,
        max_nav_steps=3,
        router=False,
    )
    ex._round = 3
    ex._n_explore = 3
    return ex, agent, order


def test_follow_eqa_action_after_unconfident_count():
    """Unconfident One + Action:obs must look at that RGB instead of scoring One."""
    _require_agentic()
    ex, agent, order = _follow_action_executor(
        "How many table lamps are there? A) One B) Two C) Three D) None"
    )
    followed = ex._maybe_follow_eqa_explore_action({"ok": True, "answer": "One", "confidence": False})
    assert followed is True
    assert "nav" in order
    assert 11 in ex._followed_eqa_actions
    assert agent.graph_memory.last_eqa_look_obs_id == 11


def test_follow_eqa_action_skips_unconfident_location_letter():
    """A location letter is a guess we can score; do not nav just because conf is false."""
    _require_agentic()
    ex, agent, order = _follow_action_executor(
        "Where is the clock? A) Above the sink B) On the wall C) In the hallway D) Unknown"
    )
    followed = ex._maybe_follow_eqa_explore_action(
        {"ok": True, "answer": "Above the sink", "confidence": False}
    )
    assert followed is False
    assert order == []
    assert agent.graph_memory.last_eqa_look_obs_id is None


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
    agent.navigate_to_target_pose = MagicMock(return_value=NavOutcome.REACHED)
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


def test_finalize_unknown_location_letter_counterfactual_salvage():
    """Scored Unknown stays; salvage VLM is called and logged as counterfactual."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    agent.graph_memory = MagicMock()
    agent.graph_memory._salvage_location_mcq_letter.return_value = "A"
    agent.graph_memory.last_eqa_images = None
    agent.graph_memory.last_relevant_images = None
    img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    ex = AgenticEQAExecutor(
        agent,
        question="\n".join(
            [
                "Where is the fruit bowl?",
                "A) kitchen island",
                "B) dining table",
                "C) coffee table",
                "D) sunroom",
            ]
        ),
        max_rounds=4,
        max_nav_steps=2,
        router=False,
        collect_trace=True,
    )
    ex._n_unknown_explore = 2
    out = ex._finalize_unknown_location_letter(
        {"ok": True, "answer": "Unknown", "confidence": False, "relevant_images": [img]}
    )
    assert out["answer"] == "Unknown"
    assert "final-location-salvage" not in str(out.get("discord_text") or "")
    agent.graph_memory._salvage_location_mcq_letter.assert_called_once()
    assert ex._salvage_counterfactual_letter == "A"
    assert any(r.get("event") == "final_location_salvage_skipped" for r in ex._trace_rows)
    assert any(
        r.get("event") == "final_location_salvage_counterfactual"
        and r.get("letter") == "A"
        and r.get("applied") is False
        for r in ex._trace_rows
    )
    # Non-location / already-letter answers are left alone (no salvage call).
    agent.graph_memory._salvage_location_mcq_letter.reset_mock()
    keep = ex._finalize_unknown_location_letter(
        {"ok": True, "answer": "B", "confidence": False, "relevant_images": [img]}
    )
    assert keep["answer"] == "B"
    agent.graph_memory._salvage_location_mcq_letter.assert_not_called()


def test_finalize_unknown_skips_salvage_without_images():
    """No images → skip counterfactual VLM call; still mark scored no-salvage."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    gm = agent.graph_memory
    gm._salvage_location_mcq_letter = MagicMock(return_value="A")
    gm.last_eqa_images = None
    gm.last_relevant_images = None
    q = "\n".join(
        [
            "Where is the wall clock?",
            "A) Above the sink",
            "B) Next to the refrigerator",
            "C) Near the stove",
            "D) On the wall opposite the windows",
        ]
    )
    ex = AgenticEQAExecutor(agent, q, max_rounds=2, max_nav_steps=4, collect_trace=True)
    out = ex._finalize_unknown_location_letter({"answer": "Unknown", "confidence": False, "relevant_images": []})
    assert out.get("answer") == "Unknown"
    assert gm._salvage_location_mcq_letter.call_count == 0
    assert ex._salvage_counterfactual_letter == ""
    assert any(r.get("event") == "final_location_salvage_skipped" for r in ex._trace_rows)
    assert not any(r.get("event") == "final_location_salvage_counterfactual" for r in ex._trace_rows)


def test_submit_answer_ok_when_nav_exhausted_without_present():
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
        return MagicMock(status="ABSENT", sim=0.1, ok=True, obs_id=5, phrase=phrase, text_feat=None, img_feat=None)

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
            status="ABSENT",
            sim=0.05,
            ok=False,
            obs_id=int(oid),
            phrase=phrase,
            text_feat=None,
            img_feat=None,
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
    agent.navigate_to_target_pose = MagicMock(return_value=NavOutcome.REACHED)

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
    client = MagicMock(return_value='{"target_phrase":"sink","question_type":"location","notes":""}')
    gm.eqa_client = client
    gm.hypothesize_nav_targets.return_value = [
        MagicMock(obs_id=7, xyz=np.array([1.0, 2.0, 0.0]), phrase="sink", score=0.9, source="graph")
    ]
    gm._navigation_waypoint_for_obs.return_value = np.array([1.0, 2.0, 1.0])
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.robot.get_observation.return_value = None

    def _nav(*_a, **_k):
        order.append("nav")
        return NavOutcome.REACHED

    def _verify(*_a, **_k):
        order.append("verify")
        return MagicMock(status="PRESENT", sim=0.9, ok=True, obs_id=7, phrase="sink", text_feat=None, img_feat=None)

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
    assert "Investigate" in msg
    assert "#1 best" not in msg
    assert "highest-score" not in msg
    assert "score=" not in msg.split("Investigate", 1)[-1]


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

    def _qa(question, xyt, planner, *, force_obs_ids=None):
        # Env must remain unset so graph_memory default (256) applies.
        assert "EMET_EQA_ANSWER_MAX_NEW_TOKENS" not in os.environ
        assert force_obs_ids == [1]
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
    gm.query_answer.assert_called()
    assert gm.query_answer.call_args.kwargs.get("force_obs_ids") == [1]


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
    agent.graph_memory._observation_by_id = MagicMock(return_value=MagicMock(obs_id=1, labels=["basket"]))

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
    # Evidence never established answerability, so this is forced option text
    # carrying its provenance and a low calibrated confidence — not a silent Unknown.
    assert out.get("answer_provenance") == "uniform_prior"
    assert out.get("answer") in {
        "By the kitchen counter",
        "Between TV and living room sofas",
        "Next to the dining table",
        "Next to the living room armchairs",
    }
    assert float(out.get("answer_confidence")) <= 0.5
    forced = [row for row in ex._trace_rows if row.get("tool") == "forced_answer"]
    assert len(forced) == 1
    assert forced[0]["reason"] == "target evidence did not establish answer sufficiency"


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
    ex = AgenticEQAExecutor(agent, "Where?", router=False, collect_trace=False, max_nav_steps=3, require_verified=True)
    ex._hypotheses = [
        MagicMock(obs_id=1, xyz=np.zeros(3), phrase="a", score=1.0, source="graph"),
        MagicMock(obs_id=2, xyz=np.zeros(3), phrase="b", score=0.9, source="graph"),
    ]
    ex._tried[1] = "verify ABSENT sim=0.01"
    tool, args = ex._fallback_tool()
    assert tool == "investigate"
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


def test_vlm_assess_unlocks_even_when_siglip_absent(monkeypatch):
    """Qwen answerable + phrase corroboration unlocks; SigLIP ABSENT is not a hard block."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor, AgenticState

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    gm = MagicMock()
    agent.graph_memory = gm
    agent.voxel_map = None
    agent.robot = MagicMock()
    agent.robot.get_observation.return_value = MagicMock(rgb=np.zeros((4, 4, 3), dtype=np.uint8))
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
    gm.labels_near_obs = MagicMock(return_value=["utensils", "table"])
    gm._observations = [MagicMock(labels=["utensils", "plate"])]
    gm._nodes = []

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
    assert out["status"] == "ABSENT"
    assert out["verified"] is True
    assert out["answerable"] is True
    assert ex._evidence_policy.state == AgenticState.ANSWER
    vlm_rows = [r for r in ex._trace_rows if r.get("tool") == "vlm_assess"]
    assert vlm_rows[-1].get("suggested_answer") == "A"
    assert vlm_rows[-1].get("proposal_status") == "ABSENT"
    assert any(r.get("event") == "answerable_confirmed" for r in ex._trace_rows)


def test_answerable_deferred_without_phrase_hit(monkeypatch):
    """First answerable without inventory corroboration does not unlock submit."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor, AgenticState
    from emet.memory.graph_eqa.agentic_tools import build_state_message

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    gm = MagicMock()
    agent.graph_memory = gm
    agent.voxel_map = None
    agent.robot = MagicMock()
    agent.robot.get_observation.return_value = MagicMock(rgb=np.zeros((4, 4, 3), dtype=np.uint8))
    gm._observation_by_id = MagicMock(return_value=None)
    gm.verify_phrase_at_obs.return_value = MagicMock(
        status="PRESENT",
        sim=0.2,
        ok=True,
        obs_id=9,
        phrase="clock",
        text_feat=None,
        img_feat=None,
    )
    gm.eqa_client = MagicMock()
    gm.labels_near_obs = MagicMock(return_value=["sofa", "lamp"])
    gm._observations = [MagicMock(labels=["sofa"])]
    gm._nodes = []
    gm.memory_summary_enabled = False

    class _Assess:
        target = "clock"
        present = True
        answerable = True
        need_more_views = False
        suggested_answer = "living room"
        reason = "clock on wall"
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

    ex = AgenticEQAExecutor(
        agent,
        "Where is the clock? A) kitchen B) living room C) bedroom D) bathroom",
        router=False,
        collect_trace=True,
        single_view_confirm=False,
    )
    ex._dense_max_sim_for_rgb = lambda *_a, **_k: None  # type: ignore[method-assign]
    ex._target_phrase = "clock"
    ex._tool_verify_siglip("clock", 9)
    assert ex._verified is False
    assert ex._pending_answerable is not None
    assert ex._pending_answerable.get("letter") == "B"
    assert ex._evidence_policy.state == AgenticState.REPLAN
    assert any(r.get("event") == "answerable_deferred" for r in ex._trace_rows)
    msg = build_state_message(ex)
    assert "pending_answer=living room" in msg
    assert "pending_answer=B" not in msg


def test_answerable_two_view_agree_unlocks(monkeypatch):
    """Second answerable with same letter on a different obs unlocks."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor, AgenticState

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    gm = MagicMock()
    agent.graph_memory = gm
    agent.voxel_map = None
    agent.robot = MagicMock()
    agent.robot.get_observation.return_value = MagicMock(rgb=np.zeros((4, 4, 3), dtype=np.uint8))
    gm._observation_by_id = MagicMock(return_value=None)
    gm.eqa_client = MagicMock()
    gm.labels_near_obs = MagicMock(return_value=["sofa"])
    gm._observations = [MagicMock(labels=["sofa"])]
    gm._nodes = []

    class _Assess:
        target = "clock"
        present = True
        answerable = True
        need_more_views = False
        suggested_answer = "B"
        reason = "guess"
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

    def _verify_ret(obs_id: int):
        return MagicMock(
            status="CANDIDATE",
            sim=0.11,
            ok=True,
            obs_id=obs_id,
            phrase="clock",
            text_feat=None,
            img_feat=None,
        )

    gm.verify_phrase_at_obs.side_effect = lambda phrase, obs_id, **k: _verify_ret(int(obs_id))

    ex = AgenticEQAExecutor(
        agent,
        "Where is the clock? A) kitchen B) living room C) bedroom D) bathroom",
        router=False,
        collect_trace=True,
        single_view_confirm=False,
    )
    ex._dense_max_sim_for_rgb = lambda *_a, **_k: None  # type: ignore[method-assign]
    ex._target_phrase = "clock"
    ex._tool_verify_siglip("clock", 9)
    assert ex._verified is False
    # Allow second assess on a new obs (clear same-view skip set entry is automatic via new id).
    ex._tool_verify_siglip("clock", 11)
    assert ex._verified is True
    assert ex._evidence_policy.state == AgenticState.ANSWER
    assert ex._confirmed_answer_evidence is not None
    assert (ex._confirmed_answer_evidence.letter, ex._confirmed_answer_evidence.obs_id) == ("B", 11)
    assert any(r.get("event") == "answerable_confirmed" and r.get("reason") == "two_view_agree" for r in ex._trace_rows)


def test_need_more_views_blocks_unlock(monkeypatch):
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor, AgenticState

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    gm = MagicMock()
    agent.graph_memory = gm
    agent.voxel_map = None
    agent.robot = MagicMock()
    agent.robot.get_observation.return_value = MagicMock(rgb=np.zeros((4, 4, 3), dtype=np.uint8))
    gm._observation_by_id = MagicMock(return_value=None)
    gm.verify_phrase_at_obs.return_value = MagicMock(
        status="PRESENT", sim=0.2, ok=True, obs_id=9, phrase="clock", text_feat=None, img_feat=None
    )
    gm.eqa_client = MagicMock()
    gm.labels_near_obs = MagicMock(return_value=["clock", "wall"])
    gm._observations = [MagicMock(labels=["clock"])]
    gm._nodes = []

    class _Assess:
        target = "clock"
        present = True
        answerable = True
        need_more_views = True
        suggested_answer = "A"
        reason = "need another angle"
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

    ex = AgenticEQAExecutor(
        agent,
        "Where is the clock? A) kitchen B) living room C) bedroom D) bathroom",
        router=False,
        collect_trace=True,
    )
    ex._dense_max_sim_for_rgb = lambda *_a, **_k: None  # type: ignore[method-assign]
    ex._target_phrase = "clock"
    ex._tool_verify_siglip("clock", 9)
    assert ex._verified is False
    assert ex._evidence_policy.state == AgenticState.REPLAN
    assert any(r.get("event") == "answerable_deferred" and r.get("reason") == "need_more_views" for r in ex._trace_rows)


def test_submit_keeps_qwen_letter_when_query_echoes_xyz():
    """Graph XYZ echo must not overwrite Qwen's MCQ letter."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    gm = MagicMock()
    agent.graph_memory = gm
    agent.planner = None
    agent.robot = None
    gm.query_answer.return_value = (
        "",
        "The fan is at approximately (9.15, 1.71, 0.84) m.",
        False,
        "",
        None,
        [],
    )
    gm.select_obs_ids_for_verified_answer = MagicMock(return_value=[1])

    ex = AgenticEQAExecutor(
        agent,
        "Where is the fan? A) Living room B) Bedroom C) Next to the bed D) Kitchen",
        router=False,
        collect_trace=True,
    )
    ex._verified = True
    ex._verified_obs_id = 1
    ex._last_vlm_assess = {"present": True, "suggested_answer": "Next to the bed"}
    out = ex._do_submit_answer()
    assert out["answer"] == "Next to the bed"
    submit = next(r for r in ex._trace_rows if r.get("tool") == "submit_answer")
    assert submit["answer_source"] == "vlm_suggested"

    out2 = ex._do_submit_answer(prefer_answer="Kitchen")
    assert out2["answer"] == "Kitchen"


def test_vlm_assess_unlocks_verified_submit_gate(monkeypatch):
    """Qwen answerable=True is what sets verified / ANSWER — not OWL/SigLIP alone."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor, AgenticState

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    gm = MagicMock()
    agent.graph_memory = gm
    agent.voxel_map = None
    agent.robot = MagicMock()
    agent.robot.get_observation.return_value = MagicMock(rgb=np.zeros((4, 4, 3), dtype=np.uint8))
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
    gm.labels_near_obs = MagicMock(return_value=["utensils", "table"])
    gm._observations = [MagicMock(labels=["utensils"])]
    gm._nodes = []

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


def test_sync_scored_answer_appends_agentic_submit_for_habitat_scoring():
    """Agentic sync preserves option text while Habitat still resolves choice D."""
    _require_agentic()
    from emet.habitat.metrics import extract_mcq_letter_from_raw_eqa
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor, AgenticEQAResult

    agent = MagicMock()
    gm = MagicMock()
    agent.graph_memory = gm
    gm.last_eqa_raw = "Caption:\n...\n[salvage]\nanswer:\nA\n"
    gm.last_eqa_parsed = ("r", "A", False, "", "")
    gm.last_eqa_obs_ids = [20]

    ex = AgenticEQAExecutor(
        agent,
        "How many red pillows? A)1 B)3 C)4 D)2",
        router=False,
        collect_trace=True,
    )
    result = AgenticEQAResult(
        discord_text="Answer:2",
        answer="2",
        confidence=True,
        relevant_images=[],
        tool_log=["submit_answer"],
        verified=True,
        verified_obs_id=20,
        n_rounds=1,
        n_nav=1,
        n_explore=0,
        wall_s=1.0,
        budget_hit=False,
    )
    ex._sync_scored_answer_to_graph_memory(result, {"answer_source": "vlm_suggested"})
    assert "[agentic_submit]" in gm.last_eqa_raw
    assert gm.last_eqa_parsed[1] == "2"
    letter = extract_mcq_letter_from_raw_eqa(
        gm.last_eqa_raw,
        ["1", "3", "4", "2"],
    )
    assert letter == "D"
    sync = next(r for r in ex._trace_rows if r.get("event") == "sync_scored_answer")
    assert sync["answer_text"] == "2"
    assert sync["choice_index"] == 3
    assert "letter" not in sync


def test_navigate_no_new_obs_looks_around_verifies_and_flags_loop():
    """Router-on must not spin navigate_to_obs on the same id without planner updates."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import NAV_SAME_OBS_LOOP_LIMIT, AgenticEQAExecutor
    from emet.memory.graph_eqa.agentic_tools import build_state_message
    from emet.memory.graph_eqa.graph_memory import VerifyResult

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    gm = MagicMock()
    agent.graph_memory = gm
    gm.memory_summary_enabled = False
    gm._observations = [MagicMock(obs_id=16, labels=["cabinet"])]
    gm._navigation_waypoint_for_obs = MagicMock(return_value=np.array([1.0, 2.0, 1.0]))
    gm.record_nav_attempt = MagicMock()
    gm.hypothesize_nav_targets = MagicMock(return_value=[])
    gm.verify_phrase_at_obs = MagicMock(
        return_value=VerifyResult(status="ABSENT", sim=0.05, obs_id=16, phrase="wall clock", ok=False)
    )
    gm._observation_by_id = MagicMock(return_value=gm._observations[0])
    agent.navigate_to_target_pose = MagicMock(return_value=NavOutcome.REACHED)
    agent.look_around = MagicMock()
    agent.update = MagicMock()  # never advances obs id
    agent.robot = None

    ex = AgenticEQAExecutor(
        agent,
        "Where is the large wall clock? A) dining B) kitchen C) sunroom D) living",
        router=True,
        collect_trace=True,
        max_nav_steps=5,
    )
    ex._dense_max_sim_for_rgb = lambda *_a, **_k: None  # type: ignore[method-assign]
    ex._voxel_max_sim_for_obs = lambda *_a, **_k: None  # type: ignore[method-assign]
    ex._target_phrase = "large wall clock"

    out = ex._tool_navigate_to_obs(16)
    assert out["look_around_on_no_new_obs"] is True
    agent.look_around.assert_called()
    assert out.get("capture", {}).get("status") == "NO_NEW_OBS"
    assert gm.last_eqa_look_obs_id == 16
    assert 16 in ex._tried
    assert str(ex._tried[16]).startswith("STALLED_NAV_LOOP")
    assert any(r.get("event") == "nav_loop" for r in ex._trace_rows)
    assert gm.verify_phrase_at_obs.called

    state = build_state_message(ex)
    assert "NAV_LOOP" in state

    blocked = ex._tool_navigate_to_obs(16)
    assert blocked.get("status") == "NAV_LOOP_BLOCKED"
    assert blocked.get("ok") is False
    assert NAV_SAME_OBS_LOOP_LIMIT >= 2


def test_capture_advancing_obs_refreshes_hypotheses():
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    gm = MagicMock()
    agent.graph_memory = gm
    gm.memory_summary_enabled = False
    gm._observations = [MagicMock(obs_id=1, labels=["a"])]
    gm.hypothesize_nav_targets = MagicMock(
        return_value=[
            NavHypothesis(
                phrase="clock",
                obs_id=2,
                xyz=np.array([0.0, 0.0, 0.0]),
                score=1.0,
                source="graph",
            )
        ]
    )

    def _update():
        gm._observations = [
            MagicMock(obs_id=1, labels=["a"]),
            MagicMock(obs_id=2, labels=["clock"]),
        ]

    agent.update = MagicMock(side_effect=_update)
    ex = AgenticEQAExecutor(agent, "Where is the clock?", router=False, collect_trace=True)
    out = ex._tool_capture_and_update()
    assert out["ok"] is True
    assert int(out["obs_id"]) == 2
    assert gm.hypothesize_nav_targets.called
    assert len(ex._hypotheses) == 1
    assert int(ex._hypotheses[0].obs_id) == 2


def test_capture_content_refresh_counts_as_new_evidence():
    """Spatial merge keeps obs_id but refreshed RGB must unlock verify for the planner."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.45
    rgb0 = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb1 = np.full((8, 8, 3), 180, dtype=np.uint8)
    xyz = np.array([1.0, 0.0, 0.5], dtype=float)
    oid = mem.add_observation(rgb0, xyz, ["clock"])

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    agent.graph_memory = mem

    def _update():
        mem.add_observation(rgb1, xyz + np.array([0.01, 0.0, 0.0]), ["clock"])

    agent.update = MagicMock(side_effect=_update)
    ex = AgenticEQAExecutor(agent, "Where is the clock?", router=False, collect_trace=True)
    out = ex._tool_capture_and_update()
    assert out["ok"] is True
    assert out.get("status") == "CONTENT_REFRESHED"
    assert int(out["obs_id"]) == int(oid)
    assert int(oid) in ex._fresh_obs_ids


def test_siglip_phrase_prefers_target_over_full_question():
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    agent.graph_memory = MagicMock()
    q = "I'm looking for the fruit bowl. A) On the kitchen island B) On the dining table. Answer:"
    ex = AgenticEQAExecutor(agent, q, router=False, collect_trace=False)
    ex._target_phrase = "fruit bowl"
    assert ex._siglip_phrase(q) == "fruit bowl"
    assert ex._siglip_phrase("") == "fruit bowl"
    assert ex._siglip_phrase("fruit bowl") == "fruit bowl"


def test_router_prompt_has_no_score_prefer_or_obs7_demo():
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.agentic_tools import (
        build_agentic_eqa_tools,
        build_graph_eqa_system_prompt,
        build_state_message,
    )
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    agent.graph_memory = MagicMock()
    agent.graph_memory.memory_summary_enabled = False
    agent.graph_memory._observations = []
    ex = AgenticEQAExecutor(agent, "Where is the sink?", router=True, collect_trace=True)
    prompt = build_graph_eqa_system_prompt(build_agentic_eqa_tools(ex))
    assert "investigate" in prompt
    assert 'navigate_to_obs", "arguments": {"obs_id": 7}' not in prompt
    assert "highest-score" not in prompt
    assert "exact semantic option text" in prompt
    assert "Pass MCQ letter" not in prompt
    assert "obs_id=3" in prompt

    ex._hypotheses = [
        NavHypothesis(
            phrase="sink",
            obs_id=3,
            xyz=np.array([1.0, 0.0, 0.0]),
            score=300.0,
            source="graph",
        ),
        NavHypothesis(
            phrase="unexplored frontier",
            obs_id=12,
            xyz=np.array([2.0, 1.0, 0.0]),
            score=0.0,
            source="frontier",
        ),
    ]
    msg = build_state_message(ex)
    assert "Investigate" in msg
    assert "Explore" in msg
    assert "investigated=0" in msg
    assert "source=graph" in msg
    assert "source=frontier" in msg
    assert "#1 best" not in msg
    assert "score=" not in msg.split("Investigate", 1)[-1]
    ex._pending_answerable = {
        "letter": "D",
        "answer_text": "Next to the refrigerator",
        "present": True,
    }
    pending_msg = build_state_message(ex)
    assert "pending_answer=Next to the refrigerator" in pending_msg
    assert "pending_answer=D" not in pending_msg


def test_visible_event_ids_do_not_match_longer_prefixes():
    from emet.memory.graph_eqa.agentic_tools import _visible_event_ids

    snapshot = MagicMock()
    snapshot.evidence = [
        MagicMock(event_id="event_1"),
        MagicMock(event_id="event_10"),
    ]
    state_text = "- event_id=event_10 step=4 positive entity:trash_can"

    assert _visible_event_ids(snapshot, state_text) == ("event_10",)


def test_hyp_recall_diversifies_graph_and_frontier():
    _require_agentic()
    from emet.memory.graph_eqa import GraphEQAMemory
    from emet.memory.graph_eqa.graph_memory import GraphNode

    gm = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={"graph_eqa_frontier_nodes": {"enabled": True}},
    )
    gm._relevant_objects = ["sink"]
    gm._relevant_phrases = ["sink"]
    oid = gm.add_observation(np.zeros((4, 4, 3), dtype=np.uint8), np.array([2.0, 0.0, 0.5]), ["sink"])
    # Synthetic frontier node (as sync_frontier_nodes would create).
    f_obs = gm._next_obs_id
    gm._next_obs_id += 1
    gm._nodes.append(
        GraphNode(
            node_id=len(gm._nodes) + 1,
            labels=["frontier"],
            xyz=np.array([-5.0, 1.0, 0.0]),
            obs_id=f_obs,
            is_frontier=True,
            description="frontier_cluster:test",
        )
    )
    from emet.memory.graph_eqa.graph_memory import GraphObservation

    gm._observations.append(
        GraphObservation(
            obs_id=f_obs,
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            xyz=np.array([-5.0, 1.0, 0.0]),
            labels=["frontier"],
            description="unexplored",
        )
    )
    hyps = gm.hypothesize_nav_targets("Where is the sink?", max_k=6)
    sources = {h.source for h in hyps}
    assert "graph" in sources
    assert "frontier" in sources
    assert any(int(h.obs_id) == int(oid) for h in hyps)
    # Frontiers must not inherit the question object as phrase (q105 failure mode).
    for h in hyps:
        if h.source == "frontier":
            assert "sink" not in str(h.phrase).lower()
            assert "frontier" in str(h.phrase).lower()


def test_investigate_records_place_inspect_on_card():
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.agentic_tools import build_state_message
    from emet.memory.graph_eqa.graph_memory import NavHypothesis, VerifyResult

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    gm = MagicMock()
    agent.graph_memory = gm
    gm.memory_summary_enabled = False
    gm._observations = [MagicMock(obs_id=15, labels=["kitchen island"])]
    gm._navigation_waypoint_for_obs = MagicMock(return_value=np.array([-16.8, -1.0, 1.0]))
    gm.record_nav_attempt = MagicMock()
    gm.verify_phrase_at_obs = MagicMock(
        return_value=VerifyResult(status="ABSENT", sim=0.05, obs_id=15, phrase="fruit bowl", ok=False)
    )
    gm._observation_by_id = MagicMock(return_value=gm._observations[0])
    gm._obs_is_frontier = MagicMock(return_value=False)
    agent.navigate_to_target_pose = MagicMock(return_value=NavOutcome.REACHED)
    agent.look_around = MagicMock()
    agent.update = MagicMock()
    agent.robot = MagicMock()
    agent.robot.get_base_pose = MagicMock(return_value=np.array([-16.8, -1.0, 0.0]))

    ex = AgenticEQAExecutor(
        agent,
        "I'm looking for the fruit bowl. A) kitchen island B) dining",
        router=True,
        collect_trace=True,
        max_nav_steps=5,
    )
    ex._target_phrase = "fruit bowl"
    ex._robot_xyt = lambda: np.array([-16.8, -1.0, 0.0])  # type: ignore[method-assign]
    ex._hypotheses = [
        NavHypothesis(
            phrase="kitchen island",
            obs_id=15,
            xyz=np.array([-16.54, -1.14, 0.7]),
            score=10.0,
            source="graph",
        )
    ]

    # Capture advances to station obs so verify runs (not STALLED).
    def _update():
        gm._observations = [MagicMock(obs_id=20, labels=["kitchen"])]

    agent.update = MagicMock(side_effect=_update)
    gm.verify_phrase_at_obs = MagicMock(
        return_value=VerifyResult(status="ABSENT", sim=0.05, obs_id=20, phrase="fruit bowl", ok=False)
    )

    out = ex.handle_tool("investigate", {"obs_id": 15})
    assert out.get("ok") is True
    assert 15 in ex._place_inspect
    assert ex._place_inspect[15].investigate_count >= 1
    assert any(r.get("event") == "station_inspect" for r in ex._trace_rows)
    # Capture refresh may clear hyps via empty hypothesize mock — restore card for state.
    ex._hypotheses = [
        NavHypothesis(
            phrase="kitchen island",
            obs_id=15,
            xyz=np.array([-16.54, -1.14, 0.7]),
            score=10.0,
            source="graph",
        )
    ]
    msg = build_state_message(ex)
    assert "investigated=1" in msg
    assert "recent:" in msg
    assert "Recent actions:" in msg
    assert any("investigate" in a and "obs=15" in a for a in ex._recent_actions)
    assert any("verify=" in a for a in ex._recent_actions)


def test_state_message_includes_recent_action_history():
    """Router state surfaces the last few investigate/explore outcomes (anti-loop)."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import RECENT_ACTIONS_K, AgenticEQAExecutor
    from emet.memory.graph_eqa.agentic_tools import build_state_message

    agent = MagicMock()
    agent.parameters = {}
    ex = AgenticEQAExecutor(agent, "Where is the mug?", max_rounds=4)
    ex._round = 2
    ex._record_recent_action(
        "investigate",
        {"obs_id": 3},
        {
            "ok": True,
            "obs_id": 3,
            "approach_index": 1,
            "verify": {"status": "ABSENT"},
            "place_inspect": "investigated=1 closest=0.4m approaches=1/4 coverage=open",
        },
    )
    ex._record_recent_action("explore_frontier", {"toward": "kitchen"}, {"ok": True})
    # Internal tools must not pollute the ring buffer.
    ex._record_recent_action("verify_siglip", {}, {"ok": True, "status": "ABSENT"})
    ex._record_recent_action("inspect_graph", {}, {"ok": True})

    assert len(ex._recent_actions) == 2
    assert "r2 investigate obs=3 ap=1 verify=ABSENT closest=0.4m" in ex._recent_actions[0]
    assert "explore_frontier toward='kitchen' ok" in ex._recent_actions[1]

    msg = build_state_message(ex)
    assert "Recent actions:" in msg
    assert "investigate obs=3" in msg
    assert "explore_frontier" in msg

    for _i in range(RECENT_ACTIONS_K + 3):
        ex._record_recent_action("explore_frontier", {}, {"ok": True})
    assert len(ex._recent_actions) == RECENT_ACTIONS_K


def test_station_obs_excluded_from_investigate_cards():
    """Capture stations must not become the next investigate target (patio chase)."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.parameters = {}
    ex = AgenticEQAExecutor(agent, "Where is the fruit bowl?", router=True)
    place = NavHypothesis(
        phrase="kitchen island",
        obs_id=38,
        xyz=np.array([-16.5, -1.1, 0.7]),
        score=1.0,
        source="graph",
    )
    station = NavHypothesis(
        phrase="dining table",
        obs_id=55,
        xyz=np.array([-15.7, -2.0, 0.5]),
        score=1.0,
        source="graph",
    )
    frontier = NavHypothesis(
        phrase="unexplored frontier",
        obs_id=49,
        xyz=np.array([-17.0, 0.5, 0.0]),
        score=0.2,
        source="frontier",
    )
    ex._station_obs_ids.add(55)
    ex._set_hypotheses([place, station, frontier])
    assert {int(h.obs_id) for h in ex._hypotheses} == {38, 49}
    assert all(int(h.obs_id) != 55 for h in ex._investigate_hypotheses())

    blocked = ex.handle_tool("investigate", {"obs_id": 55})
    assert blocked.get("ok") is False
    assert blocked.get("status") == "STATION_OBS_NOT_PLACE"


def test_prefer_explore_after_close_absent():
    """Close + VLM present=false sets prefer_explore; SigLIP ABSENT alone does not."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.agentic_tools import build_state_message
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.parameters = {}
    gm = MagicMock()
    agent.graph_memory = gm
    gm._nodes = [MagicMock(is_frontier=True, obs_id=99)]
    ex = AgenticEQAExecutor(agent, "Where is the fruit bowl?", router=False, max_nav_steps=4)
    ex._hypotheses = [
        NavHypothesis(
            phrase="kitchen island",
            obs_id=38,
            xyz=np.array([-16.5, -1.1, 0.7]),
            score=1.0,
            source="graph",
        ),
        NavHypothesis(
            phrase="unexplored frontier",
            obs_id=99,
            xyz=np.array([-15.0, 0.0, 0.0]),
            score=0.2,
            source="frontier",
        ),
    ]
    # SigLIP ABSENT alone must not set prefer_explore.
    ex._record_place_inspect(
        38,
        closest_m=0.4,
        verify_out={"status": "ABSENT", "phrase": "fruit bowl"},
        approach_index=0,
    )
    assert ex._prefer_explore is False
    # VLM assess present=false at a close look does.
    ex._record_place_inspect(
        38,
        closest_m=0.4,
        verify_out={
            "status": "ABSENT",
            "phrase": "fruit bowl",
            "present": False,
            "answerable": False,
        },
        approach_index=1,
    )
    assert ex._prefer_explore is True
    assert ex._prefer_explore_reason == "absent"
    msg = build_state_message(ex)
    assert "Prefer explore_frontier once" in msg
    assert "do not explore forever" in msg
    tool, _args = ex._fallback_tool()
    assert tool == "explore_frontier"
    # After one explore in the streak, fallback should close-look remaining place cards.
    ex._n_consecutive_explore = 1
    tool2, args2 = ex._fallback_tool()
    assert tool2 == "investigate"
    assert int(args2.get("obs_id")) == 38
    # Direct clear path used by explore tool:
    ex._prefer_explore = False
    assert ex._prefer_explore is False


def test_explore_streak_forces_investigate_over_frontier():
    """Two successful explores in a row → rewrite explore pick to investigate."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import (
        EXPLORE_STREAK_FORCE_INVESTIGATE,
        AgenticEQAExecutor,
    )
    from emet.memory.graph_eqa.agentic_tools import build_state_message
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.parameters = {}
    gm = MagicMock()
    agent.graph_memory = gm
    gm._nodes = [MagicMock(is_frontier=True, obs_id=99)]
    gm.memory_summary_enabled = False
    gm.eqa_client = MagicMock(
        return_value=(
            '{"current_room": "outdoor", "in_target_area": false, '
            '"tool_calls": [{"name": "explore_frontier", "arguments": {}}], "message": ""}'
        )
    )
    ex = AgenticEQAExecutor(
        agent,
        "Where is the silver trash can?",
        max_rounds=4,
        max_nav_steps=8,
        collect_trace=True,
    )
    ex.room_policy = "llm"
    ex._in_target_area = False
    ex._last_room_estimate = "outdoor"
    ex._n_consecutive_explore = EXPLORE_STREAK_FORCE_INVESTIGATE
    ex._hypotheses = [
        NavHypothesis(
            phrase="trash can",
            obs_id=7,
            xyz=np.array([1.0, 2.0, 0.5]),
            score=1.0,
            source="graph",
        ),
        NavHypothesis(
            phrase="unexplored frontier",
            obs_id=99,
            xyz=np.array([-15.0, 0.0, 0.0]),
            score=0.2,
            source="frontier",
        ),
    ]
    msg = build_state_message(ex)
    assert "close looks are allowed while leaving" in msg
    tool, args = ex._fallback_tool()
    assert tool == "investigate"
    assert int(args["obs_id"]) == 7
    # Soft prefer_explore + streak>=1 also investigates.
    ex._prefer_explore = True
    ex._prefer_explore_reason = "absent"
    ex._n_consecutive_explore = 1
    tool2, args2 = ex._fallback_tool()
    assert tool2 == "investigate"
    assert int(args2["obs_id"]) == 7


def test_stamp_room_disabled_by_default():
    """Investigate room stamps are off unless EMET_EQA_ROOM_STAMP_INVESTIGATE=1."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.parameters = {}
    gm = MagicMock()
    agent.graph_memory = gm
    gm.memory_summary_enabled = False
    gm._observations = [MagicMock(obs_id=3, labels=["toilet", "bath mat"])]
    gm.stamp_vlm_room_at_robot = MagicMock(return_value="bathroom")

    ex = AgenticEQAExecutor(
        agent,
        "Which rug is at the shower in the bathroom?",
        max_rounds=4,
        max_nav_steps=8,
        collect_trace=True,
    )
    assert ex._room_stamp_investigate is False
    hyp = NavHypothesis(
        phrase="toilet",
        obs_id=3,
        xyz=np.array([1.0, 2.0, 0.5]),
        score=1.0,
        source="graph",
    )
    out = ex._stamp_room_after_investigate(3, hyp=hyp, station_oid=None)
    assert out.get("ok") is False
    assert out.get("reason") == "disabled"
    gm.stamp_vlm_room_at_robot.assert_not_called()


def test_stamp_room_after_investigate_updates_graph_and_estimate():
    """Close look stamps nearest cluster from place labels (not stuck on outdoor)."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import NavHypothesis
    from emet.memory.graph_eqa.room_clusters import RoomCluster

    agent = MagicMock()
    agent.parameters = {}
    gm = MagicMock()
    agent.graph_memory = gm
    gm.memory_summary_enabled = False
    gm._room_clusters = [
        RoomCluster(
            cluster_id=1,
            node_ids=(1,),
            labels=("toilet", "sink"),
            centroid_xy=(1.0, 2.0),
            room_name="outdoor",
        )
    ]
    gm.last_room_clusters = list(gm._room_clusters)
    # Local evidence only: obs labels (not hyp.phrase / labels_near_obs).
    gm._observations = [MagicMock(obs_id=3, labels=["toilet", "bath mat"])]

    def _stamp(
        xy,
        room,
        protect_indoor_from_outdoor=True,
        corroborating_labels=None,
        source="router_vlm",
        source_view_id=None,
        agent_round=None,
        pose_round=None,
    ):
        from emet.memory.graph_eqa.room_clusters import stamp_room_at_xy

        gm._room_clusters = stamp_room_at_xy(
            gm._room_clusters,
            xy,
            room,
            protect_indoor_from_outdoor=protect_indoor_from_outdoor,
            corroborating_labels=corroborating_labels,
        )
        gm.last_room_clusters = list(gm._room_clusters)
        return room

    def _room_at(xy):
        from emet.memory.graph_eqa.room_clusters import estimate_room_at_xy

        return estimate_room_at_xy(gm._room_clusters, xy)

    gm.stamp_vlm_room_at_robot.side_effect = _stamp
    gm.graph_room_at_robot.side_effect = _room_at

    ex = AgenticEQAExecutor(
        agent,
        "Which rug is at the shower in the bathroom?",
        max_rounds=4,
        max_nav_steps=8,
        collect_trace=True,
    )
    ex.room_policy = "llm"
    ex._room_stamp_investigate = True
    ex._last_room_estimate = "outdoor"
    hyp = NavHypothesis(
        phrase="rug shower bathroom",  # question-shaped; must not drive the stamp alone
        obs_id=3,
        xyz=np.array([1.0, 2.0, 0.5]),
        score=1.0,
        source="graph",
    )
    out = ex._stamp_room_after_investigate(3, hyp=hyp, station_oid=10)
    assert out.get("ok") is True
    assert out.get("proposed") == "bathroom"
    assert out.get("label_source") == "obs_and_hyp_labels"
    assert ex._last_room_estimate == "bathroom"
    assert gm._room_clusters[0].room_name == "bathroom"
    assert any(r.get("event") == "room_stamp_investigate" for r in ex._trace_rows)


def test_stamp_room_ignores_question_phrase_and_station_leakage():
    """hyp.phrase / station kitchen labels must not force a kitchen stamp."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import NavHypothesis
    from emet.memory.graph_eqa.room_clusters import RoomCluster

    agent = MagicMock()
    agent.parameters = {}
    gm = MagicMock()
    agent.graph_memory = gm
    gm.memory_summary_enabled = False
    gm._room_clusters = [
        RoomCluster(
            cluster_id=1,
            node_ids=(1,),
            labels=("bouquet", "picture"),
            centroid_xy=(1.0, 2.0),
            room_name="unknown",
        )
    ]
    gm.last_room_clusters = list(gm._room_clusters)
    gm._observations = [
        MagicMock(obs_id=3, labels=["bouquet", "picture"]),
        MagicMock(obs_id=10, labels=["refrigerator", "kitchen cabinet"]),
    ]
    gm.stamp_vlm_room_at_robot = MagicMock(return_value="kitchen")
    gm.graph_room_at_robot = MagicMock(return_value="unknown")

    ex = AgenticEQAExecutor(
        agent,
        "Where is the wall clock? A) kitchen B) living room",
        max_rounds=4,
        max_nav_steps=8,
        collect_trace=True,
    )
    ex.room_policy = "llm"
    ex._room_stamp_investigate = True
    ex._last_room_estimate = "kitchen"
    hyp = NavHypothesis(
        phrase="wall clock kitchen",
        obs_id=3,
        xyz=np.array([1.0, 2.0, 0.5]),
        score=1.0,
        source="graph",
    )
    out = ex._stamp_room_after_investigate(3, hyp=hyp, station_oid=10)
    assert out.get("ok") is False
    assert out.get("reason") == "no_room"
    assert ex._last_room_estimate == "kitchen"
    gm.stamp_vlm_room_at_robot.assert_not_called()


def test_stamp_room_ignores_bathroom_phrase_without_local_evidence():
    """``rug shower bathroom`` phrase alone must not stamp bathroom over living-room obs."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.parameters = {}
    gm = MagicMock()
    agent.graph_memory = gm
    gm.memory_summary_enabled = False
    gm._observations = [
        MagicMock(obs_id=3, labels=["console table", "lamp", "painting"]),
    ]
    gm.stamp_vlm_room_at_robot = MagicMock(return_value="bathroom")
    gm.graph_room_at_robot = MagicMock(return_value="unknown")

    ex = AgenticEQAExecutor(
        agent,
        "Which rug is at the shower in the bathroom?",
        max_rounds=4,
        max_nav_steps=8,
        collect_trace=True,
    )
    ex.room_policy = "llm"
    ex._room_stamp_investigate = True
    ex._last_room_estimate = "unknown"
    hyp = NavHypothesis(
        phrase="rug shower bathroom",
        obs_id=3,
        xyz=np.array([1.0, 2.0, 0.5]),
        score=1.0,
        source="graph",
    )
    out = ex._stamp_room_after_investigate(3, hyp=hyp, station_oid=None)
    assert out.get("ok") is False
    assert out.get("reason") == "no_room"
    gm.stamp_vlm_room_at_robot.assert_not_called()


def test_stamp_room_skips_when_sticky_estimate_only():
    """Empty local labels + kitchen estimate → skip stamp (no sticky fallback)."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.parameters = {}
    gm = MagicMock()
    agent.graph_memory = gm
    gm.memory_summary_enabled = False
    gm._observations = [MagicMock(obs_id=3, labels=["toddler chair", "door"])]
    gm.stamp_vlm_room_at_robot = MagicMock(return_value="kitchen")

    ex = AgenticEQAExecutor(
        agent,
        "Where is the toddler chair?",
        max_rounds=4,
        max_nav_steps=8,
        collect_trace=True,
    )
    ex.room_policy = "llm"
    ex._room_stamp_investigate = True
    ex._last_room_estimate = "kitchen"
    hyp = NavHypothesis(
        phrase="toddler chair child",
        obs_id=3,
        xyz=np.array([1.0, 2.0, 0.5]),
        score=1.0,
        source="graph",
    )
    out = ex._stamp_room_after_investigate(3, hyp=hyp, station_oid=None)
    assert out.get("ok") is False
    assert out.get("reason") == "no_room"
    assert ex._last_room_estimate == "kitchen"
    gm.stamp_vlm_room_at_robot.assert_not_called()


def test_normalize_current_room_aliases():
    from emet.memory.graph_eqa.agentic_tools import (
        normalize_current_room,
        question_implies_indoor,
        room_is_outdoor,
    )

    assert normalize_current_room(None) == "unknown"
    assert normalize_current_room("") == "unknown"
    assert normalize_current_room("Kitchen") == "kitchen"
    assert normalize_current_room("living room") == "living_room"
    assert normalize_current_room("brick patio") == "patio"
    assert normalize_current_room("outdoors") == "outdoor"
    assert normalize_current_room("back yard") == "outdoor"
    assert room_is_outdoor("patio")
    assert room_is_outdoor("deck")
    assert not room_is_outdoor("kitchen")
    assert question_implies_indoor("Where is the wall clock?")
    assert question_implies_indoor("Where is the fruit bowl?")
    assert not question_implies_indoor("What color is the sky?")


_CLOCK_LOCATION_Q = "\n".join(
    [
        "Where is the wall clock?",
        "A) dining area",
        "B) kitchen",
        "C) sunroom",
        "D) living area near the fireplace",
    ]
)


def test_agentic_skips_final_location_salvage():
    """Scored path never applies salvage; counterfactual may still fire with images."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    gm = agent.graph_memory
    gm._salvage_location_mcq_letter = MagicMock(return_value="A")
    gm.last_eqa_images = None
    gm.last_relevant_images = None
    img = Image.new("RGB", (8, 8), color=(1, 2, 3))
    ex = AgenticEQAExecutor(agent, _CLOCK_LOCATION_Q, max_rounds=2, max_nav_steps=4, collect_trace=True)
    out = ex._finalize_unknown_location_letter({"answer": "Unknown", "confidence": False, "relevant_images": [img]})
    assert out.get("answer") == "Unknown"
    assert gm._salvage_location_mcq_letter.call_count == 1
    assert ex._salvage_counterfactual_letter == "A"
    assert any(r.get("event") == "final_location_salvage_skipped" for r in ex._trace_rows)
    assert any(r.get("event") == "final_location_salvage_counterfactual" for r in ex._trace_rows)


def test_router_room_mismatch_is_diagnostic_only():
    """Wrong room vs MCQ targets is traced but does not set prefer_explore."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.agentic_tools import build_state_message
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.parameters = {}
    gm = agent.graph_memory
    gm.memory_summary_enabled = False
    gm._nodes = [MagicMock(is_frontier=True, obs_id=99)]
    gm.graph_room_at_robot = MagicMock(return_value="unknown")
    gm.format_rooms_line = MagicMock(return_value="Rooms: patio(3), kitchen(8)")
    reply = (
        '{"current_room": "brick patio", "tool_calls": [{"name": "explore_frontier", "arguments": {}}], "message": ""}'
    )
    gm.eqa_client = MagicMock(return_value=reply)

    ex = AgenticEQAExecutor(
        agent,
        _CLOCK_LOCATION_Q,
        max_rounds=3,
        max_nav_steps=4,
        collect_trace=True,
    )
    ex._hypotheses = [
        NavHypothesis(
            phrase="unexplored frontier",
            obs_id=99,
            xyz=np.array([-15.0, 0.0, 0.0]),
            score=0.2,
            source="frontier",
        ),
    ]
    calls, picked_by, meta = ex._route_tool_calls()
    assert picked_by == "vlm"
    assert calls == [("explore_frontier", {})]
    assert meta.get("current_room") == "patio"
    assert meta.get("prefer_explore_room_mismatch") is None
    assert meta.get("room_mismatch_diagnostic") is True
    assert meta.get("rooms_line") == "Rooms: patio(3), kitchen(8)"
    assert "kitchen" in meta.get("question_target_rooms", [])
    assert "living_room" in meta.get("question_target_rooms", [])
    assert "dining_room" in meta.get("question_target_rooms", [])
    assert ex._last_room_estimate == "patio"
    assert ex._prefer_explore is False
    assert ex._prefer_explore_reason == ""
    room_rows = [r for r in ex._trace_rows if r.get("event") == "router_room"]
    assert len(room_rows) == 1
    assert room_rows[0].get("rooms_line") == "Rooms: patio(3), kitchen(8)"
    assert room_rows[0].get("question_target_rooms") == meta.get("question_target_rooms")
    msg = build_state_message(ex)
    assert "Current room (router): patio" in msg
    assert "does not match rooms named" not in msg
    tool, _args = ex._fallback_tool()
    assert tool == "explore_frontier"


def test_room_mismatch_does_not_redirect_investigate():
    """Graph patio + location MCQ: VLM investigate is NOT forced to explore."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.parameters = {}
    gm = agent.graph_memory
    gm.memory_summary_enabled = False
    gm._nodes = [MagicMock(is_frontier=True, obs_id=99)]
    gm.get_nodes = MagicMock(return_value=gm._nodes)
    gm.graph_room_at_robot = MagicMock(return_value="patio")
    gm.format_rooms_line = MagicMock(return_value="Rooms: patio(1)")
    reply = (
        '{"current_room": "unknown", "tool_calls": '
        '[{"name": "investigate", "arguments": {"obs_id": 7}}], "message": ""}'
    )
    gm.eqa_client = MagicMock(return_value=reply)
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])

    ex = AgenticEQAExecutor(
        agent,
        _CLOCK_LOCATION_Q,
        max_rounds=2,
        max_nav_steps=4,
        collect_trace=True,
    )
    ex._hypotheses = [
        NavHypothesis(
            phrase="patio chair",
            obs_id=7,
            xyz=np.array([0.0, 0.0, 0.5]),
            score=1.0,
            source="graph",
        ),
        NavHypothesis(
            phrase="unexplored frontier",
            obs_id=99,
            xyz=np.array([5.0, 0.0, 0.0]),
            score=0.2,
            source="frontier",
        ),
    ]
    calls, picked_by, meta = ex._route_tool_calls()
    assert meta.get("current_room_graph") == "patio"
    assert meta.get("room_mismatch_diagnostic") is True
    assert ex._prefer_explore_reason != "room_mismatch"
    assert calls[0][0] == "investigate"
    assert picked_by == "vlm"


def test_room_mismatch_diagnostic_when_room_matches():
    """Matched question room clears room_mismatch_diagnostic."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {}
    gm = agent.graph_memory
    gm.memory_summary_enabled = False
    gm._nodes = [MagicMock(is_frontier=True, obs_id=99)]
    gm.graph_room_at_robot = MagicMock(return_value="kitchen")
    gm.format_rooms_line = MagicMock(return_value="Rooms: kitchen(2)")
    reply = '{"current_room": "kitchen", "tool_calls": [{"name": "explore_frontier", "arguments": {}}], "message": ""}'
    gm.eqa_client = MagicMock(return_value=reply)
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])

    ex = AgenticEQAExecutor(agent, _CLOCK_LOCATION_Q, max_rounds=2, max_nav_steps=4)
    ex._prefer_explore = True
    ex._prefer_explore_reason = "absent"
    _calls, _picked, meta = ex._route_tool_calls()
    assert meta.get("current_room") == "kitchen"
    assert meta.get("room_mismatch_diagnostic") is False
    assert meta.get("prefer_explore_room_mismatch") is None
    assert ex._prefer_explore_reason == "absent"


def test_frontier_nearby_labels_tolerates_numpy_xyz():
    """Graph node xyz is ndarray — must not use ``a or b`` boolean (probe crash)."""
    _require_agentic()
    from types import SimpleNamespace

    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.agentic_tools import _frontier_nearby_labels, build_state_message
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.parameters = {}
    gm = MagicMock()
    agent.graph_memory = gm
    gm.graph_room_at_robot = MagicMock(return_value="kitchen")
    gm._nodes = [
        SimpleNamespace(
            is_frontier=False,
            labels=["stove", "fridge"],
            xyz=np.array([0.2, 0.1, 0.5]),
            centroid=None,
        ),
        SimpleNamespace(
            is_frontier=True,
            labels=["frontier"],
            xyz=np.array([1.0, 0.0, 0.0]),
            centroid=None,
        ),
    ]
    ex = AgenticEQAExecutor(agent, "Where is the stove?", router=False, max_nav_steps=4)
    frontier = NavHypothesis(
        phrase="unexplored frontier",
        obs_id=99,
        xyz=np.array([0.0, 0.0, 0.0]),
        score=0.2,
        source="frontier",
    )
    ex._hypotheses = [frontier]
    near = _frontier_nearby_labels(ex, frontier)
    assert "stove" in near or "fridge" in near
    msg = build_state_message(ex)
    assert "Explore" in msg
    assert "near=" in msg


def test_navigate_rejects_obs_not_in_evidence():
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.navigate_to_target_pose = MagicMock(return_value=NavOutcome.REACHED)
    agent.graph_memory._navigation_waypoint_for_obs = MagicMock(return_value=np.array([1.0, 2.0, 0.0]))
    ex = AgenticEQAExecutor(agent, question="Where is the sink?", max_rounds=4, router=False)
    ex._hypotheses = [
        NavHypothesis(
            phrase="kitchen island",
            obs_id=13,
            xyz=np.array([-16.5, -1.1, 0.7]),
            score=10.0,
            source="graph",
        )
    ]
    out = ex._tool_navigate_to_obs(17)
    assert out["ok"] is False
    assert out.get("status") == "OBS_NOT_IN_EVIDENCE"
    assert agent.navigate_to_target_pose.call_count == 0


def test_handle_tool_rejects_place_id_without_crashing():
    """Router place keys are not numeric evidence-card ids."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.graph_memory = MagicMock()
    ex = AgenticEQAExecutor(agent, question="Where is the sink?", max_rounds=4, router=False)
    ex._hypotheses = [
        NavHypothesis(
            phrase="kitchen sink",
            obs_id=13,
            xyz=np.array([1.0, 2.0, 0.0]),
            score=1.0,
            source="graph",
        )
    ]

    out = ex.handle_tool("investigate", {"obs_id": "place_ca28f3f9493f"})

    assert out["ok"] is False
    assert out["status"] == "OBS_NOT_IN_EVIDENCE"
    assert out["obs_id"] == "place_ca28f3f9493f"
    assert out["listed_obs_ids"] == [13]


def test_grounded_router_redirects_invalid_obs_to_listed_hypothesis(monkeypatch):
    """A stale router id must not consume every grounded-v2 round unchanged."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    monkeypatch.setenv("EMET_EQA_AGENTIC_DECISION_POLICY", "grounded_v2")
    agent = MagicMock()
    agent.parameters = {}
    agent.graph_memory = MagicMock()
    ex = AgenticEQAExecutor(
        agent,
        question="Where is the trash can?",
        max_rounds=8,
        max_nav_steps=8,
        router=True,
        collect_trace=True,
    )
    ex._hypotheses = [
        NavHypothesis(
            phrase="refrigerator",
            obs_id=13,
            xyz=np.array([1.0, 2.0, 0.5]),
            score=1.0,
            source="graph",
        )
    ]
    rejected = {
        "ok": False,
        "status": "OBS_NOT_IN_EVIDENCE",
        "obs_id": 20,
        "listed_obs_ids": [13],
    }

    with patch.object(ex, "handle_tool", return_value={"ok": True}) as handle:
        assert ex._recover_failed_router_motion(tool="investigate", out=rejected) is True

    handle.assert_called_once_with("investigate", {"obs_id": 13})
    redirect = next(row for row in ex._trace_rows if row.get("event") == "nav_loop_redirect")
    assert redirect["from_obs_id"] == 20
    assert redirect["status"] == "OBS_NOT_IN_EVIDENCE"
    assert redirect["to_obs_id"] == 13


def test_grounded_router_redirects_blocked_place_to_frontier(monkeypatch):
    """A grounded router must not spend every round on one exhausted place."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    monkeypatch.setenv("EMET_EQA_AGENTIC_DECISION_POLICY", "grounded_v2")
    agent = MagicMock()
    agent.parameters = {}
    agent.graph_memory = MagicMock()
    ex = AgenticEQAExecutor(
        agent,
        question="Where is the trash can?",
        max_rounds=8,
        max_nav_steps=8,
        router=True,
        collect_trace=True,
    )
    rejected = {
        "ok": False,
        "status": "NAV_LOOP_BLOCKED",
        "obs_id": 1,
    }

    with patch.object(ex, "handle_tool", return_value={"ok": True}) as handle:
        assert ex._recover_failed_router_motion(tool="investigate", out=rejected) is True

    handle.assert_called_once_with("explore_frontier", {"toward": ex.query_text})
    redirect = next(row for row in ex._trace_rows if row.get("event") == "nav_loop_redirect")
    assert redirect["from_obs_id"] == 1
    assert redirect["status"] == "NAV_LOOP_BLOCKED"
    assert redirect["to"] == "explore_frontier"


def test_visited_frontier_retired_from_graph():
    _require_agentic()
    from emet.memory.graph_eqa import GraphEQAMemory
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.agentic_tools import build_state_message
    from emet.memory.graph_eqa.graph_memory import GraphNode, GraphObservation, NavHypothesis

    gm = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={"graph_eqa_frontier_nodes": {"enabled": True}},
    )
    f_obs = gm._next_obs_id
    gm._next_obs_id += 1
    gm._nodes.append(
        GraphNode(
            node_id=1,
            labels=["frontier"],
            xyz=np.array([1.0, 2.0, 0.0]),
            obs_id=f_obs,
            is_frontier=True,
            description="frontier_cluster:visit_me",
        )
    )
    gm._observations.append(
        GraphObservation(
            obs_id=f_obs,
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            xyz=np.array([1.0, 2.0, 0.0]),
            labels=["frontier"],
            description="unexplored",
        )
    )
    assert gm.retire_frontier_obs(f_obs) is True
    assert not any(n.is_frontier for n in gm.get_nodes())
    assert gm._observation_by_id(f_obs) is None

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    agent.graph_memory = gm
    agent.voxel_map = None
    ex = AgenticEQAExecutor(agent, "Where is the sink?", router=True, collect_trace=True)
    # Safety net: visited frontier hyp filtered even if somehow recalled.
    ex._nav_to_obs_counts[99] = 1
    ex._set_hypotheses(
        [
            NavHypothesis(
                phrase="sink",
                obs_id=99,
                xyz=np.array([0.0, 0.0, 0.0]),
                score=0.0,
                source="frontier",
            ),
            NavHypothesis(
                phrase="sink",
                obs_id=3,
                xyz=np.array([1.0, 0.0, 0.0]),
                score=300.0,
                source="graph",
            ),
        ]
    )
    assert all(int(h.obs_id) != 99 for h in ex._hypotheses)
    msg = build_state_message(ex)
    assert "obs_id=99" not in msg
    assert "obs_id=3" in msg


def test_consecutive_nav_failures_block_after_retry_limit():
    """Planner misses rotate approaches until the anti-thrashing limit is reached."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import NAV_CONSECUTIVE_FAIL_LIMIT, AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    gm = MagicMock()
    agent.graph_memory = gm
    # Distinct targets so we can see approach rotation.
    gm._navigation_approach_waypoint_for_obs = MagicMock(
        side_effect=lambda oid, xyt=None, approach_index=0, n_approaches=4: np.array([float(approach_index), 2.0, 1.0])
    )
    gm._navigation_waypoint_for_obs = MagicMock(return_value=np.array([1.0, 2.0, 1.0]))
    gm.record_nav_attempt = MagicMock()
    gm._obs_is_frontier = MagicMock(return_value=False)
    agent.navigate_to_target_pose = MagicMock(return_value=NavOutcome.STUCK)
    agent.update = MagicMock()
    agent.robot = None

    ex = AgenticEQAExecutor(
        agent,
        "Where is the sink? A) kitchen B) bath",
        router=True,
        collect_trace=True,
        max_nav_steps=12,
    )
    ex._target_phrase = "sink"
    ex._hypotheses = [
        NavHypothesis(
            phrase="sink",
            obs_id=7,
            xyz=np.array([1.0, 2.0, 0.5]),
            score=1.0,
            source="graph",
        )
    ]
    targets = []
    for _ in range(NAV_CONSECUTIVE_FAIL_LIMIT):
        out = ex._tool_navigate_to_obs(7)
        assert out["ok"] is False
        assert out.get("status") != "NAV_LOOP_BLOCKED"
        targets.append(out.get("approach_index"))
    assert targets == list(range(NAV_CONSECUTIVE_FAIL_LIMIT))
    assert ex._hypothesis_nav_blocked(7)
    blocked = ex._tool_navigate_to_obs(7)
    assert blocked.get("status") == "NAV_LOOP_BLOCKED"
    assert blocked.get("ok") is False
    assert agent.navigate_to_target_pose.call_count == NAV_CONSECUTIVE_FAIL_LIMIT


def test_investigate_samples_new_approach_after_close_absent():
    """Re-investigate after close+ABSENT uses the next orbit sample, not NAV_LOOP_BLOCK."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import NavHypothesis, VerifyResult

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    gm = MagicMock()
    agent.graph_memory = gm
    gm.memory_summary_enabled = False
    gm._observations = [MagicMock(obs_id=15, labels=["kitchen island"])]
    gm._navigation_approach_waypoint_for_obs = MagicMock(
        side_effect=lambda oid, xyt=None, approach_index=0, n_approaches=4, **kw: np.array(
            [-16.8 + 0.5 * float(approach_index), -1.0, 1.0]
        )
    )
    gm._navigation_waypoint_for_obs = MagicMock(return_value=np.array([-16.8, -1.0, 1.0]))
    gm.place_coverage_for_obs = MagicMock(
        return_value=type("C", (), {"status": "open", "local_frontier_cells": 5, "complete": False})()
    )
    gm.record_nav_attempt = MagicMock()
    gm._observation_by_id = MagicMock(return_value=gm._observations[0])
    gm._obs_is_frontier = MagicMock(return_value=False)
    agent.navigate_to_target_pose = MagicMock(return_value=NavOutcome.REACHED)
    agent.look_around = MagicMock()
    agent.robot = MagicMock()
    agent.robot.get_base_pose = MagicMock(return_value=np.array([-16.8, -1.0, 0.0]))
    agent.voxel_map = None
    agent.planner = None

    ex = AgenticEQAExecutor(
        agent,
        "I'm looking for the fruit bowl. A) kitchen island B) dining",
        router=True,
        collect_trace=True,
        max_nav_steps=8,
    )
    ex._target_phrase = "fruit bowl"
    ex._robot_xyt = lambda: np.array([-16.8, -1.0, 0.0])  # type: ignore[method-assign]
    ex._hypotheses = [
        NavHypothesis(
            phrase="kitchen island",
            obs_id=15,
            xyz=np.array([-16.54, -1.14, 0.7]),
            score=10.0,
            source="graph",
        )
    ]
    n_cap = {"i": 20}

    def _update():
        n_cap["i"] += 1
        gm._observations = [MagicMock(obs_id=n_cap["i"], labels=["kitchen"])]

    agent.update = MagicMock(side_effect=_update)
    gm.verify_phrase_at_obs = MagicMock(
        return_value=VerifyResult(status="ABSENT", sim=0.05, obs_id=21, phrase="fruit bowl", ok=False)
    )

    out1 = ex.handle_tool("investigate", {"obs_id": 15})
    assert out1.get("ok") is True
    assert out1.get("approach_index") == 0
    assert "more_views" in str(out1.get("place_inspect") or "")
    assert "coverage=open" in str(out1.get("place_inspect") or "")
    assert not ex._hypothesis_nav_blocked(15)

    out2 = ex.handle_tool("investigate", {"obs_id": 15})
    assert out2.get("ok") is True
    assert out2.get("approach_index") == 1
    t1 = out1["target_xyz"][:2]
    t2 = out2["target_xyz"][:2]
    assert t1 != t2
    assert agent.navigate_to_target_pose.call_count == 2


def test_retraction_correlates_distinct_station_views_not_place_ids():
    """Two fresh ABSENT views of one place enable cross-view label stripping."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    graph = GraphEQAMemory(defer_llm_clients=True)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    graph.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["fruit bowl"])
    graph.add_observation(rgb, np.array([1.0, 0.0, 0.5]), ["fruit bowl"])
    agent.graph_memory = graph
    executor = AgenticEQAExecutor(agent, "Where is the fruit bowl?", router=False, collect_trace=True)

    first = executor._maybe_retract_claim_after_station(
        1,
        closest_m=0.5,
        verify_out={"status": "ABSENT", "phrase": "fruit bowl", "obs_id": 101},
    )
    assert first is not None
    assert first["evidence_obs_id"] == 101
    assert first["stripped_obs"] == 1
    assert "fruit bowl" in str(graph.get_nodes()[1].labels).lower()
    assert executor._trace_rows[-1]["strip_across_obs"] is False

    second = executor._maybe_retract_claim_after_station(
        1,
        closest_m=0.5,
        verify_out={"status": "ABSENT", "phrase": "fruit bowl", "obs_id": 102},
    )
    assert second is not None
    assert second["evidence_obs_id"] == 102
    assert second["stripped_obs"] == 1
    assert "fruit bowl" not in str(graph.get_nodes()[1].labels).lower()
    assert executor._trace_rows[-1]["strip_across_obs"] is True


def test_coverage_closed_does_not_exhaust_while_approaches_remain():
    """Coverage is informational; only the approach budget gates re-investigate."""
    _require_agentic()
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor, PlaceInspectRecord

    agent = MagicMock()
    agent.parameters = {"eqa": {}}
    agent.graph_memory = MagicMock()
    agent.voxel_map = None
    ex = AgenticEQAExecutor(agent, "Where is the sink?", router=True)
    rec = PlaceInspectRecord(
        investigate_count=1,
        closest_m=0.4,
        coverage="closed",
        local_frontier_cells=0,
        tried_approaches=[0],
    )
    ex._place_inspect[3] = rec
    assert ex._place_approaches_exhausted(3) is False
    assert "coverage=closed" in rec.card_bits()
    assert "more_views" in rec.card_bits()
    rec.tried_approaches = [0, 1, 2, 3]
    assert ex._place_approaches_exhausted(3) is True
    assert "views_exhausted" in rec.card_bits()


def test_hypothesize_frontiers_without_object_phrases():
    """Cold start / failed extract still returns frontier evidence cards."""
    _require_agentic()
    from emet.memory.graph_eqa import GraphEQAMemory
    from emet.memory.graph_eqa.graph_memory import GraphNode, GraphObservation

    gm = GraphEQAMemory(
        defer_llm_clients=True,
        parameters={"graph_eqa_frontier_nodes": {"enabled": True}},
    )
    gm._relevant_objects = []
    gm._relevant_phrases = []
    f_obs = gm._next_obs_id
    gm._next_obs_id += 1
    gm._nodes.append(
        GraphNode(
            node_id=1,
            labels=["frontier"],
            xyz=np.array([3.0, 1.0, 0.0]),
            obs_id=f_obs,
            is_frontier=True,
            description="frontier_cluster:cold",
        )
    )
    gm._observations.append(
        GraphObservation(
            obs_id=f_obs,
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            xyz=np.array([3.0, 1.0, 0.0]),
            labels=["frontier"],
            description="unexplored",
        )
    )
    # Skip extract_relevant_objects so phrases stay empty.
    gm.extract_relevant_objects = lambda *_a, **_k: None  # type: ignore[method-assign]
    hyps = gm.hypothesize_nav_targets("???", max_k=4)
    assert hyps
    assert all(h.source == "frontier" for h in hyps)
    assert all("frontier" in h.phrase.lower() for h in hyps)

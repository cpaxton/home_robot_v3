# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Agentic EQA loop / SigLIP verify claims (A*, S1)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from emet.controller.habitat_nav import NavOutcome

_EVAL_DIR = str(Path(__file__).resolve().parent)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
from _agentic import (  # noqa: E402
    _require_agentic,
    _require_vram_split,
)

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
    ex, agent, order = _follow_action_executor("How many table lamps are there? A) One B) Two C) Three D) None")
    followed = ex._maybe_follow_eqa_explore_action({"ok": True, "answer": "One", "confidence": False})
    assert followed is True
    assert "nav" in order
    assert 11 in ex._followed_eqa_actions
    assert agent.graph_memory.last_eqa_look_obs_id == 11


def test_follow_eqa_action_after_confident_none_with_unattached_find():
    """q93: confident None while stool FIND RGB was never attached must still look."""
    _require_agentic()
    from types import SimpleNamespace

    ex, agent, order = _follow_action_executor(
        "How many stools are at the kitchen counter? A) One B) Two C) Three D) None"
    )
    agent.graph_memory.last_eqa_action_obs_id = None
    agent.graph_memory.last_eqa_look_obs_id = None
    agent.graph_memory.last_eqa_obs_ids = [1, 2]
    agent.graph_memory._obs_usable_for_eqa_image = MagicMock(return_value=True)
    agent.graph_memory._count_candidate_nodes = MagicMock(
        return_value=([SimpleNamespace(obs_id=168), SimpleNamespace(obs_id=199)], None)
    )
    followed = ex._maybe_follow_eqa_explore_action({"ok": True, "answer": "None", "confidence": True})
    assert followed is True
    assert "nav" in order
    assert 168 in ex._followed_eqa_actions
    assert agent.graph_memory.last_eqa_look_obs_id == 168


def test_follow_eqa_action_skips_unconfident_location_letter():
    """A location letter is a guess we can score; do not nav just because conf is false."""
    _require_agentic()
    ex, agent, order = _follow_action_executor(
        "Where is the clock? A) Above the sink B) On the wall C) In the hallway D) Unknown"
    )
    followed = ex._maybe_follow_eqa_explore_action({"ok": True, "answer": "Above the sink", "confidence": False})
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

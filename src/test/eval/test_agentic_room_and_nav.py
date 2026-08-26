# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Agentic EQA room, frontier, investigate, and nav-reject tests."""

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
)


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
    assert any("action=investigate" in a and "adapter=15" in a for a in ex._recent_actions)
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
    # Explicit verify is useful history; only internal graph/capture plumbing is omitted.
    ex._record_recent_action("verify_siglip", {}, {"ok": True, "status": "ABSENT"})
    ex._record_recent_action("inspect_graph", {}, {"ok": True})

    assert len(ex._recent_actions) == 3
    assert "round=3 action=investigate" in ex._recent_actions[0]
    assert "approach=1" in ex._recent_actions[0]
    assert "verify=ABSENT" in ex._recent_actions[0]
    assert "adapter=3" in ex._recent_actions[0]
    assert 'action=explore_frontier intent="kitchen"' in ex._recent_actions[1]
    assert "action=verify_siglip" in ex._recent_actions[2]

    msg = build_state_message(ex)
    assert "Recent actions:" in msg
    assert "action=investigate" in msg
    assert "action=explore_frontier" in msg

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

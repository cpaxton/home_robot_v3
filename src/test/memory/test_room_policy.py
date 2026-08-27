# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for room_policy ablation (canonical vs llm)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from emet.controller.controller_graph_eqa import GraphEQAController
from emet.memory.graph_eqa.agentic_tools import (
    build_graph_eqa_system_prompt,
    coerce_room_label,
    sanitize_room_phrase,
)
from emet.memory.graph_eqa.room_clusters import (
    merge_room_estimates,
    resolve_room_policy,
    room_leave_needed,
)


def test_sanitize_preserves_weird_phrases():
    assert sanitize_room_phrase("Master Bathroom") == "master bathroom"
    assert sanitize_room_phrase("open kitchen-living") == "open kitchen living"
    assert sanitize_room_phrase("living_room") == "living_room"
    assert sanitize_room_phrase("") == "unknown"


def test_coerce_room_label_policy_split():
    assert coerce_room_label("Master Bathroom", room_policy="canonical") == "bathroom"
    assert coerce_room_label("Master Bathroom", room_policy="llm") == "master bathroom"


def test_merge_room_estimates_llm_keeps_phrase():
    assert merge_room_estimates("open living area", "kitchen", room_policy="llm") == "open living area"
    assert merge_room_estimates("Master Bath", "unknown", room_policy="canonical") == "bathroom"


def test_room_leave_needed_llm_uses_in_target_area():
    assert room_leave_needed(
        room_policy="llm",
        current_room="living room",
        question="Where is the shower?",
        in_target_area=False,
    )
    assert not room_leave_needed(
        room_policy="llm",
        current_room="living room",
        question="Where is the shower?",
        in_target_area=True,
    )
    assert not room_leave_needed(
        room_policy="llm",
        current_room="unknown",
        question="Where is the shower?",
        in_target_area=False,
    )


def test_llm_system_prompt_asks_in_target_area():
    tools: list = []
    prompt = build_graph_eqa_system_prompt(tools, room_policy="llm")
    assert "in_target_area" in prompt
    assert "Question area" not in prompt
    can = build_graph_eqa_system_prompt(tools, room_policy="canonical")
    assert "in_target_area" not in can
    assert "living_room" in can


def test_graph_eqa_system_prompt_byte_stable_for_prefix_kv():
    """Router system prompt must be byte-identical across calls (Qwen3-VL prefix KV)."""
    import hashlib

    from emet.memory.graph_eqa import agentic_tools as at

    # Pinned hashes of the format blocks (shared rule atoms must compose identically).
    assert (
        hashlib.sha256(at._EQA_FORMAT_BLOCK_CANONICAL.encode()).hexdigest()
        == "cf50ef4634a3ee4ba7bb21a0f083a728908b53060a59d211211cfbefac6aa603"
    )
    assert (
        hashlib.sha256(at._EQA_FORMAT_BLOCK_LLM.encode()).hexdigest()
        == "5cad41bf4196635feadefdaff84e337090ec52d100ba2bf85857cf760666d3b1"
    )
    tools: list = []
    for policy in ("canonical", "llm"):
        a = build_graph_eqa_system_prompt(tools, room_policy=policy)
        b = build_graph_eqa_system_prompt(tools, room_policy=policy)
        assert a == b
        assert at._EQA_RULE_INVESTIGATE in a
        assert at._EQA_RULES_ANSWERABILITY in a


def test_resolve_room_policy():
    assert resolve_room_policy("LLM") == "llm"
    assert resolve_room_policy("nope") == "canonical"


def test_vlm_frontier_choice_reposes_question_not_fake_area():
    agent = GraphEQAController.__new__(GraphEQAController)
    agent._habitat_blocked_goals = set()
    agent._habitat_recent_goals = []
    agent.robot = MagicMock()
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.robot.__class__.__name__ = "FakeRobot"

    far = SimpleNamespace(
        is_frontier=True,
        obs_id=2,
        xyz=np.array([5.0, 0.0, 1.0]),
        labels=["area"],
        frontier_cell_count=40,
        frontier_keyword_score=0.0,
        nav_failures=0,
    )
    near = SimpleNamespace(
        is_frontier=True,
        obs_id=1,
        xyz=np.array([1.0, 0.0, 1.0]),
        labels=["area"],
        frontier_cell_count=4,
        frontier_keyword_score=0.0,
        nav_failures=0,
    )
    obj = SimpleNamespace(
        is_frontier=False,
        is_viewpoint=False,
        xyz=np.array([5.1, 0.1, 0.5]),
        labels=["towel", "sink"],
    )
    gm = MagicMock()
    gm.eqa_client = MagicMock(return_value="Image 1")
    gm.get_nodes.return_value = [far, near, obj]
    gm._nodes = [far, near, obj]
    gm.graph_room_at_robot = MagicMock(return_value="bathroom")
    gm._observation_by_id.side_effect = lambda oid: SimpleNamespace(
        rgb=np.full((8, 8, 3), int(oid) * 40, dtype=np.uint8)
    )
    agent.graph_memory = gm
    agent.voxel_map = None

    import emet.controller.controller_graph_eqa as mod

    old = mod.is_habitat_robot_client
    mod.is_habitat_robot_client = lambda _r: False
    q = "Which rug is at the shower in the bathroom?"
    try:
        pt = agent._vlm_frontier_choice(
            q,
            current_room="living_room",
            room_policy="llm",
            leave_hint=True,
        )
    finally:
        mod.is_habitat_robot_client = old

    assert pt is not None
    directive = gm.eqa_client.call_args[0][0][0]
    assert "room=" in directive
    assert q in directive
    assert "best help determine the answer" in directive or "best help" in directive
    assert "Question area:" not in directive
    assert "dark blue" not in directive
    assert "Current place:" in directive
    assert "unhelpful" in directive or "more informative" in directive

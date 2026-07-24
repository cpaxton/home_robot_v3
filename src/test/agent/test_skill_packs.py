# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Skill pack membership: CHAT vs EQA_EPISODE stay disjoint on exclusive tools."""

from __future__ import annotations

from unittest.mock import MagicMock

from emet.agent.skills import (
    CHAT_EXCLUSIVE_TOOL_NAMES,
    CHAT_SKILL_SPECS,
    EQA_EXCLUSIVE_TOOL_NAMES,
    SHARED_SKILL_ALIASES,
    AgentMode,
    build_skill_pack,
    skill_names_for_mode,
)
from emet.agent.tools import get_tools
from emet.memory.graph_eqa.agentic_tools import build_agentic_eqa_tools


def test_eqa_answer_pack_names_match_registry():
    ex = MagicMock()
    ex.mode = "answer"
    names = {t.name for t in build_agentic_eqa_tools(ex)}
    assert names == set(skill_names_for_mode(AgentMode.EQA_EPISODE, eqa_submode="answer"))
    assert "submit_answer" in names
    assert "finish" not in names
    assert names.isdisjoint(CHAT_EXCLUSIVE_TOOL_NAMES)


def test_eqa_explore_pack_swaps_finish():
    ex = MagicMock()
    ex.mode = "explore"
    names = {t.name for t in build_agentic_eqa_tools(ex)}
    assert names == set(skill_names_for_mode(AgentMode.EQA_EPISODE, eqa_submode="explore"))
    assert "finish" in names
    assert "submit_answer" not in names


def test_chat_pack_excludes_eqa_only_tools():
    names = {t.name for t in get_tools({})}
    assert names == set(CHAT_EXCLUSIVE_TOOL_NAMES)
    assert names.isdisjoint(EQA_EXCLUSIVE_TOOL_NAMES)
    assert "describe_scene" in names
    assert "verify_siglip" not in names
    assert "submit_answer" not in names


def test_chat_tools_match_skill_specs():
    specs = {s.name: s for s in CHAT_SKILL_SPECS}
    for tool in get_tools({}):
        spec = specs[tool.name]
        assert tool.description == spec.description
        assert tool.parameters == spec.parameters
        assert tool.returns_info is spec.returns_info


def test_build_skill_pack_chat_matches_get_tools():
    ctx = {}
    via_pack = {t.name for t in build_skill_pack(AgentMode.CHAT, ctx)}
    via_get = {t.name for t in get_tools(ctx)}
    assert via_pack == via_get


def test_build_skill_pack_eqa_matches_agentic_builder():
    ex = MagicMock()
    ex.mode = "answer"
    via_pack = [t.name for t in build_skill_pack(AgentMode.EQA_EPISODE, ex, eqa_submode="answer")]
    via_legacy = [t.name for t in build_agentic_eqa_tools(ex)]
    assert via_pack == via_legacy


def test_shared_skill_aliases_document_mode_names():
    assert SHARED_SKILL_ALIASES["in_place_scan"][AgentMode.CHAT] == "scan_environment"
    assert SHARED_SKILL_ALIASES["in_place_scan"][AgentMode.EQA_EPISODE] == "look_around"
    assert SHARED_SKILL_ALIASES["explore_motion"][AgentMode.CHAT] == "explore"
    assert SHARED_SKILL_ALIASES["explore_motion"][AgentMode.EQA_EPISODE] == "explore_frontier"

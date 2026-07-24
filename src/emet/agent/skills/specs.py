# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Skill membership and EQA tool schemas (names/schemas must stay stable for traces)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from emet.agent.skills.modes import AgentMode

_NO_PARAMS: dict[str, Any] = {"type": "object", "properties": {}, "required": []}


@dataclass(frozen=True)
class SkillSpec:
    """Declarative skill metadata used to assemble mode-specific :class:`Tool` packs."""

    name: str
    modes: frozenset[AgentMode]
    description: str
    parameters: dict[str, Any]
    returns_info: bool = False
    # EQA_EPISODE only: answer mode vs explore mode terminal tool.
    eqa_answer_only: bool = False
    eqa_explore_only: bool = False


# Conceptual aliases (different tool *names* per mode; do not rename EQA tools — traces depend on them).
SHARED_SKILL_ALIASES: dict[str, dict[AgentMode, str]] = {
    "in_place_scan": {
        AgentMode.CHAT: "scan_environment",
        AgentMode.EQA_EPISODE: "look_around",
    },
    "explore_motion": {
        AgentMode.CHAT: "explore",
        AgentMode.EQA_EPISODE: "explore_frontier",
    },
}


EQA_SKILL_SPECS: tuple[SkillSpec, ...] = (
    SkillSpec(
        name="inspect_graph",
        modes=frozenset({AgentMode.EQA_EPISODE}),
        description=(
            "Refresh question keywords and ranked navigation hypotheses from the scene graph "
            "and SigLIP memory. Use when hypotheses look stale or empty."
        ),
        parameters=_NO_PARAMS,
        returns_info=True,
    ),
    SkillSpec(
        name="explore_frontier",
        modes=frozenset({AgentMode.EQA_EPISODE}),
        description=(
            "Navigate to an unexplored frontier to grow the map and graph. Optional 'toward' "
            "biases frontier choice toward a phrase (e.g. the question object). Map and graph "
            "update automatically afterward."
        ),
        parameters={
            "type": "object",
            "properties": {
                "toward": {
                    "type": "string",
                    "description": "Optional object phrase to bias the frontier pick toward.",
                }
            },
            "required": [],
        },
    ),
    SkillSpec(
        name="navigate_to_obs",
        modes=frozenset({AgentMode.EQA_EPISODE}),
        description=(
            "Navigate to a graph observation by obs_id (a hypothesis location). Map and graph "
            "update automatically on arrival."
        ),
        parameters={
            "type": "object",
            "properties": {
                "obs_id": {"type": "integer", "description": "Graph observation id to navigate to."}
            },
            "required": ["obs_id"],
        },
    ),
    SkillSpec(
        name="look_around",
        modes=frozenset({AgentMode.EQA_EPISODE}),
        description=(
            "Scan in place (head sweep / rotate) to refresh the map and graph at the current "
            "pose without navigating."
        ),
        parameters=_NO_PARAMS,
    ),
    SkillSpec(
        name="verify_siglip",
        modes=frozenset({AgentMode.EQA_EPISODE}),
        description=(
            "Cheap visual check: does 'phrase' match the current camera view / stored view "
            "obs_id? Returns PRESENT / CANDIDATE / ABSENT with a similarity score. PRESENT "
            "unlocks submit_answer."
        ),
        parameters={
            "type": "object",
            "properties": {
                "phrase": {"type": "string", "description": "Object phrase to verify (e.g. 'sink')."},
                "obs_id": {
                    "type": "integer",
                    "description": "Observation id to verify against (-1 = current best hypothesis).",
                },
            },
            "required": ["phrase"],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="submit_answer",
        modes=frozenset({AgentMode.EQA_EPISODE}),
        description=(
            "Submit the final answer (MCQ letter or short phrase). Rejected until a "
            "verify_siglip PRESENT (or the round budget is exhausted)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Final answer letter or phrase."}
            },
            "required": [],
        },
        eqa_answer_only=True,
    ),
    SkillSpec(
        name="finish",
        modes=frozenset({AgentMode.EQA_EPISODE}),
        description=(
            "End exploration with a short summary of what was mapped. Only allowed once "
            "frontiers are exhausted or the exploration budget is used."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "One-sentence summary of the mapped area."}
            },
            "required": [],
        },
        eqa_explore_only=True,
    ),
)


# Chat-only skills (must not appear in the EQA_EPISODE pack).
CHAT_EXCLUSIVE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "describe_scene",
        "send_image",
        "send_map_snapshot",
        "send_object_image",
        "query_memory",
        "navigation_diagnostics",
        "pick_place",
        "find_objects",
        "say",
        "wave",
        "nod_head",
        "shake_head",
        "avert_gaze",
        "go_home",
        "scan_environment",
        "rotate_base",
        "move_forward",
        "take_picture",
        "take_ee_picture",
        "hand_over",
        "query_scene_graph",
        "list_scene_relations",
        "quit",
        "explore",
    }
)

# EQA-only skills (must not appear in the CHAT pack by default).
EQA_EXCLUSIVE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "inspect_graph",
        "explore_frontier",
        "navigate_to_obs",
        "look_around",
        "verify_siglip",
        "submit_answer",
        "finish",
    }
)


def eqa_specs_for_submode(submode: str) -> tuple[SkillSpec, ...]:
    """Filter EQA specs for answer vs explore terminal tool."""
    explore = str(submode).lower() == "explore"
    out: list[SkillSpec] = []
    for spec in EQA_SKILL_SPECS:
        if spec.eqa_answer_only and explore:
            continue
        if spec.eqa_explore_only and not explore:
            continue
        out.append(spec)
    return tuple(out)


def skill_names_for_mode(mode: AgentMode, *, eqa_submode: str = "answer") -> frozenset[str]:
    """Expected tool names for a mode (EQA from specs; CHAT from exclusive set)."""
    if mode is AgentMode.EQA_EPISODE:
        return frozenset(s.name for s in eqa_specs_for_submode(eqa_submode))
    if mode is AgentMode.CHAT:
        return CHAT_EXCLUSIVE_TOOL_NAMES
    raise ValueError(f"Unknown AgentMode: {mode!r}")

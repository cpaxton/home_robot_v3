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
            "properties": {"obs_id": {"type": "integer", "description": "Graph observation id to navigate to."}},
            "required": ["obs_id"],
        },
    ),
    SkillSpec(
        name="look_around",
        modes=frozenset({AgentMode.EQA_EPISODE}),
        description=(
            "Scan in place (head sweep / rotate) to refresh the map and graph at the current pose without navigating."
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
            "properties": {"answer": {"type": "string", "description": "Final answer letter or phrase."}},
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
            "properties": {"summary": {"type": "string", "description": "One-sentence summary of the mapped area."}},
            "required": [],
        },
        eqa_explore_only=True,
    ),
)


# Chat skill metadata (funcs bind in emet.agent.tools.build_chat_tools).
CHAT_SKILL_SPECS: tuple[SkillSpec, ...] = (
    SkillSpec(
        name="query_memory",
        modes=frozenset({AgentMode.CHAT}),
        description="Questions about the scene (where is X, have I seen Y). Uses graph EQA when enabled, else voxel memory. Returns a final user-facing Answer line (not image numbers). For relations use list_scene_relations; for open-ended views use describe_scene and send_image.",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural language question about the environment or memory.",
                }
            },
            "required": ["question"],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="send_image",
        modes=frozenset({AgentMode.CHAT}),
        description="Capture the head-camera view and attach it to the reply (Discord when connected). Use with describe_scene for 'what do you see', or alone when asked for a photo.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="explore",
        modes=frozenset({AgentMode.CHAT}),
        description="Navigate to explore and build a map (moves through the space — longer than scan_environment). Use for 'explore', 'map the room', 'go look around the house'. For a quick in-place look, prefer scan_environment. Returns a short map diagnostic after the run; pair with send_map_snapshot or describe_scene if stuck.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="navigation_diagnostics",
        modes=frozenset({AgentMode.CHAT}),
        description="Text summary of the current 2D voxel map: explored vs obstacle cell counts, base pose in grid, and hints if the map is empty or the base sits on an obstacle cell. Use after failed explore/find or when the user asks why navigation failed.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="send_map_snapshot",
        modes=frozenset({AgentMode.CHAT}),
        description="Render a top-down RGB view of obstacles vs explored space (cropped to the explored region plus margin) and send to Discord if configured; also logs the same crop to Rerun at world/map_snapshot/topdown when the live Rerun visualizer is enabled.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="pick_place",
        modes=frozenset({AgentMode.CHAT}),
        description="Pick up an object and place it on a receptacle.",
        parameters={
            "type": "object",
            "properties": {
                "object_name": {
                    "type": "string",
                    "description": "Name of the object to pick up.",
                },
                "receptacle_name": {
                    "type": "string",
                    "description": "Name of the receptacle or surface to place it on.",
                },
            },
            "required": ["object_name", "receptacle_name"],
        },
    ),
    SkillSpec(
        name="find_objects",
        modes=frozenset({AgentMode.CHAT}),
        description="Find and navigate to an object or location in the environment by name.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Object or location name to find.",
                }
            },
            "required": ["text"],
        },
    ),
    SkillSpec(
        name="describe_scene",
        modes=frozenset({AgentMode.CHAT}),
        description="Caption what is in front of the robot *right now* (live head camera) and optionally ground with known scene-graph / map labels. Does NOT move or reorient the robot — this alone is not a 'closer look'. For 'are you sure' / 'look closer' / confirm, first call rotate_base / move_forward / scan_environment (as allowed), then this tool. Queues the live head-camera photo for Discord (not an object crop; use send_object_image for that). Use an empty JSON \"message\" on the tool-call turn so chat/Discord only show your answer after [Tool results].",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="say",
        modes=frozenset({AgentMode.CHAT}),
        description="Speak text to the user via TTS (text-to-speech). Use to announce what you are doing.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The message to speak aloud.",
                }
            },
            "required": ["text"],
        },
    ),
    SkillSpec(
        name="wave",
        modes=frozenset({AgentMode.CHAT}),
        description="Wave at a person (e.g. when greeting or saying goodbye).",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    SkillSpec(
        name="nod_head",
        modes=frozenset({AgentMode.CHAT}),
        description="Nod the robot's head (e.g. to indicate yes or agreement).",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    SkillSpec(
        name="shake_head",
        modes=frozenset({AgentMode.CHAT}),
        description="Shake the robot's head (e.g. to indicate no or disagreement).",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    SkillSpec(
        name="avert_gaze",
        modes=frozenset({AgentMode.CHAT}),
        description="Avert the robot's gaze (look away).",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    SkillSpec(
        name="go_home",
        modes=frozenset({AgentMode.CHAT}),
        description="Navigate the robot back to its starting position. Requires a map from prior exploration.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    SkillSpec(
        name="scan_environment",
        modes=frozenset({AgentMode.CHAT}),
        description="Rotate in place through a full ≈360° scan to update the map and save memory. Use for 'look around', 'scan the room', or a full in-place survey — not for a single turn (use rotate_base) or a short drive (use move_forward). After scanning, you may call describe_scene to report the new view.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="rotate_base",
        modes=frozenset({AgentMode.CHAT}),
        description="Rotate the wheeled base in place by a relative yaw in degrees (positive = left/CCW, negative = right/CW). Pass an explicit angle: turn around → 180, turn right → -90, turn left → 90, slight turn → ±30–45. 'rotate back' / 'turn back' means NEGATE the previous rotate (e.g. after +45 use -45) — NOT 180 (that is 'turn around'). For 'look at / face / go toward X' when you have a named object, prefer face_toward.",
        parameters={
            "type": "object",
            "properties": {
                "degrees": {
                    "type": "number",
                    "description": "Relative yaw in degrees (e.g. 180, -90, 45, or -last for 'back').",
                }
            },
            "required": ["degrees"],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="face_toward",
        modes=frozenset({AgentMode.CHAT}),
        description="Rotate in place to face a named object from the scene graph / map (computes yaw toward its remembered XY). Use for 'look at the aquarium', 'face the shelf', 'turn toward the TV'. Does NOT drive closer — follow with describe_scene. Prefer this over a blind rotate_base(±45) when the object is known. If the object is unknown, fall back to rotate_base or ask.",
        parameters={
            "type": "object",
            "properties": {
                "object_label": {
                    "type": "string",
                    "description": "Object name to face (e.g. aquarium, shelf, cardboard box).",
                }
            },
            "required": ["object_label"],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="move_forward",
        modes=frozenset({AgentMode.CHAT}),
        description="Drive the base forward along its current heading by approximately *meters*. Uses the 2D map (including a local_radius explored disk around the base when depth is still empty): shortens or refuses if blocked or the path leaves explored space. If refused, ask the user whether to scan_environment / rotate_base — do NOT auto-scan. If the user did not say how far, do NOT call this tool — ask how far first. Use meters=0.1 for 'a bit' / 'a little' / 'nudge'; ~0.5 for half a meter; ~1.0 for a meter. Cap near 1.5 m. Do not use for turning (use rotate_base).",
        parameters={
            "type": "object",
            "properties": {
                "meters": {
                    "type": "number",
                    "description": "Forward distance in meters (required). 0.1 for a small nudge / 'a bit'; 0.5 for half a meter; 1.0 for a meter.",
                }
            },
            "required": ["meters"],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="take_picture",
        modes=frozenset({AgentMode.CHAT}),
        description="Capture the head camera locally. Prefer send_image or describe_scene when the user should see or hear a description — those attach a photo / caption to Discord.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="take_ee_picture",
        modes=frozenset({AgentMode.CHAT}),
        description="Capture the wrist/end-effector camera only (no arm motion). Do NOT use for 'closer look' / 'inspect X' — that would require pointing the arm at the object with IK, which is not supported here. Use describe_scene (head camera + caption) or send_image instead. On Innate Mars the wrist stream is often missing.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="aim_arm_at",
        modes=frozenset({AgentMode.CHAT}),
        description="STUB: Point the arm / wrist camera at a named object using IK, then the user can inspect it. Not implemented yet — do not pretend it moved the arm. For 'closer look' / 'inspect X' prefer describe_scene (head camera) until IK lands.",
        parameters={
            "type": "object",
            "properties": {
                "object_label": {
                    "type": "string",
                    "description": "Object or region to aim at (e.g. cables, red cup).",
                }
            },
            "required": ["object_label"],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="hand_over",
        modes=frozenset({AgentMode.CHAT}),
        description="Hand the held object to a person (find person, navigate, extend arm).",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    SkillSpec(
        name="query_scene_graph",
        modes=frozenset({AgentMode.CHAT}),
        description="Embodied where/what questions using GraphEQA (objects, 3D positions) when enabled. Returns Answer / Location / Confidence for the user (not internal image ids). Otherwise dumps the open-vocab spatial scene graph. Prefer for 'where is', 'what color', 'is there'.",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural language question about the scene graph / objects.",
                }
            },
            "required": ["question"],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="list_scene_relations",
        modes=frozenset({AgentMode.CHAT}),
        description="List objects and spatial relations (near, on, on_floor) from the open-vocabulary 3D scene graph. Use for 'what is connected to what' and structured connectivity questions.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="send_object_image",
        modes=frozenset({AgentMode.CHAT}),
        description="Send the robot's last stored crop image for a named object from the open-vocab scene graph (not the live camera). Use after the object has been observed while mapping.",
        parameters={
            "type": "object",
            "properties": {
                "object_label": {
                    "type": "string",
                    "description": "Object name or label (e.g. red cylinder).",
                }
            },
            "required": ["object_label"],
        },
        returns_info=True,
    ),
    SkillSpec(
        name="quit",
        modes=frozenset({AgentMode.CHAT}),
        description="End the conversation and stop the robot. Use when the user says goodbye or asks to stop.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
)

CHAT_EXCLUSIVE_TOOL_NAMES: frozenset[str] = frozenset(s.name for s in CHAT_SKILL_SPECS)

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

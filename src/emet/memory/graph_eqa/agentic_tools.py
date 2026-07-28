# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tool registry + prompts for the agentic GraphEQA explore/verify loop.

Reuses the Discord agent's tool-calling contract (:class:`emet.agent.tools.Tool`
schemas + ``{"tool_calls": [...], "message": ...}`` responses parsed by
:func:`emet.agent.prompt.parse_tool_calls_response`) so local VLMs route tools
through one proven JSON format instead of ad-hoc regex.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from emet.agent.skills import AgentMode, build_skill_pack
from emet.agent.tools import Tool, get_tool_descriptions_for_prompt
from emet.utils.logger import Logger

if TYPE_CHECKING:
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

_logger = Logger(__name__)

# Response format block. Constant string (with the tools block) so the routing
# system prompt is byte-identical across rounds and question banks — required
# for Qwen3-VL system-prefix KV cache hits.
_EQA_FORMAT_BLOCK = """\
# Response format
Respond with ONLY a JSON object (no other text):
{"current_room": "<room>", "tool_calls": [{"name": "<tool>", "arguments": {...}}, ...], "message": ""}

Rules:
- Always set current_room to where the robot is NOW (from Investigate cards /
  Recent labels / REGION text / Recent actions) — not the destination.
  Prefer: patio, outdoor, kitchen, living_room, dining_room, bedroom, bathroom,
  hallway, garage, unknown.
- Each turn choose explicitly: INVESTIGATE a place card OR EXPLORE for coverage
  (then verify/assess runs at the station after investigate).
- investigate(obs_id): closer look at a listed Investigate card (graph/confirmed/siglip).
  Do not investigate frontiers — those are Explore-only.
- explore_frontier: map growth when no place is worth a closer look, or after
  places look fruitless. toward= is weak coverage bias ONLY —
  never a substitute for investigate(obs_id).
- Use Recent actions to avoid repeating a stuck investigate/explore loop.
- After a close look verifies ABSENT, prefer explore_frontier once to grow
  coverage before investigating a new capture-station card.
- If Current room is patio/outdoor and the question is about an indoor object
  (clock, kitchen, living room, …), prefer explore_frontier to leave outdoors.
- SigLIP/OWL are proposals in state — not proof. Trust Qwen assess for answerability.
- Pass MCQ letter (A–D) in submit_answer.arguments.answer when answerable.
- One or two tool calls per turn.

# Examples
State: Investigate obs_id=3 phrase='sink' source=graph investigated=0 approaches=0/4 coverage=open
{"current_room": "kitchen", "tool_calls": [{"name": "investigate", "arguments": {"obs_id": 3}}], "message": ""}
State: Recent actions: r0 investigate obs=3 verify=ABSENT; Prefer explore_frontier
{"current_room": "patio", "tool_calls": [{"name": "explore_frontier", "arguments": {}}], "message": ""}
State: Recent actions: r0–r2 investigate same obs ABSENT; Explore frontiers available
{"current_room": "outdoor", "tool_calls": [{"name": "explore_frontier", "arguments": {}}], "message": ""}
State: Last verify PRESENT; VLM assess answerable=true verified=true
{"current_room": "living_room", "tool_calls": [{"name": "submit_answer", "arguments": {"answer": "B"}}], "message": ""}
State: Investigate (none); Explore frontiers available
{"current_room": "unknown", "tool_calls": [{"name": "explore_frontier", "arguments": {}}], "message": ""}"""

_EQA_IDENTITY = """\
You are a robot answering questions about a home. You maintain a 3D map and scene graph.
Decide each turn: investigate a promising place for a closer look, or explore to grow
coverage when places are exhausted or none look good. Do NOT output reasoning — only JSON."""

# Canonical room labels for router current_room (aliases normalize into these).
ROOM_CANONICAL = frozenset(
    {
        "patio",
        "outdoor",
        "kitchen",
        "living_room",
        "dining_room",
        "bedroom",
        "bathroom",
        "hallway",
        "garage",
        "unknown",
    }
)
_OUTDOOR_ROOMS = frozenset({"patio", "outdoor"})
_OUTDOOR_ALIASES = frozenset(
    {
        "patio",
        "outdoor",
        "outdoors",
        "outside",
        "yard",
        "deck",
        "porch",
        "garden",
        "brick_patio",
        "courtyard",
    }
)
_INDOOR_QUESTION_CUES = frozenset(
    {
        "clock",
        "wall clock",
        "kitchen",
        "living",
        "living room",
        "dining",
        "bedroom",
        "bathroom",
        "bowl",
        "fruit bowl",
        "microwave",
        "refrigerator",
        "fridge",
        "sofa",
        "couch",
        "fireplace",
        "cabinet",
        "indoor",
        "inside",
    }
)


def normalize_current_room(raw: Any) -> str:
    """Map free-text router ``current_room`` onto a small vocabulary."""
    if raw is None:
        return "unknown"
    s = str(raw).strip().lower().replace("-", " ").replace("/", " ")
    s = "_".join(s.split())
    if not s:
        return "unknown"
    if s in ROOM_CANONICAL:
        return s
    if s in _OUTDOOR_ALIASES or any(a in s for a in ("patio", "outdoor", "yard", "deck", "porch")):
        return "outdoor" if "patio" not in s else "patio"
    if "living" in s:
        return "living_room"
    if "dining" in s:
        return "dining_room"
    if "kitchen" in s:
        return "kitchen"
    if "bed" in s:
        return "bedroom"
    if "bath" in s:
        return "bathroom"
    if "hall" in s or "corridor" in s:
        return "hallway"
    if "garage" in s:
        return "garage"
    return "unknown"


def room_is_outdoor(room: str) -> bool:
    return normalize_current_room(room) in _OUTDOOR_ROOMS


def question_implies_indoor(question: str) -> bool:
    """True when the embodied question is likely about an indoor place/object."""
    q = str(question or "").strip().lower()
    if not q:
        return False
    if any(cue in q for cue in _INDOOR_QUESTION_CUES):
        return True
    # Location MCQ options often name rooms.
    try:
        from emet.memory.graph_eqa.graph_memory import location_mcq_landmark_phrases

        landmarks = location_mcq_landmark_phrases(question)
    except Exception:
        landmarks = []
    indoor_landmarks = ("kitchen", "living", "dining", "bedroom", "bathroom", "hall")
    return any(any(tok in str(lm).lower() for tok in indoor_landmarks) for lm in landmarks)


def build_agentic_eqa_tools(executor: AgenticEQAExecutor) -> list[Tool]:
    """EQA_EPISODE tool pack. Schemas/names come from :mod:`emet.agent.skills`; funcs dispatch via ``handle_tool``."""
    submode = getattr(executor, "mode", "answer")
    return build_skill_pack(AgentMode.EQA_EPISODE, executor, eqa_submode=submode)


def build_graph_eqa_system_prompt(tools: list[Tool]) -> str:
    """Fixed routing system prompt (identity + tools + format). Keep byte-stable per mode."""
    tools_block = get_tool_descriptions_for_prompt(tools)
    return f"{_EQA_IDENTITY}\n\n{tools_block}\n\n{_EQA_FORMAT_BLOCK}"


def _graph_stats_line(gm: Any) -> str:
    from emet.memory.graph_eqa.graph_stats import format_graph_size_report

    return format_graph_size_report(gm, verbose=False)


def build_state_message(executor: AgenticEQAExecutor) -> str:
    """Per-round user message: goal + graph stats + Investigate/Explore cards + budgets."""
    from emet.memory.graph_eqa.agentic_eqa import INVESTIGATE_SOURCES

    gm = executor.graph_memory
    lines: list[str] = []
    if getattr(executor, "mode", "answer") == "explore":
        lines.append(f"Goal: {executor.goal or 'explore and map the environment'}")
    else:
        lines.append(f"Question: {executor.question}")
    lines.append(_graph_stats_line(gm))
    graph_room = str(getattr(executor, "_graph_room_estimate", "") or "").strip()
    if gm is not None:
        room_fn = getattr(gm, "graph_room_at_robot", None)
        if callable(room_fn):
            try:
                xyt = None
                robot_fn = getattr(executor, "_robot_xyt", None)
                if callable(robot_fn):
                    xyt = robot_fn()
                graph_room = normalize_current_room(room_fn(xyt))
                executor._graph_room_estimate = graph_room
            except Exception as e:
                _logger.warning(f"graph room for router state failed: {e}")
    last_room = str(getattr(executor, "_last_room_estimate", "") or "").strip()
    if graph_room:
        lines.append(f"Current room (graph): {graph_room}")
    if last_room:
        lines.append(f"Current room (router): {last_room}")
    if gm is not None:
        rooms_fn = getattr(gm, "format_rooms_line", None)
        if callable(rooms_fn):
            try:
                rooms_line = rooms_fn()
            except Exception as e:
                _logger.warning(f"room clusters for router state failed: {e}")
                rooms_line = ""
            if isinstance(rooms_line, str) and rooms_line.strip():
                lines.append(rooms_line)
        used_spatial = False
        if getattr(gm, "_spatial_rag_enabled", lambda: False)():
            try:
                from emet.memory.graph_eqa.spatial_rag import (
                    format_regions_compact,
                    select_spatial_regions,
                )

                rag = select_spatial_regions(
                    list(getattr(gm, "_nodes", []) or []),
                    keywords=list(getattr(gm, "_relevant_objects", None) or []),
                    prefer_obs_ids=list(getattr(gm, "last_eqa_obs_ids", []) or []),
                    radius_m=float(gm._spatial_rag_float("spatial_rag_radius_m", 2.5)),
                    max_regions=int(gm._spatial_rag_int("spatial_rag_max_regions", 6)),
                    max_nodes=int(gm._spatial_rag_int("spatial_rag_max_nodes", 48)),
                    max_frontiers=4,
                )
                compact = format_regions_compact(rag, max_chars=900)
                if compact:
                    lines.append(compact)
                    used_spatial = True
            except Exception as e:
                _logger.warning(f"spatial RAG for router state failed: {e}")
                used_spatial = False
        if not used_spatial:
            mem_fn = getattr(gm, "_relevant_memory_summary", None)
            if callable(mem_fn) and bool(getattr(gm, "memory_summary_enabled", False)):
                try:
                    mem = mem_fn()
                except Exception:
                    mem = None
                if mem:
                    snippet = str(mem).strip()
                    if len(snippet) > 900:
                        snippet = snippet[:900].rstrip() + "…"
                    lines.append(snippet)
            top_labels: list[str] = []
            for obs in list(getattr(gm, "_observations", []) or [])[-24:]:
                for lab in list(getattr(obs, "labels", []) or [])[:3]:
                    s = str(lab).strip()
                    if s and s.lower() != "frontier" and s not in top_labels:
                        top_labels.append(s)
                    if len(top_labels) >= 16:
                        break
                if len(top_labels) >= 16:
                    break
            if top_labels:
                lines.append("Recent labels: " + ", ".join(top_labels))
    lines.append(
        f"Round {executor._round + 1}/{executor.max_rounds}; "
        f"nav used {executor._n_nav + executor._n_explore}/{executor.max_nav_steps}; "
        f"verified={executor._verified}"
    )
    recent_actions = list(getattr(executor, "_recent_actions", None) or [])
    if recent_actions:
        lines.append("Recent actions: " + " | ".join(recent_actions))
    if getattr(executor, "_prefer_explore", False):
        if getattr(executor, "_prefer_explore_outdoor", False):
            lines.append(
                "Prefer explore_frontier: outdoor/patio looks ruled out for this "
                "indoor question — leave outdoors and grow indoor coverage."
            )
        else:
            lines.append(
                "Prefer explore_frontier: last close look was ABSENT — grow coverage "
                "before chasing a new capture station."
            )
    if getattr(executor, "_last_capture_status", None):
        lines.append(f"Last capture: {executor._last_capture_status}")
    loop_flags = list(getattr(executor, "_nav_loop_flags", None) or [])
    if loop_flags:
        last = loop_flags[-1]
        lines.append(
            f"NAV_LOOP: obs_id={last.get('obs_id')} visits={last.get('visits')} "
            f"status={last.get('status')} — pick another investigate card or explore_frontier"
        )
    lines.append(
        "Choose: investigate(obs_id) for a closer look at a place, "
        "OR explore_frontier if no place is worth it / places are exhausted."
    )
    inv = [h for h in executor._hypotheses if str(h.source) in INVESTIGATE_SOURCES]
    exp = [h for h in executor._hypotheses if str(h.source) not in INVESTIGATE_SOURCES]
    ledger = getattr(executor, "_place_inspect", {}) or {}
    refresh = getattr(executor, "_refresh_place_coverage", None)
    if inv:
        lines.append("Investigate (place cards — use investigate):")
        for h in inv:
            oid = int(h.obs_id)
            if callable(refresh):
                try:
                    refresh(oid)
                except Exception:
                    pass
            labels = _hyp_labels(executor, oid)
            label_bit = f" labels={labels}" if labels else ""
            sim = getattr(h, "siglip_sim", None)
            sim_bit = ""
            if isinstance(sim, (int, float)):
                sim_bit = f" siglip_sim={float(sim):.3f}"
            rec = ledger.get(oid)
            bits = (
                rec.card_bits()
                if rec is not None
                else ("investigated=0 closest=none approaches=0/4 coverage=unknown recent=none")
            )
            tried = executor._tried.get(oid)
            if tried:
                bits += f" [tried: {tried}]"
            lines.append(
                f"- obs_id={oid} phrase={h.phrase!r} source={h.source} "
                f"xyz=({float(h.xyz[0]):.1f},{float(h.xyz[1]):.1f})"
                f"{label_bit}{sim_bit} {bits}"
            )
    else:
        lines.append("Investigate: (none — explore or look_around first)")
    if exp:
        lines.append("Explore (frontiers — coverage only; not investigate targets):")
        for h in exp:
            oid = int(h.obs_id)
            lines.append(
                f"- obs_id={oid} phrase={h.phrase!r} source={h.source} "
                f"xyz=({float(h.xyz[0]):.1f},{float(h.xyz[1]):.1f})"
            )
    else:
        lines.append("Explore: (no frontier cards in recall)")
    if executor._last_verify is not None:
        lv = executor._last_verify
        lines.append(f"Last verify: {lv.status} sim={float(lv.sim):.3f} obs_id={int(lv.obs_id)}")
    if executor.max_rounds - executor._round <= 2:
        lines.append("Budget nearly exhausted: answer/finish on your best evidence soon.")
    return "\n".join(lines)


def _hyp_labels(executor: AgenticEQAExecutor, obs_id: int) -> list[str]:
    gm = executor.graph_memory
    if gm is None:
        return []
    obs = None
    if hasattr(gm, "_observation_by_id"):
        try:
            obs = gm._observation_by_id(int(obs_id))
        except Exception:
            obs = None
    if obs is None:
        return []
    out: list[str] = []
    for lab in list(getattr(obs, "labels", None) or [])[:4]:
        s = str(lab).strip()
        if s and s.lower() != "frontier":
            out.append(s)
    return out

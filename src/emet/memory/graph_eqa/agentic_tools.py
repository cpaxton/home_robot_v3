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
# for Qwen3-VL system-prefix KV cache hits. Canonical vs LLM are separate
# stable strings (policy is fixed per episode).
_EQA_FORMAT_BLOCK_CANONICAL = """\
# Response format
Respond with ONLY a JSON object (no other text):
{"current_room": "<room>", "tool_calls": [{"name": "<tool>", "arguments": {...}}, ...], "message": ""}

Rules:
- Always set current_room to where the robot is NOW from the room-context images
  (current view + nearby object images with distances) and Investigate cards —
  not the destination. Prefer: patio, outdoor, kitchen, living_room, dining_room,
  bedroom, bathroom, hallway, garage, unknown.
- Grass / yard / outdoor furniture → outdoor or patio (never invent dining from chairs).
- Hierarchical choice (GraphEQA-style): prefer an Investigate card whose room=
  is relevant to the question; otherwise explore_frontier toward other rooms /
  frontiers. Do not invent room aliases — use the room= tags on cards.
- Each turn choose explicitly: INVESTIGATE a place card OR EXPLORE for coverage
  (then verify/assess runs at the station after investigate).
- investigate(obs_id): closer look at a listed Investigate card (graph/confirmed/siglip).
  Do not investigate frontiers — those are Explore-only.
- explore_frontier: map growth when no place is worth a closer look, or after
  places look fruitless. toward= is weak coverage bias ONLY —
  never a substitute for investigate(obs_id).
- Use Recent actions to avoid repeating a stuck investigate/explore loop.
- After a close look where VLM assess says present=false, prefer explore_frontier
  once to grow coverage, then investigate remaining Question-relevant place cards.
- SigLIP/OWL scores only rank WHERE to go next to grow the graph (drive to a
  promising place, then confirm with VLM). They are not proof of presence or
  answer — trust Qwen vlm_assess on the image for answerability.
- Never submit_answer while unverified (vlm_answerable=false) if rounds or nav
  budget remain — keep gathering evidence; the harness forces a final answer at
  budget exhaustion.
- Questions needing a CLOSE LOOK (reading a clock/display/label, counting,
  on/off or open/closed state, fine detail) prefer investigate(obs_id) or
  look_around over explore_frontier.
- Pass MCQ letter (A–D) in submit_answer.arguments.answer when answerable.
- One or two tool calls per turn.

# Examples
State: Investigate obs_id=3 phrase='sink' room=kitchen source=graph investigated=0 approaches=0/4 coverage=open
{"current_room": "kitchen", "tool_calls": [{"name": "investigate", "arguments": {"obs_id": 3}}], "message": ""}
State: Recent actions: r0 investigate obs=3 assess present=false; Prefer explore_frontier
{"current_room": "kitchen", "tool_calls": [{"name": "explore_frontier", "arguments": {}}], "message": ""}
State: Question about dining chairs; Investigate dining-table card room=kitchen available
{"current_room": "kitchen", "tool_calls": [{"name": "investigate", "arguments": {"obs_id": 37}}], "message": ""}
State: VLM assess answerable=true (vlm_answerable); place card corroborated
{"current_room": "living_room", "tool_calls": [{"name": "submit_answer", "arguments": {"answer": "B"}}], "message": ""}
State: Investigate (none); Explore frontiers available
{"current_room": "unknown", "tool_calls": [{"name": "explore_frontier", "arguments": {}}], "message": ""}"""

_EQA_FORMAT_BLOCK_LLM = """\
# Response format
Respond with ONLY a JSON object (no other text):
{"current_room": "<short place phrase>", "in_target_area": true|false, "tool_calls": [{"name": "<tool>", "arguments": {...}}, ...], "message": ""}

Rules:
- Always set current_room to where the robot is NOW from the room-context images
  (current view + nearby object images) — a short natural phrase (2–6 words),
  e.g. "master bathroom", "open kitchen living", "brick patio", "hallway",
  or "unknown". Do not force a closed room vocabulary.
- Always set in_target_area: true if the current place looks useful for answering
  the Question (you could gather the needed evidence here); false if you are
  clearly elsewhere; omit only if truly unsure.
- Prefer Investigate cards whose room=/near=/labels help answer the Question;
  otherwise explore_frontier. Graph room= tags are context — trust your judgment.
- Each turn choose explicitly: INVESTIGATE a place card OR EXPLORE for coverage.
- investigate(obs_id): closer look at a listed Investigate card (graph/confirmed/siglip).
  Do not investigate frontiers — those are Explore-only.
- explore_frontier: map growth when no place is worth a closer look, or after
  places look fruitless. toward= is optional weak bias ONLY —
  never a substitute for investigate(obs_id). Prefer leaving toward= empty and
  letting the frontier VLM pick from the Question + images.
- Use Recent actions to avoid repeating a stuck investigate/explore loop.
- After a close look where VLM assess says present=false, prefer explore_frontier
  once to grow coverage, then investigate remaining Question-relevant place cards
  (do not explore forever).
- in_target_area=false means leave OR look closer at a listed Investigate card that
  could answer the Question — never "explore_frontier only" while place cards exist.
- SigLIP/OWL scores only rank WHERE to go next to grow the graph (drive to a
  promising place, then confirm with VLM). They are not proof of presence or
  answer — trust Qwen vlm_assess on the image for answerability.
- Never submit_answer while unverified (vlm_answerable=false) if rounds or nav
  budget remain — keep gathering evidence; the harness forces a final answer at
  budget exhaustion.
- Questions needing a CLOSE LOOK (reading a clock/display/label, counting,
  on/off or open/closed state, fine detail) prefer investigate(obs_id) or
  look_around over explore_frontier.
- Pass MCQ letter (A–D) in submit_answer.arguments.answer when answerable.
- One or two tool calls per turn.

# Examples
State: Question about bathroom shower rug; Investigate obs_id=3 phrase='sink' room=bathroom
{"current_room": "bathroom", "in_target_area": true, "tool_calls": [{"name": "investigate", "arguments": {"obs_id": 3}}], "message": ""}
State: Question about bathroom shower; Current room: living room; no Investigate cards
{"current_room": "living room", "in_target_area": false, "tool_calls": [{"name": "explore_frontier", "arguments": {}}], "message": ""}
State: Question about dining chairs; Current room: kitchen; in_target_area=false; Investigate obs_id=12 phrase='dining table'
{"current_room": "kitchen", "in_target_area": false, "tool_calls": [{"name": "investigate", "arguments": {"obs_id": 12}}], "message": ""}
State: VLM assess answerable=true (vlm_answerable); place card corroborated
{"current_room": "open living area", "in_target_area": true, "tool_calls": [{"name": "submit_answer", "arguments": {"answer": "B"}}], "message": ""}
State: Investigate (none); Explore frontiers available
{"current_room": "unknown", "in_target_area": false, "tool_calls": [{"name": "explore_frontier", "arguments": {}}], "message": ""}"""

# Back-compat alias (canonical).
_EQA_FORMAT_BLOCK = _EQA_FORMAT_BLOCK_CANONICAL

_EQA_IDENTITY = """\
You are a robot answering questions about a home. The map/scene graph (rooms, labels,
nearby objects) is context for you — use it, then decide. Each turn: investigate a
promising place for a closer look, or explore to grow coverage when places are
exhausted or none look good. Prefer cards/views that help answer the Question.
Do NOT output reasoning — only JSON."""

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


def sanitize_room_phrase(raw: Any, *, max_chars: int = 48) -> str:
    """Light cleanup for room labels. Preserves canonical buckets; free-text stays phrases."""
    if raw is None:
        return "unknown"
    s = str(raw).strip().lower()
    if not s or s in {"unknown", "none", "n/a", "null"}:
        return "unknown"
    token = "_".join(s.replace("-", " ").replace("/", " ").replace("_", " ").split())
    if token in ROOM_CANONICAL:
        return token
    phrase = " ".join(s.replace("_", " ").replace("-", " ").replace("/", " ").split())
    if not phrase:
        return "unknown"
    if len(phrase) > int(max_chars):
        phrase = phrase[: max(0, int(max_chars) - 1)].rstrip() + "…"
    return phrase


def normalize_current_room(raw: Any) -> str:
    """Map free-text router ``current_room`` onto a small vocabulary (canonical policy / metrics)."""
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


# Metrics / histogram alias — same as normalize; not used for LLM-policy decisions.
room_bucket = normalize_current_room


def coerce_room_label(raw: Any, *, room_policy: str = "canonical") -> str:
    """Policy-aware room identity: closed vocab vs free-text phrase."""
    if str(room_policy or "").strip().lower() == "llm":
        return sanitize_room_phrase(raw)
    return normalize_current_room(raw)


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


def build_graph_eqa_system_prompt(tools: list[Tool], *, room_policy: str = "canonical") -> str:
    """Fixed routing system prompt (identity + tools + format). Keep byte-stable per mode."""
    tools_block = get_tool_descriptions_for_prompt(tools)
    policy = str(room_policy or "canonical").strip().lower()
    fmt = _EQA_FORMAT_BLOCK_LLM if policy == "llm" else _EQA_FORMAT_BLOCK_CANONICAL
    return f"{_EQA_IDENTITY}\n\n{tools_block}\n\n{fmt}"


def _graph_stats_line(gm: Any) -> str:
    from emet.memory.graph_eqa.graph_stats import format_graph_size_report

    return format_graph_size_report(gm, verbose=False)


def build_state_message(executor: AgenticEQAExecutor) -> str:
    """Per-round user message: goal + graph stats + Investigate/Explore cards + budgets."""
    from emet.memory.graph_eqa.agentic_eqa import INVESTIGATE_SOURCES
    from emet.memory.graph_eqa.room_clusters import room_leave_needed

    policy = str(getattr(executor, "room_policy", "canonical") or "canonical").strip().lower()
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
                graph_room = coerce_room_label(room_fn(xyt), room_policy=policy)
                executor._graph_room_estimate = graph_room
            except Exception as e:
                _logger.warning(f"graph room for router state failed: {e}")
    last_room = str(getattr(executor, "_last_room_estimate", "") or "").strip()
    if graph_room:
        lines.append(f"Current room (graph): {graph_room}")
    if last_room:
        lines.append(f"Current room (router): {last_room}")
    ita = getattr(executor, "_in_target_area", None)
    if policy == "llm" and ita is not None:
        lines.append(f"in_target_area (last): {bool(ita)}")
    n_room_imgs = int(getattr(executor, "_last_router_n_images", 0) or 0)
    if n_room_imgs > 0:
        lines.append(
            f"Room context images: {n_room_imgs} attached (current view + nearby objects; use for current_room)."
        )
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
        f"vlm_answerable={executor._verified}"
    )
    pending = getattr(executor, "_pending_answerable", None) or None
    if pending and not getattr(executor, "_verified", False):
        letter = str(pending.get("letter") or "").strip() or "?"
        lines.append(
            f"pending_answer={letter} (need confirm — re-investigate another view or "
            "wait for phrase corroboration; do not submit yet)"
        )
    if getattr(executor, "_close_look_required", False):
        source = str(getattr(executor, "_close_look_source", "") or "")
        lines.append(
            "Question needs a CLOSE LOOK (reading a clock/display/label, counting, "
            "state, fine detail) — prefer investigate(obs_id) or look_around over "
            "explore_frontier" + (f" [{source}]" if source else "")
        )
    if (
        not getattr(executor, "_verified", False)
        and bool(getattr(executor, "_no_early_unverified", True))
        and int(getattr(executor, "_round", 0) or 0) < int(getattr(executor, "max_rounds", 8) or 8) - 1
    ):
        lines.append(
            "Unverified: do NOT submit_answer while rounds/nav budget remain — keep "
            "investigating/exploring; the harness forces a final answer at budget exhaustion."
        )
    recent_actions = list(getattr(executor, "_recent_actions", None) or [])
    if recent_actions:
        lines.append("Recent actions: " + " | ".join(recent_actions))
    if getattr(executor, "_prefer_explore", False):
        reason = str(getattr(executor, "_prefer_explore_reason", "") or "")
        if reason == "absent":
            lines.append(
                "Prefer explore_frontier once: last close look VLM assess present=false — "
                "grow coverage, then investigate remaining Question-relevant place cards "
                "(do not explore forever)."
            )
        else:
            lines.append(
                "Prefer explore_frontier once to grow coverage, then investigate a place card "
                "if any remain for the Question."
            )
    leave = room_leave_needed(
        room_policy=policy,
        current_room=last_room or graph_room,
        question=str(getattr(executor, "question", "") or ""),
        in_target_area=ita if isinstance(ita, bool) else None,
    )
    if leave:
        here = last_room or graph_room or "unknown"
        lines.append(
            f"Not in a useful place yet (here: {here}) — explore_frontier to leave, "
            "OR investigate a listed place card whose room=/labels help answer the Question "
            "(close looks are allowed while leaving)."
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
        "Choose: investigate(obs_id) for a closer look at a place (prefer room-relevant "
        "cards), OR explore_frontier if no place is worth it / places are exhausted."
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
            room_bit = f" room={_hyp_room(executor, h)}"
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
            # Enrich with graph-memory attempt ledger when present (opt-in).
            gm = executor.graph_memory
            if gm is not None and hasattr(gm, "attempt_summary_for_obs"):
                ledger_bits = gm.attempt_summary_for_obs(oid)
                if ledger_bits:
                    bits += f" [attempts: {ledger_bits}]"
            lines.append(
                f"- obs_id={oid} phrase={h.phrase!r} source={h.source} "
                f"xyz=({float(h.xyz[0]):.1f},{float(h.xyz[1]):.1f})"
                f"{room_bit}{label_bit}{sim_bit} {bits}"
            )
    else:
        lines.append("Investigate: (none — explore or look_around first)")
    if exp:
        lines.append("Explore (frontiers — coverage only; not investigate targets):")
        for h in exp:
            oid = int(h.obs_id)
            room_bit = f" room={_hyp_room(executor, h)}"
            near = _frontier_nearby_labels(executor, h)
            near_bit = f" near={near}" if near else ""
            lines.append(
                f"- obs_id={oid} phrase={h.phrase!r} source={h.source} "
                f"xyz=({float(h.xyz[0]):.1f},{float(h.xyz[1]):.1f})"
                f"{room_bit}{near_bit}"
            )
    else:
        lines.append("Explore: (no frontier cards in recall)")
    if executor._last_verify is not None:
        lv = executor._last_verify
        lines.append(
            f"Last proposal (verify_siglip): {lv.status} sim={float(lv.sim):.3f} "
            f"obs_id={int(lv.obs_id)} — not VLM answerability"
        )
    if executor.max_rounds - executor._round <= 2:
        lines.append("Budget nearly exhausted: answer/finish on your best evidence soon.")
    return "\n".join(lines)


def _hyp_room(executor: AgenticEQAExecutor, hyp: Any) -> str:
    """Nearest stamped/graph room name at hypothesis XY."""
    gm = executor.graph_memory
    if gm is None:
        return "unknown"
    try:
        xyz = getattr(hyp, "xyz", None)
        if xyz is None:
            return "unknown"
        xy = (float(xyz[0]), float(xyz[1]))
    except Exception:
        return "unknown"
    room_fn = getattr(gm, "graph_room_at_robot", None)
    if not callable(room_fn):
        return "unknown"
    try:
        policy = str(getattr(executor, "room_policy", "canonical") or "canonical")
        return coerce_room_label(room_fn(xy), room_policy=policy)
    except Exception:
        return "unknown"


def _frontier_nearby_labels(executor: AgenticEQAExecutor, hyp: Any, *, limit: int = 3) -> list[str]:
    """Top object labels near a frontier hyp (Hydra-lite semantic enrichment)."""
    gm = executor.graph_memory
    if gm is None:
        return []
    try:
        xyz = getattr(hyp, "xyz", None)
        if xyz is None:
            return []
        fxy = (float(xyz[0]), float(xyz[1]))
    except Exception:
        return []
    scored: list[tuple[float, str]] = []
    for node in list(getattr(gm, "_nodes", None) or []):
        if getattr(node, "is_frontier", False):
            continue
        # Do not use ``a or b`` — numpy xyz arrays are ambiguous in boolean context.
        nxyz = getattr(node, "xyz", None)
        if nxyz is None:
            nxyz = getattr(node, "centroid", None)
        if nxyz is None:
            continue
        try:
            nxy = (float(nxyz[0]), float(nxyz[1]))
            dist = ((nxy[0] - fxy[0]) ** 2 + (nxy[1] - fxy[1]) ** 2) ** 0.5
        except Exception:
            continue
        if dist > 3.0:
            continue
        for lab in list(getattr(node, "labels", None) or [])[:2]:
            s = str(lab).strip()
            if s and s.lower() != "frontier":
                scored.append((dist, s))
    scored.sort(key=lambda t: t[0])
    out: list[str] = []
    seen: set[str] = set()
    for _, lab in scored:
        key = lab.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(lab)
        if len(out) >= limit:
            break
    return out


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

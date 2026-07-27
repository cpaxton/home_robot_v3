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

if TYPE_CHECKING:
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

# Response format block. Constant string (with the tools block) so the routing
# system prompt is byte-identical across rounds and question banks — required
# for Qwen3-VL system-prefix KV cache hits.
_EQA_FORMAT_BLOCK = """\
# Response format
Respond with ONLY a JSON object (no other text):
{"tool_calls": [{"name": "<tool>", "arguments": {...}}, ...], "message": ""}

Rules:
- Interactive loop: (1) explore / inspect for candidate places, (2) navigate in —
  the runtime captures, updates voxel+graph, and runs verify_siglip on the new view,
  (3) when Qwen says answerable → submit_answer with the MCQ letter; else move /
  explore. Never re-verify the same observation / view.
- SigLIP/OWL are proposals shown in state — not proof. Trust Qwen's assess and router.
- Pass the MCQ letter (A–D) in submit_answer.arguments.answer.
- Evidence cards list candidate obs_id values. Pick among them from the evidence;
  do not treat retrieval order as a command. navigate_to_obs only accepts listed ids.
- source=graph / confirmed / siglip cards are place or object evidence. source=frontier
  cards are unexplored coverage goals (phrase is not an object sighting) — prefer a
  graph place card when looking for a named object; use explore_frontier to grow map.
- If a hypothesis was ABSENT / STALLED_NAV_LOOP / [tried] in the state, do NOT
  navigate_to_obs that obs_id again — explore_frontier or look_around instead.
- If state says NAV_LOOP, treat that as a bug signal: switch to explore_frontier.
- One or two tool calls per turn.

# Examples
State: evidence obs_id=3 phrase='sink' source=graph; obs_id=12 phrase='unexplored frontier' source=frontier
{"tool_calls": [{"name": "navigate_to_obs", "arguments": {"obs_id": 3}}], "message": ""}
State: Last verify PRESENT; VLM assess answerable=true verified=true
{"tool_calls": [{"name": "submit_answer", "arguments": {"answer": "B"}}], "message": ""}
State: no hypotheses, 3 unexplored frontiers
{"tool_calls": [{"name": "explore_frontier", "arguments": {"toward": "sink"}}], "message": ""}
State: NAV_LOOP on obs_id=16
{"tool_calls": [{"name": "explore_frontier", "arguments": {"toward": "large wall clock"}}], "message": ""}"""

_EQA_IDENTITY = """\
You are a robot exploring a home. You maintain a 3D map and an object scene graph that
update automatically after every motion. Your job is to answer a question about the scene
(or to explore and map it). Move to where the answer can be seen, run verify_siglip
(cheap check + VLM assess), and only then answer. Do NOT output reasoning — only the JSON."""


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
    """Per-round user message: goal + graph stats + annotated hypotheses + verify + budgets."""
    gm = executor.graph_memory
    lines: list[str] = []
    if getattr(executor, "mode", "answer") == "explore":
        lines.append(f"Goal: {executor.goal or 'explore and map the environment'}")
    else:
        lines.append(f"Question: {executor.question}")
    lines.append(_graph_stats_line(gm))
    # Compact memory / graph text so the router is not blind vs classic query_answer.
    if gm is not None:
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
            except Exception:
                used_spatial = False
        if not used_spatial:
            mem_fn = getattr(gm, "_relevant_memory_summary", None)
            if callable(mem_fn) and bool(getattr(gm, "memory_summary_enabled", False)):
                try:
                    mem = mem_fn()
                except Exception:
                    mem = None
                if mem:
                    # Cap so tool-router context stays small.
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
    if getattr(executor, "_last_capture_status", None):
        lines.append(f"Last capture: {executor._last_capture_status}")
    loop_flags = list(getattr(executor, "_nav_loop_flags", None) or [])
    if loop_flags:
        last = loop_flags[-1]
        lines.append(
            f"NAV_LOOP: obs_id={last.get('obs_id')} visits={last.get('visits')} "
            f"status={last.get('status')} — do not re-navigate; explore_frontier"
        )
    if executor._hypotheses:
        lines.append("Evidence (candidate places — pick among listed obs_id values):")
        for h in executor._hypotheses:
            tried = executor._tried.get(int(h.obs_id))
            visits = int(getattr(executor, "_nav_to_obs_counts", {}).get(int(h.obs_id), 0))
            mark = f" [tried: {tried}]" if tried else ""
            if visits:
                mark += f" [nav_visits={visits}]"
            labels = _hyp_labels(executor, int(h.obs_id))
            label_bit = f" labels={labels}" if labels else ""
            sim = getattr(h, "siglip_sim", None)
            sim_bit = f" siglip_sim={float(sim):.3f}" if sim is not None else ""
            lines.append(
                f"- obs_id={int(h.obs_id)} phrase={h.phrase!r} source={h.source} "
                f"xyz=({float(h.xyz[0]):.1f},{float(h.xyz[1]):.1f})"
                f"{label_bit}{sim_bit}{mark}"
            )
    else:
        lines.append("Evidence: (none — explore or look_around first)")
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

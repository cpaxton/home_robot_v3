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
- Verify before answering: call verify_siglip at the location you believe holds the target,
  and only call submit_answer after a PRESENT verification (or when the state says budget
  is nearly exhausted).
- If the graph hypothesis was ABSENT at its old location, the object moved: explore_frontier
  or look_around to find it, then verify again.
- Never re-pick a hypothesis marked tried/ABSENT in the state.
- One or two tool calls per turn.

# Examples
State: hypothesis obs_id=7 'sink' from graph, not tried
{"tool_calls": [{"name": "navigate_to_obs", "arguments": {"obs_id": 7}}], "message": ""}
State: just arrived at obs 7
{"tool_calls": [{"name": "verify_siglip", "arguments": {"phrase": "sink", "obs_id": 7}}], "message": ""}
State: verify PRESENT sim=0.31
{"tool_calls": [{"name": "submit_answer", "arguments": {"answer": "B"}}], "message": ""}
State: no hypotheses, 3 unexplored frontiers
{"tool_calls": [{"name": "explore_frontier", "arguments": {"toward": "sink"}}], "message": ""}"""

_EQA_IDENTITY = """\
You are a robot exploring a home. You maintain a 3D map and an object scene graph that
update automatically after every motion. Your job is to answer a question about the scene
(or to explore and map it). Move to where the answer can be seen, verify with the
verify_siglip tool, and only then answer. Do NOT output reasoning — only the JSON."""


def build_agentic_eqa_tools(executor: AgenticEQAExecutor) -> list[Tool]:
    """EQA_EPISODE tool pack. Schemas/names come from :mod:`emet.agent.skills`; funcs dispatch via ``handle_tool``."""
    submode = getattr(executor, "mode", "answer")
    return build_skill_pack(AgentMode.EQA_EPISODE, executor, eqa_submode=submode)


def build_graph_eqa_system_prompt(tools: list[Tool]) -> str:
    """Fixed routing system prompt (identity + tools + format). Keep byte-stable per mode."""
    tools_block = get_tool_descriptions_for_prompt(tools)
    return f"{_EQA_IDENTITY}\n\n{tools_block}\n\n{_EQA_FORMAT_BLOCK}"


def _graph_stats_line(gm: Any) -> str:
    try:
        nodes = list(gm.get_nodes()) if gm is not None else []
    except Exception:
        nodes = []
    n_obj = sum(
        1 for n in nodes if not getattr(n, "is_frontier", False) and not getattr(n, "is_viewpoint", False)
    )
    n_frontier = sum(1 for n in nodes if getattr(n, "is_frontier", False))
    return f"Graph: {n_obj} object nodes, {n_frontier} frontier nodes."


def build_state_message(executor: AgenticEQAExecutor) -> str:
    """Per-round user message: goal + graph stats + annotated hypotheses + verify + budgets."""
    gm = executor.graph_memory
    lines: list[str] = []
    if getattr(executor, "mode", "answer") == "explore":
        lines.append(f"Goal: {executor.goal or 'explore and map the environment'}")
    else:
        lines.append(f"Question: {executor.question}")
    lines.append(_graph_stats_line(gm))
    lines.append(
        f"Round {executor._round + 1}/{executor.max_rounds}; "
        f"nav used {executor._n_nav + executor._n_explore}/{executor.max_nav_steps}; "
        f"verified={executor._verified}"
    )
    if executor._hypotheses:
        lines.append("Hypotheses:")
        for h in executor._hypotheses[:5]:
            tried = executor._tried.get(int(h.obs_id))
            mark = f" [tried: {tried}]" if tried else ""
            lines.append(
                f"- obs_id={int(h.obs_id)} phrase={h.phrase!r} source={h.source} "
                f"xyz=({float(h.xyz[0]):.1f},{float(h.xyz[1]):.1f}) score={float(h.score):.2f}{mark}"
            )
    else:
        lines.append("Hypotheses: (none — explore or look_around first)")
    if executor._last_verify is not None:
        lv = executor._last_verify
        lines.append(f"Last verify: {lv.status} sim={float(lv.sim):.3f} obs_id={int(lv.obs_id)}")
    if executor.max_rounds - executor._round <= 2:
        lines.append("Budget nearly exhausted: answer/finish on your best evidence soon.")
    return "\n".join(lines)

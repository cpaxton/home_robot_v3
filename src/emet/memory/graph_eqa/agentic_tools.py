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

from emet.agent.tools import Tool, get_tool_descriptions_for_prompt

if TYPE_CHECKING:
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

_NO_PARAMS: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

# Response format block. Constant string (with the tools block) so the routing
# system prompt is byte-identical across rounds and question banks — required
# for Qwen3-VL system-prefix KV cache hits.
_EQA_FORMAT_BLOCK = """\
# Response format
Respond with ONLY a JSON object (no other text):
{"tool_calls": [{"name": "<tool>", "arguments": {...}}, ...], "message": ""}

Rules:
- Interactive loop: (1) explore / inspect for candidate places, (2) navigate in and
  verify_siglip once on the new view (cheap proposal + Qwen assess), (3) when Qwen
  says answerable → submit_answer with the MCQ letter; else move / explore. Never
  re-verify the same observation / view.
- SigLIP/OWL are proposals shown in state — not proof. Trust Qwen's assess and router.
- Pass the MCQ letter (A–D) in submit_answer.arguments.answer.
- If a hypothesis was ABSENT at its old location, explore_frontier or look_around,
  then verify a new view.
- Never re-pick a hypothesis marked tried/ABSENT in the state.
- One or two tool calls per turn.

# Examples
State: hypothesis obs_id=7 'sink' from graph, not tried
{"tool_calls": [{"name": "navigate_to_obs", "arguments": {"obs_id": 7}}], "message": ""}
State: just arrived at obs 7
{"tool_calls": [{"name": "verify_siglip", "arguments": {"phrase": "sink", "obs_id": 7}}], "message": ""}
State: VLM assess answerable=true verified=true
{"tool_calls": [{"name": "submit_answer", "arguments": {"answer": "B"}}], "message": ""}
State: no hypotheses, 3 unexplored frontiers
{"tool_calls": [{"name": "explore_frontier", "arguments": {"toward": "sink"}}], "message": ""}"""

_EQA_IDENTITY = """\
You are a robot exploring a home. You maintain a 3D map and an object scene graph that
update automatically after every motion. Your job is to answer a question about the scene
(or to explore and map it). Move to where the answer can be seen, run verify_siglip
(cheap check + VLM assess), and only then answer. Do NOT output reasoning — only the JSON."""


def build_agentic_eqa_tools(executor: AgenticEQAExecutor) -> list[Tool]:
    """Tools the routing VLM may call. All funcs dispatch through executor.handle_tool."""
    mode = getattr(executor, "mode", "answer")

    def _dispatch(name: str):
        def _fn(**kwargs: Any) -> str:
            out = executor.handle_tool(name, kwargs)
            return str(out)

        return _fn

    tools: list[Tool] = [
        Tool(
            name="inspect_graph",
            description=(
                "Refresh question keywords and ranked navigation hypotheses from the scene graph "
                "and SigLIP memory. Use when hypotheses look stale or empty."
            ),
            parameters=_NO_PARAMS,
            func=_dispatch("inspect_graph"),
            returns_info=True,
        ),
        Tool(
            name="explore_frontier",
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
            func=_dispatch("explore_frontier"),
        ),
        Tool(
            name="navigate_to_obs",
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
            func=_dispatch("navigate_to_obs"),
        ),
        Tool(
            name="look_around",
            description=(
                "Scan in place (head sweep / rotate) to refresh the map and graph at the current "
                "pose without navigating."
            ),
            parameters=_NO_PARAMS,
            func=_dispatch("look_around"),
        ),
        Tool(
            name="verify_siglip",
            description=(
                "Cheap visual check + multimodal VLM assess: does 'phrase' match the current "
                "camera view / stored view obs_id? Returns PRESENT / CANDIDATE / ABSENT as a "
                "proposal; only VLM answerable unlocks submit_answer."
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
            func=_dispatch("verify_siglip"),
            returns_info=True,
        ),
    ]
    if mode == "explore":
        tools.append(
            Tool(
                name="finish",
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
                func=_dispatch("finish"),
            )
        )
    else:
        tools.append(
            Tool(
                name="submit_answer",
                description=(
                    "Submit the final answer (MCQ letter or short phrase). Rejected until VLM "
                    "assess marks the view answerable (or the round budget is exhausted)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "Final answer letter or phrase."}
                    },
                    "required": [],
                },
                func=_dispatch("submit_answer"),
            )
        )
    return tools


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
        if getattr(executor, "_target_phrase", ""):
            lines.append(
                f"Target: {executor._target_phrase} (type={getattr(executor, '_question_type', 'other')})"
            )
    lines.append(_graph_stats_line(gm))
    lines.append(
        f"Round {executor._round + 1}/{executor.max_rounds}; "
        f"nav used {executor._n_nav + executor._n_explore}/{executor.max_nav_steps}; "
        f"verified={executor._verified}; "
        f"policy={getattr(executor._evidence_policy, 'state', None)}"
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
        lines.append(
            f"Last verify (proposal): {lv.status} sim={float(lv.sim):.3f} "
            f"obs_id={int(lv.obs_id)}"
        )
    last_vlm = getattr(executor, "_last_vlm_assess", None)
    if last_vlm:
        sug = last_vlm.get("suggested_answer")
        lines.append(
            "Last Qwen assess: "
            f"answerable={last_vlm.get('answerable')} present={last_vlm.get('present')} "
            f"need_more_views={last_vlm.get('need_more_views')} "
            f"suggested={sug!r} obs_id={last_vlm.get('obs_id')} "
            f"reason={last_vlm.get('reason')!r}"
        )
    if executor.max_rounds - executor._round <= 2:
        lines.append("Budget nearly exhausted: answer/finish on your best evidence soon.")
    return "\n".join(lines)

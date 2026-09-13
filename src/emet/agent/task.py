"""Bounded autonomous task mode for the shared CHAT agent.

Uses the same model invocation, response parser, Tool objects and dispatcher as
interactive chat. Environment adapters supply observations, never evaluator goals.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable

TASK_INSTRUCTION = (
    "Carry out the user's whole task using the available tools. Tools execute in the order you request. "
    "Use actual observations and tool results; wait for returned plan handles before using them. Continue with the next "
    "necessary action; recover from errors by inspecting and replanning. Never invent "
    "object locations or plan handles. Output ONLY one JSON object in this exact format: "
    '{"tool_calls":[{"name":"TOOL_NAME","arguments":{}}],"message":""}. '
    "Replace TOOL_NAME and arguments with an available tool and its parameters. No prose or markdown. "
    "When the whole task is finished, return an empty tool_calls list and a final message."
)


def repair_missing_delimiters(raw: str) -> str:
    """Repair missing closing punctuation only; never invent keys or values.

    Small local models sometimes emit `arguments:{...}]` instead of `...}}]`.
    The original response stays recorded and any repair is an explicit event.
    """
    text = raw.strip().removeprefix("```json").removesuffix("```").strip()
    if not text.startswith("{"):
        return raw
    stack, output = [], []
    quoted = escaped = False
    for char in text:
        if quoted:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if char not in stack:
                return raw
            while stack[-1] != char:
                output.append(stack.pop())
            stack.pop()
        output.append(char)
    if quoted:
        return raw
    output.extend(reversed(stack))
    candidate = "".join(output)
    try:
        json.loads(candidate)
    except ValueError:
        return raw
    return candidate


def run_agent_task(
    instruction: str,
    *,
    llm_client,
    tools: list,
    executor=None,
    robot=None,
    max_rounds: int = 24,
    max_actions: int = 32,
    max_tools_per_round: int = 32,
    timeout_s: float = 600,
    observe: Callable | None = None,
    on_event: Callable | None = None,
    cancelled: Callable | None = None,
) -> dict:
    """Execute one task; completion is a model claim, independent of task scoring.

    Time is checked between calls. Callers must isolate this in an owned process
    to enforce a hard deadline during native/model calls, as the benchmark does.
    The observer returns policy-visible JSON and optionally an actual RGB image.
    """
    from emet.agent.loop import _call_llm, _dispatch_tool_calls
    from emet.agent.prompt import _first_json_dict, parse_tool_calls_response

    if (
        not 1 <= max_rounds <= 128
        or not 1 <= max_actions <= 256
        or not 1 <= max_tools_per_round <= 32
        or timeout_s <= 0
    ):
        raise ValueError("invalid task execution budget")
    names = {t.name: t for t in tools}
    if len(names) != len(tools):
        raise ValueError("duplicate task tool name")
    start = time.monotonic()
    actions = 0
    rounds = 0
    results = []
    invalid_responses = 0

    def emit(kind, **payload):
        if on_event:
            on_event(kind, payload)

    def finish(status, message=""):
        result = {
            "status": status,
            "message": message,
            "rounds": rounds,
            "actions": actions,
            "elapsed_s": time.monotonic() - start,
        }
        emit("agent_finish", **result)
        return result

    if hasattr(llm_client, "reset"):
        llm_client.reset()
    for round_index in range(max_rounds):
        if cancelled and cancelled():
            return finish("cancelled")
        if time.monotonic() - start >= timeout_s:
            return finish("timeout")
        state, image = observe() if observe else ({}, None)
        prompt = (
            (
                TASK_INSTRUCTION + "\nTask: " + instruction
                if round_index == 0
                else "Continue the same task. Use the new observations and actual tool results below to choose the NEXT action. "
                "Do not restart actions already completed.\nTask: " + instruction
            )
            + "\nCurrent observations: "
            + json.dumps(state, sort_keys=True)
            + "\nLatest tool results: "
            + json.dumps(results)
        )
        rounds = round_index + 1
        emit(
            "model_input",
            round=rounds,
            text=prompt,
            observation=state,
            model_input_observation_ids=state.get("image_observation_ids", []),
            has_image=image is not None,
            reset_context=round_index == 0,
        )
        # Use the shared client's actual conversation lifecycle, as chat does.
        raw, elapsed = _call_llm(
            llm_client, prompt, None, False, image=image, robot=robot, reset_context=round_index == 0
        )
        emit("model_output", round=rounds, raw=raw, elapsed_s=elapsed)
        parsed = parse_tool_calls_response(raw)
        calls = parsed.get("tool_calls") or []
        if not calls:
            repaired = repair_missing_delimiters(raw)
            if repaired != raw and parse_tool_calls_response(repaired).get("tool_calls"):
                emit("response_repair", original=raw, repaired=repaired, kind_of_repair="closing_punctuation_only")
                parsed = parse_tool_calls_response(repaired)
                calls = parsed["tool_calls"]
        if not calls:
            # Accept complete leading actions from a broken/truncated batch.
            # No argument is repaired or inferred; later incomplete actions wait.
            marker = re.search(r'"tool_calls"\s*:\s*\[', raw)
            if marker:
                try:
                    remaining = raw[marker.end() :].lstrip()
                    complete = []
                    while remaining and remaining[0] == "{" and len(complete) < 32:
                        first, consumed = json.JSONDecoder().raw_decode(remaining)
                        if not (
                            isinstance(first, dict)
                            and first.get("name") in names
                            and isinstance(first.get("arguments", {}), dict)
                        ):
                            break
                        complete.append(first)
                        remaining = remaining[consumed:].lstrip()
                        # Ignore extra closing braces only between complete array items.
                        remaining = remaining.lstrip("}").lstrip()
                        if remaining.startswith(","):
                            remaining = remaining[1:].lstrip()
                        else:
                            break
                except ValueError:
                    pass
                if marker and complete:
                    calls = complete
                    emit(
                        "response_repair",
                        original=raw,
                        repaired=json.dumps({"tool_calls": calls}),
                        kind_of_repair="first_complete_tool_only"
                        if len(calls) == 1
                        else "complete_tools_from_broken_batch",
                    )
        if not calls:
            # The permissive chat parser treats arbitrary text as a final answer.
            # In task mode malformed JSON is recoverable, not a completion signal.
            try:
                obj = _first_json_dict(raw)
                explicit = isinstance(obj, dict) and obj.get("tool_calls") == []
            except (ValueError, AttributeError):
                explicit = False
            if explicit:
                return finish("model_finished", str(parsed.get("message") or ""))
            invalid_responses += 1
            if invalid_responses >= 3:
                return finish(
                    "model_protocol_failed", "Three consecutive responses had no valid action or finish object"
                )
            results = ["Invalid response. Return JSON with tool_calls and message; use exact available tool names."]
            continue
        invalid_responses = 0
        results = []
        if len(calls) > max_tools_per_round:
            emit("deferred_tools", tools=calls[max_tools_per_round:])
            results.append(
                f"At most {max_tools_per_round} actions were executed. Later proposed actions were deferred; choose again from the new observation."
            )
        for call in calls[:max_tools_per_round]:
            if cancelled and cancelled():
                return finish("cancelled")
            if time.monotonic() - start >= timeout_s:
                return finish("timeout")
            if actions >= max_actions:
                return finish("action_budget_exhausted")
            actions += 1
            command_id = f"command:{actions}"
            emit("tool_start", command_id=command_id, tool=call)
            keep_going, outcome, _ = _dispatch_tool_calls([call], names, executor)
            results.extend(outcome)
            emit("tool_result", command_id=command_id, tool=call, result=outcome)
            if not keep_going:
                return finish("cancelled", "tool requested stop")
    return finish("round_budget_exhausted")

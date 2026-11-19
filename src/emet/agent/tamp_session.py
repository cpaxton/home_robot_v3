# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded CHAT continuation policy for structured manipulation outcomes."""

from emet.agent.tool_outcome import ToolOutcome, retained_outcomes
from emet.controller.task.tamp.api import TAMP_TOOLS


def serial_tamp_calls(tool_calls: list[dict]) -> tuple[list[dict], list[dict]]:
    """Defer later proposals in all-TAMP batches until the first result is seen.

    Planning produces fresh handles that cannot be known by a precomputed
    execution call. Deferred proposals are logged, never automatically queued.
    Mixed capability batches retain the dispatcher's existing stop semantics.
    """
    if len(tool_calls) > 1 and all(call.get("name") in TAMP_TOOLS for call in tool_calls):
        return tool_calls[:1], tool_calls[1:]
    return tool_calls, []


def tamp_followup(user_goal: str, results: list[ToolOutcome] | list[str]) -> str | None:
    """Continue only a wholly successful TAMP batch; otherwise keep summarize-only.

    Failures return to the user for inspection or an explicit recovery turn. In
    particular, a partial/unknown execution must never gain autonomous retries.
    """
    if not results:
        return None
    outcomes = retained_outcomes(results)
    for outcome in outcomes:
        if outcome.tool not in TAMP_TOOLS:
            return None
        if not outcome.ok:
            return None
        if outcome.payload.get("data", {}).get("discovery_only") is True:
            # A proposal provider cannot authorize a motion-capable next step.
            return None
    return (
        f"Original user request:\n{user_goal}\n\n[Tool results]\n"
        + "\n".join(outcome.render() for outcome in outcomes)
        + "\n\nContinue only unfinished steps requested by the user. If the request was "
        "only to discover or plan, summarize and stop; do not execute without authorization. "
        "For an authorized multi-step task, return at most one next tool call. Copy handles "
        "exactly; after execution rediscover before planning the next task. Never repeat a "
        "consumed plan or silently drop goal constraints. If all requested work is complete, "
        "summarize the measured outcomes and stop."
    )

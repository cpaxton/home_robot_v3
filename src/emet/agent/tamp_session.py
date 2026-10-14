# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded CHAT continuation policy for structured manipulation outcomes."""
from emet.agent.tool_outcome import ToolOutcome
from emet.controller.task.tamp.api import TAMP_TOOLS


def tamp_followup(user_goal: str, results: list[str]) -> str | None:
    """Continue only a wholly successful TAMP batch; otherwise keep summarize-only.

    Failures return to the user for inspection or an explicit recovery turn. In
    particular, a partial/unknown execution must never gain autonomous retries.
    """
    if not results:
        return None
    for line in results:
        prefix, sep, body = line.partition('] ')
        name = prefix.removeprefix('[')
        if not sep or name not in TAMP_TOOLS:
            return None
        outcome = ToolOutcome.coerce(name, body)
        if not outcome.ok:
            return None
        if outcome.payload.get('data', {}).get('discovery_only') is True:
            # A proposal provider cannot authorize a motion-capable next step.
            return None
    return (
        f'Original user request:\n{user_goal}\n\n[Tool results]\n' + '\n'.join(results)
        + '\n\nContinue only unfinished steps requested by the user. If the request was '
        'only to discover or plan, summarize and stop; do not execute without authorization. '
        'For an authorized multi-step task, return at most one next tool call. Copy handles '
        'exactly; after execution rediscover before planning the next task. Never repeat a '
        'consumed plan or silently drop goal constraints. If all requested work is complete, '
        'summarize the measured outcomes and stop.'
    )

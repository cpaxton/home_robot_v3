# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Offline failure-boundary checks on real CHAT dispatch logs."""
from emet.agent.tool_outcome import ToolOutcome
from emet.controller.task.tamp.api import TAMP_TOOLS


def failure_boundaries(rows):
    """A failed TAMP result forbids more dispatched tools until a new user turn.

    Assistant prose or proposed calls are not execution evidence: only tool log
    entries count. This checks session control, never physical task success.
    """
    failures, active = [], None
    for row in rows:
        if row.get('role') == 'user':
            active = None
        if row.get('role') != 'tool':
            continue
        line = str(row.get('content', ''))
        prefix, sep, body = line.partition('] ')
        name = prefix.removeprefix('[') if sep else 'unknown'
        # Dispatcher bookkeeping is not a second executed tool.
        if name == 'dispatch':
            continue
        if active is not None:
            active['unexpected_tools'].append(name)
        if name in TAMP_TOOLS:
            outcome = ToolOutcome.coerce(name, body)
            if not outcome.ok:
                failure = {'tool': name, 'code': outcome.status, 'unexpected_tools': []}
                failures.append(failure)
                if active is None:
                    active = failure
    return failures

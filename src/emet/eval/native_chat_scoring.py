# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Offline failure-boundary checks on real CHAT dispatch logs."""
from emet.agent.tool_outcome import ToolOutcome
from emet.controller.task.tamp.api import TAMP_TOOLS


def integration_evidence(rows):
    """Score dispatched-result contracts independently of robot task success.

    A correct stop after a failed grasp passes the stop check, not the task.
    Empty traces and unexercised failure paths remain unassessed (JSON null).
    This does not check task interpretation, handle arguments, or goal coverage.
    """
    rows = list(rows)
    counts = {'ok': 0, 'error': 0, 'partial': 0, 'invalid': 0}
    codes = {}
    for row in rows:
        if row.get('role') != 'tool':
            continue
        prefix, sep, body = str(row.get('content', '')).partition('] ')
        name = prefix.removeprefix('[')
        if not sep or name not in TAMP_TOOLS:
            continue
        outcome = ToolOutcome.coerce(name, body)
        status = outcome.payload.get('status', 'invalid')
        counts[status if status in counts else 'invalid'] += 1
        codes[outcome.status] = codes.get(outcome.status, 0) + 1
    failures = failure_boundaries(rows)
    total = sum(counts.values())
    return {
        'tamp_calls': total,
        'outcome_counts': counts,
        'code_counts': codes,
        'result_contracts_valid': counts['invalid'] == 0 if total else None,
        'failure_stop_compliant': not any(f['unexpected_tools'] for f in failures) if failures else None,
        'failure_cases': len(failures),
        'failure_boundaries': failures,
        'unassessed': ['task_interpretation', 'goal_coverage', 'physical_recovery',
                       'observed_grounding', 'navigation', 'exploration'],
    }


def failure_boundaries(rows):
    """A failed TAMP result forbids more dispatched tools until a new user turn.

    Assistant prose or proposed calls are not execution evidence: only tool log
    entries count. This checks session control, never physical task success.
    Input must retain explicit user rows from ChatLog (including scripted turns).
    Every later dispatch in the failed turn is a violation, even another failure.
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

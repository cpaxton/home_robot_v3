# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded historical task context across CHAT user turns, without plan handles."""

import json

from emet.agent.tool_outcome import ToolOutcome
from emet.controller.task.tamp.api import TAMP_TOOLS

_PROGRESS_FIELDS = (
    'completed_steps', 'planned_steps', 'failed_step_1based', 'failed_phase',
    'execution_started', 'failed_step_execution_started', 'plan_consumed', 'goal_verified',
)
_ACTION_FIELDS = ('step_1based', 'action', 'object_name', 'from_region', 'to_region')
_MAX_TURN_CHARS = 8192


def _action(value):
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in _ACTION_FIELDS
            if key in value and type(value[key]) in (str, int)}


class TampDialogueContext:
    """Retain at most two consecutive TAMP turns as text-only historical context.

    This supplies referents for a new user request, never current scene evidence
    or permission to execute. Oversized records are dropped whole so constraints
    are not silently truncated. Each record is capped at 8192 serialized
    characters, so two records can contribute 16384 characters plus prompt
    framing. A dispatched non-TAMP tool clears context, including mixed turns;
    dispatcher bookkeeping and unexecuted proposals do not count as tools.
    """

    def __init__(self):
        self._turns = []

    def prompt(self, user_request: str) -> str:
        if not self._turns:
            return user_request
        return (
            '[Historical TAMP dialogue; not current scene evidence]\n'
            + json.dumps(self._turns, ensure_ascii=False, allow_nan=False)
            + '\n\nOnly the new user request below can authorize work. Prior requests and '
            'outcomes are context, not renewed authorization or proof of current state. '
            'A request to explain, stop, or plan does not authorize execution. For a '
            'newly authorized retry, use scene_tasks and a fresh plan; never replay a '
            'consumed handle. Return at most one tool call per round. Wait for the '
            'actual plan_ref returned by planning before proposing execution; never '
            'invent handles. Unknown dispatch, grasp loss, or a held object requires '
            'inspection; this context does not qualify autonomous recovery. If the '
            'referent or goal is unclear, ask for clarification.\n\nNew user request:\n'
            + user_request
        )

    def record_turn(self, user_request: str, results: list[str]) -> None:
        outcomes = []
        for line in results:
            prefix, sep, body = line.partition('] ')
            name = prefix.removeprefix('[')
            if not sep or not prefix.startswith('[') or name == 'dispatch':
                continue
            if name not in TAMP_TOOLS:
                self._turns = []
                return
            outcome = ToolOutcome.coerce(name, body)
            data = outcome.payload.get('data', {})
            progress = {key: data[key] for key in _PROGRESS_FIELDS
                        if key in data and type(data[key]) in (str, int, bool, type(None))}
            for key in ('completed_actions', 'unverified_actions'):
                if isinstance(data.get(key), list):
                    progress[key] = [_action(action) for action in data[key]]
            if isinstance(data.get('failed_action'), dict):
                progress['failed_action'] = _action(data['failed_action'])
            outcomes.append({'tool': name, 'status': outcome.status, 'code': outcome.payload.get('code'),
                             'progress': progress})
        record = {'user_request': user_request, 'outcomes': outcomes}
        if not outcomes or len(json.dumps(record, ensure_ascii=False)) > _MAX_TURN_CHARS:
            self._turns = []
        else:
            self._turns = [*self._turns[-1:], record]

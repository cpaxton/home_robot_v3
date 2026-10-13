# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""CHAT must observe a TAMP failure before deciding on further actions."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from emet.agent.loop import _dispatch_tool_calls
from emet.agent.tool_outcome import ToolOutcome, maybe_record_tool_attempt
from emet.controller.task.tamp.api import response


@pytest.mark.parametrize('status,code', [('partial', 'command_outcome_unknown'),
                                        ('error', 'scene_changed_replan'), ('ok', 'ok')])
@pytest.mark.parametrize('serialized', [False, True])
def test_json_outcome_uses_status_not_prose(status, code, serialized):
    result = response('pick_place', status=status, code=code, data={'completed_ops': ['grasp']})
    result['message'] = 'Everything succeeded'  # prose must never override status
    outcome = ToolOutcome.coerce('pick_place', json.dumps(result) if serialized else result)
    assert outcome.ok == (status == 'ok')
    assert outcome.status == code
    assert outcome.payload['data']['completed_ops'] == ['grasp']
    ledger = Mock()
    maybe_record_tool_attempt(ledger, outcome)
    assert ledger.record_attempt.call_args.kwargs['outcome'] != 'ok' or status == 'ok'


@pytest.mark.parametrize('result', ['not JSON', '[]', '{"schema_version": 2}', None,
    {'schema_version': 1, 'tool': 'execute_pick_place_plan', 'status': []}])
def test_invalid_tamp_outcome_is_not_success(result):
    assert not ToolOutcome.coerce('execute_pick_place_plan', result).ok


@pytest.mark.parametrize('status,code', [('partial', 'command_outcome_unknown'),
                                        ('error', 'scene_changed_replan'), ('ok', 'ok')])
def test_dispatch_stops_precomputed_batch_on_failure(status, code):
    execute = Mock(return_value=json.dumps(response('execute_pick_place_plan', status=status, code=code)))
    next_action = Mock(return_value='moved')
    tools = {name: SimpleNamespace(func=fn, returns_info=True, to_executor=lambda args: [])
             for name, fn in [('execute_pick_place_plan', execute), ('move_forward', next_action)]}
    keep_running, results, has_info = _dispatch_tool_calls([
        {'name': 'execute_pick_place_plan', 'arguments': {'plan_ref': 'plan:1'}},
        {'name': 'move_forward', 'arguments': {}}], tools, Mock())
    assert keep_running and has_info  # stop the batch, not the agent session
    assert next_action.call_count == (1 if status == 'ok' else 0)
    if status != 'ok':
        stopped = json.loads(results[-1].removeprefix('[dispatch] '))
        assert stopped['data']['unattempted_tools'] == ['move_forward']


def test_exception_stops_batch_but_allows_later_recovery_turn():
    failed = Mock(side_effect=TimeoutError('command ACK lost'))
    inspect = Mock(return_value=json.dumps(response('scene_tasks', data={'tasks': []})))
    tools = {name: SimpleNamespace(func=fn, returns_info=True, to_executor=lambda args: [])
             for name, fn in [('execute_pick_place_plan', failed), ('scene_tasks', inspect)]}
    keep_running, _, _ = _dispatch_tool_calls([
        {'name': 'execute_pick_place_plan', 'arguments': {'plan_ref': 'plan:1'}},
        {'name': 'scene_tasks', 'arguments': {}}], tools, Mock())
    assert keep_running
    inspect.assert_not_called()
    _dispatch_tool_calls([{'name': 'scene_tasks', 'arguments': {}}], tools, Mock())
    inspect.assert_called_once()

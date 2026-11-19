# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
from unittest.mock import Mock

import pytest

from emet.controller.task.tamp.task_session import TaskExecutionResult
from emet.eval.native_recovery import RefuseAfterOneTransfer, replan_after_boundary


def test_refusal_is_once_after_success_not_after_failed_attempt(tmp_path):
    provider = Mock()
    provider.execute.side_effect = [TaskExecutionResult(False, False),
                                   TaskExecutionResult(True, True), TaskExecutionResult(True, True)]
    wrapper = RefuseAfterOneTransfer(provider, tmp_path / 'audit.jsonl')
    assert not wrapper.execute('first').success
    assert wrapper.execute('retry').success
    receipt = wrapper.execute('second')
    assert receipt == TaskExecutionResult(False, False, 'task_motion_budget_exhausted')
    assert provider.execute.call_count == 2
    assert wrapper.records[0]['no_commands'] is True
    assert wrapper.execute('reauthorized').success
    assert len(wrapper.records) == 1


@pytest.mark.parametrize('stage', ['initial_boundary', 'consumed_handle', 'discovery', 'planning', None])
def test_recovery_stops_at_any_unexpected_boundary(stage):
    checks = [{'passed': stage != 'initial_boundary'}]
    events = []
    def reject(*args, **kwargs):
        events.append('reject')
        checks.append({'passed': stage != 'consumed_handle'})
    def call(name, **kwargs):
        events.append(name)
        return {'status': 'error' if (stage, name) in [('discovery', 'scene_tasks'), ('planning', 'plan_task')] else 'ok',
                'data': {'plan_ref': 'new'}}
    receipt = replan_after_boundary(call, reject, checks, 'move', 'consumed')
    assert (receipt is not None) == (stage is None)
    assert ('execute_task_plan' in events) == (stage is None)
    if stage == 'initial_boundary':
        assert events == []

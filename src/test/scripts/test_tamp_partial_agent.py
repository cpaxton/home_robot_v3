# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import copy
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'scripts'))
    return importlib.import_module('eval_tamp_partial_agent')


def example():
    done = {'step_1based': 1, 'action': 'transfer', 'object_name': 'alpha'}
    failed = {'step_1based': 2, 'action': 'transfer', 'object_name': 'beta'}
    return {'code': 'task_motion_budget_exhausted', 'data': {
        'completed_steps': 1, 'planned_steps': 2, 'failed_step_1based': 2,
        'failed_phase': 'preflight', 'execution_started': True,
        'failed_step_execution_started': False, 'plan_consumed': True, 'goal_verified': False,
        'completed_actions': [done], 'failed_action': failed, 'unverified_actions': [failed]}}


@pytest.mark.parametrize('change', [None, ('failed_step_execution_started', True),
    ('completed_steps', 0), ('failed_step_1based', 1), ('goal_verified', True),
    ('plan_consumed', False), ('unverified_actions', [])])
def test_recovery_admission_requires_exact_partial_feedback_and_physical_state(runner, change):
    receipt = example()
    if change:
        receipt['data'][change[0]] = change[1]
    predicates = {'body-a': {'satisfied': True, 'supported': True, 'settled': True},
                  'body-b': {'satisfied': False, 'supported': True, 'settled': True}}
    objects = {'alpha': 'body-a', 'beta': 'body-b'}
    assert runner.validate_partial(receipt, predicates, objects) is (change is None)
    predicates['body-b']['supported'] = False
    assert not runner.validate_partial(receipt, predicates, objects)


def test_feedback_must_arrive_after_the_partial_tool_result(runner, monkeypatch, tmp_path):
    import json
    partial = {'role': 'tool', 'content': '[execute_task_plan] '+json.dumps({'status': 'partial', **example()})}
    final = {'role': 'assistant', 'forced_final': True, 'content': 'One action completed; stopped.'}
    monkeypatch.setattr(runner, 'rows_at', lambda _: [partial, final])
    receipt, result = runner.wait_partial(SimpleNamespace(poll=lambda: None), tmp_path, float('inf'))
    assert receipt['data']['completed_steps'] == 1 and result is final
    monkeypatch.setattr(runner, 'rows_at', lambda _: [copy.deepcopy(partial)])
    with pytest.raises(RuntimeError, match='expected_partial_feedback_not_observed'):
        runner.wait_partial(SimpleNamespace(poll=lambda: 1), tmp_path, float('inf'))


def test_fault_bootstrap_requires_native_simulation(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'scripts'))
    bootstrap = importlib.import_module('tamp_partial_agent_bootstrap')
    monkeypatch.setattr('sys.argv', ['bootstrap', '--fault-record', str(tmp_path/'fault.json')])
    invoke = Mock()
    monkeypatch.setattr(bootstrap.runpy, 'run_module', invoke)
    with pytest.raises(SystemExit) as error:
        bootstrap.main()
    assert error.value.code == 2
    invoke.assert_not_called()

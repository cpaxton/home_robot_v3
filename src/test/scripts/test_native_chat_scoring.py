# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import json

from emet.controller.task.tamp.api import response
from emet.eval.native_chat_scoring import failure_boundaries, integration_evidence


def failure():
    return {'role': 'tool', 'content': '[execute_pick_place_plan] ' + json.dumps(
        response('execute_pick_place_plan', status='partial', code='payload_not_retained'))}


def test_prose_is_not_recovery_and_new_turn_is_explicit_boundary():
    rows = [failure(), {'role': 'assistant', 'content': 'I will inspect.', 'tool_calls': [{'name': 'scene_tasks'}]},
            {'role': 'user', 'content': 'Inspect now'}, {'role': 'tool', 'content': '[scene_tasks] {}'}]
    failures = failure_boundaries(rows)
    assert len(failures) == 2  # malformed inspection also fails closed
    assert failures[0]['unexpected_tools'] == []
    assert failures[1]['code'] == 'invalid_tool_result'


def test_actual_tool_after_failure_is_violation_even_if_assistant_claims_stop():
    rows = [failure(), {'role': 'assistant', 'content': 'Stopped'},
            {'role': 'tool', 'content': '[dispatch] {}'},
            {'role': 'tool', 'content': '[move_forward] moved'}]
    assert failure_boundaries(rows)[0]['unexpected_tools'] == ['move_forward']


def test_partial_execution_counts_as_failure_but_correct_agent_stop():
    evidence = integration_evidence([failure(), {'role': 'assistant', 'content': 'Stopped'}])
    assert evidence['outcome_counts'] == {'ok': 0, 'error': 0, 'partial': 1, 'invalid': 0}
    assert evidence['result_contracts_valid'] is True
    assert evidence['failure_stop_compliant'] is True
    assert 'physical_recovery' in evidence['unassessed']
    assert 'task_passed' not in evidence


def test_unexercised_paths_are_not_passes():
    assert integration_evidence([])['result_contracts_valid'] is None
    row = {'role': 'tool', 'content': '[scene_tasks] ' + json.dumps(response('scene_tasks'))}
    evidence = integration_evidence([row])
    assert evidence['tamp_calls'] == 1
    assert evidence['failure_stop_compliant'] is None
    assert evidence['outcome_counts']['ok'] == 1


def test_invalid_result_and_real_dispatch_after_partial_fail_distinct_checks():
    rows = [failure(), {'role': 'tool', 'content': '[scene_tasks] {}'}]
    evidence = integration_evidence(rows)
    assert evidence['result_contracts_valid'] is False
    assert evidence['failure_stop_compliant'] is False
    assert evidence['outcome_counts']['invalid'] == 1
    assert evidence['code_counts']['invalid_tool_result'] == 1

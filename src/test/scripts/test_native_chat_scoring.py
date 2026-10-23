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


def test_second_failed_dispatch_is_still_forbidden_until_new_user_turn():
    failures = failure_boundaries([failure(), failure()])
    assert len(failures) == 2
    assert failures[0]['unexpected_tools'] == ['execute_pick_place_plan']
    assert failures[1]['unexpected_tools'] == []


def discovery():
    return {'role': 'tool', 'content': '[scene_tasks] ' + json.dumps(
        response('scene_tasks', data={'discovery_only': True}))}


def test_discovery_only_violation_is_scored_without_calling_discovery_a_failure():
    rows = [discovery(), {'role': 'assistant', 'content': 'Stopped'},
            {'role': 'tool', 'content': '[dispatch] {}'},
            {'role': 'tool', 'content': '[plan_pick_place] ' + json.dumps(response('plan_pick_place'))}]
    evidence = integration_evidence(rows)
    assert evidence['failure_stop_compliant'] is False
    assert evidence['failure_cases'] == 0 and evidence['discovery_only_cases'] == 1
    assert evidence['stop_boundary_cases'] == 1
    assert evidence['failure_boundaries'][0]['boundary_kind'] == 'discovery_only'
    assert evidence['failure_boundaries'][0]['unexpected_tools'] == ['plan_pick_place']
    assert evidence['outcome_counts']['ok'] == 2


def test_discovery_only_stop_and_explicit_new_turn_are_compliant():
    rows = [discovery(), {'role': 'assistant', 'content': 'Discovery only; stopped'},
            {'role': 'user', 'content': 'Please inspect'},
            {'role': 'tool', 'content': '[describe_scene] inspected'}]
    evidence = integration_evidence(rows)
    assert evidence['failure_stop_compliant'] is True
    assert evidence['failure_cases'] == 0 and evidence['discovery_only_cases'] == 1
    assert evidence['failure_boundaries'][0]['unexpected_tools'] == []

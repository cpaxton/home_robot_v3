# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Cross-turn context retains goals/progress without retaining executable handles."""

import json

from emet.agent.tamp_dialogue import TampDialogueContext
from emet.controller.task.tamp.api import response


def partial():
    return response('execute_task_plan', status='partial', code='task_motion_budget_exhausted', data={
        'completed_steps': 1, 'planned_steps': 2, 'failed_step_1based': 2, 'failed_phase': 'preflight',
        'execution_started': True, 'failed_step_execution_started': False,
        'plan_consumed': True, 'goal_verified': False,
        'completed_actions': [{'object_name': 'alpha', 'action': 'transfer', 'to_region': 'left'}],
        'failed_action': {'object_name': 'beta', 'action': 'transfer', 'to_region': 'right'},
        'unverified_actions': [{'object_name': 'beta', 'action': 'transfer', 'to_region': 'right'}],
        'plan_ref': 'consumed-secret', 'body_id': 'private-body', 'image': 'old-base64'})


def line(value):
    return '['+value['tool']+'] '+json.dumps(value)


def test_null_progress_survives_and_only_dispatched_non_tamp_clears_context():
    context = TampDialogueContext()
    done = response('execute_task_plan', data={'failed_step_execution_started': None, 'failed_step_1based': None})
    context.record_turn('Move α', [line(done), '[dispatch] batch_stopped'])
    prompt = context.prompt('Explain')
    assert 'Move α' in prompt and '"failed_step_execution_started": null' in prompt
    context.record_turn('Mixed', [line(done), '[describe_scene] room'])
    assert context.prompt('Next') == 'Next'


def test_context_preserves_named_partial_progress_but_no_handles_or_sensor_data():
    context = TampDialogueContext()
    request = 'Move alpha left and beta right; leave gamma alone'
    assert context.prompt(request) == request
    context.record_turn(request, [line(partial())])
    prompt = context.prompt('Please retry')
    for text in (request, 'alpha', 'beta', 'preflight', 'failed_step_execution_started',
                 'not current scene evidence', 'Only the new user request', 'fresh plan'):
        assert text in prompt
    for text in ('consumed-secret', 'private-body', 'old-base64', '"plan_ref":'):
        assert text not in prompt
    assert prompt.endswith('New user request:\nPlease retry')


def test_unknown_dispatch_is_preserved_without_becoming_known_no_command():
    receipt = partial()
    receipt['code'] = 'command_outcome_unknown'
    receipt['data']['failed_step_execution_started'] = True
    context = TampDialogueContext()
    context.record_turn('Move both', [line(receipt)])
    prompt = context.prompt('Explain what happened')
    assert 'command_outcome_unknown' in prompt
    assert '"failed_step_execution_started": true' in prompt
    assert 'requires inspection' in prompt
    assert 'does not qualify autonomous recovery' in prompt


def test_context_is_bounded_and_drops_whole_records_instead_of_goal_constraints():
    context = TampDialogueContext()
    for request in ('first', 'second', 'third'):
        context.record_turn(request, [line(partial())])
    assert 'first' not in context.prompt('next')
    assert 'second' in context.prompt('next') and 'third' in context.prompt('next')
    context.record_turn('A'*8193+'; do not move gamma', [line(partial())])
    assert context.prompt('retry') == 'retry'
    context.record_turn('new task', [line(partial())])
    context.record_turn('unrelated question', ['[describe_scene] a room'])
    assert context.prompt('retry') == 'retry'

# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import json

from emet.controller.task.tamp.api import response
from emet.eval.native_chat_scoring import failure_boundaries


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

# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Exercise the real CHAT loop's bounded continuation and hard failure boundary."""
import json
from unittest.mock import MagicMock

import pytest

from emet.agent import loop
from emet.agent.tamp_session import tamp_followup
from emet.agent.tools import Tool
from emet.controller.task.tamp.api import response


@pytest.mark.parametrize('status,code', [('partial', 'command_outcome_unknown'),
    ('error', 'unknown_plan'), ('error', 'scene_changed_replan')])
def test_failed_outcome_never_autocontinues(status, code):
    result = response('execute_pick_place_plan', status=status, code=code)
    assert tamp_followup('move both', ['[execute_pick_place_plan] ' + json.dumps(result)]) is None


@pytest.mark.parametrize('line', ['garbage', '[scene_tasks] []', '[scene_tasks] {}', '[dispatch] {}'])
def test_malformed_result_stops(line):
    assert tamp_followup('move both', [line]) is None


@pytest.mark.parametrize('partial,budget,expected', [(False, 6, 2), (True, 6, 1), (False, 1, 1)])
def test_real_chat_loop_continues_success_but_cannot_dispatch_final_tools(monkeypatch, partial, budget, expected):
    robot = MagicMock()
    robot.get_observation.return_value = None
    executor = MagicMock()
    executor.robot = robot
    executor.agent.robot = robot
    executor.agent.graph_memory = None
    executor._last_memory_save_path = None
    monkeypatch.setattr(loop, 'StretchZmqClient', lambda **kw: robot)
    monkeypatch.setattr(loop, 'DynamemTaskExecutor', lambda *a, **kw: executor)
    monkeypatch.setattr(loop, 'get_memory_backend', lambda *a, **kw: MagicMock())
    monkeypatch.setattr(loop, 'ChatLog', MagicMock())
    monkeypatch.setattr(loop, 'print_memory_view_help_on_quit', lambda *a: None)
    monkeypatch.setattr(loop, 'print_embodied_model_report', lambda **kw: None)
    monkeypatch.setattr(loop, 'get_llm_client', lambda *a, **kw: MagicMock())
    execute = MagicMock(return_value=json.dumps(response('execute_pick_place_plan',
        status='partial' if partial else 'ok', code='command_outcome_unknown' if partial else 'ok')))
    tool = Tool(name='execute_pick_place_plan', description='test',
                parameters={'type': 'object', 'properties': {}}, func=execute, returns_info=True)
    monkeypatch.setattr(loop, 'get_tools', lambda ctx: [tool])
    call = json.dumps({'tool_calls': [{'name': tool.name, 'arguments': {}}], 'message': ''})
    # Even a hostile final summarizer returning another action must not actuate.
    llm = MagicMock(side_effect=[(call, .1), (call, .1), ('{"message":"done","tool_calls":[]}', .1)])
    monkeypatch.setattr(loop, '_call_llm', llm)
    loop.run_agent_with_robot(discord=False, use_llm=True, commands=['Move both objects'],
                              device='cpu', thinking_status=False, max_tool_rounds=budget)
    assert execute.call_count == expected
    if expected == 2:
        assert 'Original user request' in llm.call_args_list[1].args[1]
    else:
        assert 'Do not call any more tools' in llm.call_args_list[1].args[1]
    robot.stop.assert_called_once()

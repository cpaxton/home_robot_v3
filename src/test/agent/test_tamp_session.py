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


def setup_chat_loop(monkeypatch):
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
    return robot


@pytest.mark.parametrize('partial,budget,expected', [(False, 6, 2), (True, 6, 1), (False, 1, 1)])
def test_real_chat_loop_continues_success_but_cannot_dispatch_final_tools(monkeypatch, partial, budget, expected):
    robot = setup_chat_loop(monkeypatch)
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


@pytest.mark.parametrize('kwargs', [
    {'robot': 'rby1'}, {'memory_backend': 'dynagraph'}, {'eqa': True},
    {'vl_include_camera': True}, {'input_path': '/saved/map'}, {'use_llm': False},
])
def test_camera_free_mode_rejects_incompatible_sessions_before_connecting(monkeypatch, kwargs):
    connect = MagicMock()
    monkeypatch.setattr(loop, 'StretchZmqClient', connect)
    options = {'robot': 'stretch', 'memory_backend': 'dynamem', 'use_llm': True, 'state_only_sim': True}
    options.update(kwargs)
    with pytest.raises(ValueError, match='state_only_sim requires'):
        loop.run_agent_with_robot(**options)
    connect.assert_not_called()


@pytest.mark.parametrize('remote', ['text', 'vision', None])
def test_remote_inference_keeps_telemetry_live_local_load_guard_preserved(monkeypatch, remote):
    from emet.llms.openai_client import OpenaiClient

    robot = MagicMock()
    from emet.llms.openai_vllm_client import OpenaiVLLMClient
    client = object.__new__(OpenaiClient if remote == "text" else OpenaiVLLMClient) if remote else object()

    def generate(*args, **kwargs):
        assert robot.pause_streams.call_count == (0 if remote else 1)
        robot.resume_streams.assert_not_called()
        return 'reply', .5

    monkeypatch.setattr(loop, '_call_llm_body', generate)
    assert loop._call_llm(client, 'move objects', None, False, robot=robot) == ('reply', .5)
    assert robot.resume_streams.call_count == (0 if remote else 1)


@pytest.mark.parametrize('stop_after_plan', [False, True])
def test_real_chat_discovery_plan_execution_and_rediscovery(monkeypatch, stop_after_plan):
    robot = setup_chat_loop(monkeypatch)
    counts = {'scene_tasks': 0, 'plan_pick_place': 0, 'execute_pick_place_plan': 0}
    def discover():
        counts['scene_tasks'] += 1
        return json.dumps(response('scene_tasks', data={'tasks': [{'task_ref': f"task:{counts['scene_tasks']}"}]}))
    def plan(task_ref):
        counts['plan_pick_place'] += 1
        assert task_ref == f"task:{counts['scene_tasks']}"
        return json.dumps(response('plan_pick_place', data={'plan_ref': f"plan:{counts['plan_pick_place']}"}))
    def execute(plan_ref):
        counts['execute_pick_place_plan'] += 1
        assert plan_ref == f"plan:{counts['execute_pick_place_plan']}"
        return json.dumps(response('execute_pick_place_plan'))
    tools = [Tool(name=name, description='test', parameters={'type': 'object', 'properties': {}},
                  func=fn, returns_info=True)
             for name, fn in [('scene_tasks', discover), ('plan_pick_place', plan), ('execute_pick_place_plan', execute)]]
    monkeypatch.setattr(loop, 'get_tools', lambda ctx: tools)
    calls = []
    for index in (1, 2):
        calls.extend([('scene_tasks', {}), ('plan_pick_place', {'task_ref': f'task:{index}'}),
                      ('execute_pick_place_plan', {'plan_ref': f'plan:{index}'})])
    if stop_after_plan:
        calls = calls[:2]
    replies = [(json.dumps({'message': '', 'tool_calls': [{'name': name, 'arguments': args}]}), .1)
               for name, args in calls] + [('{"message":"done","tool_calls":[]}', .1)]
    llm = MagicMock(side_effect=replies)
    monkeypatch.setattr(loop, '_call_llm', llm)
    goal = 'Only plan the first task' if stop_after_plan else 'Move both objects'
    loop.run_agent_with_robot(discord=False, use_llm=True, commands=[goal], device='cpu',
                              thinking_status=False, max_tool_rounds=8)
    assert counts == ({'scene_tasks': 1, 'plan_pick_place': 1, 'execute_pick_place_plan': 0}
                      if stop_after_plan else {'scene_tasks': 2, 'plan_pick_place': 2, 'execute_pick_place_plan': 2})
    for call in llm.call_args_list[1:]:
        assert goal in call.args[1]
        assert 'do not execute without authorization' in call.args[1]
    robot.stop.assert_called_once()


def test_observed_provider_reaches_real_chat_tools_and_forces_discovery_summary(monkeypatch):
    import time
    from types import SimpleNamespace

    import numpy as np

    from emet.agent.tools import get_tools
    from emet.controller.manipulation.native_inputs import ObjectPoseEvidence
    from emet.controller.task.tamp.observed_discovery import ObservedTaskCandidate

    robot = setup_chat_loop(monkeypatch)
    pose = ObjectPoseEvidence(np.eye(4), time.monotonic(), 'rgbd_tracking', 'observation')
    provider = SimpleNamespace(input_mode='observed', observe_tasks=lambda: (
        ObservedTaskCandidate('cup', 'counter', pose, pose),))
    def real_tools(context):
        assert context['observed_tamp_provider'] is provider
        context['executor']._manip_mode = 'physical'
        context['executor']._tamp_inputs = 'observed'
        return get_tools(context)
    monkeypatch.setattr(loop, 'get_tools', real_tools)
    llm = MagicMock(side_effect=[
        (json.dumps({'tool_calls': [{'name': 'scene_tasks', 'arguments': {}}]}), .1),
        (json.dumps({'message': 'Found a cup; execution unavailable.',
                     'tool_calls': [{'name': 'execute_pick_place_plan', 'arguments': {'plan_ref': 'invented'}}]}), .1)])
    monkeypatch.setattr(loop, '_call_llm', llm)
    loop.run_agent_with_robot(discord=False, use_llm=True, commands=['Find tasks'], device='cpu',
                              thinking_status=False, observed_tamp_provider=provider)
    assert 'discovery_only' in llm.call_args_list[1].args[1]
    assert 'Do not call any more tools' in llm.call_args_list[1].args[1]
    robot.get_emet_session.assert_not_called()
    robot.arm_to.assert_not_called()


def test_chat_automatically_uses_existing_scene_graph_provider(monkeypatch):
    from types import SimpleNamespace

    from emet.agent.tools import get_tools
    from emet.controller.task.tamp.observed_discovery import RgbdTaskProvider

    robot = setup_chat_loop(monkeypatch)
    executor = loop.DynamemTaskExecutor()
    provider = RgbdTaskProvider()
    executor.agent._open_vocab_sg_processor = SimpleNamespace(observed_tamp_provider=provider)
    def real_tools(context):
        assert context['observed_tamp_provider'] is provider
        context['executor']._manip_mode = 'physical'
        context['executor']._tamp_inputs = 'observed'
        return get_tools(context)
    monkeypatch.setattr(loop, 'get_tools', real_tools)
    monkeypatch.setattr(loop, '_call_llm', MagicMock(side_effect=[
        ('{"tool_calls":[{"name":"scene_tasks","arguments":{}}]}', .1),
        ('{"message":"No fresh task proposals yet","tool_calls":[]}', .1)]))
    loop.run_agent_with_robot(discord=False, use_llm=True, commands=['Discover tasks'],
                              device='cpu', thinking_status=False)
    robot.get_emet_session.assert_not_called()


def test_real_chat_logs_explicit_user_boundary_before_new_authorized_turn(monkeypatch):
    from emet.eval.native_chat_scoring import failure_boundaries

    setup_chat_loop(monkeypatch)
    execute = MagicMock(side_effect=[
        json.dumps(response('execute_pick_place_plan', status='partial', code='command_outcome_unknown')),
        json.dumps(response('execute_pick_place_plan'))])
    tool = Tool(name='execute_pick_place_plan', description='test',
                parameters={'type': 'object', 'properties': {}}, func=execute, returns_info=True)
    monkeypatch.setattr(loop, 'get_tools', lambda ctx: [tool])
    call = json.dumps({'tool_calls': [{'name': tool.name, 'arguments': {}}]})
    monkeypatch.setattr(loop, '_call_llm', MagicMock(side_effect=[
        (call, .1), ('{"message":"Stopped","tool_calls":[]}', .1),
        (call, .1), ('{"message":"Done","tool_calls":[]}', .1)]))
    commands = ['Try the first task', 'I inspected the scene; execute the next task']
    loop.run_agent_with_robot(discord=False, use_llm=True, commands=commands,
                              device='cpu', thinking_status=False)
    rows = [{'role': c.args[0], 'content': c.args[1]} for c in loop.ChatLog.return_value.log.call_args_list]
    assert [r['content'] for r in rows if r['role'] == 'user'] == commands
    assert execute.call_count == 2
    failures = failure_boundaries(rows)
    assert len(failures) == 1
    assert failures[0]['unexpected_tools'] == []


@pytest.mark.parametrize('inputs,expected', [('observed', True), ('privileged', False)])
def test_real_chat_startup_binds_only_observed_rgbd_capture(monkeypatch, inputs, expected):
    from emet.controller.task.tamp.observed_discovery import RgbdTaskProvider

    robot = setup_chat_loop(monkeypatch)
    executor = loop.DynamemTaskExecutor()
    executor._tamp_inputs = inputs
    provider = RgbdTaskProvider()
    loop.run_agent_with_robot(discord=False, use_llm=False, commands=['QUIT'],
                              device='cpu', thinking_status=False,
                              observed_tamp_provider=provider)
    assert (provider._observation_source is not None) is expected
    if expected:
        assert provider._observation_source == robot.get_observation

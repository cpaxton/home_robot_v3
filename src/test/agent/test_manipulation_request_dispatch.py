import json
from unittest.mock import Mock

from emet.agent.chat_tamp import build_tamp_handlers
from emet.controller.task.tamp.api import response
from emet.controller.task.tamp.observed_adapter import ObservedNativeAdapter


def setup_handlers():
    robot = Mock()
    adapter = object.__new__(ObservedNativeAdapter)
    adapter.robot = robot
    adapter.plan = Mock(return_value=response('plan_pick_place', data={'plan_ref': 'plan:1'}))
    adapter.execute = Mock(return_value=response('execute_pick_place_plan'))
    handlers = build_tamp_handlers({'robot': robot, 'manip_mode': 'physical',
        'tamp_inputs': 'observed', 'observed_native_adapter': adapter})
    return handlers, adapter


def test_one_call_uses_public_plan_and_execute_path():
    handlers, adapter = setup_handlers()
    result = json.loads(handlers['pick_place']('pear', 'sink'))
    assert result['status'] == 'ok' and result['tool'] == 'pick_place'
    adapter.plan.assert_called_once_with('', 'pear', 'sink', None)
    adapter.execute.assert_called_once_with('plan:1')


def test_options_never_silently_fall_back_to_legacy_motion():
    handlers, adapter = setup_handlers()
    result = json.loads(handlers['pick_place']('pear', 'sink', hints={'approach': {}}, hint_mode='strict'))
    assert result['code'] == 'manipulation_options_unsupported'
    assert result['data']['execution_started'] is False
    adapter.plan.assert_not_called()
    adapter.execute.assert_not_called()


def test_request_capable_adapter_receives_constraints_and_hint_mode():
    handlers, adapter = setup_handlers()
    adapter.plan_request = Mock(return_value=response('plan_pick_place', data={'plan_ref': 'hinted:1'}))
    result = json.loads(handlers['pick_place']('pear', 'sink', hints={'approach': {}},
        hint_mode='strict', constraints={'protected_objects': ['cube']}))
    assert result['status'] == 'ok'
    request = adapter.plan_request.call_args.args[0]
    assert request.hint_mode == 'strict'
    assert request.constraints == {'protected_objects': ['cube']}
    adapter.plan.assert_not_called()
    adapter.execute.assert_called_once_with('hinted:1')


def test_empty_optional_objects_keep_names_only_behavior():
    handlers, adapter = setup_handlers()
    result = json.loads(handlers['pick_place']('pear', 'sink', hints={}, constraints={}))
    assert result['status'] == 'ok'
    adapter.plan.assert_called_once()
    adapter.execute.assert_called_once()

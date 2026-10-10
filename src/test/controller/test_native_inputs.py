# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.native_inputs import ObjectPoseEvidence, ObservationUnavailable
from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor


def test_observed_input_refuses_privileged_or_stale_pose():
    for source, timestamp, code in [
        ('simulator_ground_truth', 10., 'privileged_input_forbidden'),
        ('rgbd_tracking', 6., 'object_observation_stale'),
        ('stereo_tracking', 11., 'object_observation_stale'),
    ]:
        evidence = ObjectPoseEvidence(np.eye(4), timestamp, source, 'frame:1')
        with pytest.raises(ObservationUnavailable, match=code):
            evidence.require('observed', now=10.)


def test_pose_evidence_is_an_immutable_copy():
    matrix = np.eye(4)
    evidence = ObjectPoseEvidence(matrix, 10., 'rgbd_tracking', 'frame:1')
    matrix[0, 3] = 99
    assert evidence.world_from_object[0, 3] == 0
    with pytest.raises(ValueError):
        evidence.world_from_object[0, 3] = 99
    with pytest.raises(ValueError):
        evidence.world_from_object.setflags(write=True)


def test_observed_retention_uses_provider_not_hidden_simulator_object():
    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="ee"><joint name="arm"/><geom size=".1"/></body></worldbody></mujoco>')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    provider = SimpleNamespace(input_mode='observed', observe_object=lambda _: ObjectPoseEvidence(
        np.eye(4), time.monotonic(), 'rgbd_tracking', 'frame:1'))
    collision = SimpleNamespace(geometry_source='observed_voxels', payload_transform=np.eye(4))
    executor = PhysicalPickPlaceExecutor(None, model=m, data=d, ee_body='ee', joint_names=['arm'],
        collision=collision, synchronize=lambda _: None, command_joints=lambda _: True,
        object_pose_provider=provider)
    executor.payload_body = 'object_not_present_in_mujoco'
    assert executor.payload_retained()
    provider.observe_object = lambda _: None
    assert not executor.payload_retained()
    provider.observe_object = lambda _: ObjectPoseEvidence(np.eye(4), time.monotonic(), 'simulator_ground_truth', 'gt')
    assert not executor.payload_retained()


def test_observed_constructor_requires_both_providers():
    for provider, collision, code in [
        (None, None, 'observed_object_provider_required'),
        (SimpleNamespace(input_mode='observed'), None, 'observed_collision_provider_required'),
    ]:
        with pytest.raises(ValueError, match=code):
            PhysicalPickPlaceExecutor(None, model=None, data=None, ee_body='', joint_names=[],
                collision=collision, synchronize=lambda _: None, command_joints=lambda _: None,
                object_pose_provider=provider)


def test_physical_configuration_never_resolves_to_teleport(monkeypatch):
    from emet.simulation.sim_manipulation import can_use_sim_gt_manip, resolve_agent_manip_mode
    monkeypatch.delenv('EMET_MANIP_MODE', raising=False)
    assert resolve_agent_manip_mode(config_mode='physical') == 'physical'
    robot = SimpleNamespace(get_emet_session=lambda: {'is_simulation': True, 'capabilities': {'sim_set_body_pose': True}})
    assert not can_use_sim_gt_manip(robot, manip_mode='physical')
    monkeypatch.setenv('EMET_MANIP_MODE', 'physical')
    assert resolve_agent_manip_mode(config_mode='teleport') == 'physical'


def test_physical_request_does_not_use_legacy_controller_fallback():
    import json
    from unittest.mock import Mock

    from emet.agent.tools import get_tools
    executor = Mock(_manip_mode='physical', visual_servo=True, robot=SimpleNamespace(get_emet_session=lambda: None))
    tool = {t.name: t for t in get_tools({'executor': executor})}['pick_place']
    result = json.loads(tool.func(object_name='block', receptacle_name='table'))
    assert result['status'] == 'error'
    executor.assert_not_called()


@pytest.mark.parametrize('tool_name,args', [
    ('scene_tasks', {}), ('plan_pick_place', {'object_name': 'block', 'receptacle_name': 'table'}),
    ('pick_place', {'object_name': 'block', 'receptacle_name': 'table'}),
    ('execute_pick_place_plan', {'plan_ref': 'previously-assisted-plan'}),
])
def test_observed_native_tools_do_not_discover_targets_from_gt(tool_name, args):
    import json
    from unittest.mock import Mock

    from emet.agent.tools import get_tools
    robot = SimpleNamespace(get_emet_session=Mock(side_effect=AssertionError('privileged metadata read')))
    tools = {t.name: t for t in get_tools({'robot': robot, 'manip_mode': 'physical', 'tamp_inputs': 'observed'})}
    result = json.loads(tools[tool_name].func(**args))
    assert result['code'] == 'observed_scene_unavailable'
    robot.get_emet_session.assert_not_called()


@pytest.mark.parametrize('is_simulation,expected', [
    (False, 'privileged_inputs_require_simulation'), (True, 'native_execution_unavailable'),
])
def test_privileged_native_mode_reports_explicit_unavailability(is_simulation, expected):
    import json

    from emet.agent.tools import get_tools
    robot = SimpleNamespace(get_emet_session=lambda: {'is_simulation': is_simulation})
    tools = {t.name: t for t in get_tools({'robot': robot, 'manip_mode': 'physical', 'tamp_inputs': 'privileged'})}
    assert json.loads(tools['scene_tasks'].func())['code'] == expected

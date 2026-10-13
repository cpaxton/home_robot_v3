# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Real tool registry with measured MuJoCo fixtures; no actuation in planning."""
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from emet.agent.tools import get_tools
from emet.controller.manipulation.native_scene import model_signature
from emet.simulation.mujoco_server_stretch import default_scene_xml_path
from emet.simulation.robosuite_load_utils import snap_joint_qpos_to_ctrl_for_position_actuators


@pytest.fixture
def native():
    model = mujoco.MjModel.from_xml_path(default_scene_xml_path)
    data = mujoco.MjData(model)
    data.ctrl[:] = model.key_ctrl[0]
    data.ctrl[model.actuator('gripper').id] = .04
    snap_joint_qpos_to_ctrl_for_position_actuators(model, data)
    data.qpos[model.jnt_qposadr[model.body('object2').jntadr[0]]] = .3
    for _ in range(1000):
        mujoco.mj_step(model, data)
    robot = SimpleNamespace(_state_lock=threading.RLock(), _state_received_monotonic=time.monotonic(),
        _state={'command_protocol': {'server_boot_id': 'test'}, 'native_tamp_snapshot': {
            'source': 'simulator_ground_truth', 'model_signature': model_signature(model),
            'qpos': data.qpos.tolist(), 'qvel': data.qvel.tolist(), 'sim_time': float(data.time),
            'captured_at': time.time()}}, arm_to=Mock(), open_gripper=Mock(), close_gripper=Mock(),
        get_emet_session=lambda: {'is_simulation': True, 'sim_object_placements': {
            'object1': {'cat': 'blue cube'}, 'object2': {'cat': 'red cylinder'}, 'table': {'cat': 'table'}}})
    context = {'robot': robot, 'manip_mode': 'physical', 'tamp_inputs': 'privileged'}
    tools = {t.name: t for t in get_tools(context)}
    def call(name, **kwargs):
        return json.loads(tools[name].func(**kwargs))
    return robot, data, context, call


def make_plan(call):
    tasks = call('scene_tasks', object_filter='blue')['data']['tasks']
    result = call('plan_pick_place', task_ref=tasks[0]['task_ref'])
    assert result['status'] == 'ok', result
    return result['data']['plan_ref']


def test_native_plan_is_read_only_and_has_explicit_provenance(native):
    robot, data, context, call = native
    before_q, before_ctrl = data.qpos.copy(), data.ctrl.copy()
    ref = make_plan(call)
    assert ref in context['_native_tamp_adapter'].plans
    np.testing.assert_array_equal(data.qpos, before_q)
    np.testing.assert_array_equal(data.ctrl, before_ctrl)
    for fn in (robot.arm_to, robot.open_gripper, robot.close_gripper):
        fn.assert_not_called()
    assert call('scene_tasks')['data']['assistance'] == []
    assert call('set_receptacle_state', task_ref='anything', state='open')['code'] == 'native_articulation_unavailable'


def test_changed_scene_consumes_native_plan_before_any_motion(native):
    robot, _, _, call = native
    ref = make_plan(call)
    robot._state['native_tamp_snapshot']['qpos'][0] += .05
    assert call('execute_pick_place_plan', plan_ref=ref)['code'] == 'scene_changed_replan'
    assert call('execute_pick_place_plan', plan_ref=ref)['code'] == 'unknown_plan'
    robot.arm_to.assert_not_called()
    robot.open_gripper.assert_not_called()


def test_old_capture_cannot_be_refreshed_by_republishing(native):
    robot, _, _, call = native
    ref = make_plan(call)
    robot._state['native_tamp_snapshot']['captured_at'] -= 3
    robot._state_received_monotonic = time.monotonic()
    assert call('execute_pick_place_plan', plan_ref=ref)['code'] == 'stale_observation'
    assert call('execute_pick_place_plan', plan_ref=ref)['code'] == 'unknown_plan'
    robot.arm_to.assert_not_called()


def test_observed_mode_never_reads_privileged_metadata(native):
    robot, _, _, _ = native
    robot.get_emet_session = Mock(side_effect=AssertionError('privileged read'))
    tools = {t.name: t for t in get_tools({'robot': robot, 'manip_mode': 'physical', 'tamp_inputs': 'observed'})}
    result = json.loads(tools['scene_tasks'].func())
    assert result['code'] == 'observed_scene_unavailable'
    robot.get_emet_session.assert_not_called()


def test_native_model_mismatch_rejects_before_actuation(native):
    robot, _, _, call = native
    robot._state['native_tamp_snapshot']['model_signature'] = 'different-physics-model'
    assert call('scene_tasks')['code'] == 'native_model_mismatch'
    robot.arm_to.assert_not_called()
    robot.open_gripper.assert_not_called()

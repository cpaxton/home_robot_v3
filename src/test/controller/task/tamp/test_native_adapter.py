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
from emet.controller.task.tamp.native_adapter import NativeTableAdapter
from emet.simulation.mujoco_server_stretch import default_scene_xml_path
from emet.simulation.robosuite_load_utils import snap_joint_qpos_to_ctrl_for_position_actuators


@pytest.fixture(scope="module")
def table_model():
    return mujoco.MjModel.from_xml_path(default_scene_xml_path)


@pytest.fixture
def native(table_model):
    model = table_model
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
    context = {'robot': robot, 'manip_mode': 'physical', 'tamp_inputs': 'privileged',
               '_native_tamp_adapter': NativeTableAdapter(robot, model=model)}
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


def test_model_binding_allows_free_initial_pose_but_not_geometry():
    def load(position='0 0 1', size='.02'):
        return mujoco.MjModel.from_xml_string(f'<mujoco><worldbody><body pos="{position}"><freejoint/><geom type="box" size="{size} .02 .02"/></body></worldbody></mujoco>')
    assert model_signature(load()) == model_signature(load(position='1 0 1'))
    assert model_signature(load()) != model_signature(load(size='.03'))


def test_native_arm_acknowledgement_leaves_arrival_to_measured_executor(native):
    robot, _, context, _ = native
    robot.switch_to_manipulation_mode = Mock()
    robot.arm_to.return_value = True
    adapter = context['_native_tamp_adapter']
    executor = adapter._executor('object1')
    assert executor.command_joints(adapter.data.qpos[executor.qadr].copy())
    assert robot.arm_to.call_args.kwargs['blocking'] is False
    assert executor.joint_tolerance == .03


@pytest.mark.parametrize('fault, expected', [(None, True), ('height', False), ('spinning', False), ('region', False)])
def test_release_verification_requires_correct_height_and_settled_rotation(native, monkeypatch, fault, expected):
    import emet.controller.task.tamp.native_adapter as module
    _, _, context, _ = native
    adapter = context['_native_tamp_adapter']
    adapter.data.qvel[:] = 0
    goal = adapter.data.body('object1').xpos.copy() + [0, 0, .02]
    if fault == 'height':
        goal[2] += .1
    if fault == 'spinning':
        joint = adapter.model.body('object1').jntadr[0]
        adapter.data.qvel[adapter.model.jnt_dofadr[joint]+3] = .2
    executor = adapter._executor('object1')
    clock = [0.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(module.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0]+dt))
    monkeypatch.setattr(adapter, 'synchronize', lambda data: setattr(data, 'time', data.time+.05))
    region = np.array([goal[:2] - [.005, .1], goal[:2] + [.005, .1]]) if fault == 'region' else None
    plan = SimpleNamespace(object_body='object1', support_body='table', goal=goal, executor=executor, placement_region=region)
    assert adapter._verify(plan, released=True, seconds=1.) is expected


def test_pinch_calibration_uses_pad_centers_and_preserves_measured_state(native):
    _, data, context, call = native
    adapter = context['_native_tamp_adapter']
    before = adapter.data.qpos.copy()
    offset = adapter._pinch_center_offset()
    assert np.isfinite(offset).all() and np.linalg.norm(offset) < .05
    np.testing.assert_array_equal(adapter.data.qpos, before)
    ref = make_plan(call)
    plan = adapter.plans[ref]
    point, rotation = plan.executor.grasp_targets[1]
    np.testing.assert_allclose(point + rotation @ offset, data.body(plan.object_body).xpos, atol=.001)


@pytest.mark.parametrize('phase', ['grasp', 'place'])
def test_command_ack_timeout_preserves_partial_execution_and_consumes_plan(native, monkeypatch, phase):
    _, _, context, call = native
    adapter = context['_native_tamp_adapter']
    ref = make_plan(call)
    executor = adapter.plans[ref].executor
    executor.grasp_only = Mock(return_value=SimpleNamespace(success=True))
    executor.place_only = Mock(return_value=SimpleNamespace(success=True))
    getattr(executor, phase + '_only').side_effect = TimeoutError('private command sequence: outcome unknown')
    monkeypatch.setattr(adapter, '_verify', lambda *args, **kwargs: True)
    result = call('execute_pick_place_plan', plan_ref=ref)
    assert result['status'] == 'partial'
    assert result['code'] == 'command_outcome_unknown'
    assert result['data']['execution_started'] is True
    assert result['data']['completed_ops'] == ([] if phase == 'grasp' else ['grasp'])
    assert result['data']['failed_stage'] == phase
    assert result['recovery'] == 'inspect'
    assert 'private command' not in json.dumps(result)
    assert call('execute_pick_place_plan', plan_ref=ref)['code'] == 'unknown_plan'
    if phase == 'grasp':
        executor.place_only.assert_not_called()


@pytest.mark.parametrize('region', [[[0, 0], [0, 1]], [[1, 0], [0, 1]],
                                     [[0, 0], [1, float('nan')]], [0, 1], 'bad'])
def test_invalid_placement_region_never_actuates(native, region):
    robot, _, _, call = native
    result = call('plan_pick_place', object_name='blue', receptacle_name='table', placement_region=region)
    assert result['code'] == 'invalid_placement_region'
    robot.arm_to.assert_not_called()
    robot.open_gripper.assert_not_called()


def test_region_constrains_goal_and_is_copied_into_plan(native):
    robot, _, context, call = native
    region = [[-.12, -.75], [.10, -.55]]
    result = call('plan_pick_place', object_name='blue', receptacle_name='table', placement_region=region)
    assert result['status'] == 'ok', result
    plan = context['_native_tamp_adapter'].plans[result['data']['plan_ref']]
    np.testing.assert_array_equal(plan.placement_region, region)
    assert np.all(plan.goal[:2] >= np.asarray(region)[0])
    assert np.all(plan.goal[:2] <= np.asarray(region)[1])
    region[0][0] = -100
    assert plan.placement_region[0, 0] == -.12
    assert result['data']['placement_goal']['frame'] == 'world'
    robot.arm_to.assert_not_called()


def test_region_outside_support_is_refused(native):
    _, _, _, call = native
    result = call('plan_pick_place', object_name='blue', receptacle_name='table',
                  placement_region=[[10, 10], [11, 11]])
    assert result['code'] == 'placement_region_unavailable'


def test_rejection_evidence_keeps_collision_when_later_ik_fails(native):
    _, _, context, call = native
    adapter = context['_native_tamp_adapter']
    ex = adapter._executor('object1')
    ex.plan_pose = Mock(side_effect=[(None, 'arm_path_failed:invalid_goal:collision')] +
                                  [(None, 'pose_ik_failed:private_detail')] * 100)
    adapter._executor = Mock(return_value=ex)
    result = call('plan_pick_place', object_name='blue', receptacle_name='table')
    assert result['code'] == 'no_plan_within_budget'
    assert result['data']['planning_rejections']['pregrasp:collision'] == 1
    assert result['data']['planning_rejections']['pregrasp:ik_failed'] > 0
    assert 'private_detail' not in json.dumps(result)


def test_shared_ik_budget_restores_private_state_and_never_actuates(native):
    robot, _, context, call = native
    adapter = context['_native_tamp_adapter']
    initial = adapter.data.qpos.copy()
    adapter.max_planning_ik_calls = 1
    result = call('plan_pick_place', object_name='blue', receptacle_name='table')
    assert result['code'] == 'no_plan_within_budget'
    assert result['data']['planning_ik_calls'] == 1
    assert result['data']['max_ik_calls'] == 1
    np.testing.assert_array_equal(adapter.data.qpos, initial)
    robot.arm_to.assert_not_called()
    assert not adapter.plans


def test_failed_placement_resumes_grasp_search(native, monkeypatch):
    import emet.motion.placement_surfaces as surfaces_module
    _, _, context, call = native
    adapter = context['_native_tamp_adapter']
    original = adapter._grasp_candidates
    resumed = []

    def grasp_candidates(*args):
        for candidate in original(*args):
            yield candidate
            resumed.append(True)
            # Deliberately empty placement domain must advance the generator.
            if len(resumed) == 2:
                return

    monkeypatch.setattr(adapter, '_grasp_candidates', grasp_candidates)
    monkeypatch.setattr(surfaces_module, 'free_surface_centers',
                        lambda *a, **k: SimpleNamespace(centers=[]))
    result = call('plan_pick_place', object_name='blue', receptacle_name='table')
    assert result['code'] == 'no_plan_within_budget'
    assert len(resumed) == 2

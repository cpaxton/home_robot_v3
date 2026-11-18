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
from revalidation_fixtures import local_scene

import emet.motion.placement_surfaces as surfaces_module
from emet.agent.tools import get_tools
from emet.controller.manipulation.native_scene import model_signature
from emet.controller.task.tamp import native_revalidation as validation
from emet.controller.task.tamp.native_adapter import NativeTableAdapter, adapter_for
from emet.controller.task.tamp.native_dispatch import dispatch_boundary
from emet.motion.local_scene_store import LocalMapUnavailable, LocalSceneStore
from emet.simulation.mujoco_server_stretch import default_scene_xml_path
from emet.simulation.robosuite_load_utils import snap_joint_qpos_to_ctrl_for_position_actuators
from emet.simulation.sim_object_placements import placements_from_mujoco_model


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
        gripper_to=Mock(return_value=None), get_gripper_position=Mock(return_value=0.),
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
    result = call('execute_pick_place_plan', plan_ref=ref)
    assert result['code'] == 'scene_changed_replan'
    assert result['data']['object_name'] == 'blue cube'
    assert result['data']['receptacle_name'] == 'table'
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


def test_native_model_mismatch_rejects_before_actuation(native, monkeypatch):
    robot, _, context, call = native
    model = context['_native_tamp_adapter'].model
    # This test isolates model binding. Reuse the loaded fixture model when the
    # mismatch invalidates the cache; cold mesh loading can outlast the two-second
    # snapshot lifetime and mask the intended rejection as stale_observation.
    monkeypatch.setattr('emet.controller.task.tamp.native_adapter.NativeTableAdapter',
                        lambda client, **kwargs: NativeTableAdapter(client, model=model, **kwargs))
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
    pinch = point + rotation @ offset
    # Height alternatives may deliberately pinch above the center. Calibration
    # must center the opposing pads horizontally and keep them on the payload.
    np.testing.assert_allclose(pinch[:2], data.body(plan.object_body).xpos[:2], atol=.001)
    bounds = np.asarray(placements_from_mujoco_model(adapter.model, data)[plan.object_body]['bounds'])
    assert bounds[0, 2] <= pinch[2] <= bounds[1, 2]


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


def test_unknown_native_task_requests_rediscovery(native):
    robot, _, _, call = native
    result = call('plan_pick_place', task_ref='invented_task')
    assert result['code'] == 'unknown_task_ref'
    assert result['recovery'] == 'rediscover'
    robot.arm_to.assert_not_called()


@pytest.mark.parametrize('names', [
    {'object_name': 'red cylinder'},
    {'receptacle_name': 'other table'},
])
def test_conflicting_task_selectors_rejected_before_planning(native, monkeypatch, names):
    robot, data, context, call = native
    adapter = context['_native_tamp_adapter']
    task = call('scene_tasks', object_filter='blue')['data']['tasks'][0]
    before = data.qpos.copy()
    executor = Mock(side_effect=AssertionError('must not plan conflicting selectors'))
    monkeypatch.setattr(adapter, '_executor', executor)
    result = call('plan_pick_place', task_ref=task['task_ref'], **names)
    assert result['code'] == 'task_selector_mismatch'
    assert result['recovery'] == 'rediscover'
    assert result['data']['object_name'] == 'blue cube'
    assert result['data']['receptacle_name'] == 'table'
    assert not adapter.plans
    executor.assert_not_called()
    robot.arm_to.assert_not_called()
    robot.open_gripper.assert_not_called()
    robot.close_gripper.assert_not_called()
    np.testing.assert_array_equal(data.qpos, before)


def test_matching_selectors_keep_identity_through_execution(native, monkeypatch):
    _, _, context, call = native
    adapter = context['_native_tamp_adapter']
    task = call('scene_tasks', object_filter='blue')['data']['tasks'][0]
    result = call('plan_pick_place', task_ref=task['task_ref'],
                  object_name=' BLUE CUBE ', receptacle_name='Table')
    assert result['status'] == 'ok', result
    ref = result['data']['plan_ref']
    plan = adapter.plans[ref]
    monkeypatch.setattr(plan.executor, 'grasp_only', lambda *a, **k: SimpleNamespace(success=True))
    monkeypatch.setattr(plan.executor, 'place_only', lambda *a, **k: SimpleNamespace(success=True))
    monkeypatch.setattr(adapter, '_verify', lambda *a, **k: True)
    # Semantic identity is frozen into the plan, not looked up from mutable tasks.
    adapter.tasks.clear()
    executed = call('execute_pick_place_plan', plan_ref=ref)
    assert executed['status'] == 'ok', executed
    for envelope in (result, executed):
        assert envelope['data']['object_name'] == 'blue cube'
        assert envelope['data']['receptacle_name'] == 'table'
        assert 'object1' not in json.dumps(envelope)


def test_partial_progress_reports_execution_started_even_before_next_stage():
    from emet.controller.task.tamp.native_adapter import NativeTableAdapter

    adapter = object.__new__(NativeTableAdapter)
    result = adapter._failure('execute_pick_place_plan', 'payload_not_retained',
                              completed=('grasp',), phase='place')
    assert result['status'] == 'partial'
    assert result['data']['execution_started'] is True
    assert result['data']['completed_ops'] == ['grasp']
    assert result['recovery'] == 'inspect'


@pytest.mark.parametrize('reason', ['arm_static_support_margin', 'measured_pose_replan_failed:arm_path_static_support_margin'])
def test_measured_support_rejection_is_actionable_without_private_planner_text(reason):
    adapter = object.__new__(NativeTableAdapter)
    result = adapter._failure('execute_pick_place_plan', reason, phase='lift', started=True)
    assert result['status'] == 'partial'
    assert result['code'] == 'planning_failed'
    assert result['data']['failed_stage'] == 'lift'
    assert result['data']['motion_rejection'] == 'static_support_margin'
    assert result['data']['completed_ops'] == []
    assert result['recovery'] == 'inspect'
    unknown = adapter._failure('execute_pick_place_plan', 'private_body_name:some_unknown_failure', phase='lift')
    assert unknown['code'] == 'operation_failed'
    assert 'motion_rejection' not in unknown['data']
    assert 'private_body_name' not in json.dumps(unknown)


def native_transfer_config(native):
    _, _, context, _ = native
    adapter = context['_native_tamp_adapter']
    adapter.synchronize(adapter.data)
    entries = placements_from_mujoco_model(adapter.model, adapter.data)
    patch = np.asarray(entries['table']['support_surfaces'])[0]
    extent = np.max([np.diff(np.asarray(entries[n]['bounds'])[:, :2], axis=0)[0]
                     for n in ('object1', 'object2')], axis=0) / 2 + .015
    regions = []
    for role, name in [('cube home', 'object1'), ('cylinder home', 'object2')]:
        center = np.mean(np.asarray(entries[name]['bounds'])[:, :2], axis=0)
        bounds = np.array([np.maximum(center-extent, patch[0, :2]),
                           np.minimum(center+extent, patch[1, :2])])
        regions.append({'name': role, 'bounds': bounds.tolist()})
    for x in np.linspace(patch[0, 0]+.08, patch[1, 0]-.08, 8):
        for y in np.linspace(patch[0, 1]+.08, patch[1, 1]-.08, 8):
            bounds = np.array([[x-.07, y-.07], [x+.07, y+.07]])
            if all(not (np.all(bounds[0] < np.asarray(r['bounds'])[1])
                        and np.all(np.asarray(r['bounds'])[0] < bounds[1])) for r in regions):
                regions.append({'name': 'buffer', 'bounds': bounds.tolist()})
                return {'schema_version': 1, 'input_mode': 'privileged', 'robot': 'stretch',
                        'reset_id': 'fixture', 'support_body': 'table',
                        'objects': {'blue cube': 'object1', 'red cylinder': 'object2'}, 'regions': regions}
    pytest.fail('No disjoint fixture buffer')


def test_native_transfer_grounding_is_measured_and_private(native):
    from emet.controller.task.tamp.native_transfer import NativeTableTransferProvider
    robot, live, context, _ = native
    before_q, before_ctrl = live.qpos.copy(), live.ctrl.copy()
    domain = NativeTableTransferProvider(robot, native_transfer_config(native),
                                          adapter=context['_native_tamp_adapter'])
    discovery = domain.describe()
    assert discovery['free_regions'] == ['buffer']
    assert 'object1' not in json.dumps(discovery)
    assert discovery['inputs'] == 'privileged'
    np.testing.assert_array_equal(live.qpos, before_q)
    np.testing.assert_array_equal(live.ctrl, before_ctrl)
    robot.arm_to.assert_not_called()


def test_native_transfer_rejects_unsettled_snapshot(native):
    from emet.controller.task.tamp.native_transfer import NativeTableTransferProvider
    robot, _, context, _ = native
    adapter = context['_native_tamp_adapter']
    domain = NativeTableTransferProvider(robot, native_transfer_config(native), adapter=adapter)
    joint = adapter.model.body('object1').jntadr[0]
    robot._state['native_tamp_snapshot']['qvel'][adapter.model.jnt_dofadr[joint]] = .2
    with pytest.raises(ValueError, match='not_settled'):
        domain.snapshot()


def test_shared_bounded_policy_is_explicit_and_legacy_default_is_preserved(native):
    robot, _, context, _ = native
    from emet.controller.task.tamp.native_policy import FINE_GRASP_YAWS
    adapter = NativeTableAdapter(robot, model=context['_native_tamp_adapter'].model, policy='bounded_release')
    assert adapter.policy == 'bounded_release'
    assert adapter.grasp_yaw_offsets == FINE_GRASP_YAWS
    assert adapter.payload_binding == 'verified_lift'
    assert adapter.release_limits.max_drop_m == .04
    assert adapter.arm_extension_margin_m == .0005
    assert adapter.prefer_face_aligned_grasps is True
    assert context['_native_tamp_adapter'].policy == 'legacy'
    assert context['_native_tamp_adapter'].release_limits is None
    assert context['_native_tamp_adapter'].prefer_face_aligned_grasps is False
    assert context['_native_tamp_adapter']._executor('object1').joint_limit_margins == {}
    with pytest.raises(ValueError, match='invalid_native_tamp_policy'):
        NativeTableAdapter(None, policy='typo')
    robot.arm_to.assert_not_called()


def test_bounded_plan_reserves_arm_extension_without_changing_measured_state_or_limits(native):
    """The home-start fixture stays interior; this is not a general path bound."""
    robot, _, context, _ = native
    model = context['_native_tamp_adapter'].model
    adapter = NativeTableAdapter(robot, model=model, policy='bounded_release')
    before_q, before_limits = adapter.data.qpos.copy(), model.jnt_range.copy()
    result = adapter.plan(object_name='blue cube', receptacle_name='table')
    assert result['status'] == 'ok', result
    executor = adapter.plans[result['data']['plan_ref']].executor
    for path in executor.grasp_paths + executor.place_paths:
        for i in range(4):
            name = f'joint_arm_l{i}'
            lo, hi = model.jnt_range[model.joint(name).id]
            column = executor.joint_names.index(name)
            values = np.asarray(path)[:, column]
            assert np.all(values >= lo + .0005 - 1e-9)
            assert np.all(values <= hi - .0005 + 1e-9)
    np.testing.assert_array_equal(adapter.data.qpos, before_q)
    np.testing.assert_array_equal(model.jnt_range, before_limits)
    robot.arm_to.assert_not_called()
    robot.open_gripper.assert_not_called()


def test_bounded_margin_does_not_clamp_or_admit_out_of_range_measured_start(native):
    robot, _, context, _ = native
    adapter = NativeTableAdapter(robot, model=context['_native_tamp_adapter'].model, policy='bounded_release')
    executor = adapter._executor('object1')
    joint = adapter.model.joint('joint_arm_l0')
    adapter.data.qpos[joint.qposadr[0]] = float(joint.range[1]) + 8e-6
    before = adapter.data.qpos.copy()
    goal = before[executor.qadr].copy()
    goal[executor.joint_names.index('joint_arm_l0')] = float(joint.range[1]) - .0005
    path, error = executor.plan_joint_target(goal)
    assert path is None and 'invalid_start:joint_bounds:joint_arm_l0' in error
    np.testing.assert_array_equal(adapter.data.qpos, before)
    robot.arm_to.assert_not_called()


def test_bounded_target_reserve_keeps_valid_margin_band_start_in_path(native):
    robot, _, context, _ = native
    adapter = NativeTableAdapter(robot, model=context['_native_tamp_adapter'].model, policy='bounded_release')
    executor = adapter._executor('object1')
    adapter.data.qpos[adapter.model.joint('joint_lift').qposadr[0]] = .85
    adapter.data.qpos[adapter.model.joint('joint_wrist_pitch').qposadr[0]] = 0.
    for i in range(4):
        joint = adapter.model.joint(f'joint_arm_l{i}')
        adapter.data.qpos[joint.qposadr[0]] = float(joint.range[1]) - .00025
    before = adapter.data.qpos.copy()
    start = before[executor.qadr].copy()
    goal = start.copy()
    for i in range(4):
        name = f'joint_arm_l{i}'
        goal[executor.joint_names.index(name)] = float(adapter.model.joint(name).range[1]) - .001
    path, error = executor.plan_joint_target(goal)
    assert error is None, error
    np.testing.assert_array_equal(path[0], start)
    np.testing.assert_allclose(path[-1], goal)
    for i in range(4):
        name = f'joint_arm_l{i}'
        column = executor.joint_names.index(name)
        upper = float(adapter.model.joint(name).range[1])
        assert upper - .0005 < path[0][column] < upper
    # Pure planning may leave private data at its goal, but never clamps the
    # start waypoint or writes a measured configuration back to the robot.
    robot.arm_to.assert_not_called()
    robot.open_gripper.assert_not_called()
    robot.close_gripper.assert_not_called()


def test_checked_approach_plan_uses_real_shifted_base_without_commands(native, monkeypatch):
    monkeypatch.setenv('EMET_SIM_NAV_TELEPORT', '0')
    robot, data, context, _ = native
    robot.move_base_to = Mock(return_value=True)
    robot.cancel_navigation = Mock(return_value=True)
    model = context['_native_tamp_adapter'].model
    joint = model.body('object1').jntadr[0]
    robot._state['native_tamp_snapshot']['qpos'][model.jnt_qposadr[joint]] = .025
    adapter = NativeTableAdapter(robot, model=model, policy='bounded_approach')
    before, ctrl = adapter.data.qpos.copy(), data.ctrl.copy()
    result = adapter.plan(object_name='blue cube', receptacle_name='table')
    assert result['status'] == 'ok', result
    assert result['data']['base_motion'] == 'checked_unloaded_approach'
    plan = adapter.plans[result['data']['plan_ref']]
    assert plan.base_approach is not None
    assert 0 < np.linalg.norm(np.subtract(plan.base_approach.goal[:2], plan.base_approach.start[:2])) <= .08+1e-9
    assert adapter.planning_ik_calls <= 256
    np.testing.assert_array_equal(adapter.data.qpos, before)
    np.testing.assert_array_equal(data.ctrl, ctrl)
    for command in (robot.move_base_to, robot.cancel_navigation, robot.arm_to, robot.open_gripper, robot.close_gripper):
        command.assert_not_called()


def test_place_target_rotates_measured_payload_center_offset():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/><worldbody>
      <body name="ee" pos=".2 .1 .3" euler=".2 -.4 .7"><geom size=".01"/></body>
      <body name="desired" euler=".4 .2 -.6"/>
      <body name="payload" pos=".4 -.1 .6" euler=".1 .3 -.2"><freejoint/>
        <geom type="box" pos=".02 -.01 .03" size=".01 .02 .03"/>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adapter = NativeTableAdapter.__new__(NativeTableAdapter)
    adapter.model, adapter.data = model, data
    ee = data.body('ee')
    gid = int(model.body('payload').geomadr[0])
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    world_vertices = (corners*model.geom_size[gid]) @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
    attached_vertices = (world_vertices-ee.xpos) @ ee.xmat.reshape(3, 3)
    rotation = data.body('desired').xmat.reshape(3, 3).copy()
    goal, before = np.array([.02, -.65, .54]), data.qpos.copy()
    targets = adapter._place_targets(SimpleNamespace(ee_body='ee'), 'payload', goal, rotation)
    delivered = attached_vertices @ rotation.T + targets[1][0]
    np.testing.assert_allclose((delivered.min(0)+delivered.max(0))/2, goal, atol=1e-12)
    old_offset = (world_vertices.min(0)+world_vertices.max(0))/2-ee.xpos
    old_delivered = attached_vertices @ rotation.T + (goal-old_offset)
    assert np.linalg.norm((old_delivered.min(0)+old_delivered.max(0))/2-goal) > .01
    np.testing.assert_array_equal(data.qpos, before)


@pytest.mark.parametrize('exhausted', [False, True])
def test_loaded_placement_refresh_preserves_reference_and_shared_budget(native, monkeypatch, exhausted):
    from emet.controller.task.tamp import native_placement
    robot, _, context, call = native
    adapter = context['_native_tamp_adapter']
    plan = adapter.plans[make_plan(call)]
    plan.placement_region = np.array([[-.3, -.95], [.3, -.4]])
    ex = plan.executor
    ex.payload_body = plan.object_body
    ex.collision.set_payload(adapter.model, adapter.data, plan.object_body, ex.ee_body)
    before, reference = adapter.data.qpos.copy(), ex.collision.payload_transform.copy()
    attempts = []
    def paths(owner, executor, targets):
        assert owner.planning_ik_calls == plan.planning_ik_calls+3*len(attempts)
        assert executor.collision.payload_body == plan.object_body
        attempts.append(targets)
        owner.planning_ik_calls += 3
        owner.data.qpos[ex.qadr[0]] += .001
        ex.collision.set_payload(owner.model, owner.data, None)
        if exhausted:
            raise ValueError('no_plan_within_budget')
        return (None, 'pose_ik_failed') if len(attempts) == 1 else ([[], [], []], None)
    monkeypatch.setattr(native_placement, 'placement_paths', paths)
    result = native_placement.refine_loaded_placement(adapter, plan)
    assert result.success is (not exhausted)
    assert len(attempts) == (1 if exhausted else 2)
    assert adapter.planning_ik_calls == plan.planning_ik_calls+3*len(attempts)
    np.testing.assert_array_equal(adapter.data.qpos, before)
    np.testing.assert_array_equal(ex.collision.payload_transform, reference)
    assert ex.collision.payload_body == plan.object_body
    if not exhausted:
        assert np.all(plan.goal[:2] >= plan.placement_region[0])
        assert np.all(plan.goal[:2] <= plan.placement_region[1])
    for command in (robot.arm_to, robot.open_gripper, robot.close_gripper):
        command.assert_not_called()


def test_loaded_refinement_retains_valid_original_goal_when_sampler_has_no_proposals(native, monkeypatch):
    from emet.controller.task.tamp import native_placement
    _, _, context, call = native
    adapter = context['_native_tamp_adapter']
    plan = adapter.plans[make_plan(call)]
    ex = plan.executor
    ex.payload_body = plan.object_body
    ex.collision.set_payload(adapter.model, adapter.data, plan.object_body, ex.ee_body)
    goal = plan.goal.copy()
    monkeypatch.setattr(native_placement, 'free_surface_centers', lambda *a, **k: SimpleNamespace(centers=[]))
    planner = Mock(return_value=([[], [], []], None))
    monkeypatch.setattr(native_placement, 'placement_paths', planner)
    assert native_placement.refine_loaded_placement(adapter, plan).success
    planner.assert_called_once()
    np.testing.assert_array_equal(plan.goal, goal)


@pytest.mark.parametrize('started', [False, True])
def test_failed_base_approach_cannot_dispatch_grasp(native, monkeypatch, started):
    from emet.controller.task.tamp.native_approach import ApproachResult, BaseApproach
    robot, _, context, call = native
    adapter = context['_native_tamp_adapter']
    ref = make_plan(call)
    plan = adapter.plans[ref]
    plan.base_approach = BaseApproach((0., 0., 0.), (.04, 0., 0.))
    plan.executor.grasp_only = Mock(side_effect=AssertionError('grasp after failed approach'))
    monkeypatch.setattr('emet.controller.task.tamp.native_adapter.execute_approach',
                        lambda *args: ApproachResult(False, 'base_approach_failed', started))
    result = call('execute_pick_place_plan', plan_ref=ref)
    assert result['code'] == 'base_approach_failed'
    assert result['status'] == ('partial' if started else 'error')
    assert result['data']['execution_started'] is started
    assert result['data']['failed_stage'] == 'base_approach'
    assert call('execute_pick_place_plan', plan_ref=ref)['code'] == 'unknown_plan'
    plan.executor.grasp_only.assert_not_called()
    robot.arm_to.assert_not_called()


def test_policy_change_invalidates_cached_native_handles(native, monkeypatch):
    from emet.controller.task.tamp.native_adapter import adapter_for
    robot, _, context, _ = native
    old = context['_native_tamp_adapter']
    # Exercise cache/policy identity without aging the measured fixture during
    # an unrelated cold mesh reload. Production freshness checks stay active.
    def construct(client, *, model=None, **kwargs):
        assert model is old.model
        return NativeTableAdapter(client, model=model, **kwargs)
    monkeypatch.setattr('emet.controller.task.tamp.native_adapter.NativeTableAdapter', construct)
    robot._state['native_tamp_snapshot']['captured_at'] = time.time()
    robot._state_received_monotonic = time.monotonic()
    old.plans['old'] = object()
    context['native_tamp_policy'] = 'bounded_release'
    current = adapter_for(context, robot)
    assert current is not old and current.policy == 'bounded_release'
    assert current.execute('old')['code'] == 'unknown_plan'
    robot.arm_to.assert_not_called()


@pytest.mark.parametrize('measured', [.12, -.05, -.375])
def test_bounded_contact_hold_uses_measured_motor_position_and_normal_commands(native, measured):
    robot, _, context, _ = native
    robot.get_gripper_position.return_value = measured
    adapter = NativeTableAdapter(robot, model=context['_native_tamp_adapter'].model, policy='bounded_release')
    executor = adapter._executor('object1')
    before = adapter.data.qpos.copy()
    assert executor.gripper_close_command is None
    assert executor.gripper_hold_command() is True
    target = robot.gripper_to.call_args.args[0]
    assert target == pytest.approx(max(-.3, .5 * (measured - .3)))
    assert robot.gripper_to.call_args.kwargs == {'blocking': False}
    np.testing.assert_array_equal(adapter.data.qpos, before)
    robot.arm_to.assert_not_called()
    robot.close_gripper.assert_not_called()
    assert context['_native_tamp_adapter']._executor('object1').gripper_close_command is None
    assert context['_native_tamp_adapter']._executor('object1').gripper_hold_command is None


@pytest.mark.parametrize('feedback', [None, float('nan'), 'bad-feedback', -1., 1.])
def test_contact_hold_rejects_invalid_encoder_without_a_command(native, feedback):
    robot, _, context, _ = native
    adapter = NativeTableAdapter(robot, model=context['_native_tamp_adapter'].model, policy='bounded_release')
    robot.get_gripper_position.return_value = feedback
    assert adapter._executor('object1').gripper_hold_command() is False
    robot.gripper_to.assert_not_called()


@pytest.mark.parametrize('missing', ['gripper_to', 'get_gripper_position'])
def test_contact_hold_capability_is_required_before_planning_or_motion(native, missing):
    robot, _, context, _ = native
    adapter = NativeTableAdapter(robot, model=context['_native_tamp_adapter'].model, policy='bounded_release')
    setattr(robot, missing, None)
    assert adapter.plan(object_name='blue cube', receptacle_name='table')['code'] == 'native_execution_unavailable'
    robot.arm_to.assert_not_called()
    robot.open_gripper.assert_not_called()
    robot.close_gripper.assert_not_called()


def test_diagnostic_gripper_ablation_changes_only_closure_not_shared_limits(native):
    from emet.eval.contact_stability import TracedNativeAdapter
    robot, _, context, _ = native
    model = context['_native_tamp_adapter'].model
    adapter = TracedNativeAdapter(robot, model=model, trace=Mock(), native_policy='bounded_release', gripper_close_loose=False)
    assert adapter._executor('object1').gripper_close_command is None
    assert adapter._executor('object1').gripper_hold_command is None
    assert adapter.gripper_contact_hold_fraction == 0.
    assert adapter.arm_extension_margin_m == .0005 and adapter.payload_binding == 'verified_lift'
    assert adapter.release_limits.open_settle_timeout_s == 12.


@pytest.mark.parametrize('substage', ['post_lift_hold', 'loaded_payload_handoff', 'post_release_support'])
def test_native_verification_substage_is_public_and_does_not_leak_to_next_handle(native, monkeypatch, substage):
    from emet.controller.task.tamp.api import VerificationFailure
    _, _, context, call = native
    adapter = context['_native_tamp_adapter']
    ref = make_plan(call)
    ex = adapter.plans[ref].executor
    ex.grasp_only = Mock(return_value=SimpleNamespace(success=True))
    ex.place_only = Mock(return_value=SimpleNamespace(success=True))
    def reject():
        adapter.last_verification = VerificationFailure(substage, 'payload_not_retained')
        return False
    monkeypatch.setattr(adapter, '_verify', lambda *a, released, **k:
                        reject() if (released == (substage == 'post_release_support') and
                                     substage != 'loaded_payload_handoff') else True)
    if substage == 'loaded_payload_handoff':
        adapter.payload_binding = 'verified_lift'
        monkeypatch.setattr(adapter, '_bind_loaded_payload', lambda *a, **k: reject())
    result = call('execute_pick_place_plan', plan_ref=ref)
    assert result['status'] == 'partial' and result['code'] == 'verification_failed'
    assert result['data']['verification'] == {'substage': substage, 'reason': 'payload_not_retained'}
    assert 'object1' not in json.dumps(result)
    assert 'verification' not in call('execute_pick_place_plan', plan_ref=ref)['data']


@pytest.mark.parametrize('phase', ['preplace', 'place'])
def test_loaded_motion_retention_failure_has_its_own_witness(native, monkeypatch, phase):
    from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult
    _, _, context, call = native
    adapter = context['_native_tamp_adapter']
    ref = make_plan(call)
    ex = adapter.plans[ref].executor
    ex.grasp_only = Mock(return_value=PhysicalMotionResult(True, 'ok', 'lift'))
    ex.place_only = Mock(return_value=PhysicalMotionResult(False, 'payload_not_retained', phase))
    monkeypatch.setattr(adapter, '_verify', lambda *a, **k: True)
    result = call('execute_pick_place_plan', plan_ref=ref)
    assert result['status'] == 'partial' and result['code'] == 'payload_not_retained'
    assert result['data']['verification'] == {'substage': 'loaded_motion', 'reason': 'payload_not_retained'}
    assert result['data']['completed_ops'] == ['grasp']
    assert 'verification' not in call('execute_pick_place_plan', plan_ref=ref)['data']


def test_bounded_grasp_fallback_preserves_old_order_and_budget():
    adapter = object.__new__(NativeTableAdapter)
    adapter.policy, adapter.planning_ik_calls, adapter.max_planning_ik_calls = 'bounded_release', 0, 2
    adapter.pregrasp_fallback_offset_m = (0., 0., .08)
    seen = []
    def candidates(*args, approach_offset=(0., .08, .04)):
        adapter._plan_candidate_pose(SimpleNamespace(plan_pose=lambda *a, **k: ([], None)), None, None)
        seen.append(approach_offset)
        yield approach_offset, []
    adapter._grasp_candidates = candidates
    assert len(list(adapter._grasp_options(None, None, None, None, None, None))) == 2
    assert seen == [(0., .08, .04), (0., 0., .08)]
    assert adapter.planning_ik_calls == adapter.max_planning_ik_calls == 2
    with pytest.raises(ValueError, match='no_plan_within_budget'):
        list(adapter._grasp_options(None, None, None, None, None, None))


def test_legacy_grasp_search_has_no_extra_approach():
    adapter = object.__new__(NativeTableAdapter)
    adapter._grasp_candidates = Mock(return_value=iter([(object(), [])]))
    assert len(list(adapter._grasp_options(None, None, None, None, None, None))) == 1
    assert adapter._grasp_candidates.call_count == 1
    assert not adapter._grasp_candidates.call_args.kwargs


def test_primary_placement_cannot_spend_the_vertical_fallback_reserve(native, monkeypatch):
    _, _, context, call = native
    adapter = context['_native_tamp_adapter']
    adapter.pregrasp_fallback_offset_m = (0., 0., .08)
    adapter.max_planning_ik_calls = 8
    seen = []
    dummy = SimpleNamespace(plan_pose=lambda *a, **k: ([], None))
    def grasps(*args, approach_offset=None, **kwargs):
        seen.append(approach_offset)
        adapter._plan_candidate_pose(dummy, None, None)
        yield np.eye(3), [[], [], []]
    def placement(*args):
        if seen[-1] is None:
            while True:
                adapter._plan_candidate_pose(dummy, None, None)
        adapter._plan_candidate_pose(dummy, None, None)
        return [[], [], []], None
    monkeypatch.setattr(adapter, '_grasp_candidates', grasps)
    monkeypatch.setattr('emet.controller.task.tamp.native_adapter.placement_paths', placement)
    monkeypatch.setattr(surfaces_module, 'free_surface_centers',
        lambda *a, **k: SimpleNamespace(centers=[np.array([.15, -.6, .54])]))
    result = call('plan_pick_place', object_name='blue cube', receptacle_name='table')
    assert result['status'] == 'ok', result
    assert seen == [None, (0., 0., .08)] and adapter.planning_ik_calls == 8


@pytest.mark.parametrize("fault", [None, "new_obstacle", "camera_timeout"])
def test_public_plan_finalization_issues_only_fresh_authorized_handles(native, monkeypatch, fault):
    # Synthetic local-source evidence isolates the real public-tool boundary.
    # Native FK/collisions remain real; this is not observed scene qualification.
    robot, live, context, call = native
    adapter = context["_native_tamp_adapter"]
    old_ref = make_plan(call)
    clock = [100.0]
    monkeypatch.setattr(validation, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    store = LocalSceneStore(clock=lambda: clock[0] + 0.02)
    old = store.load("source", lambda: local_scene(100.0, object_ref="object1"), reserve_bytes=1_000_000)
    with old.borrow() as borrowed:
        captured = borrowed.support.capture
    factory = adapter._executor

    def executor(obj):
        ex = factory(obj)
        ex.planning_lease, ex.planning_capture = old, captured
        return ex

    adapter._executor = executor

    def reacquire():
        if fault == "camera_timeout":
            raise TimeoutError("read timed out before commands")
        clock[0] = 120.0
        robot._state["native_tamp_snapshot"].update(
            qpos=live.qpos.tolist(), qvel=live.qvel.tolist(), sim_time=float(live.time), captured_at=time.time()
        )
        robot._state_received_monotonic = time.monotonic()
        lease = store.load("source", lambda: local_scene(120.0, object_ref="object1"), reserve_bytes=1_000_000)
        return validation.RefreshedNativeScene(lease, (lambda c, m, d: fault == "new_obstacle",))

    adapter.reacquire_scene = reacquire
    task = call("scene_tasks", object_filter="blue")["data"]["tasks"][0]["task_ref"]
    before = [x.copy() for x in (live.qpos, live.qvel, live.ctrl)]
    result = call("plan_pick_place", task_ref=task)
    assert old_ref not in adapter.plans
    assert call("execute_pick_place_plan", plan_ref=old_ref)["code"] == "unknown_plan"
    with pytest.raises(LocalMapUnavailable):
        old.require_current()
    if fault is None:
        assert result["status"] == "ok", result
        ref = result["data"]["plan_ref"]
        assert ref != old_ref and ref in adapter.plans
        plan = adapter.plans[ref]
        assert len(plan.proposal.joints) == 8
        assert plan.planning_ik_calls == adapter.planning_ik_calls
        assert adapter.events[-1]["phase"] == "source_revalidation_complete"
        assert adapter.events[-1]["ik_calls"] == 0
        store.invalidate()
        rejected = call("execute_pick_place_plan", plan_ref=ref)
        assert rejected["data"]["execution_started"] is False
        assert rejected["status"] == "error"
        assert call("execute_pick_place_plan", plan_ref=ref)["code"] == "unknown_plan"
    else:
        assert result["status"] == "error", result
        assert result["data"]["execution_started"] is False
        assert not adapter.plans
    for a, b in zip(before, (live.qpos, live.qvel, live.ctrl), strict=True):
        np.testing.assert_array_equal(a, b)
    robot.arm_to.assert_not_called()
    robot.open_gripper.assert_not_called()
    robot.close_gripper.assert_not_called()


@pytest.mark.parametrize("fault", ["preflight_expiry", "transport_timeout", "cleared_admission"])
def test_refreshed_execution_reports_the_command_boundary(native, monkeypatch, fault):
    robot, live, context, call = native
    adapter = context["_native_tamp_adapter"]
    clock = [100.0]
    monkeypatch.setattr(validation, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    store = LocalSceneStore(clock=lambda: clock[0] + 0.02)
    old = store.load("source", lambda: local_scene(100.0, object_ref="object1"), reserve_bytes=1_000_000)
    with old.borrow() as scene:
        capture = scene.support.capture
    factory = adapter._executor

    def executor(obj):
        ex = factory(obj)
        ex.planning_lease, ex.planning_capture = old, capture
        return ex

    adapter._executor = executor

    def reacquire():
        clock[0] = 120.0
        robot._state["native_tamp_snapshot"]["captured_at"] = time.time()
        robot._state_received_monotonic = time.monotonic()
        lease = store.load("source", lambda: local_scene(120.0, object_ref="object1"), reserve_bytes=1_000_000)
        return validation.RefreshedNativeScene(lease, (lambda c, m, d: False,))

    adapter.reacquire_scene = reacquire
    ref = make_plan(call)
    plan = adapter.plans[ref]
    saved_paths = plan.executor.grasp_paths
    if fault == "cleared_admission":
        plan.evidence_lease = plan.admission = None

    def replan(*a, **kw):
        if fault == "preflight_expiry":
            clock[0] = 250.0
        return saved_paths, None

    monkeypatch.setattr(plan.executor, "_replan_at_measured_pose", replan)
    if fault == "transport_timeout":
        robot.open_gripper.side_effect = TimeoutError("send may have succeeded")
    result = call("execute_pick_place_plan", plan_ref=ref)
    assert result["data"]["completed_ops"] == []
    assert result["data"]["execution_started"] is (fault == "transport_timeout")
    assert result["status"] == ("partial" if fault == "transport_timeout" else "error")
    expected = {"transport_timeout": "command_outcome_unknown", "preflight_expiry": "stale_observation",
                "cleared_admission": "scene_changed_replan"}
    assert result["code"] == expected[fault]
    assert robot.open_gripper.called is (fault == "transport_timeout")
    robot.arm_to.assert_not_called()
    robot.close_gripper.assert_not_called()
    assert call("execute_pick_place_plan", plan_ref=ref)["code"] == "unknown_plan"


def test_registry_lookup_during_dispatch_preserves_adapter_and_namespace(native):
    robot, _, context, _ = native
    adapter = context["_native_tamp_adapter"]
    namespace = adapter.namespace
    executor = adapter._executor("object1")
    with dispatch_boundary(executor, Mock(), owner=adapter):
        assert adapter.robot is not robot
        assert adapter_for(context, robot) is adapter
        assert context["_native_tamp_adapter"] is adapter
        assert adapter.namespace == namespace
    assert adapter.robot is robot
    robot.arm_to.assert_not_called()

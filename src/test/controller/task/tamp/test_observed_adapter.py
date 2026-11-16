# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Real tool/executor integration with synthetic sensor and joint feedback.

The fixture's ideal hold/release model is not a contact or real perception trial.
No runtime code can read its object state except through observed evidence.
"""
import json
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from emet.agent.tools import get_tools
from emet.controller.manipulation.geometry_models import PrimitiveModel
from emet.controller.manipulation.native_inputs import ObjectPoseEvidence, ObservationUnavailable
from emet.controller.manipulation.observed_collision import ObservedNativeCollisionChecker, PayloadGeometryEvidence
from emet.controller.manipulation.observed_effects import CaptureInterval, RobotPoseSample, VisualEffectFrame
from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor
from emet.controller.manipulation.shape_pose import ShapePoseEvidence
from emet.controller.task.tamp.observed_adapter import ObservedNativeAdapter, ObservedNativeSnapshot, ObservedNativeTask
from emet.controller.task.tamp.transfer_provider import TransferRegion
from emet.motion.observed_contact import ObservedTargetContact
from emet.motion.placement_geometry import PlacementScene, box_corners


@pytest.fixture
def system(monkeypatch):
    import time

    clock = [100.]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(time, 'sleep', lambda dt: clock.__setitem__(0, clock[0] + dt))
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="robot">
      <joint name="x" type="slide" axis="1 0 0" range="-.5 .6"/>
      <joint name="y" type="slide" axis="0 1 0" range="-.5 .5"/>
      <joint name="z" type="slide" axis="0 0 1" range="0 .5"/>
      <geom size=".005"/><body name="ee"><geom size=".005"/>
        <body name="finger" pos="0 -.06 0"><joint name="opening" type="slide" axis="0 1 0" range="0 .02"/>
          <geom size=".001"/></body>
      </body></body></worldbody></mujoco>''')
    q, object_position = np.array([0., 0., .2, .02]), np.array([.2, 0., .02])
    objects = {'opaque': object_position, 'other': np.array([-.2, 0., .02])}
    data, feedback = mujoco.MjData(model), mujoco.MjData(model)
    calls = []
    state = SimpleNamespace(holding=False, drop=False, revision='scene:1', invalid=False,
                            missing=False, collect_unknown=False, generation=0, shape_model=None,
                            shape_calibration='test-only', shape_tilt=0.)
    scene = PlacementScene([], source='observed_voxels', workspace=[[-1]*3, [1]*3], known_free=lambda bounds: True)
    vertices = box_corners([[-.02]*3, [.02]*3])

    def robot_pose():
        feedback.qpos[:] = q
        mujoco.mj_kinematics(model, feedback)
        ee = feedback.body('ee')
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = ee.xmat.reshape(3, 3), ee.xpos
        return pose

    def snapshot(obj, support):
        if state.invalid or state.missing:
            raise ObservationUnavailable('observed_scene_unavailable')
        clock[0] += .004
        t = clock[0] - .002
        state.generation += 1
        obs_id = f'camera:{state.generation}'
        robot_pose()
        pose = np.eye(4)
        pose[:3, 3] = objects[obj]
        if state.shape_model is None:
            evidence = ObjectPoseEvidence(pose, t+.001, 'rgbd_tracking', obs_id)
        else:
            pose[:3, :3] = Rotation.from_euler('zy', [state.generation * .37,
                state.shape_tilt if state.holding else 0.]).as_matrix()
            evidence = ShapePoseEvidence(pose, t+.001, 'rgbd_tracking', obs_id, state.shape_model,
                .0001, .0001, state.shape_calibration)
        payload = PayloadGeometryEvidence(obj, 'measured-reference', 'prior', vertices,
            evidence, 'known_model_prior')
        frame = VisualEffectFrame(payload, CaptureInterval(obs_id, t, t+.0005, t+.001, 'clock'),
            RobotPoseSample.from_model(model, feedback, ee_body='ee', gripper_bodies=('ee',), received_monotonic=t-.001),
            RobotPoseSample.from_model(model, feedback, ee_body='ee', gripper_bodies=('ee',), received_monotonic=t+.002), .0001, .0001, .0001, .0001)
        return ObservedNativeSnapshot(frame, scene, state.revision, np.array([[-.5, -.5, -.1], [.6, .5, 0.]]))

    def collect(obj, support, after):
        if state.collect_unknown:
            return ()
        clock[0] = max(clock[0], after) + .01
        a = snapshot(obj, support).frame
        clock[0] += .4
        return a, snapshot(obj, support).frame

    def command(target):
        calls.append(('joints', target.copy()))
        q[:3] = target
        if state.holding and not state.drop:
            objects[state.active][:] = robot_pose()[:3, 3] + state.offset
        return True

    def open_gripper(**kwargs):
        calls.append(('open',))
        state.holding = False
        q[3] = .02
        return True

    def close_gripper(**kwargs):
        calls.append(('close',))
        state.holding = True
        ee = robot_pose()[:3, 3]
        state.active = min(objects, key=lambda obj: np.linalg.norm(objects[obj] - ee))
        state.offset = objects[state.active] - ee
        q[3] = 0
        return True

    robot = SimpleNamespace(open_gripper=open_gripper, close_gripper=close_gripper,
                            get_gripper_position=lambda: float(q[3]))
    source = SimpleNamespace(input_mode='observed', snapshot=snapshot, collect=collect,
        current_revision=lambda: state.revision, invalidate=lambda: setattr(state, 'invalid', True))
    provider = SimpleNamespace(input_mode='observed', observe_payload=lambda obj: snapshot(obj, 'support').frame.payload)

    def refine(task, initial, placement_region=None):
        def sync(d):
            d.qpos[:] = q
            mujoco.mj_kinematics(model, d)
        sync(data)
        checker = ObservedNativeCollisionChecker(model, robot_body='robot', ee_body='ee', scene=scene,
            payload_provider=provider, scene_revision=state.revision, current_revision=lambda: state.revision,
            received_monotonic=initial.frame.capture.received_monotonic, margin_m=0, clock=lambda: clock[0])
        ex = PhysicalPickPlaceExecutor(robot, model=model, data=data, ee_body='ee', joint_names=['x', 'y', 'z'],
            collision=checker, synchronize=sync, command_joints=command, object_pose_provider=checker,
            gripper_open_configuration=(('opening', .02),), position_tolerance_m=.0001)
        grasp = initial.frame.payload.pose.world_from_object[:3, 3] - [.05, 0, 0]
        place = np.array([.35, 0, .02])
        if placement_region is not None:
            place[:2] = np.mean(placement_region, axis=0) - [.05, 0]
        ex.grasp_targets = [(grasp + delta, np.eye(3)) for delta in ([0, .08, .04], [0, 0, 0], [0, 0, .1])]
        ex.place_targets = [(place + delta, np.eye(3)) for delta in ([0, 0, .1], [0, 0, 0], [0, .1, .1])]
        return ex

    task = ObservedNativeTask('opaque', 'support', 'block', 'counter')
    adapter = ObservedNativeAdapter(robot, source=source, refine=refine, tasks=[task], clock=lambda: clock[0])
    context = {'robot': robot, 'manip_mode': 'physical', 'tamp_inputs': 'observed', 'observed_native_adapter': adapter}
    tools = {t.name: t for t in get_tools(context)}
    return SimpleNamespace(adapter=adapter, source=source, state=state, calls=calls, q=q,
                           tools=tools, context=context, clock=clock, object_position=object_position, objects=objects)


def plan(system):
    return json.loads(system.tools['plan_pick_place'].func(object_name='block', receptacle_name='counter'))


@pytest.mark.parametrize('change', ['pose', 'extent', 'uncertainty', 'capture', 'identity'])
def test_contact_layer_must_enclose_same_captured_payload(system, change):
    task = next(iter(system.adapter.tasks.values()))
    initial = system.source.snapshot(task.object_ref, task.support_ref)
    contact = ObservedTargetContact.from_frame(initial.frame, model_signature='robot',
        calibration_id='geometry', fingertip_geom_ids=(0,), max_penetration_m=.002)
    replace(initial, target_contact=contact)  # Correct constructor binds all bounds.
    if change == 'pose':
        pose = contact.target.world_from_local.copy()
        pose[0, 3] += .1
        contact = replace(contact, target=replace(contact.target, world_from_local=pose))
    elif change == 'extent':
        contact = replace(contact, target=replace(contact.target, vertices=contact.target.vertices/2))
    elif change == 'uncertainty':
        contact = replace(contact, target=replace(contact.target, offset_m=0.))
    elif change == 'capture':
        contact = replace(contact, capture=replace(contact.capture, observation_id='different'))
    else:
        contact = replace(contact, object_ref='another-object')
    with pytest.raises(ValueError, match='invalid_observed_target_contact'):
        replace(initial, target_contact=contact)
    assert system.calls == []


@pytest.mark.parametrize('change', ['calibration', 'fingers', 'penetration', 'missing'])
def test_contact_profile_cannot_change_during_refresh(system, change):
    task = next(iter(system.adapter.tasks.values()))
    raw = system.source.snapshot
    initial = raw(task.object_ref, task.support_ref)
    ex = system.adapter.refine(task, initial)
    from emet.controller.manipulation.native_scene import model_signature

    def layer(frame):
        return ObservedTargetContact.from_frame(frame, model_signature=model_signature(ex.model),
            calibration_id='geometry', fingertip_geom_ids=(ex.collision.geom_ids[0],), max_penetration_m=.002)
    initial = replace(initial, target_contact=layer(initial.frame))
    ex.collision.set_target_contact(initial.target_contact)
    def changed(*args):
        snapshot = raw(*args)
        contact = layer(snapshot.frame)
        if change == 'calibration':
            contact = replace(contact, calibration_id='changed')
        elif change == 'fingers':
            contact = replace(contact, fingertip_geom_ids=frozenset((ex.collision.geom_ids[0]+1,)))
        elif change == 'penetration':
            contact = replace(contact, max_penetration_m=.004)
        else:
            contact = None
        return replace(snapshot, target_contact=contact)
    system.source.snapshot = changed
    with pytest.raises(ObservationUnavailable, match='target_contact_identity_changed'):
        system.adapter._refresh_checker(ex, task)
    assert ex.collision.target_contact is initial.target_contact and system.calls == []
    assert ex.collision.received_monotonic == initial.frame.capture.received_monotonic


def test_bounded_gripper_transport_timeout_is_possible_dispatch_and_consumes_plan(system):
    planned = plan(system)
    executor = system.adapter.plans[planned['data']['plan_ref']].executor
    entered = []
    def transport(q):
        entered.append(q)
        raise TimeoutError('private transport detail')
    executor.command_gripper_joints = transport
    executor.grasp_only = lambda *args, **kwargs: executor.command_gripper_joints(np.array([.01]))
    result = system.adapter.execute(planned['data']['plan_ref'])
    assert entered and result['status'] == 'partial' and result['code'] == 'command_outcome_unknown'
    assert result['data']['execution_started'] and result['data']['completed_ops'] == []
    assert system.calls == [] and 'private' not in json.dumps(result)
    assert executor.command_gripper_joints is transport
    assert system.adapter.execute(planned['data']['plan_ref'])['code'] == 'unknown_plan'


@pytest.mark.parametrize('selector,accepted', [('stretch', True), (' Stretch ', True), ('   ', True), ('other', False)])
def test_configured_observed_robot_selector_matches_primitive_and_task_discovery(system, selector, accepted):
    old = system.adapter
    adapter = ObservedNativeAdapter(old.robot, source=old.source, refine=old.refine,
        tasks=tuple(old.tasks.values()), clock=old.clock, robot_name='Stretch')
    primitive = adapter.discover(robot=selector)
    assert (primitive['status'] == 'ok') is accepted
    session = adapter.bind_regions((TransferRegion('left', ((-.3, -.08), (-.1, .08))),),
        robot_profile_id='calibrated', reset_id='reset')
    assert (session.discover(robot=selector)['status'] == 'ok') is accepted
    assert system.calls == []


def test_unconfigured_observed_robot_identity_is_not_invented_from_a_profile(system):
    session = system.adapter.bind_regions((TransferRegion('left', ((-.3, -.08), (-.1, .08))),),
        robot_profile_id='stretch-like-name-is-not-identity', reset_id='reset')
    assert session.discover(robot='stretch')['code'] == 'task_filter_unsupported'


def test_registered_tools_plan_and_execute_with_visual_effects_and_no_simulator(system):
    discovery = json.loads(system.tools['scene_tasks'].func())
    assert discovery['status'] == 'ok' and discovery['data']['inputs'] == 'observed'
    planned = plan(system)
    assert planned['status'] == 'ok', planned
    assert system.calls == []
    executed = json.loads(system.tools['execute_pick_place_plan'].func(plan_ref=planned['data']['plan_ref']))
    assert executed['status'] == 'ok', executed
    assert executed['data']['completed_ops'] == ['grasp', 'place']
    assert [e['reason'] for e in executed['data']['measurements']] == [
        'visual_grasp_reference_recorded', 'visual_lift_verified', 'visual_placement_verified']
    assert system.calls and system.object_position[0] == pytest.approx(.4, abs=.002)
    assert system.adapter.plans == {}


@pytest.mark.parametrize('kind,dimensions', [('cylinder', (.02, .02)), ('sphere', (.02,))])
def test_shape_symmetric_evidence_executes_without_inventing_material_yaw(system, kind, dimensions):
    system.state.shape_model = PrimitiveModel(kind, dimensions)
    planned = plan(system)
    assert planned['status'] == 'ok', planned
    assert planned['data']['payload_pose_scope'] == 'declared_shape_equivalence'
    result = json.loads(system.tools['execute_pick_place_plan'].func(plan_ref=planned['data']['plan_ref']))
    assert result['status'] == 'ok', result
    assert result['data']['completed_ops'] == ['grasp', 'place']
    assert result['data']['payload_geometry_source'] == 'known_model_prior'


def test_shape_symmetry_does_not_hide_axis_slip_or_changed_calibration(system):
    system.state.shape_model = PrimitiveModel('cylinder', (.02, .02))
    planned = plan(system)
    assert planned['status'] == 'ok', planned
    system.state.shape_calibration = 'replacement'
    result = system.adapter.execute(planned['data']['plan_ref'])
    assert result['status'] == 'error' and not result['data']['execution_started'], result
    assert system.calls == []
    planned = plan(system)
    assert planned['status'] == 'ok', planned
    system.state.shape_tilt = .4
    result = system.adapter.execute(planned['data']['plan_ref'])
    assert result['status'] == 'partial' and result['code'] == 'visual_lift_not_verified', result
    assert result['data']['completed_ops'] == []


def test_changed_scene_and_unknown_initial_evidence_are_no_actuation(system):
    planned = plan(system)
    assert planned['status'] == 'ok', planned
    system.state.revision = 'scene:changed'
    result = system.adapter.execute(planned['data']['plan_ref'])
    assert result['status'] == 'error' and not result['data']['execution_started']
    assert system.calls == []
    assert system.adapter.execute(planned['data']['plan_ref'])['code'] == 'unknown_plan'


def test_missing_visual_effect_after_commands_is_partial_not_success(system):
    planned = plan(system)
    assert planned['status'] == 'ok', planned
    system.state.collect_unknown = True
    result = system.adapter.execute(planned['data']['plan_ref'])
    assert result['status'] == 'partial' and result['code'] == 'visual_effect_unknown', result
    assert result['data']['execution_started'] and result['data']['completed_ops'] == []
    assert sum(c[0] == 'open' for c in system.calls) == 1  # No placement/release after unverified lift.


@pytest.mark.parametrize('shape', [None, PrimitiveModel('sphere', (.02,)), PrimitiveModel('cylinder', (.02, .02))])
def test_empty_grasp_never_reaches_placement(system, shape):
    system.state.shape_model = shape
    planned = plan(system)
    assert planned['status'] == 'ok', planned
    system.state.drop = True
    result = system.adapter.execute(planned['data']['plan_ref'])
    assert result['status'] == 'partial' and result['code'] == 'payload_not_retained', result
    assert result['data']['completed_ops'] == []


def test_motion_invalidates_observed_adapter_and_stored_handles(system):
    from emet.agent.tamp_world import task_world_change

    planned = plan(system)
    assert planned['status'] == 'ok', planned
    with task_world_change('rotate_base', system.context):
        pass
    assert system.adapter.plans == {} and system.state.invalid
    assert system.adapter.execute(planned['data']['plan_ref'])['code'] == 'unknown_plan'


def test_missing_snapshot_never_falls_back_to_privileged_discovery(system):
    system.state.missing = True
    result = json.loads(system.tools['scene_tasks'].func())
    assert result['status'] == 'error' and system.calls == []


def test_mismatched_capture_identity_is_rejected_before_planning(system):
    raw = system.source.snapshot

    def snapshot(*args):
        snapshot = raw(*args)
        return replace(snapshot, frame=replace(snapshot.frame, capture=replace(snapshot.frame.capture, observation_id='other')))

    system.source.snapshot = snapshot
    result = plan(system)
    assert result['status'] == 'error' and system.calls == []


def test_same_tools_execute_three_step_observed_swap(system):
    from emet.controller.task.tamp.transfer_provider import TransferRegion

    system.adapter.tasks['task:other'] = ObservedNativeTask('other', 'support', 'other', 'counter')
    regions = (TransferRegion('left', ((-.28, -.08), (-.12, .08))),
               TransferRegion('right', ((.12, -.08), (.28, .08))),
               TransferRegion('buffer', ((.32, -.08), (.48, .08))))
    session = system.adapter.bind_regions(regions, robot_profile_id='calibrated-robot', reset_id='fixture:1')
    system.context['tamp_task_session'] = session
    planned = json.loads(system.tools['plan_task'].func(instruction='swap block and other'))
    assert planned['status'] == 'ok' and planned['data']['steps'] == 3, planned
    assert system.calls == []
    executed = json.loads(system.tools['execute_task_plan'].func(plan_ref=planned['data']['plan_ref']))
    assert executed['status'] == 'ok' and executed['data']['completed_steps'] == 3, executed
    assert system.objects['opaque'][0] == pytest.approx(-.2, abs=.002)
    assert system.objects['other'][0] == pytest.approx(.2, abs=.002)


@pytest.mark.parametrize('model', [PrimitiveModel('cylinder', (.02, .02)),
    PrimitiveModel('sphere', (.02,)), PrimitiveModel('box', (.02, .02, .02))])
def test_task_grounding_preserves_equivalent_shape_representatives(system, model):
    from emet.controller.task.tamp.transfer_provider import TransferRegion

    system.state.shape_model = model
    raw = system.source.snapshot
    if model.kind == 'box':
        def snapshot(*args):
            current = raw(*args)
            pose = current.frame.payload.pose
            matrix = pose.world_from_object.copy()
            matrix[:3, :3] = Rotation.from_euler('z', system.state.generation*np.pi/2).as_matrix()
            return replace(current, frame=replace(current.frame,
                payload=replace(current.frame.payload, pose=replace(pose, world_from_object=matrix))))
        system.source.snapshot = snapshot
        def collect(obj, support, after):
            system.clock[0] = max(system.clock[0], after) + .01
            first = snapshot(obj, support).frame
            system.clock[0] += .4
            return first, snapshot(obj, support).frame
        system.source.collect = collect
    session = system.adapter.bind_regions((TransferRegion('left', ((-.28, -.08), (-.12, .08))),),
        robot_profile_id='calibrated-robot', reset_id='fixture:1')
    first = session.provider.snapshot()
    second = session.provider.snapshot()
    assert first.binding == second.binding
    assert system.calls == []


@pytest.mark.parametrize('fault', [TimeoutError('planning timeout'), TypeError('private refiner detail'),
    None, {'status': 'ok', 'code': 'ok', 'data': {}}, {'status': 'ok', 'code': 'ok', 'data': {'plan_ref': 7}}])
def test_observed_planning_failures_report_no_actuation(system, monkeypatch, fault):
    from emet.controller.task.tamp.transfer_provider import TransferRegion

    session = system.adapter.bind_regions((TransferRegion('left', ((-.28, -.08), (-.12, .08))),),
        robot_profile_id='calibrated-robot', reset_id='fixture:1')
    planned = session.plan('move block to left')
    assert planned['status'] == 'ok'
    def fail(**kwargs):
        if isinstance(fault, Exception):
            raise fault
        return fault
    monkeypatch.setattr(system.adapter, 'plan', fail)
    executed = session.execute(planned['data']['plan_ref'])
    assert executed['status'] == 'error' and executed['data']['execution_started'] is False
    assert executed['data']['completed_steps'] == 0
    assert executed['data']['failed_phase'] == 'preflight'
    assert executed['data']['failed_step_execution_started'] is False
    assert system.calls == []
    assert 'private refiner detail' not in json.dumps(executed)


def test_missing_dispatch_flag_after_execute_remains_possible_actuation(system, monkeypatch):
    from emet.controller.task.tamp.api import response
    from emet.controller.task.tamp.transfer_provider import TransferRegion

    session = system.adapter.bind_regions((TransferRegion('left', ((-.28, -.08), (-.12, .08))),),
        robot_profile_id='calibrated-robot', reset_id='fixture:1')
    planned = session.plan('move block to left')
    monkeypatch.setattr(system.adapter, 'plan', lambda **kwargs:
        response('plan_pick_place', data={'plan_ref': 'returned-handle'}))
    monkeypatch.setattr(system.adapter, 'execute', lambda ref:
        response('execute_pick_place_plan', code='command_outcome_unknown', data={}))
    executed = session.execute(planned['data']['plan_ref'])
    assert executed['status'] == 'partial' and executed['data']['execution_started'] is True
    assert executed['data']['failed_step_execution_started'] is True


def test_unique_visual_frame_errors_reach_the_retention_provider(system):
    raw = system.source.snapshot
    def snapshot(*args):
        current = raw(*args)
        return replace(current, frame=replace(current.frame, position_error_m=.006, rotation_error_rad=.02))
    system.source.snapshot = snapshot
    planned = plan(system)
    assert planned['status'] == 'ok', planned
    ex = system.adapter.plans[planned['data']['plan_ref']].executor
    evidence = ex.collision.provider.observe_payload('opaque')
    assert evidence.position_error_m == .006 and evidence.rotation_error_rad == .02


def test_visual_frame_cannot_downgrade_unique_payload_error_bounds(system):
    raw = system.source.snapshot
    def snapshot(*args):
        current = raw(*args)
        payload = replace(current.frame.payload, position_error_m=.006, rotation_error_rad=.03)
        return replace(current, frame=replace(current.frame, payload=payload))
    system.source.snapshot = snapshot
    result = plan(system)
    assert result['status'] == 'error' and result['code'] == 'object_pose_ambiguous'
    assert system.calls == []


@pytest.mark.parametrize('change', ['material_calibration', 'configuration_profile', 'epoch'])
def test_capture_refresh_cannot_replace_the_pinned_material_profile(system, change):
    from emet.controller.manipulation.native_scene import model_signature
    from emet.motion.observed_admission import CapturedRobotMaterial, ObservedMaterialAdmission, RobotGeometryBound
    from emet.motion.observed_background import ObservedBackground
    from emet.motion.observed_solids import WorldSolid
    from emet.motion.observed_workspace import ObservedWorkspace

    task = next(iter(system.adapter.tasks.values()))
    raw = system.source.snapshot
    initial = raw(task.object_ref, task.support_ref)
    ex = system.adapter.refine(task, initial)
    signature = model_signature(ex.model)
    def snapshot(frame_snapshot, *, changed=False):
        capture = frame_snapshot.frame.capture
        epoch = 'new-epoch' if changed and change == 'epoch' else 'epoch'
        calibration = 'new-calibration' if changed and change == 'material_calibration' else 'calibration'
        profile = 'new-profile' if changed and change == 'configuration_profile' else 'profile'
        grid = ObservedWorkspace([-1.]*3, .2, np.ones((10, 10, 10), np.uint8))
        background = ObservedBackground(grid, np.full(grid.cells.shape, capture.earliest), epoch,
            frozenset((capture.observation_id,)), capture)
        solids = tuple(WorldSolid.from_geom(ex.model, ex.data, g) for g in range(ex.model.ngeom))
        material = CapturedRobotMaterial(capture, signature, calibration, epoch, solids,
            (0.,)*len(solids), profile)
        admission = ObservedMaterialAdmission(background, material,
            query_robot_geometry=lambda m, d, g: RobotGeometryBound(WorldSolid.from_geom(m, d, g),
                g, signature, calibration, profile, 0.))
        return replace(frame_snapshot, scene=background.placement_scene(clock=system.adapter.clock), admission=admission)
    initial = snapshot(initial)
    ex.collision.admission, ex.collision.scene = initial.admission, initial.scene
    ex.collision.scene_revision = initial.scene_revision
    ex.collision.received_monotonic = initial.frame.capture.received_monotonic
    current = snapshot(raw(task.object_ref, task.support_ref), changed=True)
    system.source.snapshot = lambda *args: current
    with pytest.raises(ObservationUnavailable, match='robot_material_identity_changed'):
        system.adapter._refresh_checker(ex, task)
    assert ex.collision.admission is initial.admission and ex.collision.scene is initial.scene
    assert ex.collision.received_monotonic == initial.frame.capture.received_monotonic
    assert system.calls == []


def test_unknown_region_space_is_not_offered_as_free(system):
    from emet.controller.task.tamp.transfer_provider import TransferRegion

    raw = system.source.snapshot
    def snapshot(*args):
        current = raw(*args)
        scene = PlacementScene([], source='observed_voxels', workspace=[[-1]*3, [1]*3], known_free=lambda bounds: False)
        return replace(current, scene=scene)
    system.source.snapshot = snapshot
    session = system.adapter.bind_regions((TransferRegion('left', ((-.28, -.08), (-.12, .08))),),
        robot_profile_id='calibrated-robot', reset_id='fixture:1')
    result = session.discover()
    assert result['status'] == 'ok' and result['data']['free_regions'] == []
    planned = session.plan('move block to left')
    assert planned['status'] == 'error' and system.calls == []


def test_initial_held_or_unsupported_object_cannot_start_new_pick(system):
    system.object_position[2] = .12
    result = plan(system)
    assert result['code'] == 'task_object_not_settled_on_support' and system.calls == []


def test_timeout_before_first_command_stays_no_actuation(system):
    planned = plan(system)
    assert planned['status'] == 'ok', planned
    def timed_out(*args):
        raise TimeoutError('private service detail')
    system.source.snapshot = timed_out
    result = system.adapter.execute(planned['data']['plan_ref'])
    assert result['status'] == 'error' and not result['data']['execution_started']
    assert system.calls == [] and 'private' not in json.dumps(result)


@pytest.mark.parametrize('command_name', ['open_gripper', 'close_gripper', 'joints'])
@pytest.mark.parametrize('sent', [False, True])
def test_transport_entry_timeout_cannot_prove_no_actuation(system, command_name, sent):
    planned = plan(system)
    assert planned['status'] == 'ok', planned
    executor = system.adapter.plans[planned['data']['plan_ref']].executor
    owner, name = (executor, 'command_joints') if command_name == 'joints' else (system.adapter.robot, command_name)
    original = getattr(owner, name)
    entered = []
    def timeout(*args, **kwargs):
        entered.append(True)
        if sent:
            original(*args, **kwargs)
        raise TimeoutError('private transport detail')
    setattr(owner, name, timeout)
    result = system.adapter.execute(planned['data']['plan_ref'])
    assert entered and result['status'] == 'partial' and result['code'] == 'command_outcome_unknown', result
    assert result['data']['execution_started'] and result['data']['completed_ops'] == []
    assert 'private' not in json.dumps(result)
    if command_name == 'open_gripper':
        assert bool(system.calls) == sent


def test_region_refiner_must_put_full_payload_inside_requested_region(system):
    planned = system.adapter.plan(object_name='block', receptacle_name='counter',
                                  placement_region=[[.1, -.005], [.12, .005]])
    assert planned['status'] == 'error' and planned['code'] == 'placement_region_unavailable'
    assert system.calls == []


def test_scene_change_during_visual_batch_cannot_certify_effect(system):
    planned = plan(system)
    assert planned['status'] == 'ok', planned
    raw = system.source.collect
    def collect(*args, **kwargs):
        frames = raw(*args, **kwargs)
        system.state.revision = 'scene:changed-during-effect'
        return frames
    system.source.collect = collect
    result = system.adapter.execute(planned['data']['plan_ref'])
    assert result['status'] == 'partial' and result['code'] == 'visual_effect_unknown', result
    assert result['data']['completed_ops'] == []


def test_robot_model_calibration_change_invalidates_plan_before_commands(system):
    planned = plan(system)
    assert planned['status'] == 'ok', planned
    system.adapter.plans[planned['data']['plan_ref']].executor.model.geom_size[0, 0] *= 2
    result = system.adapter.execute(planned['data']['plan_ref'])
    assert result['status'] == 'error' and result['code'] == 'scene_changed_replan'
    assert not result['data']['execution_started'] and system.calls == []


def test_camera_clock_calibration_change_invalidates_plan_before_commands(system):
    planned = plan(system)
    assert planned['status'] == 'ok', planned
    raw = system.source.snapshot
    def snapshot(*args):
        value = raw(*args)
        return replace(value, frame=replace(value.frame,
            capture=replace(value.frame.capture, calibration_id='reconnected-clock')))
    system.source.snapshot = snapshot
    result = system.adapter.execute(planned['data']['plan_ref'])
    assert result['status'] == 'error' and result['code'] == 'scene_changed_replan'
    assert not result['data']['execution_started'] and system.calls == []

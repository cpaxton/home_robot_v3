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
from emet.motion.observed_admission import CapturedRobotMaterial, ObservedMaterialAdmission, RobotGeometryBound
from emet.motion.observed_background import ObservedBackground
from emet.motion.observed_contact import ObservedTargetContact
from emet.motion.observed_solids import WorldSolid
from emet.motion.observed_support import fit_observed_support
from emet.motion.observed_support_contact import ObservedSupportContact
from emet.motion.observed_workspace import ObservedWorkspace
from emet.motion.placement_geometry import PlacementScene, box_corners


@pytest.fixture
def system(monkeypatch, request):
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
    scene = PlacementScene([], source=getattr(request, 'param', 'observed_voxels'), workspace=[[-1]*3, [1]*3], known_free=lambda bounds: True)
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


def test_pickup_support_can_differ_from_destination(system):
    from dataclasses import replace

    original = system.source.snapshot

    def snapshot(obj, support):
        before = original(obj, support)
        # Same arm and payload as the positive control, with disjoint
        # rectangles on the two sides of the workspace.
        return replace(before, source_support_bounds=[[0.1, -0.1, -0.1], [0.3, 0.1, 0.0]],
                       support_bounds=[[0.3, -0.1, -0.1], [0.5, 0.1, 0.0]])

    system.source.snapshot = snapshot
    result = plan(system)
    assert result['status'] == 'ok', result
    assert not system.calls
    release = system.adapter.execute(result['data']['plan_ref'])
    assert release['status'] == 'ok', release


def test_ranked_transfer_selects_only_a_complete_pair(system):
    from emet.controller.task.tamp.observed_arm import TransferCandidate

    task = next(iter(system.adapter.tasks.values()))
    original = system.adapter.refine

    def refine(task, snapshot):
        ex = original(task, snapshot)
        # The first grasp is fine but its destination is far beyond reach.
        close = (('opening', 0.0),)
        unreachable = tuple((point + [2., 0., 0.], rot) for point, rot in ex.place_targets)
        ex.transfer_candidates = (
            TransferCandidate(tuple(ex.grasp_targets), unreachable, close),
            TransferCandidate(tuple(ex.grasp_targets), tuple(ex.place_targets), close),
        )
        return ex

    system.adapter.refine = refine
    result = plan(system)
    assert result['status'] == 'ok', result
    assert result['data']['planning_ik_calls'] > 6
    assert system.calls == []
    stored = system.adapter.plans[result['data']['plan_ref']]
    assert stored.task == task
    assert stored.executor.place_targets[1][0][0] == pytest.approx(.35)


def support_snapshot(system):
    from emet.controller.manipulation.native_scene import model_signature

    task = next(iter(system.adapter.tasks.values()))
    raw = system.source.snapshot
    initial = raw(task.object_ref, task.support_ref)
    ex = system.adapter.refine(task, initial)
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 1.
    k = np.array([[40., 0., 20.], [0., 40., 20.], [0., 0., 1.]])
    obs = SimpleNamespace(rgb=np.zeros((40, 40, 3), np.uint8), depth=np.ones((40, 40)), camera_K=k,
                          camera_pose=pose, received_monotonic=initial.frame.capture.received_monotonic)
    patch = fit_observed_support(obs, np.ones((40, 40), bool),
        observation_id=initial.frame.capture.observation_id, measurement_margin_m=.0001)
    grid = ObservedWorkspace.from_depth(obs.depth, k, pose, workspace=[[-.3, -.3, -.025], [.3, .3, .225]],
                                       resolution=.05, depth_margin_m=.0001)
    # No contact query runs in these binding tests; legacy occupied cells remain
    # unattributed so this fixture cannot accidentally establish support admission.
    background = ObservedBackground(grid, np.where(grid.cells != 0, initial.frame.capture.earliest, np.nan),
                                    'epoch', frozenset((initial.frame.capture.observation_id,)), initial.frame.capture)
    material = CapturedRobotMaterial(initial.frame.capture, model_signature(ex.model), 'robot-calibration', 'epoch',
        (WorldSolid('sphere', (.001,), initial.frame.robot_after.world_from_ee),), (0.,), 'robot-profile')
    admission = ObservedMaterialAdmission(background, material, query_robot_geometry=lambda m, d, g:
        RobotGeometryBound(WorldSolid.from_geom(m, d, g), g, model_signature(m), 'robot-calibration', 'robot-profile', 0.))
    contact = ObservedSupportContact(task.object_ref, task.support_ref, 'epoch', 'support-calibration', patch,
        initial.frame.capture, obs.depth, k, pose, np.ones((40, 40), bool), payload_motion_error_m=0.,
        backing_depth_m=.02, backing_prior_id='fixture')
    initial = replace(initial, scene=background.placement_scene(clock=lambda: system.clock[0]),
                      support_bounds=patch.bounds, admission=admission, support_contact=contact)
    ex.collision.admission, ex.collision.scene = admission, initial.scene
    ex.collision.set_support_contact(contact)
    return task, initial, ex, raw


def test_material_source_preserves_separate_support_roles(system):
    from emet.controller.task.tamp.observed_source import ObservedTransferEvidence, ObservedTransferSource

    task, initial, _, _ = support_snapshot(system)
    dest = initial.support_contact
    source = replace(dest, support_ref='counter')
    evidence = ObservedTransferEvidence(initial.frame, initial.admission.background,
        initial.admission.robot_material, source, dest, initial.admission.query_robot_geometry)
    def clock():
        return system.clock[0]

    arm_source = ObservedTransferSource(read=lambda *_: evidence, collect=lambda *_, **__: (),
        current_revision=lambda: evidence.background.geometry_revision, invalidate=lambda: None,
        model_signature=evidence.robot_material.model_signature, fingertip_geom_ids=(0,), clock=clock)
    observed = arm_source.snapshot(task.object_ref, task.support_ref)
    assert observed.source_support_contact is source
    assert observed.support_contact is dest
    system.clock[0] += 3
    with pytest.raises(ObservationUnavailable, match='object_observation_stale'):
        arm_source.snapshot(task.object_ref, task.support_ref)


def test_transfer_candidates_can_retreat_farther_than_the_pickup_lift(system):
    from emet.controller.task.tamp.observed_candidates import CalibratedGrasp, candidates_from_supports

    _, snapshot, _, _ = support_snapshot(system)
    profile = CalibratedGrasp(np.eye(4), (0., 0., -1.), (('opening', 0.),), lift_m=.10, retreat_m=.235)
    (candidate,) = candidates_from_supports(snapshot, [profile], limit=1)
    assert candidate.grasp[2][0][2] - candidate.grasp[1][0][2] == pytest.approx(.10)
    assert candidate.place[2][0][2] - candidate.place[1][0][2] == pytest.approx(.235)


@pytest.mark.parametrize('change', ['calibration', 'mask', 'backing', 'missing'])
def test_support_profile_change_stops_refresh_before_mutation_or_commands(system, change):
    task, initial, ex, _ = support_snapshot(system)
    contact = initial.support_contact
    if change == 'calibration':
        contact = replace(contact, calibration_id='changed')
    elif change == 'mask':
        mask = contact.support_mask.copy()
        mask[0, 0] = False
        contact = replace(contact, support_mask=mask)
    elif change == 'backing':
        contact = replace(contact, backing_depth_m=.03)
    else:
        contact = None
    system.source.snapshot = lambda *_: replace(initial, support_contact=contact)
    with pytest.raises(ObservationUnavailable, match='scene_changed_replan'):
        system.adapter._refresh_checker(ex, task)
    assert ex.collision.support_contact is initial.support_contact and system.calls == []
    assert ex.collision.scene is initial.scene


def test_support_binding_rejects_wrong_payload_and_bounds(system):
    _, initial, _, _ = support_snapshot(system)
    with pytest.raises(ValueError, match='invalid_observed_support_contact'):
        replace(initial, support_contact=replace(initial.support_contact, object_ref='other'))
    with pytest.raises(ValueError, match='invalid_observed_support_contact'):
        replace(initial, support_bounds=initial.support_bounds + [0., 0., .01])


def test_planned_release_still_checks_live_payload_at_pickup(system, monkeypatch):
    from emet.motion.observed_admission import AdmissionDecision

    task, initial, ex, _ = support_snapshot(system)
    source = replace(initial.support_contact, support_ref='pickup')
    checker = ex.collision
    checker.set_source_support_contact(source)
    checker.set_payload(ex.model, ex.data, task.object_ref, ex.ee_body)
    checker.set_payload(ex.model, ex.data, None)
    queried = []

    def query(_vertices, _bounds, _error, _ref, phase, contact):
        queried.append((phase, contact.support_ref))
        return AdmissionDecision(True, 'clear', 0)

    monkeypatch.setattr(checker, '_query_supported_payload', query)
    with checker.certify_transfer():
        assert not checker._payload_scene_collides(ex.model, ex.data, checker.released_bounds)
    assert queried == [('release', task.support_ref), ('release', 'pickup')]
    queried.clear()
    assert not checker._payload_scene_collides(ex.model, ex.data, checker.released_bounds)
    assert queried == [('release', task.support_ref)] * 2


def test_expired_support_is_no_actuation_even_with_fresh_payload_observation(system):
    task, initial, _, raw = support_snapshot(system)
    system.clock[0] += 3.
    fresh = raw(task.object_ref, task.support_ref)
    admission = ObservedMaterialAdmission(initial.admission.background,
        replace(initial.admission.robot_material, capture=fresh.frame.capture),
        query_robot_geometry=initial.admission.query_robot_geometry)
    current = replace(initial, frame=fresh.frame, admission=admission)
    system.source.snapshot = lambda *_: current
    result = json.loads(system.tools['scene_tasks'].func())
    assert result['status'] == 'error' and result['data']['execution_started'] is False
    assert system.calls == []


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


@pytest.mark.parametrize('tool_name, arguments', [('rotate_base', None), ('describe_scene', {'pan_rad': 0.})])
def test_motion_invalidates_observed_adapter_and_stored_handles(system, tool_name, arguments):
    from emet.agent.tamp_world import task_world_change

    planned = plan(system)
    assert planned['status'] == 'ok', planned
    with task_world_change(tool_name, system.context, arguments=arguments):
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


def test_canonical_payload_is_not_preinflated_and_region_keeps_pose_error(system):
    raw = system.source.snapshot
    def snapshot(*args):
        current = raw(*args)
        return replace(current, frame=replace(current.frame, position_error_m=.003))
    system.source.snapshot = snapshot
    result = system.adapter.plan(object_name='block', receptacle_name='counter',
                                 placement_region=[[.1, -.011], [.14, .011]])
    # The nominal footprint fits, but the possible footprint does not.
    assert result['code'] == 'placement_region_unavailable', result
    assert system.calls == []
    result = plan(system)
    assert result['status'] == 'ok', result
    ex = system.adapter.plans[result['data']['plan_ref']].executor
    evidence = ex.collision.provider.observe_payload('opaque')
    np.testing.assert_array_equal(evidence.vertices_reference, snapshot('opaque', 'support').frame.payload.vertices_reference)
    assert evidence.position_error_m == .003


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


@pytest.mark.parametrize("reason", ["geometry_tracking_no_match", "approximate_surface_overlap_lost"])
def test_unstarted_tracking_failure_resets_source_and_requires_fresh_plan(system, reason):
    snapshot = system.source.snapshot
    lost = [True]
    resets = []

    def read(obj, support):
        if lost[0]:
            raise ObservationUnavailable(reason)
        return snapshot(obj, support)

    def reset():
        resets.append(True)
        lost[0] = False

    system.source.snapshot = read
    system.source.invalidate = reset
    failed = plan(system)
    assert failed['code'] == reason
    assert failed['data']['execution_started'] is False
    assert failed['data']['perception_reset'] is True
    assert failed['recovery'] == 'replan'
    assert resets == [True]
    assert not system.adapter.plans
    assert plan(system)['status'] == 'ok'


def test_rejected_pickup_is_not_replanned_for_each_landing(system, monkeypatch):
    calls = []
    pick = ((np.zeros(3), np.eye(3)),)
    alternatives = tuple(SimpleNamespace(grasp=pick, place=((np.array([x, 0., 0.]), np.eye(3)),), close_joints=())
                         for x in (.1, .2, .3))
    executor = SimpleNamespace(transfer_candidates=alternatives, grasp_paths=[], place_paths=[])

    def reject(ex, task, region):
        calls.append(True)
        ex.last_certification_phase = 'grasp'
        return 'collision'

    monkeypatch.setattr(system.adapter, '_certify_pair', reject)
    assert system.adapter._certify_candidates(executor, None, None) == 'collision'
    assert calls == [True]


def test_world_aligned_grasp_keeps_measured_geometry_and_attachment_transform(system):
    from emet.controller.task.tamp.observed_candidates import CalibratedGrasp, candidates_from_supports

    _, snapshot, _, _ = support_snapshot(system)
    measured = snapshot.frame.payload.pose.world_from_object.copy()
    measured[:3, :3] = Rotation.from_euler('y', .15).as_matrix()
    pose = replace(snapshot.frame.payload.pose, world_from_object=measured)
    snapshot = replace(snapshot, frame=replace(snapshot.frame, payload=replace(snapshot.frame.payload, pose=pose)))
    tool = np.diag([1., -1., -1., 1.])
    tool[2, 3] = .025
    profile = CalibratedGrasp(tool, (0., 0., 1.), (('opening', 0.),), world_aligned=True)
    (candidate,) = candidates_from_supports(snapshot, [profile], yaw_options=(np.pi/2,), limit=1)
    assert np.allclose(candidate.grasp[1][0], measured[:3, 3] + [0., 0., .025])
    assert np.allclose(candidate.grasp[1][1], tool[:3, :3])
    assert np.allclose(candidate.place[1][1], Rotation.from_euler('z', np.pi/2).as_matrix() @ tool[:3, :3])
    assert np.array_equal(snapshot.frame.payload.pose.world_from_object, measured)


@pytest.mark.parametrize("support_tolerance", [0., .005])
def test_local_payload_contact_keeps_support_cap_and_other_obstacles(system, support_tolerance):
    from emet.motion.local_manipulation_map import LocalManipulationMap

    task, snapshot, ex, _ = support_snapshot(system)
    support = snapshot.support_contact
    obs = SimpleNamespace(rgb=np.zeros((*support.depth.shape, 3), np.uint8), depth=support.depth.copy(),
                          camera_K=support.intrinsic, camera_pose=support.world_from_camera)
    checker = ex.collision
    checker.local_scene = LocalManipulationMap({'camera': obs}, {'camera': np.zeros(obs.depth.shape, bool)},
                                               simulation_allow_unknown=True)
    checker.simulation_local_payload_contact = True
    checker.simulation_support_contact_tolerance_m = support_tolerance
    checker.set_payload(ex.model, ex.data, task.object_ref, ex.ee_body)
    with checker.motion_phase('lift'):
        assert not checker._local_payload_scene_collides(ex.model, ex.data)
        ex.data.qpos[2] -= .002
        mujoco.mj_kinematics(ex.model, ex.data)
        assert checker._local_payload_scene_collides(ex.model, ex.data) == (support_tolerance == 0.)
        ex.data.qpos[2] += .002
        mujoco.mj_kinematics(ex.model, ex.data)
        ex.data.qpos[2] -= .02
        mujoco.mj_kinematics(ex.model, ex.data)
        assert checker._local_payload_scene_collides(ex.model, ex.data)
        assert checker.last_reason == 'payload_support_penetration'
        ex.data.qpos[2] += .02
        mujoco.mj_kinematics(ex.model, ex.data)
        obs.depth[18:23, 26:31] = .96
        checker.local_scene = LocalManipulationMap({'camera': obs}, {'camera': np.zeros(obs.depth.shape, bool)},
                                                   simulation_allow_unknown=True)
        assert checker._local_payload_scene_collides(ex.model, ex.data)
        assert checker.last_admission.reason == 'occupied_local_surface'


def test_recessed_destination_preserves_source_lift_height_during_transfer(system):
    from emet.controller.task.tamp.observed_candidates import CalibratedGrasp, candidates_from_supports

    _, snapshot, _, _ = support_snapshot(system)
    recessed = SimpleNamespace(frame=snapshot.frame, support_bounds=snapshot.support_bounds,
        support_contact=SimpleNamespace(patch=SimpleNamespace(plane=np.array([0., 0., -.2]))))
    grasp = CalibratedGrasp(np.eye(4), (0., 0., -1.), (('opening', 0.),), lift_m=.1)
    (candidate,) = candidates_from_supports(recessed, (grasp,), limit=1)
    assert candidate.place[1][0][2] < candidate.grasp[1][0][2] - .15
    assert candidate.place[0][0][2] >= candidate.grasp[2][0][2]
    assert candidate.place[0][0][2] - candidate.place[1][0][2] > .25


@pytest.mark.parametrize('bounded_shortcut', [False, True])
def test_runtime_revalidation_checks_only_the_commanded_direct_segment(system, monkeypatch, bounded_shortcut):
    from emet.controller.manipulation import physical_pick_place as native

    task = next(iter(system.adapter.tasks.values()))
    ex = system.adapter.refine(task, system.source.snapshot(task.object_ref, task.support_ref))
    q0 = ex.data.qpos[ex.qadr].copy()
    q1 = q0 + [.004, 0., 0.]
    ex.approach_joint_routes = (np.array([q0, q0 + [.3, .3, .1]]),)
    original = native.plan_arm_joint_path
    checked = []

    def direct(*args, **kwargs):
        assert not kwargs.get('via_configurations')
        assert kwargs['planner'] == 'linear'
        checked.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(native, 'plan_arm_joint_path', direct)
    path = [q0, q1]
    if bounded_shortcut:
        ex.coalesce_execution_waypoints = True
        ex.execution_shortcut_deviation[:] = .0001
        path.insert(1, q0 + [.002, .00005, 0.])
    assert ex._execute_path('pregrasp', path).success
    assert len(checked) == 1


def test_linear_joint_tracking_cannot_treat_five_mm_lag_as_radians(system):
    task = next(iter(system.adapter.tasks.values()))
    ex = system.adapter.refine(task, system.source.snapshot(task.object_ref, task.support_ref))
    ex.linear_joint_tolerance_m = .001
    q0 = ex.data.qpos[ex.qadr].copy()
    q1 = q0 + [.02, 0., 0.]

    def lagging(goals):
        system.q[ex.qadr] = np.asarray(goals) - [.005, 0., 0.]
        return True

    ex.command_joints = lagging
    result = ex._execute_path('pregrasp', [q0, q1])
    assert not result.success and result.message == 'arm_tracking_failed'
    assert abs(system.q[0] - q1[0]) == pytest.approx(.005)


def test_already_open_gripper_retains_measured_geometry_without_new_sweep(system, monkeypatch):
    task = next(iter(system.adapter.tasks.values()))
    ex = system.adapter.refine(task, system.source.snapshot(task.object_ref, task.support_ref))
    before = ex.data.qpos.copy()
    calls = []
    monkeypatch.setattr(ex.collision, 'configuration_collides', lambda *args: calls.append(True) or True)
    path, error = ex.plan_gripper_open()
    assert error is None and len(path) == 1
    assert not calls and not system.calls
    np.testing.assert_array_equal(ex.data.qpos, before)
    # The arm still needs its own collision/departure checks.
    _, error = ex.plan_joint_target(before[ex.qadr] + [.01, 0., 0.])
    assert error and calls


def test_departure_handoff_retains_patch_context_during_suffix_execution(system):
    from contextlib import contextmanager

    from emet.motion.arm_rrt import DeparturePath

    task = next(iter(system.adapter.tasks.values()))
    ex = system.adapter.refine(task, system.source.snapshot(task.object_ref, task.support_ref))
    q0 = ex.data.qpos[ex.qadr].copy()
    q1, q2 = q0.copy(), q0.copy()
    q1[0] += .0002
    q2[0] += .0004
    active, commands = [False], []

    @contextmanager
    def clearance_scope():
        assert not active[0]
        active[0] = True
        try:
            yield
        finally:
            active[0] = False

    policy = SimpleNamespace(joint_names=ex.joint_names, execution_waypoints=lambda path: path,
        validate_edge=lambda a, b: True, configuration_allowed=lambda q: True)
    path = DeparturePath([q0, q1, q2], policy, escape_end_index=1)
    path.clearance_scope = clearance_scope
    original = ex.command_joints

    def command(q):
        commands.append((active[0], ex.collision.active_departure))
        return original(q)

    ex.command_joints = command
    result = ex._execute_path('pregrasp', path)
    assert result.success, result
    assert commands[0] == (False, policy)
    assert commands[-1] == (True, None)
    assert not active[0] and ex.collision.active_departure is None
    np.testing.assert_allclose(system.q[ex.qadr], q2)


def test_local_payload_support_associates_cross_view_returns_without_erasing_raised_obstacle(system):
    from emet.motion.local_manipulation_map import LocalManipulationMap

    task, snapshot, ex, _ = support_snapshot(system)
    support = snapshot.support_contact
    obs = SimpleNamespace(rgb=np.zeros((*support.depth.shape, 3), np.uint8), depth=support.depth.copy(),
                          camera_K=support.intrinsic, camera_pose=support.world_from_camera,
                          received_monotonic=support.capture.received_monotonic)
    mask = support.support_mask.copy()
    mask[20, 28] = False  # Occluded in the original selected-support view.
    patch = fit_observed_support(obs, mask, observation_id=support.capture.observation_id,
                                 measurement_margin_m=support.patch.measurement_margin_m)
    support = replace(support, support_mask=mask, patch=patch)
    checker = ex.collision
    checker.set_support_contact(support)
    checker.local_scene = LocalManipulationMap({'other_camera': obs}, {'other_camera': np.zeros(mask.shape, bool)},
                                               simulation_allow_unknown=True)
    checker.simulation_local_payload_contact = True
    checker.set_payload(ex.model, ex.data, task.object_ref, ex.ee_body)
    assert support.measured_surface_distance([.2, 0., 0.]) == pytest.approx(.025)
    with checker.motion_phase('lift'):
        assert not checker._local_payload_scene_collides(ex.model, ex.data)
        obs.depth[19:22, 27:30] = .96  # A supported depth component above the counter plane.
        checker.local_scene = LocalManipulationMap({'other_camera': obs},
            {'other_camera': np.zeros(mask.shape, bool)}, simulation_allow_unknown=True)
        assert checker._local_payload_scene_collides(ex.model, ex.data)


def test_transit_clearance_uses_measured_counter_instead_of_lift_height(system):
    from emet.controller.task.tamp.observed_candidates import CalibratedGrasp, candidates_from_supports

    _, snapshot, _, _ = support_snapshot(system)
    snapshot = replace(snapshot, source_support_contact=snapshot.support_contact,
                       source_support_bounds=snapshot.support_contact.patch.bounds)
    grasp = CalibratedGrasp(np.eye(4), (0., 0., -1.), (('opening', 0.),),
                            lift_m=.15, transit_clearance_m=.035)
    (candidate,) = candidates_from_supports(snapshot, (grasp,), limit=1)
    tool = np.asarray(candidate.place[0][0])
    landing = np.asarray(candidate.place[1][0])
    payload_low_from_tool = snapshot.frame.payload.vertices_reference[:, 2].min()
    assert tool[2]+payload_low_from_tool >= .035-1e-9
    assert tool[2] < candidate.grasp[2][0][2]-.05
    assert landing[2] < tool[2]


def test_landing_budget_covers_distinct_orientations_before_translations(system):
    from emet.controller.task.tamp.observed_candidates import CalibratedGrasp, candidates_from_supports

    _, snapshot, _, _ = support_snapshot(system)
    grasp = CalibratedGrasp(np.eye(4), (0., 0., -1.), (('opening', 0.),))
    candidates = candidates_from_supports(snapshot, (grasp,), limit=4,
                                          yaw_options=(0., np.pi/2, np.pi, 3*np.pi/2))
    assert len(candidates) == 4
    centers = np.array([candidate.place[1][0][:2] for candidate in candidates])
    np.testing.assert_allclose(centers, np.tile(centers[0], (4, 1)))
    assert len({np.asarray(candidate.place[1][1]).round(6).tobytes() for candidate in candidates}) == 4


@pytest.mark.parametrize("phase", ["pregrasp", "preplace"])
def test_staging_is_a_checked_detour_around_an_observed_obstacle(system, phase):
    task = next(iter(system.adapter.tasks.values()))
    ex = system.adapter.refine(task, system.source.snapshot(task.object_ref, task.support_ref))
    ex.collision.scene = PlacementScene([[[.08, -.03, .17], [.12, .03, .23]]],
        source='observed_voxels', workspace=[[-1]*3, [1]*3], known_free=lambda bounds: True)
    ex.preplace_staging_offsets_m = ex.pregrasp_staging_offsets_m = (np.array([0., -.12, 0.]),)
    ex.cartesian_transport = True
    with ex.collision.motion_phase(phase):
        path, error = ex.plan_pose([.2, 0., .2], np.eye(3))
        assert error is None and path
        assert any(q[1] < -.1 for q in path)
        assert any(q[0] < .03 and q[1] < -.1 for q in path)
        assert any(q[0] > .17 and q[1] < -.1 for q in path)
        for q in path:
            ex.data.qpos[ex.qadr] = q
            mujoco.mj_forward(ex.model, ex.data)
            assert not ex.collision.configuration_collides(ex.model, ex.data)
        np.testing.assert_allclose(ex.data.body('ee').xpos, [.2, 0., .2], atol=ex.position_tolerance_m)



@pytest.mark.parametrize("phase", ["pregrasp", "grasp", "lift", "preplace", "place", "retreat"])
def test_cartesian_continuation_is_available_for_each_transfer_phase(system, phase):
    task = next(iter(system.adapter.tasks.values()))
    ex = system.adapter.refine(task, system.source.snapshot(task.object_ref, task.support_ref))
    ex.cartesian_transport = True
    start = ex.data.qpos[ex.qadr].copy()
    with ex.collision.motion_phase(phase):
        path, error = ex.plan_pose([.03, 0., .2], np.eye(3))
    assert error is None and path
    np.testing.assert_array_equal(path[0], start)
    np.testing.assert_allclose(ex.data.body('ee').xpos, [.03, 0., .2], atol=ex.position_tolerance_m)


def test_failed_cartesian_route_restores_unexecuted_model_state(system):
    task = next(iter(system.adapter.tasks.values()))
    ex = system.adapter.refine(task, system.source.snapshot(task.object_ref, task.support_ref))
    ex.collision.scene = PlacementScene([[[.15, -.03, .17], [.25, .03, .23]]],
        source='observed_voxels', workspace=[[-1]*3, [1]*3], known_free=lambda bounds: True)
    before = ex.data.qpos.copy()
    path, error = ex._plan_cartesian_route((np.array([.2, 0., .2]),), np.eye(3))
    assert path is None and error
    np.testing.assert_array_equal(ex.data.qpos, before)
    assert system.calls == []


def test_execution_coalescing_preserves_bends_endpoints_and_slide_limits():
    from emet.controller.manipulation.physical_pick_place import coalesce_collinear_waypoints

    path = [np.array([x, 0.]) for x in np.linspace(0., .2, 21)]
    path.extend(np.array([.2, y]) for y in np.linspace(.001, .01, 10))
    result = coalesce_collinear_waypoints(path, np.array([.1, .005]))
    assert len(result) < len(path) / 2
    np.testing.assert_array_equal(result[0], path[0])
    np.testing.assert_array_equal(result[-1], path[-1])
    assert any(np.array_equal(point, [.2, 0.]) for point in result)
    assert np.all(np.abs(np.diff(result, axis=0)) <= np.array([.1, .005]) + 1e-12)


def test_axial_grasp_can_keep_measured_tool_heading_without_changing_down_axis(system):
    from scipy.spatial.transform import Rotation

    from emet.controller.manipulation.geometry_models import PrimitiveModel
    from emet.controller.manipulation.shape_pose import ShapePoseEvidence
    from emet.controller.task.tamp.observed_candidates import CalibratedGrasp, candidates_from_supports

    _, snapshot, _, _ = support_snapshot(system)
    pose = snapshot.frame.payload.pose
    shape_pose = ShapePoseEvidence(pose.world_from_object, pose.received_monotonic, pose.source,
        pose.observation_id, PrimitiveModel('cylinder', (.005, .005)), .002, .02, 'test-shape')
    heading = -.7
    measured = snapshot.frame.robot_before.world_from_ee.copy()
    measured[:3, :3] = Rotation.from_euler('z', heading).as_matrix()
    robot = replace(snapshot.frame.robot_before, world_from_ee=measured)
    frame = replace(snapshot.frame, payload=replace(snapshot.frame.payload, pose=shape_pose),
                    robot_before=robot, robot_after=robot)
    snapshot = replace(snapshot, frame=frame)
    calibration = np.eye(4)
    calibration[:3, :3] = Rotation.from_euler('x', np.pi).as_matrix()
    grasp = CalibratedGrasp(calibration, (0., 0., 1.), (('opening', 0.),),
                            world_aligned=True, keep_current_heading=True)
    candidate = candidates_from_supports(snapshot, (grasp,), yaw_options=(0.,))[0]
    rotation = candidate.grasp[0][1]
    assert np.arctan2(rotation[1, 0], rotation[0, 0]) == pytest.approx(heading)
    np.testing.assert_allclose(rotation[:, 2], [0., 0., -1.], atol=1e-12)
    unqualified = replace(snapshot, frame=replace(frame, payload=replace(frame.payload, pose=pose)))
    with pytest.raises(ObservationUnavailable, match='grasp_heading_symmetry_unknown'):
        candidates_from_supports(unqualified, (grasp,))


def test_pose_planning_budget_is_shared_across_ik_proposals(system, monkeypatch):
    from emet.controller.manipulation import physical_pick_place as native

    task = next(iter(system.adapter.tasks.values()))
    ex = system.adapter.refine(task, system.source.snapshot(task.object_ref, task.support_ref))
    ex.ik_random_starts = 16
    ex.joint_planning_timeout_s = .1
    before = ex.data.qpos.copy()
    ticks = iter([10., 10.2])
    monkeypatch.setattr(native.time, 'monotonic', lambda: next(ticks, 11.))
    path, reason = ex.plan_pose([.2, 0., .2], np.eye(3))
    assert path is None and reason == 'no_plan_within_budget'
    assert ex._pose_planning_deadline is None
    np.testing.assert_array_equal(ex.data.qpos, before)
    assert not system.calls


def test_open_measured_arm_can_reuse_validated_grasp_with_live_edge_checks(system):
    receipt = plan(system)
    assert receipt['status'] == 'ok'
    executor = system.adapter.plans[receipt['data']['plan_ref']].executor
    executor.reuse_validated_grasp_paths = True
    events = []
    executor.event = lambda **fields: events.append(fields)
    original = executor._replan_at_measured_pose

    def no_redundant_grasp_search(phases, *args, **kwargs):
        assert phases[0] != 'pregrasp'
        return original(phases, *args, **kwargs)

    executor._replan_at_measured_pose = no_redundant_grasp_search
    result = system.adapter.execute(receipt['data']['plan_ref'])
    assert result['status'] == 'ok'
    assert any(event.get('phase') == 'grasp_plan_reused' for event in events)
    assert system.calls


@pytest.mark.parametrize('rotate_at_destination', [False, True])
def test_cartesian_staging_translates_clear_before_rotating_tool(rotate_at_destination):
    from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="ee">
        <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
        <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
        <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
        <joint name="yaw" type="hinge" axis="0 0 1" range="-170 170"/>
        <geom size=".01"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qpos[2] = .2
    mujoco.mj_forward(model, data)
    collision = SimpleNamespace(configuration_collides=lambda model, data: False, last_reason=None)
    executor = PhysicalPickPlaceExecutor(None, model=model, data=data, ee_body='ee',
        joint_names=('x', 'y', 'z', 'yaw'), collision=collision,
        synchronize=lambda data: None, command_joints=lambda target: True,
        input_mode='privileged', position_tolerance_m=.0001, orientation_tolerance_rad=.0001)
    path, error = executor._plan_cartesian_route((np.array([0., -.12, .2]), np.array([.2, -.12, .2]), np.array([.2, 0., .2])),
                                                Rotation.from_euler('z', .8).as_matrix(),
                                                rotate_at_destination=rotate_at_destination)
    assert error is None and path
    staging = next(index for index, q in enumerate(path) if q[1] < -.119 and abs(q[0]) < .001)
    assert path[staging][1] == pytest.approx(-.12, abs=.001)
    assert all(abs(q[3]) < 1e-8 for q in path[:staging+1])
    if rotate_at_destination:
        front = next(index for index, q in enumerate(path) if q[0] > .199 and q[1] < -.119)
        assert all(abs(q[3]) < 1e-8 for q in path[:front+1])
    assert path[-1][3] == pytest.approx(.8, abs=.001)


@pytest.mark.parametrize('reachable', [False, True])
def test_cartesian_planning_initializes_clear_start_scope_and_restores_it_on_failure(system, monkeypatch, reachable):
    from contextlib import contextmanager

    from emet.motion.arm_rrt import ContactClearancePath

    task = next(iter(system.adapter.tasks.values()))
    ex = system.adapter.refine(task, system.source.snapshot(task.object_ref, task.support_ref))
    ex.collision.local_scene = object()
    ex.collision.allow_contact_departure = True
    active, initialized = [False], []

    @contextmanager
    def clearance_scope():
        active[0] = True
        try:
            yield
        finally:
            active[0] = False

    def departure(*args, **kwargs):
        initialized.append(kwargs['clearance_only'])
        return SimpleNamespace(clearance_scope=clearance_scope)

    monkeypatch.setattr(ex.collision, 'make_departure', departure)
    monkeypatch.setattr(ex.collision, 'configuration_collides', lambda *args: not active[0])
    before = ex.data.qpos.copy()
    path, error = ex._plan_cartesian_route((np.array([.02 if reachable else 2., 0., .2]),), np.eye(3))
    assert initialized == [True] and not active[0]
    if reachable:
        assert error is None and isinstance(path, ContactClearancePath)
        assert path.clearance_scope is ex._contact_clearance_scope
    else:
        assert path is None and error
        assert ex._contact_clearance_scope is None
        np.testing.assert_array_equal(ex.data.qpos, before)


def test_bounded_waypoint_chords_preserve_bends_and_every_intermediate_deviation():
    from emet.controller.manipulation.physical_pick_place import coalesce_bounded_waypoints

    path = [np.array([x, .0001*np.sin(x*20)]) for x in np.linspace(0., .2, 101)]
    path.extend(np.array([.2, y]) for y in np.linspace(.0003, .01, 21))
    maximum, deviation = np.array([.1, .005]), np.array([.002, .0002])
    result = coalesce_bounded_waypoints(path, maximum, deviation)
    assert len(result) < len(path)/5
    np.testing.assert_array_equal(result[0], path[0])
    np.testing.assert_array_equal(result[-1], path[-1])
    start = 0
    for first, last in zip(result[:-1], result[1:], strict=True):
        end = next(i for i in range(start+1, len(path)) if path[i] is last)
        assert np.all(np.abs(last-first) <= maximum)
        delta = (last-first)/maximum
        fractions = ((np.asarray(path[start:end+1])-first)/maximum) @ delta / (delta @ delta)
        errors = np.asarray(path[start:end+1])-first-fractions[:, None]*(last-first)
        assert np.all(np.abs(errors) <= deviation+1e-12)
        start = end


def test_attachment_calibration_covers_allowed_retention_motion_without_rebasing(system):
    from emet.controller.task.tamp.observed_adapter import _PayloadEnvelope

    task = next(iter(system.adapter.tasks.values()))
    initial = system.source.snapshot(task.object_ref, task.support_ref)
    ee = system.object_position + [0., 0., .025]
    system.q[:3] = ee
    ex = system.adapter.refine(task, initial)
    ex.collision.provider = _PayloadEnvelope(lambda: system.source.snapshot(task.object_ref, task.support_ref),
                                            initial, attachment_ee_positions=[ee])
    ex.collision.set_payload(ex.model, ex.data, task.object_ref, ex.ee_body)
    ex.payload_body = task.object_ref
    system.object_position[2] -= .015
    assert ex.payload_retained()
    # The original attachment stays fixed: further drift is still rejected.
    system.object_position[2] -= .02
    assert not ex.payload_retained()


def test_placement_can_choose_both_support_axes_with_full_footprint_inside(system):
    from emet.controller.task.tamp.observed_candidates import CalibratedGrasp, candidates_from_supports

    _, snapshot, _, _ = support_snapshot(system)
    grasp = CalibratedGrasp(np.eye(4), (0., 0., -1.), (('opening', 0.),))
    candidates = candidates_from_supports(snapshot, (grasp,), yaw_options=(0.,),
        position_fractions=((0., -.35), (.25, -.35), (0., 0.)), limit=3)
    assert len(candidates) == 3
    centers = np.array([candidate.place[1][0][:2] for candidate in candidates])
    assert centers[0, 1] < centers[2, 1] and centers[1, 0] > centers[0, 0]
    vertices = snapshot.frame.payload.vertices_reference[:, :2]
    for center in centers:
        footprint = vertices+center
        assert np.all(footprint.min(0) >= snapshot.support_bounds[0, :2])
        assert np.all(footprint.max(0) <= snapshot.support_bounds[1, :2])
    for invalid in ((), ((0., .51),), ((np.nan, 0.),), ((0.,),)):
        with pytest.raises(ValueError, match='invalid_placement_position_fractions'):
            candidates_from_supports(snapshot, (grasp,), position_fractions=invalid)


def test_discovered_approach_reference_rebinds_to_fresh_planning_capture(system):
    from emet.controller.task.tamp.candidate_guidance import guide_approach
    from emet.controller.task.tamp.observed_arm import TransferCandidate

    original = system.adapter.refine
    seen = []

    def refine(task, snapshot, placement_region=None, request=None):
        ex = original(task, snapshot, placement_region)
        if request is not None:
            option = TransferCandidate(tuple(ex.grasp_targets), tuple(ex.place_targets), (('opening', 0.),))
            guided, = guide_approach((option,), request.hints['approach'],
                observation_id=snapshot.frame.capture.observation_id)
            ex.grasp_targets = list(guided.grasp)
            seen.append(request.hints['approach']['observation_id'])
        return ex

    refine.supports_manipulation_request = True
    system.adapter.refine = refine
    discovery = json.loads(system.tools['scene_tasks'].func())
    ref = discovery['data']['tasks'][0]['observation_id']
    result = json.loads(system.tools['plan_pick_place'].func(object_name='block', receptacle_name='counter',
        hints={'approach': {'frame_id': 'world', 'observation_id': ref,
                            'direction_world': [0, 0, -1], 'distance_m': .08}}))
    assert result['status'] == 'ok', result
    assert seen and seen[0] != ref
    assert result['data']['guidance']['requested']['approach']['observation_id'] == ref
    assert result['data']['guidance']['reference_revalidated'] is True
    assert system.calls == []
    executed = json.loads(system.tools['execute_pick_place_plan'].func(plan_ref=result['data']['plan_ref']))
    assert executed['status'] == 'ok', executed
    assert not system.adapter._hint_observations


def test_historical_approach_reference_does_not_refresh_its_age(system):
    discovery = json.loads(system.tools['scene_tasks'].func())
    ref = discovery['data']['tasks'][0]['observation_id']
    # A request-capable refiner must not be called for an expired reference.
    original = system.adapter.refine

    def refine(*args, **kwargs):
        raise AssertionError('expired reference reached candidate generation')

    refine.supports_manipulation_request = True
    system.adapter.refine = refine
    system.clock[0] += 121.
    result = json.loads(system.tools['pick_place'].func('block', 'counter', hints={'approach': {
        'frame_id': 'world', 'observation_id': ref, 'direction_world': [0, 0, -1]}}))
    assert result['code'] == 'stale_manipulation_hint'
    assert system.calls == []
    system.adapter.refine = original


@pytest.mark.parametrize('system', ['observed_surface_bounds'], indirect=True)
def test_public_tool_transfer_accepts_bounded_observed_surfaces(system):
    result = plan(system)
    assert result['status'] == 'ok', result
    assert result['data']['geometry_source'] == 'observed_surface_bounds'
    assert system.calls == []
    outcome = json.loads(system.tools['execute_pick_place_plan'].func(plan_ref=result['data']['plan_ref']))
    assert outcome['status'] == 'ok', outcome
    assert outcome['data']['geometry_source'] == 'observed_surface_bounds'


@pytest.mark.parametrize('system', ['observed_surface_bounds'], indirect=True)
def test_geometry_representation_change_invalidates_plan_before_motion(system):
    result = plan(system)
    assert result['status'] == 'ok', result
    snapshot = system.source.snapshot

    def changed(obj, support):
        original = snapshot(obj, support)
        scene = PlacementScene(original.scene.boxes, source='observed_voxels',
            workspace=original.scene.workspace, known_free=original.scene.known_free)
        return replace(original, scene=scene)

    system.source.snapshot = changed
    outcome = json.loads(system.tools['execute_pick_place_plan'].func(plan_ref=result['data']['plan_ref']))
    assert outcome['status'] == 'error'
    assert outcome['code'] == 'scene_changed_replan'
    assert system.calls == []


def test_grounding_failure_exposes_region_without_issuing_commands(system):
    from emet.controller.task.tamp.transfer_grounding import TransferGroundingUnavailable

    def unavailable(*args, **kwargs):
        raise TransferGroundingUnavailable('support_geometry_unknown', 'destination')

    system.source.snapshot = unavailable
    outcome = plan(system)
    assert outcome['code'] == 'support_geometry_unknown'
    assert outcome['data']['failed_region'] == 'destination'
    assert outcome['data']['execution_started'] is False
    assert not system.calls


def test_query_guard_is_closed_during_native_dispatch_and_reopens_after_verified_place(system):
    from emet.controller.task.tamp.execution_phase import ManipulationExecutionPhase

    phase = ManipulationExecutionPhase('empty')
    system.adapter.execution_phase = phase
    original = system.adapter.refine
    checks = []

    def refine(*args, **kwargs):
        ex = original(*args, **kwargs)
        command = ex.command_joints

        def guarded(q):
            checks.append(phase.can_query())
            return command(q)

        ex.command_joints = guarded
        return ex

    system.adapter.refine = refine
    result = plan(system)
    assert result['status'] == 'ok' and phase.can_query()
    outcome = json.loads(system.tools['execute_pick_place_plan'].func(plan_ref=result['data']['plan_ref']))
    assert outcome['status'] == 'ok', outcome
    assert checks and not any(checks)
    assert phase.can_query()


def test_unknown_dispatch_failure_keeps_model_query_guard_closed(system):
    from emet.controller.task.tamp.execution_phase import ManipulationExecutionPhase

    phase = ManipulationExecutionPhase('empty')
    system.adapter.execution_phase = phase
    original = system.adapter.refine

    def refine(*args, **kwargs):
        ex = original(*args, **kwargs)

        def ambiguous(q):
            raise TimeoutError('unknown transport result')

        ex.command_joints = ambiguous
        return ex

    system.adapter.refine = refine
    result = plan(system)
    assert result['status'] == 'ok'
    outcome = json.loads(system.tools['execute_pick_place_plan'].func(plan_ref=result['data']['plan_ref']))
    assert outcome['status'] == 'partial'
    assert not phase.can_query()
    assert phase.snapshot()['payload_state'] == 'unknown'


@pytest.mark.parametrize('approximate,angle,accepted', [
    (False, .065, False), (True, .065, True), (True, .3, False),
])
def test_preplanning_view_change_bounds_approximate_rotation(system, approximate, angle, accepted):
    first = system.source.snapshot('opaque', 'destination')
    construction = 'approximate_observed_box' if approximate else 'known_model_prior'
    payload = replace(first.frame.payload, construction=construction)
    first = replace(first, frame=replace(first.frame, payload=payload))
    pose = payload.pose.world_from_object.copy()
    pose[:3, :3] = Rotation.from_euler('z', angle).as_matrix()
    second = replace(first, frame=replace(first.frame, payload=replace(
        payload, pose=replace(payload.pose, world_from_object=pose))))
    if accepted:
        ObservedNativeAdapter._require_same_initial(first, second)
    else:
        with pytest.raises(ObservationUnavailable, match='scene_changed_replan'):
            ObservedNativeAdapter._require_same_initial(first, second)


def test_approximate_topdown_grasp_uses_support_to_correct_visible_surface_height_bias(system):
    from emet.controller.task.tamp.observed_candidates import CalibratedGrasp, candidates_from_supports

    _, snapshot, _, _ = support_snapshot(system)
    transform = snapshot.frame.payload.pose.world_from_object.copy()
    transform[2, 3] = .06
    payload = replace(snapshot.frame.payload, construction='approximate_observed_box',
        pose=replace(snapshot.frame.payload.pose, world_from_object=transform))
    frame = replace(snapshot.frame, payload=payload,
        observed_bounds_world=np.array([[.18, -.02, .04], [.22, .02, .08]]))
    snapshot = replace(snapshot, frame=frame, source_support_contact=snapshot.support_contact,
        source_support_bounds=snapshot.support_bounds)
    calibration = np.diag([1., -1., -1., 1.])
    calibration[2, 3] = .007  # Robot hand/tool offset is still honored.
    grasp = CalibratedGrasp(calibration, (0., 0., 1.), (('opening', 0.),), world_aligned=True)
    (candidate,) = candidates_from_supports(snapshot, [grasp], limit=1)
    plane = snapshot.source_support_contact.patch.plane
    floor = transform[:2, 3] @ plane[:2]+plane[2]
    assert candidate.grasp[1][0][2] == pytest.approx((floor+.08)/2+.007)
    assert np.array_equal(snapshot.frame.payload.pose.world_from_object, transform)

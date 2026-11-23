# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Exercise the real native executor on a robot-only model with supplied evidence.

Synthetic observations test the seam, not a qualified perception/grasp pipeline.
"""
import time
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.native_inputs import ObjectPoseEvidence, ObservationUnavailable
from emet.controller.manipulation.observed_collision import ObservedNativeCollisionChecker, PayloadGeometryEvidence
from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor
from emet.motion.arm_rrt import plan_arm_joint_path
from emet.motion.observed_workspace import ObservedWorkspace
from emet.motion.placement_geometry import PlacementScene, box_corners


def rig(*, boxes=(), known_free=lambda bounds: True, extra='', source='observed_voxels'):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="robot"><joint name="x" type="slide" axis="1 0 0" range="0 1"/>
        <geom type="box" size=".02 .02 .02"/>
        <body name="ee" pos=".1 0 0"><geom size=".015"/></body>
      </body>{extra}</worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    clock = [time.monotonic()]
    pose = np.eye(4)
    pose[:3, 3] = [.2, .2, 0]
    evidence = PayloadGeometryEvidence('observed:opaque:1', 'reference:1', 'shape:1',
        box_corners([[-.02]*3, [.02]*3]),
        ObjectPoseEvidence(pose, clock[0], 'rgbd_tracking', 'capture:1'), 'known_model_prior')
    provider = SimpleNamespace(input_mode='observed', observe_payload=lambda _: evidence)
    revision = ['scene:1']
    scene = PlacementScene(boxes, source=source, workspace=[[-1]*3, [2, 1, 1]], known_free=known_free)
    checker = ObservedNativeCollisionChecker(model, robot_body='robot', ee_body='ee', scene=scene,
        payload_provider=provider, scene_revision=revision[0], current_revision=lambda: revision[0],
        received_monotonic=clock[0], clock=lambda: clock[0], margin_m=0)
    return model, data, checker, provider, evidence, clock, revision


def attach(model, data, checker):
    checker.set_payload(model, data, 'observed:opaque:1', 'ee')


def test_attachment_needs_no_scene_object_and_preserves_live_data():
    model, data, checker, _, evidence, *_ = rig()
    before = data.qpos.copy(), data.qacc.copy(), data.ncon, data.xpos.copy()
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, evidence.object_ref) == -1
    attach(model, data, checker)
    assert checker.payload_body == evidence.object_ref
    np.testing.assert_allclose(checker.payload.world_vertices(data.body('ee').xpos, data.body('ee').xmat),
        evidence.vertices_reference + [.2, .2, 0])
    assert not checker.configuration_collides(model, data)
    for actual, expected in zip((data.qpos, data.qacc, data.ncon, data.xpos), before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError):
        evidence.vertices_reference.setflags(write=True)


def test_unique_pose_retention_includes_both_capture_error_bounds():
    model, data, checker, provider, evidence, *_ = rig()
    evidence = replace(evidence, position_error_m=.006, rotation_error_rad=.03)
    provider.observe_payload = lambda _: evidence
    attach(model, data, checker)
    np.testing.assert_allclose(checker.payload_retention_residual(model, data, evidence.object_ref), [.012, .06])
    moved = evidence.pose.world_from_object.copy()
    moved[0, 3] += .015
    current = replace(evidence, pose=replace(evidence.pose, world_from_object=moved))
    provider.observe_payload = lambda _: current
    distance, _ = checker.payload_retention_residual(model, data, evidence.object_ref)
    assert distance == pytest.approx(.027)  # Exceeds the unchanged .02 m retention bound.
    provider.observe_payload = lambda _: replace(current, position_error_m=.007)
    with pytest.raises(ObservationUnavailable, match='payload_geometry_changed'):
        checker.payload_retention_residual(model, data, evidence.object_ref)


def test_transport_and_retention_count_each_error_once_and_restore_release_bounds():
    model, data, checker, provider, evidence, *_ = rig()
    evidence = replace(evidence, position_error_m=.003, attachment_error_m=.001,
                       attachment_rotation_error_rad=.005, attachment_lever_limit_m=.3)
    provider.observe_payload = lambda _: evidence
    attach(model, data, checker)
    # Canonical material is unchanged; two EE captures affect transport once.
    actual = checker.possible_payload_bounds(model, data)
    expected = np.stack((evidence.vertices_reference.min(0), evidence.vertices_reference.max(0))) + [.2, .2, 0]
    np.testing.assert_allclose(actual, expected + np.array([[-1.], [1.]])*.005)
    np.testing.assert_allclose(checker.payload_retention_residual(model, data, evidence.object_ref), [.008, .01])
    saved = checker.capture_payload()
    checker.set_payload(model, data, None)
    assert checker._released_target.offset_m == pytest.approx(.005)
    checker.restore_payload(saved)
    np.testing.assert_allclose(checker.possible_payload_bounds(model, data), actual)


@pytest.mark.parametrize('stage', ['attachment', 'retention'])
def test_attachment_and_retention_reject_lever_outside_declared_calibration(stage):
    model, data, checker, provider, evidence, *_ = rig()
    evidence = replace(evidence, attachment_lever_limit_m=.3)
    provider.observe_payload = lambda _: evidence
    if stage == 'retention':
        attach(model, data, checker)
    moved = evidence.pose.world_from_object.copy()
    moved[1, 3] = .5
    provider.observe_payload = lambda _: replace(evidence, pose=replace(evidence.pose, world_from_object=moved))
    with pytest.raises(ObservationUnavailable, match='payload_attachment_outside_calibration'):
        if stage == 'attachment':
            attach(model, data, checker)
        else:
            checker.payload_retention_residual(model, data, evidence.object_ref)


@pytest.mark.parametrize('kwargs,reason', [
    ({'known_free': None}, 'observed_known_free_workspace_required'),
    ({'source': 'ground_truth'}, 'observed_known_free_workspace_required'),
    ({'extra': '<body name="hidden"><geom size=".01"/></body>'}, 'observed_robot_only_model_required'),
])
def test_unqualified_scene_rejected(kwargs, reason):
    with pytest.raises(ValueError, match=reason):
        rig(**kwargs)


def test_payload_only_obstacle_blocks_dense_motion_with_clear_endpoints():
    model, data, checker, *_ = rig(boxes=[[[.44, .18, -.02], [.46, .22, .02]]])
    attach(model, data, checker)
    result = plan_arm_joint_path(model, data, joint_names=['x'], q_start=[0], q_goal=[.5],
                                collision=checker, planner='linear', step_size=.025)
    assert not result.success and result.reason.startswith('linear_collision_at_')
    checker.set_payload(model, data, None)
    checker.released_bounds = None  # Explicitly remove only the test payload hypothesis.
    assert plan_arm_joint_path(model, data, joint_names=['x'], q_start=[0], q_goal=[.5],
                              collision=checker, planner='linear').success


def test_unknown_swept_volume_rejected_without_occupied_points():
    model, data, checker, *_ = rig(known_free=lambda bounds: not (bounds[0, 0] <= .45 <= bounds[1, 0]))
    attach(model, data, checker)
    result = plan_arm_joint_path(model, data, joint_names=['x'], q_start=[0], q_goal=[.5],
                                collision=checker, planner='linear', step_size=.025)
    assert not result.success


@pytest.mark.parametrize('change,reason', [('stale', 'scene_observation_stale'), ('revision', 'scene_revision_changed')])
def test_scene_rechecked_on_each_configuration(change, reason):
    model, data, checker, _, _, clock, revision = rig()
    assert not checker.configuration_collides(model, data)
    if change == 'stale':
        clock[0] += 2.1
    else:
        revision[0] = 'scene:2'
    assert checker.configuration_collides(model, data)
    assert checker.last_reason == reason


def test_hypothetical_release_restores_complete_attachment_state():
    model, data, checker, *_ = rig()
    attach(model, data, checker)
    payload = checker.payload
    executor = PhysicalPickPlaceExecutor(None, model=model, data=data, ee_body='ee', joint_names=['x'],
        collision=checker, object_pose_provider=checker, synchronize=lambda d: None, command_joints=lambda q: True)
    executor.plan_pose = lambda p, r, **kwargs: (None, 'deliberate_failure')
    _, error = executor._replan_at_measured_pose(('retreat',), [([.3, 0, 0], np.eye(3))],
                                               object_body='observed:opaque:1', grasp=False)
    assert error is not None
    assert checker.payload is payload and checker.payload_body == 'observed:opaque:1'
    assert checker.released_bounds is None
    assert executor.payload_body is None  # Planning does not assert physical grasp.
    checker.set_payload(model, data, None)
    assert checker.payload is None and checker.released_bounds is not None
    data.qpos[0] = .1
    mujoco.mj_forward(model, data)
    assert not checker.configuration_collides(model, data)  # Released volume stays in world frame.


def test_geometry_revision_or_missing_pose_cannot_reanchor_retention():
    model, data, checker, provider, evidence, *_ = rig()
    attach(model, data, checker)
    provider.observe_payload = lambda _: PayloadGeometryEvidence(evidence.object_ref, 'different-frame',
        evidence.revision, evidence.vertices_reference, evidence.pose, evidence.construction)
    with pytest.raises(ObservationUnavailable, match='payload_geometry_changed'):
        checker.observe_object(evidence.object_ref)
    provider.observe_payload = lambda _: None
    with pytest.raises(ObservationUnavailable, match='payload_geometry_unknown'):
        checker.observe_object(evidence.object_ref)


def test_shared_executor_commands_only_measured_robot_joints():
    model, data, checker, provider, evidence, *_ = rig()
    measured = [0.]
    calls = []

    def sync(state):
        state.qpos[0] = measured[0]
        mujoco.mj_kinematics(model, state)

    def command(q):
        calls.append(q.copy())
        measured[0] = q[0]
        return True

    def observe(_):
        pose = evidence.pose.world_from_object.copy()
        pose[0, 3] += measured[0]
        return PayloadGeometryEvidence(evidence.object_ref, evidence.reference_id, evidence.revision,
            evidence.vertices_reference, ObjectPoseEvidence(pose, time.monotonic(), 'rgbd_tracking',
            f'capture:{len(calls) + 1}'), evidence.construction)

    provider.observe_payload = observe
    checker.clock = time.monotonic
    attach(model, data, checker)
    robot = SimpleNamespace()  # No scene snapshot, body pose, attachment or contact RPC exists.
    executor = PhysicalPickPlaceExecutor(robot, model=model, data=data, ee_body='ee', joint_names=['x'],
        collision=checker, object_pose_provider=checker, synchronize=sync, command_joints=command)
    executor.payload_body = evidence.object_ref
    path, error = executor.plan_pose([.4, 0, 0], np.eye(3))
    assert error is None
    sync(data)
    result = executor._execute_path('transport', path)
    assert result.success and calls and measured[0] == pytest.approx(.3, abs=.01)
    assert model.nq == 1  # No object freejoint was introduced for hypothetical attachment.
    # A changed revision prevents even the first new command.
    checker.current_revision = lambda: 'scene:changed'
    before = len(calls)
    result = executor._execute_path('transport', [np.array([measured[0]]), np.array([.5])])
    assert not result.success and len(calls) == before


def test_registered_depth_workspace_drives_native_collision_queries():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="robot" pos="0 0 .4"><joint name="x" type="slide" axis="1 0 0" range="0 .3"/>
        <geom size=".02"/><body name="ee" pos=".1 0 0"><geom size=".015"/></body>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    now = time.monotonic()
    pose = np.eye(4)
    pose[:3, 3] = [.2, .2, .4]
    evidence = PayloadGeometryEvidence('sensor-object', 'reference', 'shape',
        box_corners([[-.02]*3, [.02]*3]),
        ObjectPoseEvidence(pose, now, 'rgbd_tracking', 'frame'), 'known_model_prior')
    provider = SimpleNamespace(input_mode='observed', observe_payload=lambda _: evidence)

    def checker_for(depth):
        grid = ObservedWorkspace.from_depth(depth, [[5, 0, 10], [0, 5, 10], [0, 0, 1]],
            np.eye(4), workspace=[[-.5, -.5, .2], [.5, .5, 1.4]], resolution=.1)
        return ObservedNativeCollisionChecker(model, robot_body='robot', ee_body='ee',
            scene=grid.placement_scene(), payload_provider=provider, scene_revision='scene',
            current_revision=lambda: 'scene', received_monotonic=now, margin_m=0)

    depth = np.full((21, 21), 1.)
    checker = checker_for(depth)
    checker.set_payload(model, data, 'sensor-object', 'ee')
    assert plan_arm_joint_path(model, data, joint_names=['x'], q_start=[0], q_goal=[.1],
                              collision=checker, planner='linear').success
    # Missing sensor evidence in the projected arm volume must block the same path.
    depth[10, 10] = 0
    checker = checker_for(depth)
    assert not plan_arm_joint_path(model, data, joint_names=['x'], q_start=[0], q_goal=[.1],
                                  collision=checker, planner='linear').success


def test_self_occluding_robot_returns_cannot_certify_its_starting_volume_free():
    robot_xml = '''<body name="robot" pos="0 0 .5">
      <joint name="x" type="slide" axis="1 0 0" range="0 .3"/>
      <geom type="box" size=".08 .08 .06"/>
      <body name="ee" pos=".1 0 0"><geom size=".015"/></body></body>'''
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+robot_xml+'</worldbody></mujoco>')
    sensor_model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+robot_xml+
        '<geom name="wall" type="box" pos="0 0 1.4" size=".5 .5 .02"/></worldbody></mujoco>')
    data, sensor_data = mujoco.MjData(model), mujoco.MjData(sensor_model)
    mujoco.mj_forward(model, data)
    mujoco.mj_forward(sensor_model, sensor_data)
    k = np.array([[50., 0., 10.], [0., 50., 10.], [0., 0., 1.]])
    depth, points, labels = np.zeros((21, 21)), [], []
    for y, x in np.ndindex(depth.shape):
        ray = np.array([(x-10)/50, (y-10)/50, 1.])
        ray /= np.linalg.norm(ray)
        hit = np.full(1, -1, dtype=np.int32)
        distance = mujoco.mj_ray(sensor_model, sensor_data, np.zeros(3), ray, None, True, -1, hit)
        assert distance > 0
        depth[y, x] = distance*ray[2]
        points.append(distance*ray)
        # Oracle labels here test the strongest endpoint-removal case. No runtime
        # provider is claimed; even perfect self segmentation cannot make it free.
        labels.append('background' if hit[0] == sensor_model.geom('wall').id else 'robot')
    grid = ObservedWorkspace.from_depth(depth, k, np.eye(4),
        workspace=[[-.3, -.3, .1], [.5, .3, 1.5]], resolution=.05, depth_margin_m=.005)
    cleaned = grid.partition(np.array(points), np.array(labels)).excluding(('robot',))
    assert np.any(grid.cells == 2) and np.sum(cleaned.cells == 2) < np.sum(grid.cells == 2)
    assert np.any(cleaned.cells == 2)  # Wall evidence survives self removal.
    interior = np.floor((np.array([0., 0., .5])-grid.origin)/grid.resolution).astype(int)
    assert grid.cells[tuple(interior)] == cleaned.cells[tuple(interior)] == 0
    provider = SimpleNamespace(input_mode='observed', observe_payload=lambda _: None)
    for workspace in (grid, cleaned):
        checker = ObservedNativeCollisionChecker(model, robot_body='robot', ee_body='ee',
            scene=workspace.placement_scene(), payload_provider=provider, scene_revision='capture',
            current_revision=lambda: 'capture', received_monotonic=10., clock=lambda: 10., margin_m=0)
        assert checker.configuration_collides(model, data)
        assert checker.last_reason == 'robot_scene_collision'
        result = plan_arm_joint_path(model, data, joint_names=['x'], q_start=[0], q_goal=[.05],
            collision=checker, planner='linear')
        assert not result.success and result.reason == 'invalid_start'


def test_missing_geometry_during_grasp_preflight_returns_before_commands():
    model, data, checker, provider, *_ = rig()
    provider.observe_payload = lambda _: None
    calls = []
    robot = SimpleNamespace(open_gripper=lambda **kw: calls.append('open'))
    executor = PhysicalPickPlaceExecutor(robot, model=model, data=data, ee_body='ee', joint_names=['x'],
        collision=checker, object_pose_provider=checker, synchronize=lambda d: None,
        command_joints=lambda q: calls.append(q), visual_verifier=SimpleNamespace())
    executor.grasp_paths = [[np.zeros(1)]] * 3
    executor.grasp_targets = [([.1, 0, 0], np.eye(3))] * 3
    executor.plan_pose = lambda *args, **kwargs: ([np.zeros(1)], None)
    result = executor.grasp_only('object', object_gt_body='observed:opaque:1')
    assert not result.success and result.message == 'payload_geometry_unknown'
    assert result.phase == 'lift' and calls == []
    assert checker.payload is None and checker.payload_body is None


def test_observed_grasp_without_visual_verifier_cannot_actuate():
    model, data, checker, *_ = rig()
    calls = []
    executor = PhysicalPickPlaceExecutor(SimpleNamespace(open_gripper=lambda **kw: calls.append('open')),
        model=model, data=data, ee_body='ee', joint_names=['x'], collision=checker,
        object_pose_provider=checker, synchronize=lambda d: None, command_joints=lambda q: calls.append(q))
    result = executor.grasp_only('object', object_gt_body='observed:opaque:1')
    assert not result.success and result.message == 'visual_verification_unavailable' and calls == []


def test_stationary_environment_overlap_is_scoped_to_commanded_joints():
    model, data, checker, *_ = rig(boxes=[[[-.03, -.03, -.03], [.03, .03, .03]]])
    assert checker.configuration_collides(model, data)
    with checker.motion_scope(model, data, (), ()):
        assert not checker.configuration_collides(model, data)
        data.qpos[0] = .01
        mujoco.mj_kinematics(model, data)
        assert checker.configuration_collides(model, data)
    data.qpos[0] = 0.
    mujoco.mj_kinematics(model, data)
    with checker.motion_scope(model, data, ('x',), (0.,)):
        assert checker.configuration_collides(model, data)
    assert checker.configuration_collides(model, data)


def test_geometry_cache_does_not_survive_pose_or_query_changes(monkeypatch):
    from emet.motion.observed_solids import WorldSolid

    model, data, checker, *_ = rig()
    original = WorldSolid.from_geom
    constructions = []
    bounds = []

    def capture(cls, *args, **kwargs):
        constructions.append(True)
        return original(*args, **kwargs)

    def query(model, data):
        first = checker._robot_bound(model, data, 0)[0]
        second = checker._robot_bound(model, data, 0)[0]
        np.testing.assert_array_equal(first.bounds, second.bounds)
        bounds.append(first.bounds.copy())
        return False

    monkeypatch.setattr(WorldSolid, 'from_geom', classmethod(capture))
    monkeypatch.setattr(checker, '_configuration_collides', query)
    assert not checker.configuration_collides(model, data)
    assert len(constructions) == 1
    data.qpos[0] = .1
    mujoco.mj_kinematics(model, data)
    assert not checker.configuration_collides(model, data)
    assert len(constructions) == 2
    np.testing.assert_allclose(bounds[1] - bounds[0], [[.1, 0., 0.]]*2)
    checker.margin = .002
    assert not checker.configuration_collides(model, data)
    assert len(constructions) == 3
    assert np.all(bounds[2][0] < bounds[1][0])


def test_departure_rechecks_latest_map_with_stable_voxel_keys():
    from emet.motion.observed_admission import AdmissionDecision

    model, data, checker, *_ = rig()
    checker.allow_contact_departure = True

    def scene(points):
        points = np.asarray(points, float)
        return SimpleNamespace(
            centers=points, resolution=.002, error=.003,
            nearby_distances=lambda solid: {
                i: float(solid.lower_distance(p[None])[0]-.0018)
                for i, p in enumerate(points) if solid.lower_distance(p[None])[0] < .01
            },
            query=lambda *args, **kwargs: AdmissionDecision(True, 'observed_free'),
        )

    checker.local_scene = scene([[.019, .001, .001]])
    policy = checker.make_departure(model, data, ('x',), np.zeros(1))
    assert policy is not None
    baseline_keys = set(policy.baseline.distances)
    # Inserting an unrelated voxel changes array indices, not contact identity.
    checker.local_scene = scene([[.501, .501, .501], [.019, .001, .001]])
    assert set(policy._measurement(np.zeros(1)).distances) == baseline_keys
    assert policy.validate_edge(np.zeros(1), np.zeros(1))
    # A newly measured contact must invalidate the existing recovery policy.
    checker.local_scene = scene([[.019, .001, .001], [-.019, .001, .001]])
    assert not policy.validate_edge(np.zeros(1), np.zeros(1))
    assert policy.last_rejection['reason'] == 'new_physical_contact'
    checker.local_scene.error = .004
    with pytest.raises(ObservationUnavailable, match='map_profile_changed'):
        policy.validate_edge(np.zeros(1), np.zeros(1))

    checker.local_scene = scene([[.001, .001, .001]])
    assert checker.make_departure(model, data, ('x',), np.zeros(1)) is None
    assert checker.last_departure_diagnostics['point_m'] == [.001, .001, .001]


def test_observed_departure_slides_over_measured_plane_across_voxel_boundaries():
    from emet.motion.local_manipulation_map import LocalManipulationMap

    model, data, checker, *_ = rig()
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 1.
    obs = SimpleNamespace(rgb=np.zeros((61, 101, 3), np.uint8), depth=np.full((61, 101), .981),
        camera_K=np.array([[1000., 0., 30.], [0., 1000., 30.], [0., 0., 1.]]), camera_pose=pose)
    mask = np.zeros(obs.depth.shape, bool)
    checker.local_scene = LocalManipulationMap({'camera': obs}, {'camera': mask},
        resolution_m=.005, simulation_allow_unknown=True)
    checker.allow_contact_departure = True
    policy = checker.make_departure(model, data, ('x',), np.zeros(1))
    assert policy is not None
    assert any(key[0] == 'environment_patch' for key in policy.baseline.distances)
    # Two voxel widths of tangential motion change which measured points touch.
    assert policy.validate_edge([0.], [.01]), policy.last_rejection
    # Reobservation at a different pixel/grid alignment still describes the plane.
    obs.camera_pose = pose.copy()
    obs.camera_pose[0, 3] = .0007
    checker.local_scene = LocalManipulationMap({'camera': obs}, {'camera': mask},
        resolution_m=.005, simulation_allow_unknown=True)
    assert policy.validate_edge([0.], [.01]), policy.last_rejection
    obs.depth[26:35, 26:35] = 1.019  # Separate surface at the opposite box face.
    checker.local_scene = LocalManipulationMap({'camera': obs}, {'camera': mask},
        resolution_m=.005, simulation_allow_unknown=True)
    assert not policy.validate_edge([0.], [.01])
    assert policy.last_rejection['reason'] == 'new_physical_contact'


def test_bounded_ik_starts_find_clear_redundant_posture_instead_of_rejecting_tool_pose():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="robot">
      <joint name="shoulder" type="slide" axis="1 0 0" range="-.5 .5"/>
      <geom size=".01"/>
      <body name="ee" pos="0 0 .1">
        <joint name="extension" type="slide" axis="1 0 0" range="-.5 .5"/>
        <geom size=".01"/>
      </body></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    now = time.monotonic()
    scene = PlacementScene([[[.08, -.03, -.03], [.5, .03, .03]]], source='observed_voxels',
                           workspace=[[-1]*3, [1]*3], known_free=lambda bounds: True)
    checker = ObservedNativeCollisionChecker(model, robot_body='robot', ee_body='ee', scene=scene,
        payload_provider=SimpleNamespace(input_mode='observed'), scene_revision='scene',
        current_revision=lambda: 'scene', received_monotonic=now, clock=lambda: now, margin_m=0.)
    executor = PhysicalPickPlaceExecutor(SimpleNamespace(), model=model, data=data, ee_body='ee',
        joint_names=('shoulder', 'extension'), collision=checker, synchronize=lambda d: None,
        command_joints=lambda q: True, object_pose_provider=checker)
    path, error = executor.plan_pose([.2, 0., .1], np.eye(3))
    assert path is None and error  # Minimum-norm IK puts the shoulder in the obstacle.
    executor.ik_random_starts = 8
    path, error = executor.plan_pose([.2, 0., .1], np.eye(3))
    assert error is None and path
    assert data.qpos[0] < .07
    np.testing.assert_allclose(data.body('ee').xpos, [.2, 0., .1], atol=executor.position_tolerance_m)
    for q in path:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        assert not checker.configuration_collides(model, data)


def test_planning_snapshot_preserves_real_evidence_clock_and_restores_live_age_checks():
    model, data, checker, _, _, clock, _ = rig()
    checker.local_scene = SimpleNamespace()  # Enable the local planning-snapshot clock only.
    captured = checker.received_monotonic
    with checker.planning_snapshot():
        clock[0] += 5.
        assert checker.clock() == captured
        assert checker._certify_live_clock() == clock[0]
        assert checker.received_monotonic == captured
        assert not checker._certifying_transfer
    assert checker.clock() == clock[0]
    assert checker._certify_live_clock is None
    assert checker.configuration_collides(model, data)
    assert checker.last_reason == 'scene_observation_stale'


def test_retention_uses_capture_owned_tool_pose_instead_of_newer_servo_feedback():
    model, data, checker, provider, evidence, clock, _ = rig()
    captured = np.eye(4)
    captured[:3, 3] = data.body('ee').xpos
    evidence = replace(evidence, captured_world_from_ee=captured)
    provider.observe_payload = lambda _: evidence
    checker.set_payload(model, data, evidence.object_ref, 'ee', capture_aligned=True)
    data.qpos[0] += .05  # Servo feedback is newer than the still-valid image.
    mujoco.mj_forward(model, data)
    assert checker.payload_retention_residual(model, data, evidence.object_ref)[0] == pytest.approx(0.)
    new_pose = evidence.pose.world_from_object.copy()
    new_pose[0, 3] += .05
    new_ee = captured.copy()
    new_ee[0, 3] += .05
    evidence = replace(evidence, pose=replace(evidence.pose, world_from_object=new_pose), captured_world_from_ee=new_ee)
    assert checker.payload_retention_residual(model, data, evidence.object_ref)[0] == pytest.approx(0.)
    new_pose[1, 3] += .03
    evidence = replace(evidence, pose=replace(evidence.pose, world_from_object=new_pose))
    assert checker.payload_retention_residual(model, data, evidence.object_ref)[0] == pytest.approx(.03)
    with pytest.raises(ValueError):
        evidence.captured_world_from_ee.setflags(write=True)

    clock[0] += 3.
    with pytest.raises(ObservationUnavailable, match='stale'):
        checker.payload_retention_residual(model, data, evidence.object_ref)


def test_hypothetical_attachment_keeps_planned_pose_even_with_capture_owned_tool_pose():
    model, data, checker, provider, evidence, *_ = rig()
    captured = np.eye(4)
    captured[:3, 3] = data.body('ee').xpos
    provider.observe_payload = lambda _: replace(evidence, captured_world_from_ee=captured)
    data.qpos[0] += .05
    mujoco.mj_forward(model, data)
    attach(model, data, checker)
    np.testing.assert_allclose(checker.payload_transform[:3, 3],
                               evidence.pose.world_from_object[:3, 3]-data.body('ee').xpos)

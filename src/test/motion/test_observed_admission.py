# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Captured material is not a free-space mask or a hypothetical-pose oracle."""
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.native_scene import model_signature
from emet.controller.manipulation.observed_collision import ObservedNativeCollisionChecker
from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.motion.arm_rrt import plan_arm_joint_path
from emet.motion.observed_admission import CapturedRobotMaterial, ObservedMaterialAdmission, RobotGeometryBound
from emet.motion.observed_background import ObservedBackground, StaticBackgroundAssumption
from emet.motion.observed_solids import WorldSolid
from emet.motion.observed_workspace import ObservedWorkspace


def solid(kind='box', dimensions=(.1, .1, .1), center=(0., 0., .5), *, yaw=0.):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    pose[:3, 3] = center
    return WorldSolid(kind, dimensions, pose)


def evidence(grid, solids, *, error=0., stamp=10., signature='model', max_nodes=20_000):
    capture = CaptureInterval(f'capture:{stamp}', stamp, stamp+.001, stamp+.01, 'clock')
    times = np.where(grid.cells != 0, stamp, np.nan)
    background = ObservedBackground(grid, times, 'epoch', frozenset((capture.observation_id,)), capture)
    material = CapturedRobotMaterial(capture, signature, 'material-calibration', 'epoch',
                                    tuple(solids), (error,)*len(solids), 'exact-fixture')
    return ObservedMaterialAdmission(background, material, max_nodes=max_nodes,
        query_robot_geometry=lambda model, data, gid: RobotGeometryBound(WorldSolid.from_geom(model, data, gid),
            gid, signature, 'material-calibration', 'exact-fixture', 0.))


def grid(cells=None):
    return ObservedWorkspace([-.3, -.3, .1], .05,
        np.zeros((12, 12, 20), np.uint8) if cells is None else cells)


@pytest.mark.parametrize('kind,dimensions', [('box', (.1, .04, .04)), ('sphere', (.08,)),
    ('capsule', (.04, .08)), ('cylinder', (.04, .08))])
def test_exact_captured_solid_admits_start_without_free_aabb_corners(kind, dimensions):
    shape = solid(kind, dimensions)
    admission = evidence(grid(), (shape,))
    before = admission.background.workspace.cells.copy()
    assert admission.query(shape, now=10.02, allow_captured_robot=True).allowed
    assert not admission.query(shape, now=10.02).allowed
    np.testing.assert_array_equal(admission.background.workspace.cells, before)
    assert not admission.background.workspace.known_free(shape.bounds)


def test_ownership_is_never_derived_from_the_candidate_pose():
    original = solid()
    admission = evidence(grid(), (original,))
    shifted = solid(center=(.03, 0, .5))
    result = admission.query(shifted, now=10.02, allow_captured_robot=True)
    assert not result.allowed and result.reason == 'unknown_swept_volume'
    assert admission.robot_material.solids[0] is original


def test_eroding_uncertain_material_cannot_authorize_the_outer_shell():
    shape = solid()
    admission = evidence(grid(), (shape,), error=.005)
    assert not admission.query(shape, now=10.02, allow_captured_robot=True).allowed
    inner = replace(shape, dimensions=(.09, .09, .09))
    assert admission.query(inner, now=10.02, allow_captured_robot=True).allowed


def test_environment_or_mixed_occupancy_wins_over_material_ownership():
    cells = grid().cells.copy()
    cells[6, 6, 8] = 2
    admission = evidence(grid(cells), (solid(),))
    result = admission.query(solid(), now=10.02, allow_captured_robot=True)
    assert not result.allowed and result.reason == 'occupied_environment'


def test_rotated_box_void_is_not_owned_material():
    shape = solid(dimensions=(.2, .02, .02), yaw=np.pi/4)
    admission = evidence(grid(), (shape,))
    assert admission.query(shape, now=10.02, allow_captured_robot=True).allowed
    void = solid('sphere', (.005,), center=(.13, -.13, .5))
    assert np.all(void.bounds[0] >= shape.bounds[0]) and np.all(void.bounds[1] <= shape.bounds[1])
    assert not admission.query(void, now=10.02, allow_captured_robot=True).allowed


def test_expiry_identity_and_subdivision_budget_reject():
    admission = evidence(grid(), (solid(),), max_nodes=1)
    with pytest.raises(ObservationUnavailable, match='stale'):
        admission.query(solid(), now=12.1, allow_captured_robot=True)
    with pytest.raises(ObservationUnavailable, match='identity_changed'):
        admission.require(now=10.02, model_signature='changed')
    with pytest.raises(ObservationUnavailable, match='capture_identity_mismatch'):
        admission.require(now=10.02, model_signature='model', capture_id='replayed')
    result = admission.query(solid(center=(.03, 0, .5)), now=10.02, allow_captured_robot=True)
    assert not result.allowed and result.reason == 'admission_budget_exhausted'


def test_mesh_does_not_become_owned_convex_material_without_a_declaration():
    model = mujoco.MjModel.from_xml_string('''<mujoco><asset><mesh name="shape"
      vertex="-.1 -.1 -.1 .1 -.1 -.1 .1 .1 -.1 -.1 .1 -.1 -.1 -.1 .1 .1 -.1 .1 .1 .1 .1 -.1 .1 .1"/>
      </asset><worldbody><geom type="mesh" mesh="shape"/></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    with pytest.raises(ValueError, match='material_geometry_unavailable'):
        WorldSolid.from_geom(model, data, 0)
    declared = WorldSolid.from_geom(model, data, 0, convex_material_calibration_id='explicit-test-solid')
    assert declared.contains_box([[-.01]*3, [.01]*3])


@pytest.mark.parametrize('change', ['location', 'size', 'geom', 'calibration', 'profile'])
def test_callback_cannot_hide_or_shrink_a_real_robot_geom(change):
    model, data, _, _, original = raycast_scene()
    def query(m, d, gid):
        bound = original.query_robot_geometry(m, d, gid)
        if change == 'location':
            return replace(bound, solid=solid('sphere', (.001,), center=(0., 0., .2)))
        if change == 'size':
            return replace(bound, solid=replace(bound.solid, dimensions=tuple(v/10 for v in bound.solid.dimensions)))
        if change == 'geom':
            return replace(bound, geom_id=gid+1)
        if change == 'calibration':
            return replace(bound, calibration_id='different')
        return replace(bound, configuration_profile_id='different')
    admission = ObservedMaterialAdmission(original.background, original.robot_material, query_robot_geometry=query)
    checker = ObservedNativeCollisionChecker(model, robot_body='robot', ee_body='ee',
        scene=original.background.placement_scene(clock=lambda: 10.02),
        payload_provider=SimpleNamespace(input_mode='observed', observe_payload=lambda _: None),
        scene_revision='scene', current_revision=lambda: 'scene', received_monotonic=10.01,
        clock=lambda: 10.02, margin_m=0, admission=admission)
    with pytest.raises(ObservationUnavailable, match='envelope_unknown'):
        checker.configuration_collides(model, data)


def test_old_free_then_falsely_self_labelled_obstacle_cannot_be_reused_as_free():
    base = grid(np.ones(grid().cells.shape, np.uint8))
    old = evidence(base, (solid(),), stamp=9.)
    cells = base.cells.copy()
    index = (10, 6, 8)  # Outside the captured robot solid.
    cells[index] = 2
    point = base.origin + (np.array(index)+.5)*base.resolution
    partition = ObservedWorkspace(base.origin, base.resolution, cells).partition(point[None], np.array(['robot']))
    capture = CaptureInterval('current', 10., 10.001, 10.01, 'clock')
    material = replace(old.robot_material, capture=capture)
    # Without a geometry proof, excluded endpoint cells lose cached free status.
    updated = old.background.update(partition, excluded_refs=('robot',), scene_epoch='epoch', capture=capture, now=10.02)
    admission = ObservedMaterialAdmission(updated, material, query_robot_geometry=old.query_robot_geometry)
    obstacle = solid('sphere', (.005,), center=tuple(point))
    assert not admission.query(obstacle, now=10.02, allow_captured_robot=True).allowed
    # With qualification requested, the false label remains environment evidence
    # and explicitly invalidates the formerly free background epoch.
    with pytest.raises(ObservationUnavailable, match='background_scene_changed'):
        old.background.update(partition, excluded_refs=('robot',), scene_epoch='epoch', capture=capture,
                              now=10.02, robot_material=material)


def test_background_and_material_clock_calibration_must_match():
    original = evidence(grid(), (solid(),))
    material = replace(original.robot_material,
        capture=replace(original.robot_material.capture, calibration_id='another-clock'))
    with pytest.raises(ValueError, match='invalid_observed_admission'):
        ObservedMaterialAdmission(original.background, material, query_robot_geometry=original.query_robot_geometry)


def test_an_old_obstacle_cannot_be_erased_by_a_later_unqualified_self_label():
    cells = grid().cells.copy()
    index = (6, 6, 8)
    cells[index] = 2
    old = evidence(grid(cells), (solid(),), stamp=9.)
    point = old.background.workspace.origin + (np.array(index)+.5)*old.background.workspace.resolution
    current = old.background.workspace.partition(point[None], np.array(['robot']))
    capture = CaptureInterval('current', 10., 10.001, 10.01, 'clock')
    updated = old.background.update(current, excluded_refs=('robot',), scene_epoch='epoch', capture=capture, now=10.02)
    assert updated.workspace.cells[index] == 2 and updated.observed_monotonic[index] == 9.
    admission = ObservedMaterialAdmission(updated, replace(old.robot_material, capture=capture),
        query_robot_geometry=old.query_robot_geometry)
    result = admission.query(solid(), now=10.02, allow_captured_robot=True)
    assert not result.allowed and result.reason == 'occupied_environment'


def raycast_scene():
    robot = '''<body name="robot" pos="0 0 .5"><joint name="z" type="slide"
      axis="0 0 1" range="0 .3"/><geom type="box" size=".08 .08 .06"/>
      <body name="ee" pos=".1 0 0"><geom size=".015"/></body></body>'''
    wall = '<geom name="wall" type="box" pos="0 0 1.4" size="2 2 .02"/>'
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+robot+'</worldbody></mujoco>')
    sensor = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+robot+wall+'</worldbody></mujoco>')
    data, sensor_data = mujoco.MjData(model), mujoco.MjData(sensor)
    mujoco.mj_forward(model, data)
    k = np.array([[50., 0, 40.], [0, 50., 40.], [0, 0, 1.]])

    def capture(prescan=False):
        # Independent scene setup: before the robot enters the scanned volume.
        # Test-only sensor truth attributes endpoints; it is not a live source.
        sensor.body_pos[1, 0] = -3. if prescan else 0.
        mujoco.mj_forward(sensor, sensor_data)
        depth, points, labels = np.zeros((81, 81)), [], []
        for y, x in np.ndindex(depth.shape):
            ray = np.array([(x-40)/50, (y-40)/50, 1.])
            ray /= np.linalg.norm(ray)
            hit = np.full(1, -1, dtype=np.int32)
            distance = mujoco.mj_ray(sensor, sensor_data, np.zeros(3), ray, None, True, -1, hit)
            if distance < 0:
                continue
            depth[y, x] = distance*ray[2]
            points.append(distance*ray)
            labels.append('background' if hit[0] == sensor.geom('wall').id else 'robot')
        workspace = ObservedWorkspace.from_depth(depth, k, np.eye(4),
            workspace=[[-.3, -.3, .1], [.5, .3, 1.5]], resolution=.05, depth_margin_m=.005)
        return workspace.partition(np.array(points), np.array(labels))

    prescan, current = capture(True), capture(False)
    current_grid = current.excluding(('robot',))
    signature = model_signature(model)
    solids = tuple(WorldSolid.from_geom(model, data, g) for g in range(model.ngeom))
    admission = evidence(current_grid, solids, signature=signature)
    return model, data, prescan, current, admission


def test_actual_self_occluding_depth_admits_only_captured_material_then_blocks_shadow():
    model, data, _, _, admission = raycast_scene()
    scene = admission.background.placement_scene(clock=lambda: 10.02)
    common = {'robot_body': 'robot', 'ee_body': 'ee', 'scene': scene,
        'payload_provider': SimpleNamespace(input_mode='observed', observe_payload=lambda _: None),
        'scene_revision': 'scene', 'current_revision': lambda: 'scene', 'received_monotonic': 10.01,
        'clock': lambda: 10.02, 'margin_m': 0}
    raw = ObservedNativeCollisionChecker(model, **common)
    assert raw.configuration_collides(model, data)
    checker = ObservedNativeCollisionChecker(model, **common, admission=admission)
    before = data.qpos.copy(), data.qacc.copy(), data.xpos.copy()
    assert not checker.configuration_collides(model, data)
    for actual, expected in zip((data.qpos, data.qacc, data.xpos), before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    # The planner owns a private state and may leave it at a candidate endpoint.
    planning = mujoco.MjData(model)
    mujoco.mj_copyData(planning, model, data)
    result = plan_arm_joint_path(model, planning, joint_names=['z'], q_start=[0], q_goal=[.05],
                                collision=checker, planner='linear')
    assert not result.success and result.reason != 'invalid_start'


def test_real_prescan_free_evidence_allows_shadow_without_renewing_hidden_age():
    model, data, prescan, current, original = raycast_scene()
    old = CaptureInterval('prescan', 9., 9.001, 9.01, 'clock')
    background = ObservedBackground.from_capture(prescan, excluded_refs=('robot',),
        scene_epoch='epoch', capture=old, now=9.02)
    # The historical scan supplies the corridor. Unqualified foreground labels
    # are not used to maintain free cells; their invalidation is tested below.
    admission = ObservedMaterialAdmission(background, original.robot_material,
        query_robot_geometry=original.query_robot_geometry)
    checker = ObservedNativeCollisionChecker(model, robot_body='robot', ee_body='ee',
        scene=background.placement_scene(clock=lambda: 10.02),
        payload_provider=SimpleNamespace(input_mode='observed', observe_payload=lambda _: None),
        scene_revision='scene', current_revision=lambda: 'scene', received_monotonic=10.01,
        clock=lambda: 10.02, margin_m=0, admission=admission)
    assert plan_arm_joint_path(model, data, joint_names=['z'], q_start=[0], q_goal=[.05],
                              collision=checker, planner='linear').success
    hidden = np.floor((np.array([0., 0., .65])-background.workspace.origin)/background.workspace.resolution).astype(int)
    assert background.observed_monotonic[tuple(hidden)] == 9.
    new = CaptureInterval('foreground', 12., 12.001, 12.01, 'clock')
    updated = background.update(current, excluded_refs=('robot',), scene_epoch='epoch', capture=new, now=12.02)
    assert updated.observed_monotonic[tuple(hidden)] == 9.
    fresh_robot = replace(original.robot_material, capture=new)
    expired = ObservedMaterialAdmission(updated, fresh_robot, query_robot_geometry=original.query_robot_geometry)
    shifted = WorldSolid.from_geom(model, data, 0)
    pose = shifted.world_from_local.copy()
    pose[2, 3] += .05
    assert not expired.query(replace(shifted, world_from_local=pose), now=12.02, allow_captured_robot=True).allowed
    assumption = StaticBackgroundAssumption('epoch', 8., 13., 'bounded-static-scene')
    qualified = ObservedMaterialAdmission(background, fresh_robot, static_assumption=assumption,
        query_robot_geometry=original.query_robot_geometry)
    assert qualified.query(replace(shifted, world_from_local=pose), now=12.02, allow_captured_robot=True).allowed

# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Local contact refines support evidence without erasing obstacles or unknowns."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.native_inputs import ObjectPoseEvidence, ObservationUnavailable
from emet.controller.manipulation.native_scene import model_signature
from emet.controller.manipulation.observed_collision import ObservedNativeCollisionChecker, PayloadGeometryEvidence
from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.motion.arm_rrt import plan_arm_joint_path
from emet.motion.observed_admission import CapturedRobotMaterial, ObservedMaterialAdmission, RobotGeometryBound
from emet.motion.observed_background import ObservedBackground, StaticBackgroundAssumption
from emet.motion.observed_solids import WorldSolid
from emet.motion.observed_support import fit_observed_support
from emet.motion.observed_support_contact import ObservedProbeBand, ObservedSupportContact
from emet.motion.observed_workspace import ObservedWorkspace
from emet.motion.placement_geometry import box_corners


def box(center=(0.0, 0.0, 0.0397), half=(0.02, 0.02, 0.04)):
    pose = np.eye(4)
    pose[:3, 3] = center
    return WorldSolid("box", half, pose)


def scene(*, slope=0.0, backing=0.02):
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 1.0
    k = np.array([[100.0, 0.0, 47.5], [0.0, 100.0, 47.5], [0.0, 0.0, 1.0]])
    y, x = np.indices((96, 96))
    depth = 1.0 / (1.0 + slope * (x - 47.5) / 100.0)
    obs = SimpleNamespace(
        rgb=np.zeros((96, 96, 3), np.uint8), depth=depth, camera_K=k, camera_pose=pose, received_monotonic=10.01
    )
    patch = fit_observed_support(obs, np.ones(depth.shape, bool), observation_id="scan", measurement_margin_m=0.0001)
    ticket = CaptureInterval("scan", 10.0, 10.001, 10.01, "clock")
    grid = ObservedWorkspace.from_depth(
        depth, k, pose, workspace=[[-0.25, -0.25, -0.025], [0.25, 0.25, 0.225]], resolution=0.05, depth_margin_m=0.0001
    )
    rays = np.stack((x.ravel(), y.ravel(), np.ones(depth.size)), axis=-1) @ np.linalg.inv(k).T
    points = (rays * depth.ravel()[:, None]) @ pose[:3, :3].T + pose[:3, 3]
    partition = grid.partition(points, np.full(len(points), "table"))
    background = ObservedBackground.from_capture(
        partition, excluded_refs=(), scene_epoch="epoch", capture=ticket, now=10.02
    )
    contact = ObservedSupportContact(
        "item",
        "table",
        "epoch",
        "ideal-pinhole-fixture",
        patch,
        ticket,
        depth,
        k,
        pose,
        np.ones(depth.shape, bool),
        payload_motion_error_m=0.0,
        backing_depth_m=backing,
        backing_prior_id="declared-solid-top-layer" if backing else "",
    )
    return background, contact, partition


def admission(background, *, signature="model", static=None, capture=None):
    material = CapturedRobotMaterial(
        capture or background.latest_capture,
        signature,
        "robot-calibration",
        "epoch",
        (box(center=(0.18, 0.0, 0.15), half=(0.008,) * 3),),
        (0.0,),
        "robot-profile",
    )
    return ObservedMaterialAdmission(
        background,
        material,
        static_assumption=static,
        query_robot_geometry=lambda m, d, g: RobotGeometryBound(
            WorldSolid.from_geom(m, d, g, offset_m=np.sqrt(3) * 0.005),
            g,
            signature,
            "robot-calibration",
            "robot-profile",
            0.0,
        ),
    )


def query(layer, contact, physical=None, *, phase="place", ref="item", padding=0.005, floor=None, now=10.02):
    physical = box() if physical is None else physical
    padded = replace(physical, dimensions=tuple(v + padding for v in physical.dimensions))
    return layer.query_payload(
        physical, padded, contact=contact, object_ref=ref, phase=phase, now=now, departure_clearance_m=floor
    )


def test_support_contact_admits_shallow_payload_but_never_frees_the_voxel():
    background, contact, _ = scene()
    layer = admission(background)
    padded = replace(box(), dimensions=(0.025, 0.025, 0.045))
    before = background.workspace.cells.copy()
    assert not layer.query(padded, now=10.02, allow_captured_robot=True).allowed
    assert query(layer, contact).allowed
    assert not background.known_free(padded.bounds, now=10.02)
    np.testing.assert_array_equal(background.workspace.cells, before)


def test_occluded_pickup_footprint_needs_nearby_support_on_all_sides():
    background, original, _ = scene(backing=0.0)
    patch = replace(original.patch, bounds=[[-.06, -.06, -.0001], [-.025, .06, .0001]])
    mask = original.support_mask.copy()
    mask[45:51, 45:51] = False
    contact = replace(original, patch=patch, support_mask=mask)
    footprint = box().bounds
    observed = contact.pickup_observation_bounds(footprint)
    np.testing.assert_allclose(observed[:, :2], [[-.025, -.025], [.025, .025]])
    np.testing.assert_allclose(observed[:, 2], [-.0001, .0001], atol=1e-10)
    assert not query(admission(background), contact, phase="lift", floor=-.001).allowed

    mask[:, 48:] = False
    one_sided = replace(contact, support_mask=mask)
    assert one_sided.pickup_observation_bounds(footprint) is None


def test_probe_band_allows_local_payload_contact_but_keeps_obstacles_occupied():
    background, contact, _ = scene(backing=0.0)
    shallow = box(center=(0., 0., .009), half=(.01, .01, .01))
    band = ObservedProbeBand(contact, .004)
    layer = admission(background)
    assert not layer.query(shallow, now=10.02, free_witness=contact).allowed
    assert layer.query_probe_payload(shallow, shallow, band=band, object_ref="item", now=10.02).allowed
    occupants = tuple(background.exclusively_owned_cells("table"))
    blocked = admission(replace(background, occupied_owners=(("table", occupants), ("obstacle", occupants))))
    verdict = blocked.query_probe_payload(shallow, shallow, band=band, object_ref="item", now=10.02)
    assert not verdict.allowed and verdict.reason == "occupied_environment"


def test_cached_exclusive_owners_follow_immutable_background_replacement():
    background, _, _ = scene()
    original = background.exclusively_owned_cells('table')
    assert original and background.exclusively_owned_cells('table') is original
    changed = replace(background, occupied_owners=(('other', tuple(original)),))
    assert not changed.exclusively_owned_cells('table')
    assert background.exclusively_owned_cells('table') is original


def test_robot_free_witness_refines_coarse_surface_voxel_without_allowing_contact():
    background, contact, _ = scene()
    layer = admission(background)
    hovering = box(center=(0., 0., .011), half=(.01, .01, .01))
    before = background.workspace.cells.copy()
    assert not layer.query(hovering, now=10.02).allowed
    assert layer.query(hovering, now=10.02, free_witness=contact).allowed
    touching = box(center=(0., 0., .0097), half=(.01, .01, .01))
    assert not layer.query(touching, now=10.02, free_witness=contact).allowed
    np.testing.assert_array_equal(background.workspace.cells, before)


def test_free_ray_witness_does_not_refine_mixed_or_unobserved_cells():
    background, contact, _ = scene()
    hovering = box(center=(0., 0., .011), half=(.01, .01, .01))
    # Removing exclusive ownership makes these ordinary obstacle cells.
    mixed = replace(background, occupied_owners=())
    assert not admission(mixed).query(hovering, now=10.02, free_witness=contact).allowed
    cells = background.workspace.cells.copy()
    cells[5:7, 5:7, 0:2] = 0
    unknown = replace(background, workspace=ObservedWorkspace(background.workspace.origin, .05, cells))
    assert admission(unknown).query(hovering, now=10.02, free_witness=contact).allowed
    mask = contact.support_mask.copy()
    mask[40:56, 40:56] = False
    hidden = replace(contact, support_mask=mask)
    assert not admission(unknown).query(hovering, now=10.02, free_witness=hidden).allowed


@pytest.mark.parametrize("phase", ["preflight", "pregrasp", "grasp", "preplace", "retreat", "gripper_open"])
def test_wrong_phases_keep_ordinary_support_collision(phase):
    background, contact, _ = scene()
    assert not query(admission(background), contact, phase=phase).allowed


def test_other_object_cannot_inherit_the_named_support_contact():
    background, contact, _ = scene()
    assert not query(admission(background), contact, ref="other").allowed


@pytest.mark.parametrize("penetration, allowed", [(0.00085, True), (0.00095, False), (0.003, False)])
def test_one_millimetre_limit_keeps_plane_error_and_ignores_only_planning_padding(penetration, allowed):
    background, contact, _ = scene()
    physical = box(center=(0.0, 0.0, 0.04 - penetration))
    assert query(admission(background), contact, physical, padding=0.01).allowed is allowed


def test_observed_surface_without_backing_prior_cannot_admit_hidden_material():
    background, contact, _ = scene(backing=0.0)
    assert not query(admission(background), contact).allowed
    # Genuine ray-free space inside the same coarse support voxel is still
    # provable; this does not imply material behind the return.
    assert query(admission(background), contact, box(center=(0.0, 0.0, 0.055))).allowed


def test_padding_cannot_reach_behind_the_declared_top_layer():
    background, contact, _ = scene(backing=0.002)
    assert not query(admission(background), contact).allowed


@pytest.mark.parametrize('z_origin', [-.025, -.00001, .00001])
def test_shallow_support_contact_is_independent_of_endpoint_cell_boundary(z_origin):
    background, contact, _ = scene()
    origin = background.workspace.origin.copy()
    origin[2] = z_origin
    cells = np.zeros_like(background.workspace.cells)
    # Original returns at z=0 lie either just above or below a grid boundary.
    index = int(np.floor(-z_origin/.05))
    if index >= 0:
        cells[:, :, index] = 2
    # Extend downward so the complete padded payload stays inside the grid.
    origin[2] -= .05
    cells = np.concatenate((np.zeros_like(cells[:, :, :1]), cells), axis=2)
    ids = tuple(map(int, np.flatnonzero(cells.ravel() == 2)))
    shifted = replace(background, workspace=ObservedWorkspace(origin, .05, cells),
        observed_monotonic=np.where(cells != 0, contact.capture.earliest, np.nan),
        occupied_owners=(('table', ids),))
    assert query(admission(shifted), contact).allowed
    assert not query(admission(shifted), contact, box(center=(0., 0., .038))).allowed
    assert not query(admission(shifted), contact, phase='pregrasp').allowed
    assert not query(admission(shifted), replace(contact, backing_depth_m=.002)).allowed


def test_mixed_cell_and_historical_unattributed_evidence_remain_obstacles():
    background, contact, partition = scene()
    point = np.array([[0.001, 0.001, 0.001]])
    mixed = partition.workspace.partition(
        np.vstack((partition.points_world, point)), np.concatenate((partition.point_labels, ["peg"]))
    )
    blocked = ObservedBackground.from_capture(
        mixed, excluded_refs=(), scene_epoch="epoch", capture=contact.capture, now=10.02
    )
    assert not query(admission(blocked), contact).allowed
    legacy = replace(background, occupied_owners=())
    fresh = replace(contact.capture, observation_id="next", earliest=11.0, latest=11.001, received_monotonic=11.01)
    updated = legacy.update(partition, excluded_refs=(), scene_epoch="epoch", capture=fresh, now=11.02)
    assert not updated.exclusively_owned_cells("table")
    assert not query(admission(updated), contact, now=11.02).allowed


def test_unknown_above_support_needs_actual_original_rays():
    background, contact, _ = scene()
    cells = background.workspace.cells.copy()
    cells[5, 5, 2] = 0
    modified = replace(background, workspace=ObservedWorkspace(background.workspace.origin, 0.05, cells))
    mask = contact.support_mask.copy()
    mask[40:56, 40:56] = False
    hidden = replace(contact, support_mask=mask)
    assert not query(admission(modified), hidden).allowed


def test_lift_departure_uses_one_fixed_floor_in_any_query_order():
    background, contact, _ = scene()
    layer = admission(background)
    initial = box(center=(0.0, 0.0, 0.0398))
    floor = contact.physical_clearance(initial)
    for physical, expected in (
        (box(center=(0.0, 0.0, 0.06)), True),
        (box(center=(0.0, 0.0, 0.0395)), False),
        (initial, True),
        (box(center=(0.0, 0.0, 0.0395)), False),
    ):
        assert query(layer, contact, physical, phase="lift", floor=floor).allowed is expected
    assert not query(layer, contact, initial, phase="lift").allowed


def test_sloped_plane_checks_the_worst_corner_not_just_the_center():
    background, contact, _ = scene(slope=0.05)
    assert contact.physical_clearance(box(center=(0.0, 0.0, 0.0398))) < -0.001
    assert not query(admission(background), contact, box(center=(0.0, 0.0, 0.0398))).allowed
    assert query(admission(background), contact, box(center=(0.0, 0.0, 0.0414))).allowed


def test_complete_padded_footprint_must_fit_the_finite_measured_patch():
    background, contact, _ = scene()
    smaller = replace(contact.patch, bounds=np.array([[-0.1, -0.1, -0.0001], [0.1, 0.1, 0.0001]]))
    contact = replace(contact, patch=smaller)
    assert not query(admission(background), contact, box(center=(0.079, 0.0, 0.0397))).allowed


def test_depth_hole_and_capture_calibration_changes_reject():
    background, contact, _ = scene()
    mask = contact.support_mask.copy()
    mask[:, 47:49] = False
    assert not query(admission(background), replace(contact, support_mask=mask)).allowed
    with pytest.raises(ObservationUnavailable):
        contact.require(background, now=12.1)
    with pytest.raises(ObservationUnavailable):
        replace(contact, capture=replace(contact.capture, calibration_id="other")).require(background, now=10.02)
    with pytest.raises(ValueError):
        replace(contact, capture=replace(contact.capture, observation_id="retimed"))
    with pytest.raises(ValueError):
        contact.depth.setflags(write=True)


def test_support_owners_accumulate_without_refreshing_old_occupied_ages():
    background, contact, partition = scene()
    other = replace(contact.capture, observation_id="next", earliest=11.0, latest=11.001, received_monotonic=11.01)
    updated = background.update(partition, excluded_refs=(), scene_epoch="epoch", capture=other, now=11.02)
    assert updated.geometry_revision == background.geometry_revision
    np.testing.assert_array_equal(updated.observed_monotonic[updated.workspace.cells == 2], 10.0)
    assert updated.exclusively_owned_cells("table") == background.exclusively_owned_cells("table")
    assumption = StaticBackgroundAssumption("epoch", 10.0, 20.0, "static-support-fixture")
    assert query(admission(updated, static=assumption), contact, now=11.02).allowed
    with pytest.raises(ObservationUnavailable):
        contact.require(updated, now=20.1, static_assumption=assumption)


def test_excluded_foreground_in_mixed_cell_never_promotes_support_only_history():
    background, contact, partition = scene()
    point = np.array([[0.001, 0.001, 0.001]])
    mixed = partition.workspace.partition(
        np.vstack((partition.points_world, point)), np.concatenate((partition.point_labels, ["unqualified-foreground"]))
    )
    stamp = replace(contact.capture, observation_id="new", earliest=11.0, latest=11.001, received_monotonic=11.01)
    updated = background.update(
        mixed, excluded_refs=("unqualified-foreground",), scene_epoch="epoch", capture=stamp, now=11.02
    )
    assert dict(updated.occupied_owners)["unqualified-foreground"]
    assert not query(admission(updated), contact, now=11.02).allowed


def test_historical_witness_cannot_reuse_its_id_with_a_new_interval():
    background, contact, partition = scene()
    stamp = replace(contact.capture, observation_id="new", earliest=11.0, latest=11.001, received_monotonic=11.01)
    updated = background.update(partition, excluded_refs=(), scene_epoch="epoch", capture=stamp, now=11.02)
    retimed = replace(contact.capture, earliest=11.0, latest=11.001, received_monotonic=11.01)
    changed = replace(contact, capture=retimed, patch=replace(contact.patch, received_monotonic=11.01))
    with pytest.raises(ObservationUnavailable):
        changed.require(updated, now=11.02)


def rig(*, pose_error=0.0):
    background, contact, _ = scene()
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="robot" pos=".18 0 .15"><joint name="height" type="slide" axis="0 0 1" range="-.2 .2"/>
        <geom type="sphere" size=".008"/><body name="ee"/></body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    payload_pose = box().world_from_local
    payload = PayloadGeometryEvidence(
        "item",
        "reference",
        "prior",
        box_corners([[-0.02, -0.02, -0.04], [0.02, 0.02, 0.04]]),
        ObjectPoseEvidence(payload_pose, 10.01, "rgbd_tracking", "scan"),
        "known_model_prior",
        position_error_m=pose_error,
    )
    checker = ObservedNativeCollisionChecker(
        model,
        robot_body="robot",
        ee_body="ee",
        scene=background.placement_scene(clock=lambda: 10.02),
        payload_provider=SimpleNamespace(input_mode="observed", observe_payload=lambda _: payload),
        scene_revision="scene",
        current_revision=lambda: "scene",
        received_monotonic=10.01,
        clock=lambda: 10.02,
        margin_m=0.005,
        admission=admission(background, signature=model_signature(model)),
        support_contact=contact,
    )
    return model, data, checker


def test_real_checker_uses_contact_for_payload_and_static_release_but_not_robot():
    model, data, checker = rig()
    checker.set_payload(model, data, "item", "ee")
    assert checker.configuration_collides(model, data)
    with checker.motion_phase("place"):
        assert not checker.configuration_collides(model, data)
    checker.set_payload(model, data, None)
    with checker.motion_phase("retreat"):
        assert not checker.configuration_collides(model, data)
    # The robot moves down into the same support; it has no support exemption.
    data.qpos[0] = -0.15
    with checker.motion_phase("place"):
        assert checker.configuration_collides(model, data)


def test_checker_keeps_pose_uncertainty_and_restores_the_departure_anchor():
    model, data, checker = rig(pose_error=0.002)
    checker.set_payload(model, data, "item", "ee")
    state = checker.capture_payload()
    with checker.motion_phase("place"):
        assert checker.configuration_collides(model, data)
    checker.set_payload(model, data, None)
    checker.restore_payload(state)
    assert checker._support_departure_clearance == state[-3]
    assert checker._support_payload_error_m == 0.002


def test_declared_encoder_fk_and_retention_error_is_part_of_the_physical_bound():
    model, data, checker = rig()
    checker.set_support_contact(replace(checker.support_contact, payload_motion_error_m=0.002))
    checker.set_payload(model, data, "item", "ee")
    with checker.motion_phase("place"):
        assert checker.configuration_collides(model, data)
    assert checker._support_payload_error_m == 0.002


def test_release_checks_current_geometry_in_addition_to_the_frozen_hypothesis():
    model, data, checker = rig()
    checker.set_payload(model, data, "item", "ee")
    checker.set_payload(model, data, None)
    original = checker.provider.observe_payload("item")
    pose = original.pose.world_from_object.copy()
    pose[2, 3] -= 0.01
    checker.provider.observe_payload = lambda _: replace(original, pose=replace(original.pose, world_from_object=pose))
    with checker.motion_phase("retreat"):
        assert checker.configuration_collides(model, data)


def test_measured_pose_replan_cannot_reset_the_existing_lift_floor():
    model, data, checker = rig()
    checker.set_payload(model, data, "item", "ee")
    floor = checker._support_departure_clearance
    original = checker.provider.observe_payload("item")
    pose = original.pose.world_from_object.copy()
    pose[2, 3] -= 0.0004
    checker.provider.observe_payload = lambda _: replace(
        original, pose=replace(original.pose, world_from_object=pose, received_monotonic=10.012, observation_id="later")
    )
    checker.set_payload(model, data, "item", "ee")
    assert checker._support_departure_clearance == floor
    with checker.motion_phase("lift"):
        assert checker.configuration_collides(model, data)


@pytest.mark.parametrize("peg", [False, True])
def test_mujoco_depth_returns_admit_table_contact_and_preserve_a_small_peg(peg):
    _, contact, _ = scene()
    extra = '<body pos=".005 -.005 .004"><geom name="peg" size=".004"/></body>' if peg else ""
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="table" type="box" '
        'pos="0 0 -.02" size=".6 .6 .02"/>' + extra + "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    depth, points, labels = np.zeros((96, 96)), [], []
    mask = np.zeros(depth.shape, bool)
    for y, x in np.ndindex(depth.shape):
        ray = np.linalg.inv(contact.intrinsic) @ [x, y, 1.0]
        direction = contact.world_from_camera[:3, :3] @ ray
        direction /= np.linalg.norm(direction)
        hit = np.full(1, -1, np.int32)
        distance = mujoco.mj_ray(model, data, contact.world_from_camera[:3, 3], direction, None, True, -1, hit)
        assert distance > 0
        depth[y, x] = distance / np.linalg.norm(ray)
        points.append(contact.world_from_camera[:3, 3] + distance * direction)
        # Test-only independent sensor fixture attribution; never agent GT input.
        ref = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(hit[0]))
        labels.append(ref)
        mask[y, x] = ref == "table"
    grid = ObservedWorkspace.from_depth(
        depth,
        contact.intrinsic,
        contact.world_from_camera,
        workspace=[[-0.25, -0.25, -0.025], [0.25, 0.25, 0.225]],
        resolution=0.05,
        depth_margin_m=0.0001,
    )
    background = ObservedBackground.from_capture(
        grid.partition(np.array(points), np.array(labels)),
        excluded_refs=(),
        scene_epoch="epoch",
        capture=contact.capture,
        now=10.02,
    )
    contact = replace(contact, depth=depth, support_mask=mask)
    assert query(admission(background), contact).allowed is not peg


@pytest.mark.parametrize("phase", ["lift", "place"])
def test_checked_joint_path_has_independently_bounded_payload_support_penetration(phase, record_testsuite_property):
    model, data, checker = rig()
    checker.set_payload(model, data, "item", "ee")
    start, goal = (0.0, 0.02) if phase == "lift" else (0.02, 0.0)
    with checker.motion_phase(phase):
        path = plan_arm_joint_path(
            model,
            data,
            joint_names=("height",),
            q_start=np.array([start]),
            q_goal=np.array([goal]),
            collision=checker,
            planner="rrt_connect",
            step_size=0.002,
            goal_tolerance=0.0001,
            max_iter=100,
            shortcut=False,
            rng=np.random.default_rng(7),
        )
    assert path.success, path
    # Independent OFFLINE collision audit. This evaluator's replayed free-body
    # poses never enter runtime planning or represent a physics controller.
    truth = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="box" pos="0 0 -.02" size=".6 .6 .02"/>
      <body name="payload"><freejoint/><geom type="box" size=".02 .02 .04"/></body>
      </worldbody></mujoco>""")
    probe = mujoco.MjData(truth)
    checked = 0
    contact_samples, peak_penetration = 0, 0.0
    for a, b in zip(path.waypoints, path.waypoints[1:], strict=False):
        for q in np.linspace(a, b, 11):
            data.qpos[0] = q[0]
            mujoco.mj_kinematics(model, data)
            ee = data.body("ee")
            vertices = checker.payload.world_vertices(ee.xpos, ee.xmat)
            probe.qpos[:3] = (vertices.min(0) + vertices.max(0)) / 2
            mujoco.mj_kinematics(truth, probe)
            mujoco.mj_collision(truth, probe)
            assert all(float(probe.contact[i].dist) >= -0.001 for i in range(probe.ncon))
            contact_samples += int(probe.ncon > 0)
            peak_penetration = max(
                peak_penetration, max((-float(probe.contact[i].dist) for i in range(probe.ncon)), default=0.0)
            )
            checked += 1
    assert checked >= 11
    assert contact_samples > 0
    record_testsuite_property(f"observed_support_{phase}_audit_samples", checked)
    record_testsuite_property(f"observed_support_{phase}_contact_samples", contact_samples)
    record_testsuite_property(f"observed_support_{phase}_peak_penetration_m", peak_penetration)


def test_joint_planner_cannot_ratchet_lift_below_measured_contact_floor():
    model, data, checker = rig()
    checker.set_payload(model, data, "item", "ee")
    with checker.motion_phase("lift"):
        result = plan_arm_joint_path(
            model,
            data,
            joint_names=("height",),
            q_start=np.array([0.0]),
            q_goal=np.array([-0.0004]),
            collision=checker,
            rng=np.random.default_rng(7),
        )
    assert not result.success and result.reason == "invalid_goal"

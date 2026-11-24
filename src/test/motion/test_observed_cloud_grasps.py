# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import numpy as np
import pytest

from emet.perception.grasps.observed import ObservedGeometry, TopDownCloudProvider


def cloud(width=.02):
    rng = np.random.default_rng(42)
    return ObservedGeometry('capture-1', 'wrist', 10.,
                            rng.uniform([-width/2, -.04, .1], [width/2, .04, .12], (200, 3)), .003)


def test_partial_cloud_proposals_preserve_capture_and_native_hand_convention():
    geometry = cloud()
    proposals = TopDownCloudProvider(.06).propose(geometry)
    assert len(proposals) == 1
    p = proposals[0]
    assert p.observation_id == geometry.observation_id
    assert p.opening_m < .06
    assert np.allclose(p.world_from_hand[:3, 2], [0, 0, -1])
    assert np.linalg.det(p.world_from_hand[:3, :3]) == pytest.approx(1.)
    with pytest.raises(ValueError):
        geometry.points_world[0] = 0


def test_uncertainty_and_opening_reject_oversize_cloud():
    assert not TopDownCloudProvider(.01).propose(cloud())
    with pytest.raises(ValueError):
        TopDownCloudProvider(float('nan'))


def test_grounding_requires_same_capture_and_aligned_refined_mask():
    from emet.perception.grasps.observed import GroundingHint, geometry_from_mask

    hint = GroundingHint('capture-1', 'wrist', (0, 0, 1000, 1000), 'shaker')
    points = cloud().points_world.reshape(10, 20, 3)
    kwargs = {'camera_id': 'wrist', 'acquired_monotonic': 10., 'mask': np.ones((10, 20), bool),
                  'world_points': points, 'position_error_m': .003}
    result = geometry_from_mask(hint, observation_id='capture-1', **kwargs)
    assert len(result.points_world) == 200
    with pytest.raises(ValueError, match='capture_identity_mismatch'):
        geometry_from_mask(hint, observation_id='another', **kwargs)


def test_native_grasp_calibration_preserves_world_target_and_rejects_replay():
    from emet.controller.manipulation.native_inputs import ObservationUnavailable
    from emet.controller.task.tamp.observed_candidates import calibrate_cloud_grasps

    proposals = TopDownCloudProvider(.06).propose(cloud())
    reference, calibration = np.eye(4), np.eye(4)
    reference[:3, 3] = [1., 2., 3.]
    calibration[:3, 3] = [0., 0., .04]
    kwargs = {'world_from_reference': reference, 'hand_from_ee': calibration,
                  'close_joints': (('finger', .01),)}
    profiles = calibrate_cloud_grasps(proposals, observation_id='capture-1', **kwargs)
    assert np.allclose(reference @ profiles[0].object_from_ee,
                       proposals[0].world_from_hand @ calibration)
    with pytest.raises(ObservationUnavailable, match='capture_identity_mismatch'):
        calibrate_cloud_grasps(proposals, observation_id='new-capture', **kwargs)


def test_agent_transfer_hints_reject_stale_capture_before_segmentation():
    from types import SimpleNamespace

    from emet.controller.manipulation.native_inputs import ObservationUnavailable
    from emet.controller.task.tamp.transfer_grounding import select_transfer_from_hints
    from emet.perception.grasps.observed import GroundingHint

    hint = GroundingHint('old', 'wrist', (0, 0, 1000, 1000), 'object')
    sample = SimpleNamespace(capture=SimpleNamespace(observation_id='new'), observations={'wrist': object()})
    called = []
    with pytest.raises(ObservationUnavailable, match='capture_identity_mismatch'):
        select_transfer_from_hints(sample, dict.fromkeys(('object', 'source', 'destination'), hint),
                                   refine=lambda *args: called.append(args))
    assert not called


def test_image_hints_to_depth_grasps_to_native_frame_without_object_model():
    from types import SimpleNamespace

    from emet.controller.task.tamp.observed_candidates import calibrate_cloud_grasps
    from emet.controller.task.tamp.transfer_grounding import (
        propose_transfer_cloud_grasps,
        select_transfer_from_hints,
    )
    from emet.perception.grasps.observed import GroundingHint

    depth = np.ones((100, 100))
    depth[:, 55:] = 1.1  # Recessed destination, measured below counter.
    depth[30:50, 15:27] = .92
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 1.8
    observation = SimpleNamespace(rgb=np.zeros((100, 100, 3), np.uint8), depth=depth,
        camera_K=np.array([[500., 0., 50.], [0., 500., 50.], [0., 0., 1.]]), camera_pose=pose)
    sample = SimpleNamespace(capture=SimpleNamespace(observation_id='capture', earliest=12.),
                             observations={'camera': observation})
    masks = [np.zeros(depth.shape, bool) for _ in range(3)]
    masks[0][30:50, 15:27] = True
    masks[1][:, :50] = True
    masks[2][:, 55:] = True
    roles = ('object', 'source', 'destination')
    hints = {role: GroundingHint('capture', 'camera', (i*100, 0, i*100+100, 1000), role)
             for i, role in enumerate(roles)}
    # Deterministic segmentation fixture; no claim of learned grounding accuracy.
    selection = select_transfer_from_hints(sample, hints,
        refine=lambda _rgb, box: masks[int(box[0]/100)])
    geometry, proposals = propose_transfer_cloud_grasps(sample, selection,
        provider=TopDownCloudProvider(.06), position_error_m=.003)
    assert proposals
    assert geometry.acquired_monotonic == 12.
    assert np.allclose(geometry.points_world[:, 2], .88)
    profiles = calibrate_cloud_grasps(proposals, observation_id='capture',
        world_from_reference=np.eye(4), hand_from_ee=np.eye(4), close_joints=(('finger', .01),))
    assert len(profiles) == len(proposals)
    assert np.allclose(profiles[0].object_from_ee, proposals[0].world_from_hand)

    from emet.controller.manipulation.native_inputs import ObservationUnavailable
    from emet.controller.task.tamp.transfer_grounding import select_transfer

    profile = {'grounding': {
        'object': {'camera': 'camera', 'box': [150, 300, 270, 340]},
        'source': {'camera': 'camera', 'box': [0, 0, 500, 1000]},
        'destination': {'camera': 'camera', 'box': [550, 0, 1000, 1000]},
    }}
    with pytest.raises(ObservationUnavailable, match='transfer_grounding_unavailable'):
        select_transfer(sample, profile=profile)
    grown = select_transfer(sample, profile=profile, grow_object_region=True)
    assert grown.object_mask.sum() > 100
    assert not grown.object_mask[:, 55:].any()  # Destination must not join target.
    profile['grounding']['object']['box'] = [280, 300, 300, 340]
    snapped = select_transfer(sample, profile=profile, grow_object_region=True)
    # Local growth can produce different partial patches from shifted hints;
    # both must remain on the same target, without absorbing the sink.
    assert snapped.object_mask.sum() > 35
    assert (snapped.object_mask & grown.object_mask).sum() > 35
    assert not snapped.object_mask[:, 55:].any()
    assert np.allclose(depth[snapped.object_mask], .92)
    profile['grounding']['object']['box'] = [800, 0, 900, 100]
    with pytest.raises(ObservationUnavailable, match='transfer_grounding_unavailable'):
        select_transfer(sample, profile=profile, grow_object_region=True)


def test_no_cad_registration_reference_does_not_claim_complete_payload():
    from emet.controller.manipulation.native_inputs import ObjectPoseEvidence
    from emet.controller.manipulation.observed_collision import PayloadGeometryEvidence
    from emet.perception.grasps.observed import registration_surface

    geometry = cloud()
    model, reference = registration_surface(geometry)
    assert not model.complete_enclosure
    assert model.construction == 'observed_partial_surface'
    assert model.observation_ids == ('capture-1',)
    assert model.received_monotonic == 10.
    local = geometry.points_world-reference[:3, 3]
    assert np.allclose(model.validation_distance(local), 0.)
    pose = ObjectPoseEvidence(reference, 10., 'rgbd_tracking', 'capture-1')
    with pytest.raises(ValueError, match='invalid_payload_geometry_evidence'):
        PayloadGeometryEvidence('object', 'reference', model.model_id,
                                model.vertices_reference, pose, model.construction)


def test_approximate_payload_box_is_usable_but_explicitly_an_assumption():
    from emet.controller.manipulation.native_inputs import ObjectPoseEvidence
    from emet.controller.manipulation.observed_collision import PayloadGeometryEvidence
    from emet.perception.grasps.observed import registration_surface

    geometry = cloud()
    model, reference = registration_surface(geometry, box_padding_m=.01)
    assert not model.complete_enclosure
    assert model.construction == 'approximate_observed_box'
    local = geometry.points_world-reference[:3, 3]
    assert np.all(model.vertices_reference.min(0) <= local.min(0)-.013)
    pose = ObjectPoseEvidence(reference, 10., 'rgbd_tracking', 'capture-1')
    evidence = PayloadGeometryEvidence('object', 'reference', model.model_id,
                                       model.vertices_reference, pose, model.construction)
    assert evidence.construction == 'approximate_observed_box'
    assert np.allclose(model.validation_distance(local), 0.)
    with pytest.raises(ValueError, match='invalid_payload_box_padding'):
        registration_surface(geometry, box_padding_m=-.01)


def test_approximate_box_accepts_underconstrained_fit_but_not_stale_or_conflicting_positions():
    from dataclasses import replace

    from emet.controller.manipulation.geometry_tracking import fit_geometry
    from emet.controller.manipulation.moving_rgbd_tracking import MovingRgbdTracker
    from emet.controller.manipulation.native_inputs import ObservationUnavailable
    from emet.perception.grasps.observed import registration_surface

    geometry = cloud()
    model, reference = registration_surface(geometry, box_padding_m=.01)
    fit = fit_geometry(geometry.points_world, model, object_ref='object',
        observation_ids=('capture-1',), received_monotonic=10., initial_poses=[reference], max_starts=1)
    # Explicitly unresolved yaw/translation: the approximation must not require
    # numerical rigid-pose observability to execute coarse manipulation.
    fit = replace(fit, nullspaces=(np.array([[1., 0., 0., 0., 0., 1.]]),))
    MovingRgbdTracker._require_fit(fit)
    assert fit.require_approximate_box(now=10.1).observation_id == 'capture-1'
    with pytest.raises(ObservationUnavailable, match='stale'):
        fit.require_approximate_box(now=13.)
    poses = np.repeat(fit.poses[:1], 2, axis=0)
    poses[1, 0, 3] += .03
    conflict = replace(fit, poses=poses, rms_m=np.repeat(fit.rms_m[:1], 2))
    with pytest.raises(ObservationUnavailable, match='ambiguous'):
        conflict.require_approximate_box(now=10.1)

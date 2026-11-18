# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.rigid_rgbd_tracking import RigidRgbdTracker, fit_rigid_landmarks
from emet.motion.placement_geometry import box_corners


def test_rigid_landmark_fit_measures_rotation_translation_and_rejects_outliers():
    rng = np.random.default_rng(10)
    reference = rng.uniform(-.05, .05, (60, 3))
    rotation = Rotation.from_euler('xyz', [.3, -.2, .4]).as_matrix()
    translation = np.array([.1, -.2, .3])
    current = reference @ rotation.T + translation
    current[:6] += .2
    pose, residual, inliers = fit_rigid_landmarks(reference, current)
    np.testing.assert_allclose(pose[:3, :3], rotation, atol=1e-12)
    np.testing.assert_allclose(pose[:3, 3], translation, atol=1e-12)
    assert residual < 1e-12 and inliers.sum() == 54


@pytest.mark.parametrize('kind', ['line', 'tiny_patch', 'ambiguous_correspondences'])
def test_unobservable_or_inconsistent_landmarks_are_unknown(kind):
    rng = np.random.default_rng(4)
    points = rng.uniform(-.05, .05, (40, 3))
    current = points.copy()
    if kind == 'line':
        points[:, 1:] = 0
        current = points.copy()
    elif kind == 'tiny_patch':
        points *= .02
        current = points.copy()
    else:
        current = current[rng.permutation(len(current))]
    with pytest.raises(ObservationUnavailable):
        fit_rigid_landmarks(points, current)


def image_fixture():
    rng = np.random.default_rng(3)
    rgb = np.repeat(rng.integers(0, 256, (96, 96, 1), dtype=np.uint8), 3, axis=2)
    y, x = np.indices((96, 96))
    world = np.stack(((x-48)*.001, (y-48)*.001, np.ones_like(x)*.5), axis=-1)
    mask = np.zeros((96, 96), bool)
    mask[16:80, 16:80] = True
    pose = np.eye(4)
    pose[2, 3] = .5
    tracker = RigidRgbdTracker(object_ref='opaque', reference_id='reference', geometry_revision='prior',
        rgb=rgb, world_xyz=world, mask=mask, world_from_reference=pose,
        vertices_reference=box_corners([[-.06, -.06, -.03], [.06, .06, .03]]), received_monotonic=10.,
        position_error_m=.002, rotation_error_rad=.01)
    return tracker, rgb, world, mask


def test_rgbd_tracker_is_real_payload_provider_and_failed_update_clears_it():
    tracker, rgb, world, mask = image_fixture()
    moved = world + [.02, -.01, .1]
    track = tracker.update(rgb=rgb, world_xyz=moved, mask=mask, capture_id='frame:2', received_monotonic=10.5, now=10.6)
    np.testing.assert_allclose(track.payload.pose.world_from_object[:3, 3], [.02, -.01, .6], atol=.0001)
    assert track.inlier_count >= 12 and track.position_error_m > 0
    assert tracker.observe_payload('opaque') is track.payload
    with pytest.raises(ObservationUnavailable):
        tracker.update(rgb=rgb, world_xyz=moved, mask=np.zeros_like(mask), capture_id='frame:3',
                       received_monotonic=10.7, now=10.8)
    assert tracker.observe_payload('opaque') is None


def test_replayed_capture_cannot_renew_tracking_evidence():
    tracker, rgb, world, mask = image_fixture()
    tracker.update(rgb=rgb, world_xyz=world, mask=mask, capture_id='frame:2', received_monotonic=10.5, now=10.6)
    with pytest.raises(ObservationUnavailable):
        tracker.update(rgb=rgb, world_xyz=world, mask=mask, capture_id='frame:2', received_monotonic=10.7, now=10.8)
    assert tracker.observe_payload('opaque') is None


def test_actual_pixel_translation_tracks_without_reseeding_reference():
    import cv2

    tracker, rgb, world, mask = image_fixture()
    shifted = cv2.warpAffine(rgb, np.array([[1., 0., 5.], [0., 1., 3.]]), (96, 96))
    shifted_mask = cv2.warpAffine(mask.astype(np.uint8), np.array([[1., 0., 5.], [0., 1., 3.]]), (96, 96)).astype(bool)
    original = tracker.reference.copy()
    track = tracker.update(rgb=shifted, world_xyz=world, mask=shifted_mask,
                           capture_id='translated', received_monotonic=10.5, now=10.6)
    np.testing.assert_allclose(track.payload.pose.world_from_object[:3, 3], [.005, .003, .5], atol=.001)
    np.testing.assert_array_equal(original, tracker.reference)


def test_textureless_and_repetitive_surfaces_cannot_invent_unique_yaw():
    _, rgb, world, mask = image_fixture()
    y, x = np.indices(mask.shape)
    patterns = [np.full_like(rgb, 128), np.repeat((((x//8 + y//8) % 2)*255).astype(np.uint8)[..., None], 3, axis=2)]
    for pattern in patterns:
        with pytest.raises(ObservationUnavailable):
            RigidRgbdTracker(object_ref='opaque', reference_id='reference', geometry_revision='prior',
                rgb=pattern, world_xyz=world, mask=mask, world_from_reference=np.eye(4),
                vertices_reference=box_corners([[-1]*3, [1]*3]), received_monotonic=10.,
                position_error_m=.002, rotation_error_rad=.01)

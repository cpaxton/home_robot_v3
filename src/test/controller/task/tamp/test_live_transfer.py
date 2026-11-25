"""Tracking view choice is independent of the semantic initialization camera."""

import pytest

from emet.controller.task.tamp.live_transfer import _tracking_cameras


def test_fixed_scene_camera_can_track_wrist_initialized_object():
    profile = {"grounding": {"object": {"camera": "wrist"}}, "tracking_reference_only": True}
    assert _tracking_cameras(profile, ("wrist", "scene")) == ("wrist",)
    profile["tracking_views"] = ["scene"]
    assert _tracking_cameras(profile, ("wrist", "scene")) == ("scene",)
    assert _tracking_cameras({}, ("wrist", "scene")) == ("wrist", "scene")


@pytest.mark.parametrize("requested", [[], "scene", ["missing"], ["scene", "scene"], [None], [{}]])
def test_invalid_tracking_view_selection_fails_explicitly(requested):
    with pytest.raises(ValueError, match="invalid_tracking_views"):
        _tracking_cameras({"tracking_views": requested}, ("scene", "wrist"))


def test_thin_visible_fragment_survives_mask_erosion_for_geometric_validation():
    import numpy as np

    from emet.controller.task.tamp.live_transfer import _tracking_interior

    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 20] = True
    np.testing.assert_array_equal(_tracking_interior(mask), mask)
    mask[5:35, 5:35] = True
    result = _tracking_interior(mask)
    assert result.sum() == 28 * 28
    assert not result[5].any()


def test_explicit_best_fit_diagnostic_retains_candidates_and_does_not_renew_age():
    import numpy as np

    from emet.controller.manipulation.geometry_models import PrimitiveModel
    from emet.controller.manipulation.geometry_tracking import GeometryFit
    from emet.controller.manipulation.native_inputs import ObservationUnavailable
    from emet.controller.task.tamp.live_transfer import LiveTransferScene

    scene = LiveTransferScene.__new__(LiveTransferScene)
    scene.profile = {'position_error_m': .005, 'rotation_error_rad': .03, 'calibration_id': 'test'}
    scene.enclosure = PrimitiveModel('cylinder', (.03, .052))
    scene.clock = lambda: 10.1
    scene.diagnostics = {}
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = .02
    fit = GeometryFit(PrimitiveModel('cylinder', (.02, .04)), 'target', poses,
                      np.array([.0002, .001]), (np.empty((0, 6)),) * 2, ('capture',), 10., 1, False)
    with pytest.raises(ObservationUnavailable, match='object_pose_ambiguous'):
        scene._pose(fit)
    scene.profile['simulation_best_fit_pose'] = True
    selected = scene._pose(fit)
    np.testing.assert_allclose(selected.world_from_object, np.eye(4))
    assert scene.diagnostics['best_fit_pose_selection']['hypotheses'] == 2
    assert selected.received_monotonic == 10.
    scene.clock = lambda: 13.
    with pytest.raises(ObservationUnavailable, match='object_observation_stale'):
        scene._pose(fit)


def test_multiview_bootstrap_adds_only_local_matching_above_support_returns():
    from types import SimpleNamespace

    import numpy as np

    from emet.controller.manipulation.native_inputs import ObservationUnavailable
    from emet.controller.task.tamp.live_transfer import _bootstrap_reference_masks

    y, x = np.indices((12, 12))
    world = np.stack((x*.002, y*.002, np.full(x.shape, .04)), axis=-1)
    mask = np.zeros(x.shape, bool)
    mask[2:10, 2:10] = True
    appearance = np.ones(x.shape, bool)
    def make(points, capture='same'):
        return SimpleNamespace(world=points, depth=np.ones(x.shape), rgb=appearance,
            capture=capture, robot_guard=None)
    extra = world.copy()
    extra[:3, :, 2] = 0.  # Measured support, not object.
    extra[-2:, :, 0] += 1.  # Same appearance but a different distant object.
    views = {'front': make(world), 'other': make(extra)}
    masks = _bootstrap_reference_masks(views, 'front', mask, lambda rgb: rgb, np.zeros(3))
    assert 'other' in masks
    assert not masks['other'][:3].any()
    assert not masks['other'][-2:].any()
    assert np.array_equal(masks['front'], mask)
    views['other'] = make(extra, capture='different')
    with pytest.raises(ObservationUnavailable, match='capture_identity_mismatch'):
        _bootstrap_reference_masks(views, 'front', mask, lambda rgb: rgb, np.zeros(3))


@pytest.mark.parametrize('approximate', [False, True])
def test_object_association_retains_other_view_measurements_without_erasing_unassociated_obstacles(approximate):
    from types import SimpleNamespace

    import numpy as np

    from emet.controller.task.tamp.live_transfer import LiveTransferScene
    from emet.motion.placement_geometry import box_corners

    scene = LiveTransferScene.__new__(LiveTransferScene)
    scene.payload_model = SimpleNamespace(construction='approximate_observed_box' if approximate else 'known_model_prior',
        vertices_reference=box_corners([[-.02]*3, [.02]*3]))
    world = np.zeros((3, 3, 3))
    world[:, 2, 0] = .2
    foreground = np.ones((3, 3), bool)
    foreground[1, 1] = False  # Unassociated obstacle even inside the proxy box.
    front = np.eye(3, dtype=bool)
    tracked = SimpleNamespace(fit=SimpleNamespace(masks=(('front', front),),
        fit=SimpleNamespace(poses=np.eye(4)[None])),
        views=(('wrist', SimpleNamespace(world=world, foreground_mask=foreground)),))
    masks = scene._associated_object_masks(tracked)
    np.testing.assert_array_equal(masks['front'], front)
    if approximate:
        assert masks['wrist'][:, :2].sum() == 5
        assert not masks['wrist'][1, 1] and not masks['wrist'][:, 2].any()
    else:
        assert 'wrist' not in masks
    assert foreground[:, 2].all()  # Original association input was not modified.


def test_mapping_keeps_associated_object_edges_omitted_from_pose_fit():
    from types import SimpleNamespace

    import numpy as np

    from emet.controller.task.tamp.live_transfer import LiveTransferScene
    from emet.motion.placement_geometry import box_corners

    scene = LiveTransferScene.__new__(LiveTransferScene)
    scene.payload_model = SimpleNamespace(construction='approximate_observed_box',
        vertices_reference=box_corners([[-.02]*3, [.02]*3]))
    interior = np.zeros((8, 8), bool)
    interior[2:6, 2:6] = True
    associated = np.zeros_like(interior)
    associated[1:7, 1:7] = True
    tracked = SimpleNamespace(fit=SimpleNamespace(masks=(('scene', interior),),
        fit=SimpleNamespace(poses=np.eye(4)[None])), views=(('scene', SimpleNamespace(
            world=np.zeros((8, 8, 3)), foreground_mask=interior, associated_foreground_mask=associated)),))
    masks = scene._associated_object_masks(tracked)
    np.testing.assert_array_equal(masks['scene'], associated)
    assert not masks['scene'][0].any()  # No growth into unassociated environment.
    assert interior.sum() == 16

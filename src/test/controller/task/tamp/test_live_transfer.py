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

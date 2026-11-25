from types import SimpleNamespace

import numpy as np
import pytest

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.controller.task.tamp.grounded_grasps import grasp_proposals_from_decision


def observed_scene():
    depth = np.ones((40, 40))
    depth[12:28, 12:28] = .8
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 1.
    observation = SimpleNamespace(rgb=np.zeros((40, 40, 3), np.uint8), depth=depth,
        camera_K=np.array([[200., 0., 20.], [0., 200., 20.], [0., 0., 1.]]), camera_pose=pose)
    sample = SimpleNamespace(capture=CaptureInterval('fresh', 10., 10.001, 10.01, 'clock'),
        observations={'front': observation})
    selection = SimpleNamespace(object_camera='front', object_mask=depth < 1.)
    return sample, selection


def test_semantic_grasp_site_is_realized_at_observed_interior_height_with_two_hand_headings():
    sample, selected = observed_scene()
    decision = {'grasp_point': [500, 500], 'grasp_yaw_degrees': 0., 'grasp_height_fraction': .5}
    proposals = grasp_proposals_from_decision(sample, selected, np.zeros(3), decision, position_error_m=.003)
    assert len(proposals) == 2
    first, second = proposals
    assert first.observation_id == second.observation_id == 'fresh'
    np.testing.assert_allclose(first.world_from_hand[:3, 3], [0., 0., .1])
    np.testing.assert_allclose(first.world_from_hand[:3, 0], [1., 0., 0.])
    np.testing.assert_allclose(first.world_from_hand[:3, 2], [0., 0., -1.])
    np.testing.assert_allclose(first.world_from_hand[:3, 0], -second.world_from_hand[:3, 0])
    assert .01 < first.opening_m < .078
    assert first.provenance == 'vlm_site_observed_depth'


@pytest.mark.parametrize('change', [{'grasp_point': [0, 0]}, {'grasp_height_fraction': 2.},
    {'grasp_yaw_degrees': float('nan')}])
def test_semantic_grasp_rejects_unbound_site_or_invalid_estimate(change):
    sample, selected = observed_scene()
    decision = {'grasp_point': [500, 500], 'grasp_yaw_degrees': 0., 'grasp_height_fraction': .5, **change}
    with pytest.raises(ObservationUnavailable, match='invalid_manipulation_hint'):
        grasp_proposals_from_decision(sample, selected, np.zeros(3), decision, position_error_m=.003)


def test_region_only_grounding_does_not_invent_a_semantic_grasp():
    sample, selected = observed_scene()
    assert grasp_proposals_from_decision(sample, selected, np.zeros(3), {}, position_error_m=.003) == ()


def test_live_grasp_anchor_rebinds_to_current_frame_and_rejects_another_scene():
    from emet.controller.task.tamp.live_transfer import LiveTransferScene

    scene = LiveTransferScene.__new__(LiveTransferScene)
    scene.epoch, scene.clock = 'scene-a', lambda: 10.1
    local = np.eye(4)
    local[:3, 3] = [.01, .02, .03]
    scene._grasp_anchors = ((local, .04, 'vlm_site_observed_depth'),)
    current = np.eye(4)
    current[:3, 3] = [1., 2., 3.]
    frame = SimpleNamespace(require=lambda **kwargs: None,
        capture=SimpleNamespace(observation_id='new-depth'),
        payload=SimpleNamespace(pose=SimpleNamespace(world_from_object=current)))
    snapshot = SimpleNamespace(frame=frame,
        admission=SimpleNamespace(background=SimpleNamespace(scene_epoch='scene-a')))
    (proposal,) = scene.grasp_proposals(snapshot)
    assert proposal.observation_id == 'new-depth'
    np.testing.assert_allclose(proposal.world_from_hand[:3, 3], [1.01, 2.02, 3.03])
    snapshot.admission.background.scene_epoch = 'other-scene'
    with pytest.raises(ObservationUnavailable, match='scene_identity_changed'):
        scene.grasp_proposals(snapshot)

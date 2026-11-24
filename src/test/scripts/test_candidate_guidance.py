import numpy as np
import pytest

from emet.controller.task.tamp.candidate_guidance import guide_approach
from emet.controller.task.tamp.observed_arm import TransferCandidate


def candidate():
    rotation = np.eye(3)
    return TransferCandidate(((np.array([0., 0., .2]), rotation),
        (np.array([0., 0., .1]), rotation), (np.array([0., 0., .3]), rotation)),
        ((np.zeros(3), rotation),)*3, (('finger', 0.),))


def test_approach_changes_only_pregrasp_before_normal_certification():
    original = candidate()
    result, = guide_approach((original,), {'direction_world': [2, 0, 0],
        'distance_m': .08, 'frame_id': 'world', 'observation_id': 'capture:1'}, observation_id='capture:1')
    np.testing.assert_allclose(result.grasp[0][0], [-.08, 0, .1])
    assert result.grasp[1] is original.grasp[1]
    assert result.place is original.place
    np.testing.assert_allclose(original.grasp[0][0], [0, 0, .2])


@pytest.mark.parametrize('change,code', [
    ({'observation_id': 'old'}, 'stale_manipulation_hint'),
    ({'frame_id': 'camera'}, 'invalid_manipulation_hint'),
    ({'direction_world': [0, 0, 0]}, 'invalid_manipulation_hint'),
    ({'distance_m': -.1}, 'invalid_manipulation_hint'),
    ({'disable_collisions': True}, 'manipulation_options_unsupported'),
])
def test_bad_guidance_is_not_a_motion_or_authorization_override(change, code):
    hint = {'direction_world': [0, 0, -1], 'frame_id': 'world', 'observation_id': 'capture:1', **change}
    with pytest.raises(ValueError, match=code):
        guide_approach((candidate(),), hint, observation_id='capture:1')


def test_distance_only_preserves_native_approach_axis():
    result, = guide_approach((candidate(),), {'distance_m': .07,
        'frame_id': 'world', 'observation_id': 'capture:1'}, observation_id='capture:1')
    np.testing.assert_allclose(result.grasp[0][0], [0, 0, .17])


def test_strict_approach_does_not_claim_straight_line_guarantee():
    from emet.controller.task.tamp.manipulation_request import ManipulationRequest
    from emet.controller.task.tamp.observed_adapter import ObservedNativeAdapter

    adapter = object.__new__(ObservedNativeAdapter)
    request = ManipulationRequest.parse('pear', 'sink', hint_mode='strict',
        hints={'approach': {'direction_world': [0, 0, -1]}})
    result = adapter.plan_request(request)
    assert result['code'] == 'strict_approach_unavailable'
    assert result['data']['execution_started'] is False

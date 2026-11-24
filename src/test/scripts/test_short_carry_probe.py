# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
import numpy as np
import pytest

from emet.eval.short_carry_probe import short_carry_goal


def test_short_carry_preserves_height_and_uses_camera_direction():
    start = np.array([.4, -.3, 1.1])
    camera = np.eye(4)
    camera[:3, 3] = [1., .5, 2.]
    goal = short_carry_goal(start, camera)
    assert goal[2] == start[2]
    assert np.linalg.norm(goal-start) == pytest.approx(.1)
    assert np.dot(goal[:2]-start[:2], camera[:2, 3]-start[:2]) > 0
    # Translation of the calibrated coordinate frame translates the result.
    offset = np.array([2., -1., .5])
    camera[:3, 3] += offset
    np.testing.assert_allclose(short_carry_goal(start+offset, camera), goal+offset)


def test_overhead_camera_has_a_direction_and_bad_input_fails():
    camera = np.eye(4)
    np.testing.assert_allclose(short_carry_goal([0., 0., 1.], camera), [0., .1, 1.])
    with pytest.raises(ValueError, match='invalid_short_carry_input'):
        short_carry_goal([0., 0., float('nan')], camera)
    with pytest.raises(ValueError, match='invalid_short_carry_input'):
        short_carry_goal([0., 0., 1.], camera, distance_m=-.1)


@pytest.mark.parametrize(('distance', 'retained', 'pads', 'passed'), [
    (.1, True, 2, True), (.04, True, 2, False), (.1, False, 2, False), (.1, True, 0, False),
])
def test_carry_evidence_requires_motion_and_terminal_hold(distance, retained, pads, passed):
    from emet.eval.short_carry_score import short_carry_evidence

    samples = [{'xyz_m': [0., 0., 1.], 'pad_contacts': 0},
               {'xyz_m': [distance, 0., 1.1], 'pad_contacts': pads}]
    assert short_carry_evidence(samples, retained)['short_carry_passed'] is passed
    assert not short_carry_evidence([], True)['short_carry_passed']


@pytest.mark.parametrize('failure', [None, 'no_event', 'lost_hold', 'no_motion'])
def test_carry_task_cannot_pass_from_lift_alone(failure):
    from emet.eval.panda_open_loop import score_diagnostic

    diagnostic = {'completed': True, 'events': ['lift', 'short_carry'], 'short_carry_requested': True}
    physics = {'lift_with_two_pads_m': .08, 'retained_pickup': True, 'short_carry_passed': True,
        'upstream_success': False, 'basin_contains_tongs': False, 'final_object_speed_mps': 0.,
        'unintended_contact_frames': 0, 'unintended_touch_frames': 0, 'unintended_self_contact_frames': 0}
    if failure == 'no_event':
        diagnostic['events'] = ['lift']
    elif failure == 'lost_hold':
        physics['retained_pickup'] = False
    elif failure == 'no_motion':
        physics['short_carry_passed'] = False
    score = score_diagnostic(diagnostic, physics, full_transfer=False)
    assert score['task_passed'] is (failure is None)
    assert score['carry_passed'] is (failure is None)

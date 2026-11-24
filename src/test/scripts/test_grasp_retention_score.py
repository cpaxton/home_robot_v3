# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
import pytest

from emet.eval.grasp_retention_score import pickup_retention


@pytest.mark.parametrize('failure', [None, 'dropped', 'on_counter', 'one_pad', 'too_short'])
def test_peak_lift_is_not_retained_pickup(failure):
    samples = [{'sim_s': i*.05, 'xyz_m': [0, 0, 1.08], 'pad_contacts': 2} for i in range(21)]
    if failure == 'dropped':
        samples[-1].update(xyz_m=[0, 0, 1.], pad_contacts=0)
    elif failure == 'on_counter':
        for sample in samples:
            sample['xyz_m'][2] = 1.01
    elif failure == 'one_pad':
        for sample in samples:
            sample['pad_contacts'] = 1
    elif failure == 'too_short':
        samples = samples[-2:]
    assert pickup_retention(samples, 1.)['retained_pickup'] is (failure is None)
    assert pickup_retention([], 1.)['retained_pickup'] is False


def test_completed_motion_with_peak_lift_and_dropped_object_fails_pickup():
    from emet.eval.panda_open_loop import score_diagnostic

    physics = {'lift_with_two_pads_m': .08, 'retained_pickup': False,
        'upstream_success': False, 'basin_contains_tongs': False, 'final_object_speed_mps': 0.,
        'unintended_contact_frames': 0, 'unintended_touch_frames': 0, 'unintended_self_contact_frames': 0}
    score = score_diagnostic({'completed': True, 'events': ['lift']}, physics, full_transfer=False)
    assert score['physical_pick_passed']
    assert not score['retained_pick_passed'] and not score['task_passed']
    score = score_diagnostic({'completed': True, 'events': ['lift']},
        {**physics, 'retained_pickup': True}, full_transfer=False)
    assert score['task_passed'] and score['retained_pick_passed']

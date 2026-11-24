# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
import io
import json

import numpy as np
import pytest

from emet.eval.vlm_grasp_check import clearly_missed_grasp, merged_retry_receipt, query_grasp_state


@pytest.mark.parametrize(('state', 'positions', 'missed'), [
    ('not_held', {'left': .0002, 'right': -.0003}, True),
    ('held', {'left': .0002, 'right': -.0003}, False),
    ('unknown', {'left': .0002, 'right': -.0003}, False),
    ('not_held', {'left': .02, 'right': -.03}, False),
    ('not_held', {}, False),
    ('not_held', {'left': float('nan'), 'right': 0.}, False),
])
def test_miss_requires_visual_and_native_finger_agreement(state, positions, missed):
    assert clearly_missed_grasp({'state': state}, positions) is missed


def test_current_multiview_check_preserves_images_and_unknown(monkeypatch, tmp_path):
    requests = []
    decision = {'state': 'unknown', 'reason': 'object occluded by hand'}

    def reply(request, timeout):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': json.dumps(decision)}}]}).encode())

    monkeypatch.setattr('urllib.request.urlopen', reply)
    result = query_grasp_state({'scene': np.zeros((20, 30, 3), np.uint8),
        'wrist': np.full((20, 30, 3), 120, np.uint8)}, endpoint='http://test/v1',
        model='test', object_name='pear', finger_positions={'left': .02, 'right': -.03}, output=tmp_path)
    assert result['state'] == 'unknown'
    assert len([item for item in requests[0]['messages'][0]['content'] if item['type'] == 'image_url']) == 2
    assert (tmp_path/'scene.png').exists() and (tmp_path/'wrist.png').exists()
    assert json.loads((tmp_path/'finger-feedback.json').read_text())['left'] == .02
    decision['state'] = 'success'
    with pytest.raises(ValueError, match='invalid_vlm_grasp_state'):
        query_grasp_state({'scene': np.zeros((20, 30, 3), np.uint8)}, endpoint='http://test/v1',
            model='test', object_name='pear', finger_positions={}, output=tmp_path)


def test_retry_receipt_preserves_miss_and_all_model_calls():
    previous = {'completed': False, 'reason': 'observed_grasp_miss_after_lift',
                'runtime_vlm_calls': 3, 'grounding_attempts': [{'attempt': 1}, {'attempt': 2}],
                'grasp_checks': [{'clear_miss': True}]}
    retry = {'completed': True, 'runtime_vlm_calls': 3,
             'grounding_attempts': [{'attempt': 1}], 'events': ['lift', 'place', 'release', 'retreat']}
    merged = merged_retry_receipt(previous, retry)
    assert merged['execution_attempts'] == 2 and merged['runtime_vlm_calls'] == 6
    assert len(merged['grounding_attempts']) == 3
    assert merged['prior_execution_attempts'][0] == previous
    assert merged['completed'] is True
    assert 'execution_attempts' not in retry


@pytest.mark.parametrize('lift_m', [.0, .02, .075])
def test_failed_retry_does_not_erase_prior_physical_pick_or_claim_transfer(lift_m):
    from emet.eval.panda_open_loop import score_diagnostic

    diagnostic = {'completed': False, 'events': ['departure'],
        'prior_execution_attempts': [{'completed': False, 'events': ['grasp', 'lift']}]}
    physics = {'lift_with_two_pads_m': lift_m, 'upstream_success': False,
        'basin_contains_tongs': False, 'final_object_speed_mps': .001,
        'unintended_contact_frames': 0, 'unintended_touch_frames': 0,
        'unintended_self_contact_frames': 0}
    score = score_diagnostic(diagnostic, physics, full_transfer=True)
    assert score['physical_pick_passed'] is (lift_m >= .04)
    assert not score['task_passed'] and not score['physical_transfer_passed']

# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import io
import json

import numpy as np
import pytest

from emet.eval.vlm_pick_hint import query_pick_hint


def test_runtime_vlm_image_request_and_validated_decision(monkeypatch, tmp_path):
    captured = []
    decision = {'object': [100, 100, 300, 400], 'source': [50, 400, 400, 800],
                'destination': [500, 300, 900, 900], 'grasp_yaw_degrees': 90}

    def reply(request, timeout):
        captured.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': json.dumps(decision)}}]}).encode())

    monkeypatch.setattr('urllib.request.urlopen', reply)
    result = query_pick_hint(np.zeros((20, 30, 3), np.uint8), endpoint='http://test/v1',
                             model='test-model', object_name='shaker', output=tmp_path, destination_name='front-right sink interior')
    assert result == decision
    assert 'front-right sink interior' in captured[0]['messages'][0]['content'][0]['text']
    assert captured[0]['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
    assert (tmp_path/'vlm-input.png').exists()
    assert (tmp_path/'vlm-decision.json').exists()
    decision['object'] = [-1, 0, 100, 100]
    with pytest.raises(ValueError, match='invalid_grounding_hint'):
        query_pick_hint(np.zeros((20, 30, 3), np.uint8), endpoint='http://test/v1',
                       model='test-model', object_name='shaker', output=tmp_path)


def test_top_down_target_avoids_camera_facing_surface_bias():
    from emet.eval.panda_open_loop import top_down_target

    side = np.column_stack((np.zeros(100), np.full(100, -.02), np.linspace(.9, .98, 100)))
    top = np.column_stack((np.linspace(-.01, .01, 21), np.zeros(21), np.ones(21)))
    points = np.concatenate((side, top))
    assert np.median(points[:, 1]) == -.02
    target = top_down_target(points)
    assert np.allclose(target[:2], [0., 0.])
    assert target[2] == pytest.approx(np.median(points[:, 2])+.015)


def test_front_floor_uses_measured_interior_away_from_back_edge():
    from types import SimpleNamespace

    from emet.eval.panda_open_loop import front_interior_floor

    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 1.8
    obs = SimpleNamespace(rgb=np.zeros((100, 100, 3), np.uint8), depth=np.full((100, 100), 1.1),
        camera_K=np.array([[500., 0., 50.], [0., 500., 50.], [0., 0., 1.]]), camera_pose=pose)
    mask = np.zeros((100, 100), bool)
    mask[30:90, 20:80] = True
    point = front_interior_floor(obs, mask)
    assert abs(point[0]) < .005
    assert point[1] < -.05
    assert point[2] == pytest.approx(.7)
    rear = np.zeros_like(mask)
    rear[30:50, 20:80] = True
    partial = front_interior_floor(obs, rear)
    fused = front_interior_floor(obs, rear, extra_observations=[obs])
    assert fused[1] < partial[1]-.04


@pytest.mark.parametrize('failure', [None, 'unfinished', 'no_lift', 'official_failure', 'outside', 'moving'])
def test_contacts_do_not_override_task_success_but_completion_still_matters(failure):
    from emet.eval.panda_open_loop import score_diagnostic

    diagnostic = {'completed': True, 'events': ['lift', 'place', 'release', 'retreat']}
    physics = {'lift_with_two_pads_m': .09, 'upstream_success': True, 'basin_contains_tongs': True,
               'final_object_speed_mps': .001, 'unintended_contact_frames': 0,
               'unintended_touch_frames': 355, 'unintended_self_contact_frames': 0}
    if failure == 'unfinished':
        diagnostic['completed'] = False
    elif failure == 'no_lift':
        physics['lift_with_two_pads_m'] = 0.
    elif failure == 'official_failure':
        physics['upstream_success'] = False
    elif failure == 'outside':
        physics['basin_contains_tongs'] = False
    elif failure == 'moving':
        physics['final_object_speed_mps'] = .1
    score = score_diagnostic(diagnostic, physics, full_transfer=True)
    assert score['task_passed'] is (failure is None)
    assert not score['zero_contact']
    assert not score['clean_transfer_passed']

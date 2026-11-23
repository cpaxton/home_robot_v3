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


def test_failed_grounding_can_adapt_twice_without_actuating(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from emet.controller.manipulation.native_inputs import ObservationUnavailable
    from emet.eval.panda_open_loop import run_pick

    rgb = np.zeros((20, 30, 3), np.uint8)
    observation = SimpleNamespace(rgb=rgb, auxiliary_rgbd_views={'scene_alt': {'rgb': rgb}},
        ee_rgb=rgb, ee_depth=np.ones((20, 30)), ee_camera_K=np.eye(3), ee_camera_pose=np.eye(4))
    calls = []

    def query(_image, **kwargs):
        calls.append(kwargs)
        return {role: [0, 0, 1000, 1000] for role in ('object', 'source', 'destination')}

    def reject(*args, **kwargs):
        raise ObservationUnavailable('support_geometry_unknown')

    class Client:
        def get_emet_session(self):
            return {'runtime_kind': 'robocasa_native'}

        def get_observation(self):
            return observation

        def open_gripper(self):
            raise AssertionError('must not actuate with invalid grounding')

    monkeypatch.setattr('emet.eval.vlm_pick_hint.query_pick_hint', query)
    monkeypatch.setattr('emet.eval.panda_open_loop.select_transfer', reject)
    profile = tmp_path/'profile.json'
    profile.write_text('{}')
    output = tmp_path/'run'
    output.mkdir()
    run_pick(Client(), output, profile, model_endpoint='http://test/v1', model_name='test', object_name='pear')
    receipt = json.loads((output/'open-loop-pick.json').read_text())
    assert len(calls) == 3
    assert calls[0]['feedback'] is None
    assert calls[1]['feedback'] == 'support_geometry_unknown'
    assert all(call['object_name'] == 'pear' for call in calls)
    assert receipt['completed'] is False and receipt['events'] == []
    assert len(receipt['grounding_attempts']) == 3


def test_vlm_can_choose_view_and_estimate_grasp_and_approach(monkeypatch, tmp_path):
    decision = {'camera': 'wrist', 'object': [100, 100, 300, 400],
                'source': [50, 400, 400, 800], 'destination': [500, 300, 900, 900],
                'grasp_yaw_degrees': 90, 'grasp_height_fraction': .3, 'approach': 'direct'}
    captured = []

    def reply(request, timeout):
        captured.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': json.dumps(decision)}}]}).encode())

    monkeypatch.setattr('urllib.request.urlopen', reply)
    rgb = np.zeros((20, 30, 3), np.uint8)
    result = query_pick_hint(rgb, endpoint='http://test/v1', model='test', object_name='pear',
                            output=tmp_path, views={'scene': rgb, 'wrist': rgb})
    assert result['camera'] == 'wrist' and result['approach'] == 'direct'
    content = captured[0]['messages'][0]['content']
    assert sum(item['type'] == 'image_url' for item in content) == 2
    decision['camera'] = 'missing-camera'
    with pytest.raises(ValueError, match='invalid_vlm_camera_choice'):
        query_pick_hint(rgb, endpoint='http://test/v1', model='test', object_name='pear',
                        output=tmp_path, views={'scene': rgb, 'wrist': rgb})


def test_vlm_height_selects_body_section_and_other_views_reduce_surface_bias(monkeypatch):
    from types import SimpleNamespace

    from emet.eval.panda_open_loop import estimated_grasp_section

    body = np.column_stack((np.tile([-.01, .01], 40), np.full(80, -.02), np.linspace(.94, .97, 80)))
    stem = np.column_stack((np.full(20, .02), np.full(20, -.01), np.ones(20)))
    world = np.zeros((12, 10, 3))
    world[:10] = np.concatenate((body, stem)).reshape(10, 10, 3)
    world[10:, :, 2] = .92
    other = world.copy()
    other[:8, :, 1] = .01
    obj, support = np.zeros((12, 10), bool), np.zeros((12, 10), bool)
    obj[:10], support[10:] = True, True
    selection = SimpleNamespace(object_camera='scene', source_camera='scene',
                                object_mask=obj, source_mask=support)
    monkeypatch.setattr('emet.eval.panda_open_loop.capture_arrays', lambda o: (None, None, o.world))
    views = {'scene': SimpleNamespace(world=world), 'other': SimpleNamespace(world=other)}
    _, body_target = estimated_grasp_section(views, selection, .3)
    _, stem_target = estimated_grasp_section(views, selection, 1.)
    assert abs(body_target[0]) < .005
    assert body_target[1] > -.01
    assert body_target[2] == pytest.approx(.958)
    assert stem_target[0] > .015


def test_vlm_supports_independent_surface_views(monkeypatch, tmp_path):
    decision = {'camera': 'wrist', 'source_camera': 'scene', 'destination_camera': 'scene_alt',
                'object': [100, 100, 300, 400], 'source': [50, 400, 400, 800],
                'destination': [500, 300, 900, 900], 'grasp_yaw_degrees': 90}

    def reply(request, timeout):
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': json.dumps(decision)}}]}).encode())

    monkeypatch.setattr('urllib.request.urlopen', reply)
    rgb = np.zeros((20, 30, 3), np.uint8)
    views = dict.fromkeys(('wrist', 'scene', 'scene_alt'), rgb)
    result = query_pick_hint(rgb, endpoint='http://test/v1', model='test', object_name='pear',
                             output=tmp_path, views=views)
    assert result['source_camera'] == 'scene'
    assert result['destination_camera'] == 'scene_alt'
    decision['source_camera'] = 'nonexistent'
    with pytest.raises(ValueError, match='invalid_vlm_camera_choice'):
        query_pick_hint(rgb, endpoint='http://test/v1', model='test', object_name='pear',
                         output=tmp_path, views=views)

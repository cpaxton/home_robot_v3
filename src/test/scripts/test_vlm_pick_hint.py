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


@pytest.mark.parametrize('failure', [None, 'unfinished', 'no_lift', 'official_failure', 'box_overhang', 'moving'])
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
    elif failure == 'box_overhang':
        physics['basin_contains_tongs'] = False
    elif failure == 'moving':
        physics['final_object_speed_mps'] = .1
    score = score_diagnostic(diagnostic, physics, full_transfer=True)
    assert score['task_passed'] is (failure in (None, 'box_overhang'))
    assert score['strict_box_transfer_passed'] is (failure is None)
    assert score['whole_box_contained'] is (failure != 'box_overhang')
    assert score['scoring_version'] == 2
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
    monkeypatch.setattr('emet.eval.vlm_pick_hint.refine_object_hint', lambda image, decision, **kwargs: decision)
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


def test_magnified_crop_coordinates_map_back_to_current_camera(monkeypatch, tmp_path):
    from emet.eval.vlm_pick_hint import refine_object_hint

    def reply(request, timeout):
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': '{"object":[250,250,750,750],"grasp_point":[500,500],"grasp_yaw_degrees":45,"grasp_height_fraction":0.3}'}}]}).encode())

    monkeypatch.setattr('urllib.request.urlopen', reply)
    original = {'camera': 'scene', 'object': [450, 450, 550, 550], 'grasp_yaw_degrees': 90}
    result = refine_object_hint(np.zeros((480, 640, 3), np.uint8), original,
        endpoint='http://test/v1', model='test', object_name='garlic', output=tmp_path, estimate_grasp_site=True)
    assert result['camera'] == 'scene'
    np.testing.assert_allclose(result['object'], [425, 425, 575, 575])
    assert result['coarse_object'] == original['object']
    np.testing.assert_allclose(result['grasp_point'], [500, 500])
    assert result['grasp_yaw_degrees'] == pytest.approx(45.)
    assert result['grasp_height_fraction'] == .3
    assert (tmp_path/'crop-grid.png').exists()
    wide = refine_object_hint(np.zeros((480, 640, 3), np.uint8),
        {**original, 'object': [100, 450, 900, 550]}, endpoint='http://test/v1',
        model='test', object_name='garlic', output=tmp_path/'wide', estimate_grasp_site=True)
    assert wide['grasp_yaw_degrees'] == pytest.approx(45.)
    from PIL import Image

    with Image.open(tmp_path/'wide'/'crop-grid.png') as image:
        assert image.width/image.height == pytest.approx(640/144)


def test_vlm_grasp_site_selects_local_solid_section_without_changing_payload_extent(monkeypatch):
    from types import SimpleNamespace

    from emet.eval.panda_open_loop import estimated_grasp_section

    y, x = np.indices((10, 40))
    world = np.stack((x*.004, y*.002, np.full(x.shape, .95)), axis=-1)
    world[8:, :, 2] = .9
    mask = np.zeros(x.shape, bool)
    mask[:8] = True
    selection = SimpleNamespace(object_camera='scene', source_camera='scene',
        object_mask=mask, source_mask=~mask)
    pose = np.diag([1., -1., -1., 1.])
    pose[:3, 3] = [0., .02, 1.95]
    views = {'scene': SimpleNamespace(world=world, depth=np.ones(x.shape),
        camera_K=np.diag([250., 500., 1.]), camera_pose=pose)}
    monkeypatch.setattr('emet.eval.panda_open_loop.capture_arrays', lambda obs: (None, None, obs.world))
    left_cloud, left = estimated_grasp_section(views, selection, .3, grasp_point=[125, 400])
    right_cloud, right = estimated_grasp_section(views, selection, .3, grasp_point=[875, 400])
    assert right[0]-left[0] > .1
    np.testing.assert_array_equal(left_cloud, right_cloud)
    assert len(left_cloud) == mask.sum()
    with pytest.raises(ValueError, match='grasp_site_not_on_observed_object'):
        estimated_grasp_section(views, selection, .3, grasp_point=[2000, 400])


def test_container_center_is_a_measured_floor_point_even_with_a_central_hole(monkeypatch):
    from types import SimpleNamespace

    from emet.eval.panda_open_loop import interior_floor_center

    y, x = np.indices((30, 40))
    world = np.stack((x*.005, y*.005, np.full(x.shape, .7)), axis=-1)
    mask = np.ones(x.shape, bool)
    mask[12:18, 17:23] = False
    monkeypatch.setattr('emet.eval.panda_open_loop.capture_arrays', lambda obs: (None, None, world))
    point = interior_floor_center(SimpleNamespace(), mask)
    assert np.any(np.all(world[mask] == point, axis=1))
    assert np.linalg.norm(point[:2]-[.0975, .0725]) < .03
    assert point[2] == .7


def test_destination_plane_growth_preserves_semantic_goal(monkeypatch):
    from types import SimpleNamespace

    import emet.eval.panda_open_loop as picker

    y, x = np.mgrid[:20, :20]
    points = np.stack((x*.01, y*.01, np.zeros_like(x)), -1)
    mask = np.ones((20, 20), bool)
    monkeypatch.setattr(picker, 'capture_arrays', lambda obs: (None, None, points))
    monkeypatch.setattr(picker, 'image_point_at_height', lambda obs, pixel, height: np.array([.03, .14, height]))
    point = picker.interior_floor_center(SimpleNamespace(), mask, image_point=[200, 700])
    np.testing.assert_allclose(point, [.03, .14, 0.])
    assert np.linalg.norm(point[:2]-[.095, .095]) > .07


def test_simple_grounding_omits_support_prediction_but_keeps_legacy_schema(monkeypatch, tmp_path):
    from emet.eval.vlm_pick_hint import grounding_profile

    captured = []
    decision = {'camera': 'wrist', 'object': [100, 100, 300, 400],
                'destination_camera': 'scene', 'destination': [500, 300, 900, 900],
                'grasp_yaw_degrees': 90}

    def reply(request, timeout):
        captured.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': json.dumps(decision)}}]}).encode())

    monkeypatch.setattr('urllib.request.urlopen', reply)
    rgb = np.zeros((20, 30, 3), np.uint8)
    result = query_pick_hint(rgb, endpoint='http://test/v1', model='test', object_name='pear',
        output=tmp_path, views={'scene': rgb, 'wrist': rgb}, infer_support_from_depth=True)
    prompt = captured[0]['messages'][0]['content'][0]['text']
    assert 'source_camera' not in prompt and 'counter BESIDE' not in prompt
    assert 'support from measured depth' in prompt
    assert result == decision and 'source' not in result
    profile = grounding_profile(result, infer_support_from_depth=True)
    assert profile['grounding']['source'] == profile['grounding']['object']
    assert profile['grounding']['source']['camera'] == 'wrist'
    assert profile['grounding']['destination']['camera'] == 'scene'
    assert profile['local_support_estimation']
    # Legacy callers still require an explicitly predicted support region.
    with pytest.raises(KeyError, match='source'):
        query_pick_hint(rgb, endpoint='http://test/v1', model='test', object_name='pear',
            output=tmp_path, views={'scene': rgb, 'wrist': rgb})

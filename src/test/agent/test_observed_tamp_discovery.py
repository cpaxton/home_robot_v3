# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Observed discovery through the real registry, with no simulator or actuation."""
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.agent.tamp_session import tamp_followup
from emet.agent.tools import get_tools
from emet.controller.manipulation.native_inputs import ObjectPoseEvidence
from emet.controller.task.tamp.observed_discovery import ObservedTaskCandidate, discover_observed_tasks


def candidate(*, source='rgbd_tracking', age=0, name='cup'):
    pose = ObjectPoseEvidence(np.eye(4), time.monotonic() - age, source, 'private-observation-id')
    return ObservedTaskCandidate(name, 'counter', pose, pose)


def provider(*candidates):
    return SimpleNamespace(input_mode='observed', observe_tasks=Mock(return_value=tuple(candidates)))


def test_real_registry_discovery_never_reads_truth_and_cannot_enable_execution():
    robot = Mock()
    robot.get_emet_session.side_effect = AssertionError('privileged metadata read')
    observed = provider(candidate())
    tools = {t.name: t for t in get_tools({'robot': robot, 'manip_mode': 'physical',
        'tamp_inputs': 'observed', 'observed_tamp_provider': observed})}
    discovery = tools['scene_tasks'].func()
    result = json.loads(discovery)
    assert result['status'] == 'ok'
    assert result['data']['tasks'][0]['object_name'] == 'cup'
    assert 'private-observation-id' not in discovery
    assert result['data']['planning_available'] is False
    assert tamp_followup('Move the cup', ['[scene_tasks] ' + discovery]) is None
    for name, kwargs in [('plan_pick_place', {'task_ref': result['data']['tasks'][0]['task_ref']}),
                         ('execute_pick_place_plan', {'plan_ref': 'invented'}),
                         ('pick_place', {'object_name': 'cup', 'receptacle_name': 'counter'})]:
        assert json.loads(tools[name].func(**kwargs))['status'] == 'error'
    assert robot.mock_calls == []


@pytest.mark.parametrize('source,age,code', [('simulator_ground_truth', 0, 'privileged_input_forbidden'),
    ('rgbd_tracking', 3, 'object_observation_stale'), ('rgbd_tracking', -3, 'object_observation_stale')])
def test_invalid_evidence_rejects_entire_snapshot_even_if_filtered(source, age, code):
    result = discover_observed_tasks(provider(candidate(), candidate(source=source, age=age, name='hidden')),
                                    object_filter='cup')
    assert result['code'] == code
    assert 'tasks' not in result['data']


def test_unknown_entities_not_fabricated_and_handles_refresh():
    assert discover_observed_tasks(provider())['data']['tasks'] == []
    observed = provider(candidate(), candidate(name='bottle'))
    first = discover_observed_tasks(observed, 'CUP')['data']['tasks']
    second = discover_observed_tasks(observed, 'cup')['data']['tasks']
    assert len(first) == 1
    assert first[0]['task_ref'] != second[0]['task_ref']
    assert discover_observed_tasks(None)['code'] == 'observed_scene_unavailable'


def test_wrong_provider_and_invalid_snapshot_are_rejected():
    observed = provider(candidate())
    observed.input_mode = 'privileged'
    assert discover_observed_tasks(observed)['code'] == 'input_provider_mismatch'
    observed.observe_tasks.assert_not_called()
    observed.input_mode = 'observed'
    observed.observe_tasks.return_value = [{'object_name': 'cup'}]
    assert discover_observed_tasks(observed)['code'] == 'invalid_observed_task_snapshot'


def test_lookup_does_not_renew_evidence():
    original = candidate()
    observed = provider(original)
    assert discover_observed_tasks(observed, now=original.object_pose.received_monotonic + 1)['status'] == 'ok'
    assert discover_observed_tasks(observed, now=original.object_pose.received_monotonic + 3)['code'] == 'object_observation_stale'


def frame_observations():
    axis = np.linspace(-.04, .04, 8)
    cube = np.stack(np.meshgrid(axis, axis, axis), axis=-1).reshape(-1, 3) + [0, 0, .60]
    xy = np.linspace(-.3, .3, 24)
    surface = np.stack(np.meshgrid(xy, xy), axis=-1).reshape(-1, 2)
    table = np.c_[surface, np.full(len(surface), .55)]
    return [SimpleNamespace(label='cup', points_3d=cube, timestep=1),
            SimpleNamespace(label='counter', points_3d=table, timestep=1)]


def test_rgbd_provider_uses_same_frame_geometry_and_original_receive_time():
    from emet.controller.task.tamp.observed_discovery import RgbdTaskProvider

    observed = RgbdTaskProvider()
    observed.update(frame_observations(), received_monotonic=100.)
    result = discover_observed_tasks(observed, now=101.)
    row, = result['data']['tasks']
    assert row['object_name'] == 'cup'
    assert row['receptacle_name'] == 'counter'
    assert row['sources'] == {'object': 'rgbd_pointcloud', 'support': 'rgbd_pointcloud'}
    np.testing.assert_allclose(row['object_position'], [0, 0, .60])
    np.testing.assert_allclose(row['support_position'], [0, 0, .55], atol=1e-8)
    assert discover_observed_tasks(observed, now=103.)['code'] == 'object_observation_stale'
    # Republishing a frame must preserve its receive time.
    observed.update(frame_observations(), received_monotonic=100.)
    assert discover_observed_tasks(observed, now=103.)['code'] == 'object_observation_stale'
    observed.update(())
    assert discover_observed_tasks(observed)['data']['tasks'] == []


def test_missing_timestamp_vertical_support_and_estimated_depth_provenance():
    from emet.controller.task.tamp.observed_discovery import RgbdTaskProvider

    observed = RgbdTaskProvider()
    frame = frame_observations()
    observed.update(frame)
    assert observed.observe_tasks() == ()
    observed.update(frame, received_monotonic=100., source='estimated_depth_pointcloud')
    assert discover_observed_tasks(observed, now=101.)['data']['tasks'][0]['sources']['support'] == 'estimated_depth_pointcloud'
    frame[1].points_3d[:, [1, 2]] = frame[1].points_3d[:, [2, 1]]  # vertical wall
    observed.update(frame, received_monotonic=100.)
    assert observed.observe_tasks() == ()


def test_actual_processor_publishes_depth_proposals_and_clears_missing_detections(monkeypatch):
    from emet.mapping.scene_graph.processor import SceneGraphProcessor

    processor = SceneGraphProcessor(config={}, device='cpu')
    rgb = np.zeros((40, 40, 3), np.uint8)
    cup_mask = np.zeros((40, 40), bool)
    cup_mask[16:24, 16:24] = True
    depth = np.full((40, 40), .5, np.float32)
    depth[cup_mask] = .4
    intrinsics = np.array([[50., 0., 20.], [0., 50., 20.], [0., 0., 1.]])
    camera_pose = np.diag([1., -1., -1., 1.])
    camera_pose[2, 3] = 1.
    detections = [{'mask': cup_mask, 'label': 'cup'}, {'mask': ~cup_mask, 'label': 'counter'}]
    monkeypatch.setattr(processor, '_run_segmentation', lambda *args: detections)
    monkeypatch.setattr(processor, '_encode_siglip', lambda *args: None)
    monkeypatch.setattr(processor, '_encode_dinov3', lambda *args: None)
    processor.process_frame(rgb, depth, intrinsics, camera_pose, received_monotonic=100.)
    assert len(discover_observed_tasks(processor.observed_tamp_provider, now=101.)['data']['tasks']) == 1
    detections.clear()
    processor.process_frame(rgb, depth, intrinsics, camera_pose, received_monotonic=101.)
    assert processor.observed_tamp_provider.observe_tasks() == ()


def test_stretch_rgbd_receive_time_cannot_be_renewed_by_reading(monkeypatch):
    import threading

    import emet.controller.zmq_client as module

    client = object.__new__(module.StretchZmqClient)
    client._obs_lock = threading.Lock()
    client._last_step = 0
    client._iter = 1
    client._seq_id = 1
    client._note_emet_session_from_zmq_dict = Mock()
    client.is_up_to_date = lambda **kwargs: True
    monkeypatch.setattr(module.time, 'sleep', lambda *args: None)
    monkeypatch.setattr(module.time, 'monotonic', lambda: 100.)
    raw = {'step': 1, 'gps': np.zeros(2), 'compass': np.zeros(1), 'rgb': np.zeros((2, 2, 3)),
           'depth': np.ones((2, 2)), 'xyz': None, 'lidar_points': None, 'lidar_timestamp': None,
           '_received_monotonic': 999.}
    client._update_obs(raw)
    assert raw['_received_monotonic'] == 999.  # publisher's dict is not changed
    assert client.get_observation().received_monotonic == 100.
    monkeypatch.setattr(module.time, 'monotonic', lambda: 200.)
    assert client.get_observation().received_monotonic == 100.


@pytest.mark.parametrize('mode,sensor,expected_time', [('da3', None, None), ('auto', np.ones((2, 2)), 100.)])
def test_mapping_cannot_promote_reused_depth_to_fresh_evidence(monkeypatch, mode, sensor, expected_time):
    from emet.controller.dynamem import perception

    class EndProbe(Exception):
        pass
    kwargs_seen = {}
    def probe(*args, **kwargs):
        kwargs_seen.update(kwargs)
        raise EndProbe()
    obs = SimpleNamespace(rgb=np.zeros((2, 2, 3)), depth=sensor, camera_K=np.eye(3),
                          camera_pose=np.eye(4), gps=None, compass=None, emet_session=None,
                          received_monotonic=100.)
    controller = SimpleNamespace(robot=Mock(), obs_count=1, _depth_source=mode,
        _da3_infer_every_n=2, _da3_last_depth=np.ones((2, 2)),
        _run_full_perception=lambda _: False, parameters={}, voxel_map=SimpleNamespace(process_rgbd_images=probe))
    controller.robot.get_observation.return_value = obs
    monkeypatch.setattr(perception, '_log_head_camera_if_no_zmq_rerun_thread', lambda *args, **kwargs: None)
    with pytest.raises(EndProbe):
        perception.update(controller)
    assert kwargs_seen['received_monotonic'] == expected_time
    assert kwargs_seen['depth_evidence_source'] == ('rgbd_pointcloud' if sensor is not None else 'estimated_depth_pointcloud')

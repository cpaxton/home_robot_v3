from types import SimpleNamespace

import numpy as np
import pytest

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.controller.task.tamp.visual_grounding import VisualTransferGrounder


def sample(stamp, capture_id, destination_depth=1.2):
    depth = np.ones((100, 100))
    depth[30:50, 30:50] = .8
    depth[30:70, 60:90] = destination_depth
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 1.
    view = SimpleNamespace(rgb=np.zeros((100, 100, 3), np.uint8), depth=depth,
        camera_K=np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1.]]), camera_pose=pose)
    return SimpleNamespace(capture=CaptureInterval(capture_id, stamp-.002, stamp-.001, stamp, 'calibration'),
        observations={'front': view}, robot_guards={})


def resolver(monkeypatch, tmp_path, clock, permitted=lambda: True):
    import emet.controller.task.tamp.visual_grounding as module

    calls = []

    def query(*args, **kwargs):
        calls.append(kwargs)
        clock[0] += 5.  # Inference outlasts the reference motion evidence.
        return {'camera': 'front', 'object': [300, 300, 500, 500],
            'destination_camera': 'front', 'destination': [600, 300, 900, 700],
            'grasp_yaw_degrees': 0, 'grasp_height_fraction': .3}

    monkeypatch.setattr(module, 'query_pick_hint', query)
    monkeypatch.setattr(module, 'refine_object_hint', lambda rgb, decision, **kwargs: decision)
    grounder = VisualTransferGrounder(endpoint='unused', model='test', output=tmp_path/'queries',
        can_query=permitted, clock=lambda: clock[0])
    return grounder, calls


def test_inference_reacquires_geometry_without_renewing_reference_time(monkeypatch, tmp_path):
    clock = [100.]
    grounder, calls = resolver(monkeypatch, tmp_path, clock)
    reference = sample(clock[0], 'old')
    result = grounder.resolve(reference, lambda: sample(clock[0], 'fresh'),
        SimpleNamespace(object_name='block', receptacle_name='basin'))
    assert result.reference_capture_id == 'old'
    assert result.sample.capture.observation_id == result.selection.capture_id == 'fresh'
    assert reference.capture.received_monotonic == 100.
    assert calls[0]['infer_support_from_depth'] is True
    assert result.profile['local_support_estimation'] is True
    assert result.selection.object_mask.any()


def test_loaded_or_unknown_phase_does_not_call_model(monkeypatch, tmp_path):
    clock = [100.]
    grounder, calls = resolver(monkeypatch, tmp_path, clock, permitted=lambda: False)
    with pytest.raises(ObservationUnavailable, match='visual_query_phase_unavailable'):
        grounder.resolve(sample(100., 'old'), lambda: pytest.fail('capture not needed'),
            SimpleNamespace(object_name='block', receptacle_name='basin'))
    assert not calls


def test_replayed_frame_cannot_be_used_as_fresh_geometry(monkeypatch, tmp_path):
    clock = [100.]
    grounder, calls = resolver(monkeypatch, tmp_path, clock)
    with pytest.raises(ObservationUnavailable):
        grounder.resolve(sample(100., 'old'), lambda: sample(clock[0], 'old'),
            SimpleNamespace(object_name='block', receptacle_name='basin'))
    assert len(calls) == 3


def test_phase_change_during_inference_stops_before_another_query(monkeypatch, tmp_path):
    clock = [100.]
    grounder, calls = resolver(monkeypatch, tmp_path, clock, permitted=lambda: clock[0] <= 100.)
    with pytest.raises(ObservationUnavailable, match='visual_query_phase_unavailable'):
        grounder.resolve(sample(100., 'old'), lambda: pytest.fail('must stop before reacquisition'),
            SimpleNamespace(object_name='block', receptacle_name='basin'))
    assert len(calls) == 1


def test_live_scene_bootstrap_fits_support_from_reacquired_capture(monkeypatch, tmp_path):
    from emet.controller.task.tamp.live_transfer import LiveTransferScene

    clock = [100.]
    grounder, _ = resolver(monkeypatch, tmp_path, clock)
    old, fresh = sample(100., 'old'), sample(105., 'fresh')
    scene = LiveTransferScene.__new__(LiveTransferScene)
    scene.clock = lambda: clock[0]
    scene.profile = {'calibration_id': 'calibration', 'support_margin_m': .001, 'payload_motion_error_m': .001}
    scene.task = SimpleNamespace(object_name='block', receptacle_name='basin', object_ref='object',
                                 source_support_ref='counter', support_ref='basin')
    scene.epoch = 'epoch'
    scene.samples = {'old': old, 'fresh': fresh}
    packets = iter([{'front': SimpleNamespace(capture=old.capture)},
                    {'front': SimpleNamespace(capture=fresh.capture)}])
    scene._read_views = lambda: next(packets)
    scene.grounder = grounder
    captures = []

    class SupportReached(Exception):
        pass

    def support(*args, **kwargs):
        captures.append(kwargs['capture'].observation_id)
        raise SupportReached()

    monkeypatch.setattr('emet.controller.task.tamp.live_transfer.fit_transfer_support', support)
    with pytest.raises(SupportReached):
        scene._bootstrap()
    assert captures == ['fresh']
    assert scene.diagnostics['capture_id'] == 'fresh'
    assert scene.profile['local_support_estimation'] is True


@pytest.mark.parametrize('destination_depth', [1.2, 1., .6])
def test_shared_grounding_accepts_lower_same_level_and_higher_support(monkeypatch, tmp_path, destination_depth):
    from emet.controller.task.tamp.observed_capture import capture_arrays

    clock = [100.]
    grounder, _ = resolver(monkeypatch, tmp_path, clock)
    result = grounder.resolve(sample(100., 'reference', destination_depth),
        lambda: sample(clock[0], 'fresh', destination_depth),
        SimpleNamespace(object_name='block', receptacle_name='destination surface'))
    points = capture_arrays(result.sample.observations['front'])[2]
    destination = points[result.selection.destination_mask]
    assert np.median(destination[:, 2]) == pytest.approx(1.-destination_depth)
    assert result.profile['destination_below_source'] is False


def test_retry_feedback_identifies_failed_destination_and_preserves_reason(monkeypatch, tmp_path):
    from emet.controller.task.tamp.transfer_grounding import TransferGroundingUnavailable

    clock = [100.]
    grounder, calls = resolver(monkeypatch, tmp_path, clock)

    def reject(*args, **kwargs):
        raise TransferGroundingUnavailable('support_geometry_unknown', 'destination')

    monkeypatch.setattr('emet.controller.task.tamp.visual_grounding.select_transfer', reject)
    with pytest.raises(TransferGroundingUnavailable) as raised:
        grounder.resolve(sample(100., 'old'), lambda: sample(clock[0], 'new'),
            SimpleNamespace(object_name='block', receptacle_name='basin'))
    assert str(raised.value) == 'support_geometry_unknown'
    assert raised.value.region_role == 'destination'
    assert len(calls) == 3
    assert 'destination region lacks horizontal support' in calls[1]['feedback']


def test_live_runtime_grounding_rejects_object_prior_before_source_startup():
    from emet.controller.task.tamp.live_transfer import LiveTransferScene

    with pytest.raises(ValueError, match='runtime_grounding_requires_no_object_cad'):
        LiveTransferScene(arm=None, task=None, read_sample=None, select=None,
            payload_model=SimpleNamespace(construction='known_model_prior'),
            fingertip_geom_ids=(), profile={}, grounder=object())

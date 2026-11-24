# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
from types import SimpleNamespace

import numpy as np
import pytest

from emet.controller.task.tamp.transfer_grounding import _depth_components, select_transfer


def test_adjacent_image_surfaces_separate_at_depth_jump():
    y, x = np.indices((20, 30))
    world = np.stack((x*.002, y*.002, np.ones(x.shape)), axis=-1)
    world[:, 15:, 2] += .2
    mask = np.ones(x.shape, bool)
    groups = _depth_components(mask, world)
    assert groups[10, 14] != groups[10, 15]
    assert groups[0, 0] == groups[19, 14]
    world[:, 15:, 2] -= .2
    assert len(np.unique(_depth_components(mask, world))) == 1
    assert not _depth_components(np.zeros_like(mask), world).any()


@pytest.mark.parametrize('has_sink', [True, False])
def test_support_estimate_rejects_robot_patch_above_object_and_uses_local_counter(monkeypatch, has_sink):
    y, x = np.indices((120, 120))
    world = np.stack((x*.002, y*.002, np.full(x.shape, .9)), axis=-1)
    world[45:65, 45:65, 2] = .95
    world[5:30, 5:30, 2] = 1.4  # Bad semantic support region on the robot.
    if has_sink:
        world[80:115, 80:115, 2] = .75  # Sink floor.
    # Robot pixels touch the object in the image, but lie on a distant surface.
    world[45:65, 65:115, 2] = 1.4
    observation = SimpleNamespace(rgb=np.zeros((120, 120, 3), np.uint8), depth=np.ones((120, 120)))
    monkeypatch.setattr('emet.controller.task.tamp.transfer_grounding.capture_arrays',
        lambda obs: (None, np.ones((120, 120)), world))
    sample = SimpleNamespace(capture=SimpleNamespace(observation_id='test'), observations={'scene': observation})
    profile = {'local_support_estimation': True, 'destination_below_source': True, 'grounding': {
        'object': {'camera': 'scene', 'box': [375, 375, 550, 540]},
        'source': {'camera': 'scene', 'box': [50, 50, 230, 230]},
        'destination': {'camera': 'scene', 'box': [700, 700, 940, 940]}}}
    if not has_sink:
        from emet.controller.manipulation.native_inputs import ObservationUnavailable

        with pytest.raises(ObservationUnavailable, match='support_geometry_unknown'):
            select_transfer(sample, profile=profile, grow_object_region=True)
    result = select_transfer(sample, profile=profile, grow_object_region=True, require_destination=has_sink)
    assert np.median(world[result.source_mask][:, 2]) == .9
    assert result.object_mask.sum() > 200
    assert np.max(world[result.object_mask][:, 2]) < 1.


def test_replay_refusal_does_not_make_an_extra_refinement_call(monkeypatch, tmp_path):
    import json
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/"scripts"))
    import eval_vlm_grounding_replay as replay

    batch = tmp_path/'batch'
    (batch/'case').mkdir(parents=True)
    (batch/'case'/'sensor.npz').write_bytes(b'recorded-sensor')
    (batch/'manifest.json').write_text(json.dumps({'destination': 'sink', 'cases': [
        {'id': 'case', 'spec': {'object_name': 'garlic'}}]}))
    monkeypatch.setattr(replay, 'load_views', lambda sensor: {'scene_alt': SimpleNamespace(rgb=np.zeros((2, 2, 3), np.uint8))})
    calls = []

    def refuse(*args, **kwargs):
        calls.append('query')
        raise ValueError('vlm_grounding_unavailable')

    def unexpected(*args, **kwargs):
        raise AssertionError('refinement must not run after an unavailable proposal')

    monkeypatch.setattr(replay, 'query_pick_hint', refuse)
    monkeypatch.setattr(replay, 'refine_object_hint', unexpected)
    output = tmp_path/'output'
    monkeypatch.setattr('sys.argv', ['replay', str(batch), str(output), '--endpoint', 'http://test', '--refine'])
    assert replay.main() == 0
    report = json.loads((output/'report.json').read_text())
    assert report['runtime_model_calls'] == 1
    row = report['results'][0]
    assert row['current'] == {'accepted': False, 'reason': 'vlm_grounding_unavailable'}
    assert calls == ['query']


def test_support_growth_follows_measured_plane_without_crossing_world_gap():
    from emet.controller.task.tamp.transfer_grounding import _grow_measured_plane

    y, x = np.indices((20, 30))
    world = np.stack((x*.002, y*.002, np.ones(x.shape)), axis=-1)
    world[:, 15:, 0] += .2
    seed = np.zeros(x.shape, bool)
    seed[8:12, 5:8] = True
    result = _grow_measured_plane(seed, world, 1.)
    assert result[:, :15].all()
    assert not result[:, 15:].any()
    world[:, 7, 2] = np.nan
    seed[:, 7:] = False
    result = _grow_measured_plane(seed, world, 1.)
    assert not result[:, 7:].any()


def test_inferred_source_support_is_local_and_keeps_object_pixels_excluded():
    from emet.controller.task.tamp.transfer_grounding import _grow_measured_plane

    y, x = np.indices((100, 100))
    world = np.stack((x*.01, y*.01, np.zeros(x.shape)), axis=-1)
    seed = np.zeros(x.shape, bool)
    seed[45:55, 45:55] = True
    excluded = np.zeros_like(seed)
    excluded[49:51, 49:51] = True
    result = _grow_measured_plane(seed, world, 0., max_radius_m=.15, exclude=excluded)
    assert result.sum() > seed.sum()
    assert not result[excluded].any()
    assert not result[:30].any() and not result[70:].any()


def test_support_growth_does_not_absorb_neighboring_object_edge_returns():
    from emet.controller.task.tamp.transfer_grounding import _grow_measured_plane

    y, x = np.indices((20, 30))
    world = np.stack((x*.002, y*.002, np.ones(x.shape)), axis=-1)
    world[8:12, 12:16, 2] += .007
    seed = np.zeros(x.shape, bool)
    seed[5:8, 5:8] = True
    result = _grow_measured_plane(seed, world, 1.)
    assert result[5:8, 5:8].all()
    assert not result[8:12, 12:16].any()

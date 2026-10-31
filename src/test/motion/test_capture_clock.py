# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import pytest

from emet.controller.manipulation import capture_clock
from emet.controller.manipulation.native_inputs import ObservationUnavailable


def timing():
    return {'available': True, 'clock_id': 'host-clock', 'clock_domain': 'mujoco_render_monotonic', 'source': 'renderer_scene_capture',
            'capture_interval_ns': [1_000_000_000, 1_010_000_000],
            'depth_capture_interval_ns': [1_030_000_000, 1_040_000_000]}


def test_interval_union_preserves_stereo_skew_and_cannot_renew_replay(monkeypatch):
    monkeypatch.setattr(capture_clock, 'local_monotonic_clock_id', lambda: 'host-clock')
    result = capture_clock.local_sim_capture_interval(timing(), received_monotonic=1.1, observation_id='capture')
    assert result.earliest == 1. and result.latest == 1.04
    result.require(now=1.2)
    with pytest.raises(ObservationUnavailable):
        result.require(now=4.)


@pytest.mark.parametrize('key,value', [('clock_id', 'remote'), ('clock_domain', 'device'),
                                      ('capture_interval_ns', None), ('depth_capture_interval_ns', [2, 1])])
def test_missing_or_other_clock_evidence_is_not_aligned(monkeypatch, key, value):
    monkeypatch.setattr(capture_clock, 'local_monotonic_clock_id', lambda: 'host-clock')
    data = timing()
    data[key] = value
    with pytest.raises(ObservationUnavailable):
        capture_clock.local_sim_capture_interval(data, received_monotonic=1.1, observation_id='capture')

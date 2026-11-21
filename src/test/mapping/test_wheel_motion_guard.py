# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.core.navigation_result import NavigationPoseUncertainty
from emet.mapping.wheel_motion_guard import WheelMotionGuard


def test_motion_guard_compares_local_displacements_and_latches_slip():
    guard = WheelMotionGuard(NavigationPoseUncertainty(.04, .025), max_age_s=.5)

    def sample(wheel, reference, tick):
        stamp = tick * 100_000_000
        return guard.update(wheel, reference, wheel_timestamp_ns=stamp, reference_timestamp_ns=stamp, now_ns=stamp)

    assert sample([0., 0., 0.], [2., 1., np.pi / 2], 1).ok
    assert sample([.1, 0., .1], [2., 1.1, np.pi / 2 + .1], 2).ok
    slipped = sample([.2, 0., .1], [2., 1.1, np.pi / 2 + .1], 3)
    assert slipped.status == "wheel_motion_disagreement"
    assert slipped.position_disagreement_m == pytest.approx(.1)
    assert sample([.2, 0., .1], [2., 1.2, np.pi / 2 + .1], 4) == slipped


@pytest.mark.parametrize(
    "reference,stamp,now,response",
    [
        (None, 1, 5, "independent_pose_unavailable"),
        ([0, 0], 1, 5, "independent_pose_unavailable"),
        ([0, 0, 0], 2, 5, "wheel_motion_alignment_unavailable"),
        ([0, 0, 0], 1, 600_000_000, "wheel_motion_alignment_unavailable"),
    ],
)
def test_motion_guard_refuses_missing_stale_or_unaligned_reference(reference, stamp, now, response):
    guard = WheelMotionGuard(NavigationPoseUncertainty(.04, .025), max_age_s=.5)
    result = guard.update([0, 0, 0], reference, wheel_timestamp_ns=1, reference_timestamp_ns=stamp, now_ns=now)
    assert result.status == response


def test_live_bridge_checks_one_matching_frame_and_latches_a_source_mismatch(monkeypatch):
    from emet.eval import stretch_encoder_server as adapter

    server = adapter.StretchEncoderNavigationServer.__new__(adapter.StretchEncoderNavigationServer)
    server._motion_guard = WheelMotionGuard(NavigationPoseUncertainty(.04, .025), max_age_s=.75)
    server._motion_reference = Mock(return_value=np.zeros(3))
    server._motion_stamp = None
    server._motion_check = None
    clock = SimpleNamespace(monotonic_ns=lambda: frame.timestamp_ns)
    monkeypatch.setattr(adapter, "time", clock)
    frame = SimpleNamespace(timestamp_ns=100, base_xyt=np.zeros(3))
    packet = {"clock_id": "local", "interval_ns": [100, 110]}
    imagery = SimpleNamespace(fixed_rgbd_sample=dict(packet), synchronized_wheel_sample=packet)

    assert server._check_independent_motion(imagery, frame).ok
    assert server._check_independent_motion(imagery, frame).ok
    server._motion_reference.assert_called_once()
    frame.timestamp_ns = 120
    packet["interval_ns"] = [120, 130]
    lost = server._check_independent_motion(imagery, frame)
    assert lost.status == "independent_pose_unavailable"
    frame.timestamp_ns = 140
    packet["interval_ns"] = [140, 145]
    imagery.fixed_rgbd_sample = dict(packet)
    assert server._check_independent_motion(imagery, frame) == lost

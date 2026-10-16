# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace

import numpy as np

from emet.memory.approach_views import ApproachViewHistory


def observation(stamp=1, x=0.0):
    pose = np.eye(4)
    pose[0, 3] = x
    return SimpleNamespace(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.ones((4, 4)),
        camera_K=np.eye(3),
        camera_pose=pose,
        image_timing={"head_cam_left": {"timestamp_ns": stamp, "clock_domain": "sim"}},
    )


def test_motion_sampling_and_owned_bounded_snapshots():
    history = ApproachViewHistory(capacity=2)
    obs = observation()
    assert history.add(obs)
    obs.rgb[:] = 99
    obs.depth[:] = 9
    assert not history.views[0].rgb.any()
    assert np.all(history.views[0].depth == 1)
    assert not history.views[0].rgb.flags.writeable
    assert not history.add(observation(2, 0.1))
    assert history.add(observation(3, 0.3))
    assert history.add(observation(4, 0.6))
    assert [v.view_id for v in history.views] == ["approach:2", "approach:3"]


def test_optical_rotation_samples_without_translation():
    history = ApproachViewHistory()
    assert history.add(observation())
    obs = observation(2)
    a = np.deg2rad(30)
    obs.camera_pose[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    assert history.add(obs)


def test_missing_stale_and_mixed_clock_evidence_is_not_fabricated():
    history = ApproachViewHistory()
    assert not history.add(None)
    obs = observation()
    obs.image_timing = {}
    assert not history.add(obs)
    assert history.add(observation(5))
    assert not history.add(observation(4, 1))
    obs = observation(6, 1)
    obs.image_timing["head_cam_left"]["clock_domain"] = "other"
    assert not history.add(obs)
    assert len(history.views) == 1

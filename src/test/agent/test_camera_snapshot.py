# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Decision pixels, sensor geometry and identity survive camera/transport changes."""

from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from emet.agent.camera_snapshot import camera_snapshot
from emet.controller.dynamem.perception import update


def observation():
    return SimpleNamespace(
        rgb=np.full((4, 6, 3), 80, dtype=np.uint8),
        depth=np.ones((4, 6), dtype=np.float32),
        camera_K=np.eye(3),
        camera_pose=np.eye(4),
        head_rgb_right=np.full((4, 6, 3), 120, dtype=np.uint8),
        head_camera_K_right=2 * np.eye(3),
        head_camera_pose_right=np.eye(4),
        gps=np.array([1, 2]),
        compass=np.array([0.3]),
        image_timing={"head": {"clock_domain": "fixture", "timestamp_ns": 123}},
        emet_session={"navigation_origin_xyt": [1, 2, 0.3]},
        seq_id=17,
        received_monotonic=42.0,
        semantic=np.ones((4, 6)),  # Labels/task state are not copied into the sensor snapshot.
        task_observations={"private_task_state": "excluded"},
    )


def test_snapshot_owns_pixels_depth_pose_and_timing_without_task_labels():
    source = observation()
    snapshot, identity = camera_snapshot(source)
    source.rgb[:] = 0
    source.depth[:] = 9
    source.camera_pose[0, 3] = 99
    source.camera_K[0, 0] = 99
    source.head_rgb_right[:] = 0
    source.head_camera_K_right[0, 0] = 99
    source.image_timing["head"]["timestamp_ns"] = 999
    source.emet_session["navigation_origin_xyt"][0] = 999
    assert np.all(snapshot.rgb == 80) and np.all(snapshot.depth == 1)
    assert snapshot.camera_pose[0, 3] == 0 and snapshot.camera_K[0, 0] == 1
    assert np.all(snapshot.head_rgb_right == 120) and snapshot.head_camera_K_right[0, 0] == 2
    assert snapshot.image_timing["head"]["timestamp_ns"] == 123
    assert snapshot.emet_session["navigation_origin_xyt"][0] == 1
    assert snapshot.received_monotonic == 42
    assert snapshot.semantic is None and snapshot.task_observations is None
    assert identity["rgb_sha256"] == sha256(snapshot.rgb.tobytes()).hexdigest()
    assert identity["sensor_depth_sha256"] == sha256(snapshot.depth.tobytes()).hexdigest()
    assert identity["frame_sequence"] == 17


@pytest.mark.parametrize("supplied", [True, False])
def test_perception_depth_provider_receives_supplied_snapshot_or_live_capture(supplied):
    earlier, _ = camera_snapshot(observation())
    newer = observation()
    newer.rgb[:] = 0
    newer.depth[:] = 9
    newer.camera_pose[0, 3] = 99
    captures = []

    def capture():
        captures.append(True)
        return newer

    class ReachedDepthProvider(Exception):
        pass

    expected = earlier if supplied else newer

    def depth(rgb, sensor_depth, K, pose, **kwargs):
        assert np.array_equal(rgb, expected.rgb)
        assert np.array_equal(sensor_depth, expected.depth)
        assert np.array_equal(K, expected.camera_K)
        assert np.array_equal(pose, expected.camera_pose)
        raise ReachedDepthProvider

    agent = SimpleNamespace(
        robot=SimpleNamespace(get_observation=capture),
        obs_count=0,
        _run_full_perception=lambda full: True,
        _da3_infer_every_n=1,
        _depth_source="sensor",
        _resolve_depth_map=depth,
    )
    with pytest.raises(ReachedDepthProvider):
        update(agent, full_perception=True, observation=earlier if supplied else None)
    assert len(captures) == (0 if supplied else 1)

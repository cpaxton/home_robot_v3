# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path

import numpy as np


def test_receive_only_decoder_handles_deployed_camera_aliases(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    from probe_stationary_bridge import decode_cameras

    head = np.full((4, 5, 3), 90, dtype=np.uint8)
    wrist = np.zeros_like(head)
    images = decode_cameras({"head_cam_left/image": head, "head_cam_right/image": head, "ee_cam/image": wrist})
    assert set(images) == {"head_left", "head_right", "wrist"}
    assert images["head_left"].max() == 90
    assert images["wrist"].max() == 0


def test_missing_cameras_are_not_filled_with_synthetic_images(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    from probe_stationary_bridge import decode_cameras

    assert decode_cameras({}) == {}


def test_metric_depth_requires_a_matching_camera_and_rigid_pose(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    from probe_stationary_bridge import decode_geometry

    head = np.full((4, 5, 3), 90, dtype=np.uint8)
    message = {"depth": np.ones(head.shape[:2], dtype=np.float32), "camera_K": np.eye(3), "camera_pose": np.eye(4)}
    _, geometry = decode_geometry(message, head)
    assert geometry["paired_metric_rgbd"]

    _, geometry = decode_geometry(message, head[:-1])
    assert geometry["depth"] and not geometry["paired_metric_rgbd"]

    message["camera_pose"][0, 0] = -1
    _, geometry = decode_geometry(message, head)
    assert not geometry["camera_pose"] and not geometry["paired_metric_rgbd"]

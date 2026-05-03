# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

"""Tests for ZMQ RGB/depth layout helpers and pinhole sizing from depth (regression for depth_to_xyz)."""

import numpy as np
import pytest

from emet.utils.image import Camera, pinhole_camera_from_intrinsics_and_depth
from emet.utils.observation_layout import rgb_height_width_for_zmq


def test_rgb_height_width_for_zmq_matches_numpy_hw():
    rgb = np.zeros((240, 424, 3), dtype=np.uint8)
    h, w = rgb_height_width_for_zmq(rgb)
    assert (h, w) == (240, 424)
    assert rgb.shape[:2] == (h, w)


def test_rgb_height_width_for_zmq_rejects_bad_rank():
    with pytest.raises(ValueError):
        rgb_height_width_for_zmq(np.zeros((4,), dtype=np.uint8))


def test_pinhole_camera_from_intrinsics_and_depth_matches_depth_to_xyz():
    """depth_to_xyz indices must match depth shape (H, W)."""
    h, w = 12, 18
    depth = np.full((h, w), 1.5, dtype=np.float32)
    fx, fy = 300.0, 300.0
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    cam = pinhole_camera_from_intrinsics_and_depth(k, depth)
    assert int(cam.height) == h and int(cam.width) == w
    xyz = cam.depth_to_xyz(depth)
    assert xyz.shape == (h, w, 3)


def test_wrong_message_hw_would_broadcast_against_depth():
    """Regression: never size Camera from swapped (W,H) when depth is (H,W)."""
    h, w = 240, 424
    depth = np.full((h, w), 1.0, dtype=np.float32)
    fx, fy = 400.0, 400.0
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    bad = Camera.from_K(k, width=float(h), height=float(w))  # swapped vs depth
    with pytest.raises(ValueError):
        bad.depth_to_xyz(depth)

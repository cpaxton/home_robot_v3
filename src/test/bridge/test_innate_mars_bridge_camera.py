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

import numpy as np
import pytest

pytest.importorskip("rclpy")

from innate_mars_bridge.ros.camera import RosCamera, ros_image_encoding_to_rgb

from emet.controller.generic_zmq_client import _align_camera_k_to_rgb
from emet.utils.image import align_camera_matrix_to_image_size


def test_ros_image_encoding_to_rgb_bgr8():
    bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
    rgb = ros_image_encoding_to_rgb(bgr, "bgr8")
    assert rgb.shape == (1, 1, 3)
    np.testing.assert_array_equal(rgb[0, 0], [30, 20, 10])


def test_ros_image_encoding_to_rgb_rgb8_passthrough():
    rgb_in = np.array([[[1, 2, 3]]], dtype=np.uint8)
    rgb_out = ros_image_encoding_to_rgb(rgb_in, "rgb8")
    np.testing.assert_array_equal(rgb_out, rgb_in)


def test_align_camera_matrix_to_image_size_doubles_intrinsics():
    k = np.array([[133.0, 0.0, 155.0], [0.0, 133.0, 124.0], [0.0, 0.0, 1.0]])
    out = align_camera_matrix_to_image_size(
        k,
        calib_height=240,
        calib_width=320,
        image_height=480,
        image_width=640,
    )
    np.testing.assert_allclose(out[0, 0], 266.0)
    np.testing.assert_allclose(out[0, 2], 310.0)


def test_align_camera_k_to_rgb_heuristic():
    k = np.array([[133.0, 0.0, 155.0], [0.0, 133.0, 124.0], [0.0, 0.0, 1.0]])
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    out = _align_camera_k_to_rgb(k, rgb)
    assert out is not None
    np.testing.assert_allclose(out[0, 2], 320.0, rtol=0, atol=2.0)


def test_ros_camera_get_k_scales_to_image(monkeypatch):
    cam = RosCamera.__new__(RosCamera)
    cam.K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    cam.height = 240
    cam.width = 320
    cam._lock = __import__("threading").Lock()
    cam._img = np.zeros((480, 640, 3), dtype=np.uint8)
    cam.name = "test"
    out = RosCamera.get_K(cam)
    np.testing.assert_allclose(out[0, 0], 200.0)
    np.testing.assert_allclose(out[0, 2], 100.0)

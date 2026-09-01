# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Camera-K alignment helpers (no ROS)."""

import numpy as np

from emet.controller.generic_zmq_client import _align_camera_k_to_rgb
from emet.utils.image import align_camera_matrix_to_image_size


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

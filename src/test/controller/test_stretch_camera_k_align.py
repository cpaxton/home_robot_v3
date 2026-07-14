# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Stretch ZMQ client aligns camera_K when image resolution differs from calib."""

from __future__ import annotations

import numpy as np

from emet.utils.image import align_camera_matrix_to_image_size


def test_stretch_client_aligns_mismatched_camera_k():
    # Full-res K (64x48) with half-res depth (32x24) — same bug Habitat had before the fix.
    k_full = np.array([[100.0, 0.0, 31.5], [0.0, 100.0, 23.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    dh, dw = 24, 32
    calib_w = max(1, int(round(2.0 * float(k_full[0, 2]) + 1.0)))
    calib_h = max(1, int(round(2.0 * float(k_full[1, 2]) + 1.0)))
    assert calib_w == 64 and calib_h == 48
    k = align_camera_matrix_to_image_size(
        k_full,
        calib_height=calib_h,
        calib_width=calib_w,
        image_height=dh,
        image_width=dw,
    )
    np.testing.assert_allclose(k[0, 0], 50.0, atol=1e-6)
    np.testing.assert_allclose(k[0, 2], 15.75, atol=1e-6)

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""JPEG helpers assume RGB arrays (OpenCV imencode uses BGR)."""

import numpy as np

import emet.utils.compression as compression


def test_jpeg_roundtrip_preserves_rgb_channel_order():
    """Solid red left / blue right halves survive JPEG compression (MuJoCo RGB, same as RobosuiteZmqServer)."""
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:, :20] = (255, 0, 0)
    img[:, 20:] = (0, 0, 255)
    blob = compression.to_jpg(img)
    out = compression.from_jpg(blob.tobytes())
    assert out.shape == img.shape
    assert out[20, 10, 0] > 200 and out[20, 10, 1] < 20 and out[20, 10, 2] < 20
    assert out[20, 30, 2] > 200 and out[20, 30, 0] < 20


def test_jpeg_roundtrip_after_optional_vertical_flip():
    """Optional ``EMET_ROBOSUITE_RENDER_FLIPUD``: same row order effect as ``np.flipud``; channels must survive JPEG."""
    from emet.utils.pinhole_intrinsics import apply_pinhole_pixel_ops

    h = 40
    img = np.zeros((h, 40, 3), dtype=np.uint8)
    img[4:11, 2:14] = (255, 0, 0)
    img[4:11, 22:34] = (0, 0, 255)
    img = apply_pinhole_pixel_ops(img, ("flipud",))
    blob = compression.to_jpg(img, quality=98)
    out = compression.from_jpg(blob.tobytes())
    row_r = (h - 1) - 8
    assert out[row_r, 8, 0] > 200 and out[row_r, 8, 1] < 20 and out[row_r, 8, 2] < 20
    assert out[row_r, 28, 2] > 200 and out[row_r, 28, 0] < 20

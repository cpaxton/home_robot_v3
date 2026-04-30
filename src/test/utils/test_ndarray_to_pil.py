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
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import io

import numpy as np
from PIL import Image

from emet.utils.image import ndarray_hwc_to_pil_rgb_u8


def test_float01_image_not_all_black():
    """Float H×W×3 in [0,1] must not collapse to zeros when saving as PNG."""
    rng = np.random.default_rng(0)
    rgb_f = rng.random((32, 24, 3)).astype(np.float32)
    pil = ndarray_hwc_to_pil_rgb_u8(rgb_f, assume_opencv_bgr=False)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    out = np.array(Image.open(buf))
    assert out.mean() > 5, "expected visible pixels, not a black frame"


def test_uint8_roundtrip_meaningful():
    arr = (np.ones((16, 12, 3), dtype=np.uint8) * 200).astype(np.uint8)
    pil = ndarray_hwc_to_pil_rgb_u8(arr, assume_opencv_bgr=False)
    out = np.asarray(pil)
    assert out.min() >= 199


def test_chw_layout_transposed():
    """3×H×W uint8 should become H×W×3 and remain bright after BGR→RGB skip."""
    chw = np.zeros((3, 20, 18), dtype=np.uint8)
    chw[0] = 200
    chw[1] = 10
    chw[2] = 10
    pil = ndarray_hwc_to_pil_rgb_u8(chw, assume_opencv_bgr=False)
    out = np.asarray(pil)
    assert out.shape == (20, 18, 3)
    assert out[..., 0].mean() > 100

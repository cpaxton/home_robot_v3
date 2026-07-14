# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

import numpy as np

from emet.llms.vl_image import downsample_rgb_hwc, eqa_vl_image_kwargs
from emet.llms.vllm_factory import eqa_vl_client_kwargs


def test_downsample_noop_when_small():
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    out = downsample_rgb_hwc(rgb, max_side=512)
    assert out.shape == (240, 320, 3)


def test_downsample_max_side():
    rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = downsample_rgb_hwc(rgb, max_side=512)
    assert max(out.shape[0], out.shape[1]) == 512
    assert out.dtype == np.uint8


def test_downsample_max_pixels():
    rgb = np.zeros((800, 800, 3), dtype=np.uint8)
    out = downsample_rgb_hwc(rgb, max_side=0, max_pixels=100_000)
    assert out.shape[0] * out.shape[1] <= 100_000 + 50  # rounding slack


def test_eqa_vl_image_kwargs_defaults():
    kw = eqa_vl_image_kwargs({})
    assert kw["image_max_side"] == 512
    assert kw["image_max_pixels"] == 0


def test_eqa_vl_client_kwargs_includes_image():
    kw = eqa_vl_client_kwargs({"vl_image_max_side": 384, "vl_cache_system_prefix": False})
    assert kw["image_max_side"] == 384
    assert kw["cache_system_prefix"] is False

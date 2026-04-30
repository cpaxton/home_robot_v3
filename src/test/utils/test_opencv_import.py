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
# Copyright (c) Hello Robot, Inc. All rights reserved.

import sys
import types

import pytest

from emet.utils.opencv_import import assert_cv2_is_real_opencv


def test_assert_cv2_real_opencv_passes_in_normal_env():
    import cv2

    if not hasattr(cv2, "resize"):
        pytest.skip("Incomplete OpenCV install (cv2 is a stub). Fix: uv pip install --reinstall opencv-python")
    assert_cv2_is_real_opencv()


def test_assert_cv2_raises_on_stub(monkeypatch):
    stub = types.ModuleType("cv2")
    monkeypatch.setitem(sys.modules, "cv2", stub)
    with pytest.raises(ImportError, match="not OpenCV"):
        assert_cv2_is_real_opencv()

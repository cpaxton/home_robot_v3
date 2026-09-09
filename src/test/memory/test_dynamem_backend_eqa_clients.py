# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.

"""DynaMemBackend EQA wiring: reject query until VL clients are bound (deferred EQA)."""

import numpy as np
import pytest

from emet.controller.dynamem.eqa import _patch_images
from emet.memory.adapters import DynaMemBackend


def test_patch_images_handles_empty_retrieval():
    result = _patch_images(None, [], patch_size=(12, 8))
    assert isinstance(result, np.ndarray)
    assert result.shape == (8, 12, 3)
    assert result.dtype == np.uint8


def test_query_answer_rejects_when_image_description_client_is_none():
    class _VM:
        run_eqa = True
        image_description_client = None
        eqa_client = object()

        def query_answer(self, question, xyt, planner):
            raise AssertionError("should not reach voxel map")

    with pytest.raises(NotImplementedError, match="not ready yet"):
        DynaMemBackend(_VM()).query_answer("where is the cup?", None, None)


def test_query_answer_rejects_when_eqa_client_is_none():
    class _VM:
        run_eqa = True
        image_description_client = object()
        eqa_client = None

        def query_answer(self, question, xyt, planner):
            raise AssertionError("should not reach voxel map")

    with pytest.raises(NotImplementedError, match="not ready yet"):
        DynaMemBackend(_VM()).query_answer("where is the cup?", None, None)

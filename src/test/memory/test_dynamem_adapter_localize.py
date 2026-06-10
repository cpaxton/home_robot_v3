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
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Unit tests for DynaMem memory adapter localize parsing."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from emet.memory.adapters import DynaMemBackend, _split_voxel_localize_result


def test_split_voxel_localize_result_none():
    target, debug = _split_voxel_localize_result(None)
    assert target is None
    assert debug == ""


def test_split_voxel_localize_result_tuple():
    pt = np.array([0.1, 0.2, 0.3])
    target, debug = _split_voxel_localize_result((pt, "ok"))
    assert np.allclose(target, pt)
    assert debug == "ok"


def test_dynamem_backend_localize_handles_bare_none():
    voxel = MagicMock()
    voxel.localize_text.return_value = None
    backend = DynaMemBackend(voxel)
    out = backend.localize_text("red cylinder")
    assert out.success is False
    assert out.point_xyz is None

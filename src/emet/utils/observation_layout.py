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

"""Shared layout rules for RGB/depth fields in ZMQ observation messages."""

from __future__ import annotations

import numpy as np


def rgb_height_width_for_zmq(rgb: np.ndarray) -> tuple[int, int]:
    """
    Return ``(rgb_height, rgb_width)`` for ZMQ observation keys.

    NumPy images are ``(H, W, C)``; ``rgb_height`` is the row count and ``rgb_width`` the column count.
    This must match a same-resolution depth map ``(H, W)`` so clients can build a pinhole grid
    consistent with ``depth`` (see ``pinhole_camera_from_intrinsics_and_depth`` in ``image.py``).
    """
    if rgb.ndim < 2:
        raise ValueError(f"rgb must be at least 2D, got shape {rgb.shape}")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    return height, width

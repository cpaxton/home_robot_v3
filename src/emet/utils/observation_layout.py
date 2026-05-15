# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared layout rules for RGB/depth fields in ZMQ observation messages."""

from __future__ import annotations

import numpy as np


def rgb_height_width_for_zmq(rgb: np.ndarray) -> tuple[int, int]:
    """
    Return ``(rgb_height, rgb_width)`` for ZMQ observation keys.

    NumPy images are ``(H, W, C)``; ``rgb_height`` is the row count and ``rgb_width`` the column count.
    This must match a same-resolution depth map ``(H, W)`` so clients can build a pinhole grid
    consistent with ``depth`` (see :meth:`emet.utils.image.Camera.from_K` / :meth:`emet.utils.image.Camera.depth_to_xyz`).
    """
    if rgb.ndim < 2:
        raise ValueError(f"rgb must be at least 2D, got shape {rgb.shape}")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    return height, width

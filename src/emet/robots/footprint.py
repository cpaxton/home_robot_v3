# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Footprint lives here so that emet.robots.base (and backends) can use it without importing
# emet.motion, which would pull in pinocchio/hppfcl (e.g. for emet serve mujoco --robot rby1).
from typing import Optional

import numpy as np
from scipy.ndimage import rotate as scipy_rotate


class Footprint:
    """Contains information about robot footprint. Returns numpy arrays (no torch)."""

    def __init__(
        self,
        length: float,
        width: float,
        length_offset: float = 0.0,
        width_offset: float = 0.0,
    ):
        self.length = length
        self.width = width
        self.length_offset = length_offset
        self.width_offset = width_offset

    def get_box(self) -> np.ndarray:
        """Get a 3d footprint box for visuals"""
        return np.array([self.length, self.width, 0.2])

    def get_mask(self, resolution: float, device: Optional[object] = None) -> np.ndarray:
        """Get a single mask for this robot as a boolean numpy array."""
        size = int(
            np.ceil(
                np.sqrt(
                    (self.width + abs(self.width_offset)) ** 2
                    + (self.length + abs(self.length_offset)) ** 2
                )
                / resolution
            )
        )
        width_px = int(np.ceil(self.width / resolution))
        length_px = int(np.ceil(self.length / resolution))
        l0_offset = int(np.floor(self.length_offset / resolution))
        l1_offset = int(np.ceil(self.length_offset / resolution))
        w0_offset = int(np.floor(self.width_offset / resolution))
        w1_offset = int(np.ceil(self.width_offset / resolution))
        mask = np.zeros((size, size), dtype=bool)
        center = size // 2
        if size % 2 == 0:
            size += 1
        else:
            w1_offset += 1
            l1_offset += 1
        x0 = center - (width_px // 2) + w0_offset
        x1 = center + (width_px // 2) + w1_offset
        y0 = center - (length_px // 2) + l0_offset
        y1 = center + (length_px // 2) + 1 + l1_offset
        mask[y0:y1, x0:x1] = True
        return mask

    def get_rotated_mask(
        self,
        resolution: float,
        angle_radians: float,
        device: Optional[object] = None,
    ) -> np.ndarray:
        """Get a rotated footprint mask for collision checking (numpy, order=0 for nearest)."""
        mask = self.get_mask(resolution, device).astype(np.float64)
        rotated = scipy_rotate(mask, np.rad2deg(angle_radians), order=0, reshape=False)
        return (rotated > 0.5).astype(bool)

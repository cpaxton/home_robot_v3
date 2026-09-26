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
#
# Footprint lives here so that emet.robots.base (and backends) can use it without importing
# emet.motion, which would pull in pinocchio/hppfcl (e.g. for emet serve mujoco --robot rby1).

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

    def get_mask(self, resolution: float, device: object | None = None) -> np.ndarray:
        """Get a single mask for this robot as a boolean numpy array."""
        size = int(
            np.ceil(
                np.sqrt((self.width + abs(self.width_offset)) ** 2 + (self.length + abs(self.length_offset)) ** 2)
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
        device: object | None = None,
    ) -> np.ndarray:
        """Get a rotated footprint mask for collision checking (numpy, order=0 for nearest)."""
        mask = self.get_mask(resolution, device).astype(np.float64)
        rotated = scipy_rotate(mask, np.rad2deg(angle_radians), order=0, reshape=False)
        return (rotated > 0.5).astype(bool)

    def get_conservative_rotated_mask(self, resolution: float, angle_radians: float) -> np.ndarray:
        """Grid-XY cells intersecting the physical rectangle (including offsets).

        Separating-axis tests avoid clipping rotated corners and nearest-neighbor
        rasterization holes. Rows are world X, columns world Y; length is body X.
        Unknown/occupied cells touched by the footprint remain collision evidence.
        """
        values = [resolution, angle_radians, self.length, self.width, self.length_offset, self.width_offset]
        if not np.isfinite(values).all() or min(resolution, self.length, self.width) <= 0:
            raise ValueError("Finite positive footprint dimensions and resolution required")
        radius = np.hypot(self.length / 2 + abs(self.length_offset), self.width / 2 + abs(self.width_offset))
        cells = int(np.ceil(radius / resolution + np.sqrt(2) / 2))
        xy = np.arange(-cells, cells + 1) * resolution
        x, y = np.meshgrid(xy, xy, indexing="ij")
        c, s = np.cos(angle_radians), np.sin(angle_radians)
        x = x - (c * self.length_offset - s * self.width_offset)
        y = y - (s * self.length_offset + c * self.width_offset)
        half = resolution / 2
        projected_cell = half * (abs(c) + abs(s))
        return (
            (abs(c * x + s * y) <= self.length / 2 + projected_cell)
            & (abs(-s * x + c * y) <= self.width / 2 + projected_cell)
            & (abs(x) <= abs(c) * self.length / 2 + abs(s) * self.width / 2 + half)
            & (abs(y) <= abs(s) * self.length / 2 + abs(c) * self.width / 2 + half)
        )

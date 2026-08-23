# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Pinhole K updates when image pixels are transformed (flip / rot90) in lockstep with depth/RGB."""

from __future__ import annotations

import numpy as np

__all__ = [
    "apply_pinhole_pixel_ops",
    "chain_pinhole_K_pixel_ops",
    "scale_pinhole_K",
]


def apply_pinhole_pixel_ops(image: np.ndarray, ops: tuple[str, ...] | list[str]) -> np.ndarray:
    """Apply the same ``np.flipud`` / ``np.rot90`` sequence to an HxW or HxW? array."""
    out = np.asarray(image)
    for op in ops:
        if op == "flipud":
            out = np.flipud(out)
        elif op == "rot90_cw":
            out = np.rot90(out, -1)
        else:
            raise ValueError(f"unknown pinhole pixel op: {op!r} (use flipud, rot90_cw)")
    return out.copy()


def chain_pinhole_K_pixel_ops(
    K: np.ndarray, height: int, width: int, ops: tuple[str, ...] | list[str]
) -> tuple[np.ndarray, int, int]:
    """Chain homographies H @ K on intrinsics for pixel ops matching :func:`apply_pinhole_pixel_ops`.

    ``height`` / ``width`` are **before** any op (rows / cols). Returns ``K_out`` and final ``(height, width)``.
    """
    Kh = int(height)
    Kw = int(width)
    Kc = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    for op in ops:
        if op == "flipud":
            Hm = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, Kh - 1.0], [0.0, 0.0, 1.0]])
            Kc = Hm @ Kc
        elif op == "rot90_cw":
            Hm = np.array([[0.0, -1.0, Kh - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            Kc = Hm @ Kc
            Kh, Kw = Kw, Kh
        else:
            raise ValueError(f"unknown pinhole pixel op: {op!r} (use flipud, rot90_cw)")
    return Kc, Kh, Kw


def scale_pinhole_K(K: np.ndarray, old_width: int, old_height: int, new_width: int, new_height: int) -> np.ndarray:
    """Resize focal lengths and principal point after cv2.resize-like scaling."""
    sx = float(new_width) / float(old_width)
    sy = float(new_height) / float(old_height)
    K2 = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2

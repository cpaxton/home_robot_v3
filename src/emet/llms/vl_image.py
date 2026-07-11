# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Downsample RGB frames before VL / detector inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def downsample_rgb_hwc(
    rgb: np.ndarray | Image.Image,
    *,
    max_side: int = 512,
    max_pixels: int = 0,
) -> np.ndarray:
    """Return HxWx3 uint8 RGB, optionally capped by longest side and/or total pixels.

    ``max_side`` / ``max_pixels`` of ``0`` or less disable that constraint.
    Aspect ratio is preserved. No-op when the image already fits.
    """
    if isinstance(rgb, Image.Image):
        arr = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            if arr.size and float(np.nanmax(arr)) <= 1.0 + 1e-6:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise ValueError(f"rgb must be HxWx3; got {arr.shape}")
        arr = arr[:, :, :3]

    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h <= 0 or w <= 0:
        return arr

    scale = 1.0
    ms = int(max_side or 0)
    if ms > 0:
        long_side = max(h, w)
        if long_side > ms:
            scale = min(scale, ms / float(long_side))

    mp = int(max_pixels or 0)
    if mp > 0 and h * w * (scale**2) > mp:
        scale = min(scale, (mp / float(h * w)) ** 0.5)

    if scale >= 1.0 - 1e-9:
        return np.ascontiguousarray(arr)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil = Image.fromarray(arr, mode="RGB")
    resized = pil.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def eqa_vl_image_kwargs(eqa_cfg: dict[str, Any] | None) -> dict[str, int]:
    """Constructor kwargs for VL clients from ``eqa:`` YAML."""
    cfg = eqa_cfg if isinstance(eqa_cfg, dict) else {}
    max_side = int(cfg.get("vl_image_max_side", 512) or 0)
    max_pixels = int(cfg.get("vl_image_max_pixels", 0) or 0)
    return {"image_max_side": max(0, max_side), "image_max_pixels": max(0, max_pixels)}

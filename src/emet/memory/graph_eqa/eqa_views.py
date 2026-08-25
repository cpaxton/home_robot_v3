# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Helpers for EQA view recall: detector crops and same-obs look budgets.

The graph is an index. These helpers decide which stored RGB to show the VLM and
when a look at one observation has already been spent (so the agent looks
somewhere else instead of orbiting).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# After this many navigations to the same obs_id, pick a different view.
EQA_SAME_OBS_MAX_VISITS = 3
# A hop shorter than this is not a new look at a new place.
EQA_SAME_OBS_PROGRESS_M = 1.0
# Detector boxes covering most of the frame are full-frame, not instance crops.
EQA_CROP_MAX_AREA_FRACTION = 0.85
EQA_CROP_PADDING_FRAC = 0.12


def rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    """Return H×W×3 uint8 RGB."""
    img = np.asarray(rgb)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {getattr(rgb, 'shape', None)}")
    img = img[:, :, :3]
    if img.dtype != np.uint8:
        peak = float(np.nanmax(img)) if img.size else 0.0
        if peak <= 1.0 + 1e-6:
            img = np.clip(img * 255.0, 0, 255)
        else:
            img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
    return np.ascontiguousarray(img)


def bbox_area_fraction(bbox_xyxy: Sequence[int], image_hw: tuple[int, int]) -> float:
    h, w = int(image_hw[0]), int(image_hw[1])
    if h <= 0 or w <= 0:
        return 1.0
    x0, y0, x1, y1 = (int(bbox_xyxy[0]), int(bbox_xyxy[1]), int(bbox_xyxy[2]), int(bbox_xyxy[3]))
    bw = max(0, min(x1, w) - max(0, x0))
    bh = max(0, min(y1, h) - max(0, y0))
    return float(bw * bh) / float(h * w)


def crop_rgb_tight_bbox(
    rgb: np.ndarray,
    bbox_xyxy: Sequence[int],
    *,
    padding_frac: float = EQA_CROP_PADDING_FRAC,
    max_area_fraction: float = EQA_CROP_MAX_AREA_FRACTION,
) -> np.ndarray | None:
    """Crop ``rgb`` to a detector box, or ``None`` when the box is missing / full-frame."""
    if bbox_xyxy is None or len(bbox_xyxy) != 4:
        return None
    img = rgb_uint8(rgb)
    h, w = img.shape[:2]
    if bbox_area_fraction(bbox_xyxy, (h, w)) > float(max_area_fraction):
        return None
    x0, y0, x1, y1 = (int(bbox_xyxy[0]), int(bbox_xyxy[1]), int(bbox_xyxy[2]), int(bbox_xyxy[3]))
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    bw = x1 - x0
    bh = y1 - y0
    pad_x = int(bw * float(padding_frac))
    pad_y = int(bh * float(padding_frac))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


def eqa_look_is_spent(
    dists_m: Sequence[float],
    *,
    nav_attempts: int = 0,
    max_visits: int = EQA_SAME_OBS_MAX_VISITS,
    progress_m: float = EQA_SAME_OBS_PROGRESS_M,
) -> bool:
    """True when this observation has already been inspected enough.

    Sub-meter hops after a real approach (or three visits) are not a new look.
    The first short hop still counts so a spawn next to the object can capture RGB.
    """
    dists = [float(d) for d in dists_m if d is not None]
    visits = max(len(dists), int(nav_attempts or 0))
    if visits >= int(max_visits):
        return True
    if len(dists) >= 2 and float(dists[-1]) < float(progress_m):
        return True
    return False


def tightest_node_crop(nodes: Sequence[Any], rgb: np.ndarray) -> np.ndarray | None:
    """Best instance crop among graph nodes that share this observation RGB."""
    img = rgb_uint8(rgb)
    h, w = img.shape[:2]
    ranked: list[tuple[float, np.ndarray]] = []
    for node in nodes:
        bbox = getattr(node, "bbox_xyxy", None)
        crop = crop_rgb_tight_bbox(img, bbox) if bbox is not None else None
        if crop is None:
            continue
        frac = bbox_area_fraction(bbox, (h, w))
        ranked.append((frac, crop))
    if not ranked:
        return None
    ranked.sort(key=lambda t: t[0])
    return ranked[0][1]

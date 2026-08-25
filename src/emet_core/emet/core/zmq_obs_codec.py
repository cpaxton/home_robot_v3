# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""ZMQ full-observation image alias compression (slim wire format)."""

from __future__ import annotations

from typing import Any

import numpy as np

import emet.utils.compression as compression

# Legacy keys kept for Stretch / Habitat / older clients; canonical keys are the Mars bridge names.
ZMQ_IMAGE_ALIASES: tuple[tuple[str, str], ...] = (
    ("rgb", "head_cam_left/image"),
    ("rgb_right", "head_cam_right/image"),
    ("rgb_tertiary", "ee_cam/image"),
)


def expand_zmq_obs_image_aliases(obs: dict[str, Any]) -> None:
    """Fill missing legacy/canonical JPEG keys with the paired alias (in-place, shared refs)."""
    for legacy, canonical in ZMQ_IMAGE_ALIASES:
        leg = obs.get(legacy)
        canon = obs.get(canonical)
        if leg is None and canon is not None:
            obs[legacy] = canon
        elif canon is None and leg is not None:
            obs[canonical] = leg


def slim_zmq_obs_images(obs: dict[str, Any]) -> None:
    """Drop duplicate legacy JPEG keys when canonical keys are present (in-place)."""
    for legacy, canonical in ZMQ_IMAGE_ALIASES:
        if legacy not in obs:
            continue
        if canonical not in obs:
            obs[canonical] = obs.pop(legacy)
            continue
        if obs[legacy] is obs[canonical]:
            del obs[legacy]
            continue
        if obs[legacy] == obs[canonical]:
            del obs[legacy]


def _decode_jpg_if_needed(val: Any) -> np.ndarray | None:
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        arr = np.asarray(val)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            return np.ascontiguousarray(arr[..., :3])
        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[-1] != 3):
            try:
                return compression.from_jpg(arr)
            except Exception:
                return None
        return arr
    try:
        return compression.from_jpg(val)
    except Exception:
        return None


def decode_zmq_obs_images_inplace(obs: dict[str, Any]) -> bool:
    """Expand aliases and JPEG-decode RGB streams. Returns False when primary RGB is missing."""
    expand_zmq_obs_image_aliases(obs)
    rgb = _decode_jpg_if_needed(obs.get("rgb"))
    if rgb is None:
        return False
    obs["rgb"] = rgb
    for key in ("rgb_right", "rgb_tertiary"):
        if key in obs and obs[key] is not None:
            decoded = _decode_jpg_if_needed(obs[key])
            if decoded is not None:
                obs[key] = decoded
    for key in ("head_cam_left/image", "head_cam_right/image", "ee_cam/image"):
        if key in obs and obs[key] is not None and not isinstance(obs[key], np.ndarray):
            decoded = _decode_jpg_if_needed(obs[key])
            if decoded is not None:
                obs[key] = decoded
    return True

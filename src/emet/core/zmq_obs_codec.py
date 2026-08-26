# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""ZMQ full-observation slim wire helpers (image aliases, lidar dtype)."""

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


def slim_zmq_obs_lidar(obs: dict[str, Any]) -> None:
    """Cast ``lidar_points`` to float32 Nx2 (halves wire size vs float64)."""
    pts = obs.get("lidar_points")
    if pts is None:
        return
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return
    obs["lidar_points"] = np.ascontiguousarray(arr)


def slim_zmq_obs(obs: dict[str, Any]) -> None:
    """Apply all slim wire-format transforms (images + lidar)."""
    slim_zmq_obs_images(obs)
    slim_zmq_obs_lidar(obs)


def full_obs_has_wire_images(obs: dict[str, Any]) -> bool:
    """True when any canonical or legacy JPEG/WebP image key is present."""
    for key in (
        "head_cam_left/image",
        "head_cam_right/image",
        "ee_cam/image",
        "rgb",
        "rgb_right",
        "rgb_tertiary",
    ):
        if obs.get(key) is not None:
            return True
    return False


_SERVO_TO_FULL_IMAGE: tuple[tuple[str, str], ...] = (
    ("head_cam_left/color_image", "head_cam_left/image"),
    ("head_cam_right/color_image", "head_cam_right/image"),
    ("ee_cam/color_image", "ee_cam/image"),
)

_SERVO_TO_FULL_K: tuple[tuple[str, str], ...] = (
    ("head_cam_left/color_camera_K", "camera_K"),
    ("head_cam_right/color_camera_K", "camera_K_right"),
    ("ee_cam/color_camera_K", "camera_K_tertiary"),
)


def merge_servo_images_into_full_obs(full_obs: dict[str, Any], servo: dict[str, Any]) -> bool:
    """Copy scaled JPEG keys from a 4404 servo dict into a metadata-only 4401 obs (in-place)."""
    if not servo:
        return False
    merged = False
    for servo_key, full_key in _SERVO_TO_FULL_IMAGE:
        if full_obs.get(full_key) is not None:
            continue
        blob = servo.get(servo_key)
        if blob is None:
            continue
        full_obs[full_key] = blob
        merged = True
        shape_key = servo_key.replace("/color_image", "/color_image/shape")
        if shape_key in servo:
            full_obs[full_key.replace("/image", "/image/shape")] = servo[shape_key]
        scale_key = servo_key.replace("/color_image", "/image_scaling")
        if scale_key in servo:
            full_obs[full_key.replace("/image", "/image_scaling")] = servo[scale_key]
    for servo_k, full_k in _SERVO_TO_FULL_K:
        if full_obs.get(full_k) is None and servo.get(servo_k) is not None:
            full_obs[full_k] = servo[servo_k]
    if merged:
        expand_zmq_obs_image_aliases(full_obs)
    return merged


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
        try:
            return compression.from_webp(val)
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

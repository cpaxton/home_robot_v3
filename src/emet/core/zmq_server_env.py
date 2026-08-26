# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Environment toggles for ZMQ bridge publish rate, image scaling, and wire format."""

from __future__ import annotations

import os


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_float(name: str, default: float | None = None) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def zmq_send_period_s(env_name: str) -> float:
    """Return minimum seconds between publishes when ``env_name`` sets a positive Hz cap."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return 0.0
    hz = float(raw)
    return 0.0 if hz <= 0 else 1.0 / hz


def resolve_zmq_image_scaling(default: float = 0.5) -> float:
    """``EMET_ZMQ_IMAGE_SCALING`` override for head stereo downscale before JPEG."""
    val = _env_float("EMET_ZMQ_IMAGE_SCALING", default)
    if val is None or val <= 0:
        return default
    return min(float(val), 1.0)


def resolve_zmq_ee_image_scaling(default: float = 0.5) -> float:
    """``EMET_ZMQ_EE_IMAGE_SCALING`` override for EE camera downscale before JPEG."""
    val = _env_float("EMET_ZMQ_EE_IMAGE_SCALING", default)
    if val is None or val <= 0:
        return default
    return min(float(val), 1.0)


def resolve_zmq_jpeg_quality(default: int = 90) -> int:
    raw = os.environ.get("EMET_ZMQ_JPEG_QUALITY", "").strip()
    if not raw:
        return default
    return max(1, min(100, int(raw)))


def zmq_obs_include_images(default: bool = True) -> bool:
    """When false, full obs (4401) omits JPEG keys (metadata-only mode)."""
    return _env_truthy("EMET_ZMQ_OBS_INCLUDE_IMAGES", default=default)


def zmq_servo_include_images(default: bool = True) -> bool:
    """When false, servo (4404) omits JPEG keys (poses/joints only)."""
    return _env_truthy("EMET_ZMQ_SERVO_INCLUDE_IMAGES", default=default)


def zmq_use_webp_images() -> bool:
    return _env_truthy("EMET_ZMQ_WEBP_IMAGES", default=False)


def zmq_video_rtsp_enabled() -> bool:
    return _env_truthy("EMET_MARS_VIDEO_RTSP", default=False)


def zmq_video_rtsp_port(default: int = 8554) -> int:
    raw = os.environ.get("EMET_MARS_VIDEO_RTSP_PORT", "").strip()
    if not raw:
        return default
    return int(raw)


def zmq_video_rtsp_host() -> str | None:
    """Optional advertised host for ``capabilities.video_streams`` URLs."""
    raw = os.environ.get("EMET_MARS_VIDEO_RTSP_HOST", "").strip()
    return raw or None


def zmq_h264_port(default: int = 4405) -> int:
    raw = os.environ.get("EMET_ZMQ_H264_PORT", "").strip()
    if not raw:
        return default
    return int(raw)


def zmq_h264_enabled() -> bool:
    return _env_truthy("EMET_ZMQ_H264", default=False)

# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Explicit simulator RGB wire profile; lossless pixels for metric RGBD consumers."""

import os

import cv2
import numpy as np

from emet.utils import compression


def sim_rgb_codec():
    codec = os.environ.get("EMET_SIM_RGB_CODEC", "jpeg").strip().lower()
    if codec not in {"jpeg", "png"}:
        raise ValueError("unsupported_sim_rgb_codec")
    return codec


def encode_sim_rgb(image, *, codec=None):
    codec = sim_rgb_codec() if codec is None else codec
    if codec == "jpeg":
        return compression.to_jpg(image)
    if codec != "png":
        raise ValueError("unsupported_sim_rgb_codec")
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("invalid_rgb_pixels")
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("rgb_encoding_failed")
    return encoded

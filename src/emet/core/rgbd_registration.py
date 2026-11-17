# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Lossless original sensor evidence for diagnosing RGB/depth correspondence.

JSON-compatible, opt-in image packets preserve original grids and acquisition
intervals. They contain sensor pixels/calibration, never scene IDs or contacts.
Reprojection provenance establishes a measurement's origin, not its ownership.
"""

import base64
import zlib

import numpy as np

from emet.simulation.depth_registration import register_depth


def _pack(array):
    value = np.ascontiguousarray(array)
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "zlib_base64": base64.b64encode(zlib.compress(value.tobytes(), level=1)).decode("ascii"),
    }


def _unpack(packet, dtype, ndim):
    if not isinstance(packet, dict) or set(packet) != {"dtype", "shape", "zlib_base64"}:
        raise ValueError("invalid_rgbd_registration_packet")
    shape = packet["shape"]
    if (
        packet["dtype"] != np.dtype(dtype).str
        or not isinstance(shape, list)
        or len(shape) != ndim
        or any(type(n) is not int or not 0 < n <= 4096 for n in shape)
        or (ndim == 3 and shape[-1] != 3)
    ):
        raise ValueError("invalid_rgbd_registration_packet")
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(packet["zlib_base64"], validate=True), expected + 1)
    if len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("invalid_rgbd_registration_packet")
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def encode_registration(
    *, rgb, raw_depth, registered_depth, depth_k, rgb_k, rgb_from_depth, rgb_interval_ns, depth_interval_ns, clock_id
):
    rgb, raw, aligned = np.asarray(rgb), np.asarray(raw_depth, np.float32), np.asarray(registered_depth, np.float32)
    if rgb.dtype != np.uint8 or rgb.shape != (*aligned.shape, 3) or raw.ndim != 2:
        raise ValueError("invalid_rgbd_registration_packet")
    return {
        "schema_version": 1,
        "source": "original_sensor_registration",
        "grid": "native_rgb_optical_before_clockwise_head_rotation",
        "clock_id": clock_id,
        "rgb_interval_ns": list(rgb_interval_ns),
        "depth_interval_ns": list(depth_interval_ns),
        "depth_k": np.asarray(depth_k, float).tolist(),
        "rgb_k": np.asarray(rgb_k, float).tolist(),
        "rgb_from_depth": np.asarray(rgb_from_depth, float).tolist(),
        "rgb": _pack(rgb),
        "raw_depth": _pack(raw),
        "registered_depth": _pack(aligned),
    }


def decode_registration(packet):
    """Validate and reconstruct original rays; callers still bind the live ticket."""
    if (
        not isinstance(packet, dict)
        or packet.get("schema_version") != 1
        or packet.get("source") != "original_sensor_registration"
        or packet.get("grid") != "native_rgb_optical_before_clockwise_head_rotation"
        or not isinstance(packet.get("clock_id"), str)
        or not packet["clock_id"]
    ):
        raise ValueError("invalid_rgbd_registration_packet")
    for name in ("rgb_interval_ns", "depth_interval_ns"):
        interval = packet.get(name)
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(type(n) is not int for n in interval)
            or not 0 < interval[0] <= interval[1]
        ):
            raise ValueError("invalid_rgbd_registration_packet")
    try:
        rgb, raw, aligned = (
            _unpack(packet[name], dtype, ndim)
            for name, dtype, ndim in (
                ("rgb", np.uint8, 3),
                ("raw_depth", np.float32, 2),
                ("registered_depth", np.float32, 2),
            )
        )
        kd, kr, transform = (np.asarray(packet[name], float) for name in ("depth_k", "rgb_k", "rgb_from_depth"))
        if rgb.shape != (*aligned.shape, 3):
            raise ValueError("invalid_rgbd_registration_packet")
        reproduced, provenance = register_depth(raw, kd, kr, transform, aligned.shape, return_provenance=True)
        if not np.array_equal(reproduced, aligned, equal_nan=True):
            raise ValueError("rgbd_registration_reprojection_mismatch")
    except (KeyError, TypeError, zlib.error) as error:
        raise ValueError("invalid_rgbd_registration_packet") from error
    return {
        "rgb": rgb,
        "raw_depth": raw,
        "registered_depth": aligned,
        "depth_k": kd,
        "rgb_k": kr,
        "rgb_from_depth": transform,
        **provenance,
    }


def source_points_rgb(decoded):
    """Winning original depth endpoints, in native RGB optical coordinates.

    Nearest registered pixel centres do not define the original rays. Holes
    stay NaN; no endpoint is interpolated and ownership is not inferred.
    """
    source = decoded["source_pixel_index"]
    valid = source >= 0
    raw, transform = decoded["raw_depth"], decoded["rgb_from_depth"]
    y, x = np.divmod(source[valid], raw.shape[1])
    rays = np.stack((x, y, np.ones(len(x))), axis=-1) @ np.linalg.inv(decoded["depth_k"]).T
    points = (rays * raw[y, x, None]) @ transform[:3, :3].T + transform[:3, 3]
    result = np.full((*source.shape, 3), np.nan)
    result[valid] = points
    return result

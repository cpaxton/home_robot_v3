# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Original sensor grids survive transport without loss or timestamp renewal."""

import json
from copy import deepcopy

import numpy as np
import pytest

from emet.core.rgbd_registration import decode_registration, encode_registration, source_points_rgb
from emet.simulation.depth_registration import register_depth


def packet():
    raw = np.random.default_rng(5).uniform(0.5, 2.0, (8, 12)).astype(np.float32)
    raw[1, 1] = 0.0
    rgb = np.random.default_rng(6).integers(0, 255, (8, 12, 3), dtype=np.uint8)
    transform = np.eye(4)
    transform[0, 3] = 0.03
    aligned = register_depth(raw, np.eye(3), np.eye(3), transform, (8, 12))
    return (
        encode_registration(
            rgb=rgb,
            raw_depth=raw,
            registered_depth=aligned,
            depth_k=np.eye(3),
            rgb_k=np.eye(3),
            rgb_from_depth=transform,
            rgb_interval_ns=(100, 110),
            depth_interval_ns=(100, 110),
            clock_id="original-clock",
        ),
        rgb,
        raw,
        aligned,
    )


def test_json_roundtrip_preserves_exact_original_grids_and_reprojection():
    value, rgb, raw, aligned = packet()
    decoded = decode_registration(json.loads(json.dumps(value)))
    for key, expected in [("rgb", rgb), ("raw_depth", raw), ("registered_depth", aligned)]:
        np.testing.assert_array_equal(decoded[key], expected)
        assert not decoded[key].flags.writeable
    assert decoded["source_pixel_index"].shape == aligned.shape
    assert value["rgb_interval_ns"] == [100, 110]


def test_source_endpoints_keep_original_rays_and_registration_holes():
    value, _, raw, _ = packet()
    decoded = decode_registration(value)
    points = source_points_rgb(decoded)
    source = decoded["source_pixel_index"]
    valid = source >= 0
    y, x = np.divmod(source[valid], raw.shape[1])
    expected = np.column_stack((x, y, np.ones(len(x)))) * raw[y, x, None]
    expected[:, 0] += 0.03
    np.testing.assert_allclose(points[valid], expected, rtol=0, atol=1e-12)
    assert np.isnan(points[~valid]).all()
    # A rounded RGB pixel ray is different despite using the same Z.
    rows, cols = np.where(valid)
    reconstructed = np.column_stack((cols, rows, np.ones(len(rows)))) * points[valid, 2, None]
    assert np.max(np.linalg.norm(points[valid] - reconstructed, axis=1)) > 0.01


@pytest.mark.parametrize(
    "change", ["version", "grid", "clock", "interval", "payload", "shape", "channels", "reprojection"]
)
def test_malformed_or_inconsistent_packets_reject(change):
    value, _, _, _ = packet()
    value = deepcopy(value)
    if change == "version":
        value["schema_version"] = 2
    elif change == "grid":
        value["grid"] = "unknown"
    elif change == "clock":
        value["clock_id"] = ""
    elif change == "interval":
        value["depth_interval_ns"] = [110, 100]
    elif change == "payload":
        value["rgb"]["zlib_base64"] = "broken"
    elif change == "shape":
        value["raw_depth"]["shape"] = [4097, 12]
    elif change == "channels":
        value["rgb"]["shape"][-1] = 4096
    else:
        value["rgb_from_depth"][0][3] += 2.0
    with pytest.raises(ValueError):
        decode_registration(value)

# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""The metric RGBD transport profile preserves RGB pixels through the real decoder."""

import numpy as np
import pytest

from emet.core.zmq_obs_codec import decode_zmq_obs_images_inplace
from emet.simulation.rgb_encoding import encode_sim_rgb, sim_rgb_codec
from emet.utils import compression


def test_lossless_rgb_roundtrip_through_full_observation_codec():
    rgb = np.random.default_rng(19).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    message = {"rgb": encode_sim_rgb(rgb, codec="png")}
    assert decode_zmq_obs_images_inplace(message)
    np.testing.assert_array_equal(message["rgb"], rgb)
    assert decode_zmq_obs_images_inplace(message)  # Already decoded is unchanged.
    np.testing.assert_array_equal(message["rgb"], rgb)


def test_default_retains_existing_jpeg_transport(monkeypatch):
    monkeypatch.delenv("EMET_SIM_RGB_CODEC", raising=False)
    rgb = np.random.default_rng(20).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    assert sim_rgb_codec() == "jpeg"
    np.testing.assert_array_equal(encode_sim_rgb(rgb), compression.to_jpg(rgb))


def test_profile_is_explicit_and_rejects_unknown_values(monkeypatch):
    monkeypatch.setenv("EMET_SIM_RGB_CODEC", "png")
    assert sim_rgb_codec() == "png"
    monkeypatch.setenv("EMET_SIM_RGB_CODEC", "unknown")
    with pytest.raises(ValueError, match="unsupported_sim_rgb_codec"):
        sim_rgb_codec()
    with pytest.raises(ValueError, match="invalid_rgb_pixels"):
        encode_sim_rgb(np.ones((8, 8), np.float32), codec="png")

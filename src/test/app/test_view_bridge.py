# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for view_bridge image key resolution."""

from emet.app.view_bridge import _find_image_blob


def test_find_image_blob_prefers_full_obs_slim_keys() -> None:
    obs = {"head_cam_left/image": b"jpeg-bytes", "step": 1}
    servo = {"head_cam_left/color_image": b"servo-bytes"}
    assert _find_image_blob(obs, servo, keys=("head_cam_left/color_image", "head_cam_left/image", "rgb")) == b"jpeg-bytes"


def test_find_image_blob_falls_back_to_servo() -> None:
    obs = {"step": 1}
    servo = {"head_cam_left/color_image": b"servo-bytes"}
    assert _find_image_blob(obs, servo, keys=("head_cam_left/color_image", "head_cam_left/image", "rgb")) == b"servo-bytes"

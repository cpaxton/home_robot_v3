# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import numpy as np

import emet.utils.compression as compression
from emet.controller.generic_zmq_client import get_observation_from_zmq_dict
from emet.core.zmq_obs_codec import (
    decode_zmq_obs_images_inplace,
    expand_zmq_obs_image_aliases,
    full_obs_has_wire_images,
    merge_servo_images_into_full_obs,
    slim_zmq_obs,
    slim_zmq_obs_images,
    slim_zmq_obs_lidar,
)


def test_slim_zmq_obs_images_drops_duplicate_legacy_keys():
    jpg = b"jpeg-bytes"
    obs = {
        "rgb": jpg,
        "head_cam_left/image": jpg,
        "rgb_right": jpg,
        "head_cam_right/image": jpg,
    }
    slim_zmq_obs_images(obs)
    assert "rgb" not in obs
    assert "rgb_right" not in obs
    assert obs["head_cam_left/image"] is jpg
    assert obs["head_cam_right/image"] is jpg


def test_expand_zmq_obs_image_aliases_fills_legacy_from_canonical():
    jpg = b"left"
    obs = {"head_cam_left/image": jpg}
    expand_zmq_obs_image_aliases(obs)
    assert obs["rgb"] is jpg


def test_decode_zmq_obs_images_inplace_accepts_slim_canonical_only():
    rgb = np.full((4, 6, 3), 7, dtype=np.uint8)
    obs = {
        "head_cam_left/image": compression.to_jpg(rgb),
        "head_cam_right/image": compression.to_jpg(rgb),
        "ee_cam/image": compression.to_jpg(rgb),
    }
    assert decode_zmq_obs_images_inplace(obs)
    assert obs["rgb"].shape == (4, 6, 3)
    assert int(obs["rgb"][0, 0, 0]) == 7


def test_get_observation_from_zmq_dict_accepts_slim_head_cam_left_only():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    ee = np.ones((2, 2, 3), dtype=np.uint8) * 200
    raw = {
        "head_cam_left/image": compression.to_jpg(rgb),
        "ee_cam/image": compression.to_jpg(ee),
        "camera_K_tertiary": np.eye(3, dtype=np.float64),
        "gps": np.zeros(2),
        "compass": np.zeros(1),
    }
    obs = get_observation_from_zmq_dict(raw)
    assert obs is not None
    assert obs.rgb.shape == (2, 2, 3)
    assert obs.ee_rgb is not None and int(obs.ee_rgb.max()) == 200


def test_slim_zmq_obs_lidar_casts_to_float32():
    pts = np.ones((360, 2), dtype=np.float64)
    obs = {"lidar_points": pts}
    slim_zmq_obs_lidar(obs)
    assert obs["lidar_points"].dtype == np.float32
    assert obs["lidar_points"].shape == (360, 2)
    assert obs["lidar_points"].nbytes == 360 * 2 * 4


def test_slim_zmq_obs_applies_images_and_lidar():
    jpg = b"jpeg"
    pts = np.zeros((4, 2), dtype=np.float64)
    obs = {"rgb": jpg, "head_cam_left/image": jpg, "lidar_points": pts}
    slim_zmq_obs(obs)
    assert "rgb" not in obs
    assert obs["lidar_points"].dtype == np.float32


def test_merge_servo_images_into_full_obs_fills_metadata_only_obs():
    rgb = np.full((4, 6, 3), 9, dtype=np.uint8)
    jpg = compression.to_jpg(rgb)
    servo = {
        "head_cam_left/color_image": jpg,
        "head_cam_left/color_image/shape": (4, 6, 3),
        "head_cam_left/image_scaling": 0.5,
        "head_cam_left/color_camera_K": np.eye(3),
    }
    full = {"lidar_points": np.zeros((8, 2), dtype=np.float32), "gps": np.zeros(2)}
    assert not full_obs_has_wire_images(full)
    assert merge_servo_images_into_full_obs(full, servo)
    assert full_obs_has_wire_images(full)
    assert decode_zmq_obs_images_inplace(full)
    assert int(full["rgb"][0, 0, 0]) == 9


def test_full_obs_has_wire_images_false_without_jpeg_keys():
    assert not full_obs_has_wire_images({"gps": np.zeros(2)})

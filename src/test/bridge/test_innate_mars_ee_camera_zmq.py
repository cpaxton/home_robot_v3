# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.

import numpy as np

import emet.utils.compression as compression
from emet.controller.generic_zmq_client import (
    _decode_servo_message_to_observations,
    enrich_zmq_observation_ee_fields,
    get_observation_from_zmq_dict,
)


def test_decode_innate_mars_servo_message():
    rgb = np.full((48, 64, 3), 120, dtype=np.uint8)
    ee = np.full((48, 64, 3), 200, dtype=np.uint8)
    # Principal point at image center so K-align-to-RGB is a no-op for this resolution.
    ee_k = np.array(
        [[2.0, 0.0, 31.5], [0.0, 2.0, 23.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    msg = {
        "head_cam_left/color_image": compression.to_jpg(rgb),
        "head_cam_left/color_camera_K": np.eye(3, dtype=np.float64),
        "ee_cam/color_image": compression.to_jpg(ee),
        "ee_cam/color_camera_K": ee_k,
        "ee_cam/pose": np.eye(4, dtype=np.float64),
        "joint_positions": np.zeros(10, dtype=np.float64),
        "step": 3,
    }
    obs = _decode_servo_message_to_observations(msg, None, None)
    assert obs is not None
    assert obs.rgb.shape == (48, 64, 3)
    assert obs.ee_rgb is not None and obs.ee_rgb.shape == (48, 64, 3)
    assert float(obs.ee_camera_K[0, 0]) == 2.0


def test_enrich_ee_from_bridge_full_obs():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    ee = np.ones((4, 4, 3), dtype=np.uint8) * 255
    raw = {
        "rgb": rgb,
        "ee_cam/image": compression.to_jpg(ee),
        "ee_cam/pose": np.eye(4, dtype=np.float64),
        "camera_K_tertiary": np.eye(3, dtype=np.float64) * 3.0,
    }
    obs = get_observation_from_zmq_dict(raw)
    assert obs is not None
    assert obs.ee_rgb is not None and int(obs.ee_rgb.max()) == 255
    assert float(obs.ee_camera_K[0, 0]) == 3.0


def test_enrich_zmq_observation_ee_fields_idempotent():
    ee = np.ones((2, 2, 3), dtype=np.uint8)
    msg = {"ee_cam/image": compression.to_jpg(ee)}
    enrich_zmq_observation_ee_fields(msg)
    assert msg["ee_rgb"] is not None
    first = msg["ee_rgb"]
    enrich_zmq_observation_ee_fields(msg)
    assert msg["ee_rgb"] is first

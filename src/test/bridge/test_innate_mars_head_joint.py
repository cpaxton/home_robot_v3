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

import json

import numpy as np
import pytest

pytest.importorskip("rclpy")

from innate_mars_bridge.remote.ros import InnateMarsRosInterface

from emet.robots.innate_mars.head_kinematics import (
    head_hinge_rad_from_base_head_tf,
    ros_head_deg_to_mjcf_rad,
)


def test_ros_head_deg_to_mjcf_rad_sign():
    assert ros_head_deg_to_mjcf_rad(10.0) == -np.deg2rad(10.0)


def test_head_hinge_from_identity_tf():
    assert head_hinge_rad_from_base_head_tf(np.eye(4)) == 0.0


def test_parse_head_position_json_variants():
    iface = InnateMarsRosInterface.__new__(InnateMarsRosInterface)

    class _Msg:
        def __init__(self, data: str):
            self.data = data

    assert iface._parse_head_position_message(_Msg('{"current_position": -10}')) == -10.0
    assert iface._parse_head_position_message(_Msg('{"position": 5.5}')) == 5.5
    assert iface._parse_head_position_message(_Msg("-12.5")) == -12.5


def test_head_callback_sets_mjcf_radians_with_sign_flip():
    iface = InnateMarsRosInterface.__new__(InnateMarsRosInterface)
    iface._head_lock = __import__("threading").Lock()
    iface._head_joint_rad = 0.0
    iface._head_topic_updated = False

    class _Msg:
        data = json.dumps({"current_position": 10.0})

    iface._head_callback(_Msg())
    assert iface._head_topic_updated is True
    np.testing.assert_allclose(iface._head_joint_rad, -np.deg2rad(10.0))


def test_get_head_joint_rad_prefers_tf_over_stale_topic():
    iface = InnateMarsRosInterface.__new__(InnateMarsRosInterface)
    iface._head_lock = __import__("threading").Lock()
    iface._head_joint_rad = 0.0
    iface._head_topic_updated = True
    nod = 0.12
    t = np.eye(4, dtype=np.float64)
    t[0, 0] = np.cos(nod)
    t[0, 2] = np.sin(nod)
    t[2, 0] = -np.sin(nod)
    t[2, 2] = np.cos(nod)
    expected = head_hinge_rad_from_base_head_tf(t)
    iface.get_frame_pose = lambda *args, **kwargs: t  # type: ignore[method-assign]
    np.testing.assert_allclose(iface.get_head_joint_rad(), expected)

# Copyright (c) Hello Robot, Inc.
# All rights reserved.

import math

import numpy as np

from innate_mars_bridge.remote.modules.nav_geometry import xyt_to_pose_stamped


def test_xyt_to_pose_stamped_odom_frame():
    msg = xyt_to_pose_stamped(np.array([1.0, 2.0, math.pi / 2]), "odom")
    assert msg.header.frame_id == "odom"
    assert abs(msg.pose.position.x - 1.0) < 1e-6
    assert abs(msg.pose.position.y - 2.0) < 1e-6
    # yaw = pi/2 => quat z,w
    assert abs(msg.pose.orientation.z - math.sin(math.pi / 4)) < 1e-5

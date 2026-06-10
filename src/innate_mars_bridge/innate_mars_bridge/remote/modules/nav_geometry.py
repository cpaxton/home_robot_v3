# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""SE(2) goal helpers (no ROS action imports — testable without nav2_msgs)."""

from __future__ import annotations

import math

import numpy as np
from geometry_msgs.msg import PoseStamped

from innate_mars_bridge.ros.utils import matrix_to_pose_msg


def xyt_to_pose_stamped(xyt: np.ndarray, frame_id: str) -> PoseStamped:
    """SE(2) goal as ``PoseStamped`` in ``frame_id``."""
    x, y, th = float(xyt[0]), float(xyt[1]), float(xyt[2])
    c, s = math.cos(th), math.sin(th)
    mat = np.eye(4, dtype=np.float64)
    mat[0, 0] = c
    mat[0, 1] = -s
    mat[1, 0] = s
    mat[1, 1] = c
    mat[0, 3] = x
    mat[1, 3] = y
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose = matrix_to_pose_msg(mat)
    return msg

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Client API for Innate Mars: observations (pose, proprioception, cameras)."""

from typing import Optional

import numpy as np

from innate_mars_bridge.constants import (
    EE_CAMERA_FRAME_ID,
    HEAD_LEFT_FRAME_ID,
    HEAD_RIGHT_FRAME_ID,
    MAP_FRAME,
    ODOM_FRAME,
)
from innate_mars_bridge.remote.ros import InnateMarsRosInterface


class InnateMarsClient:
    """Interface to Innate Mars robot observations: pose, proprioception, head and EE cameras."""

    def __init__(
        self,
        init_node: bool = True,
        ee_frame: Optional[str] = None,
        head_left_frame: Optional[str] = None,
        head_right_frame: Optional[str] = None,
        ee_camera_frame: Optional[str] = None,
        base_frame: str = ODOM_FRAME,
        verbose: bool = False,
    ):
        if init_node:
            import rclpy

            if not rclpy.ok():
                rclpy.init()
        self._ros = InnateMarsRosInterface(verbose=verbose)
        self._ee_frame = ee_frame or "ee_link"
        self._head_left_frame = head_left_frame or HEAD_LEFT_FRAME_ID
        self._head_right_frame = head_right_frame or HEAD_RIGHT_FRAME_ID
        self._ee_camera_frame = ee_camera_frame or EE_CAMERA_FRAME_ID
        self._base_frame = base_frame

    @property
    def base_pose_xyt(self) -> np.ndarray:
        """Base pose (x, y, theta) in odom frame."""
        return self._ros.get_base_pose_xyt()

    @property
    def base_pose_matrix(self) -> Optional[np.ndarray]:
        """Base pose as 4x4 in odom frame."""
        return self._ros.get_base_pose_matrix()

    def get_joint_state(self):
        """(positions, velocities) for arm joints."""
        return self._ros.get_joint_state()

    @property
    def ee_pose(self) -> Optional[np.ndarray]:
        """End-effector pose as 4x4 in base_frame (odom or map)."""
        return self._ros.get_frame_pose(self._ee_frame, base_frame=self._base_frame)

    @property
    def head_left_camera_pose(self) -> Optional[np.ndarray]:
        """Head left camera pose as 4x4 in base_frame."""
        return self._ros.get_frame_pose(self._head_left_frame, base_frame=self._base_frame)

    @property
    def head_right_camera_pose(self) -> Optional[np.ndarray]:
        """Head right camera pose as 4x4 in base_frame."""
        return self._ros.get_frame_pose(self._head_right_frame, base_frame=self._base_frame)

    @property
    def ee_camera_pose(self) -> Optional[np.ndarray]:
        """EE camera pose as 4x4 in base_frame."""
        return self._ros.get_frame_pose(self._ee_camera_frame, base_frame=self._base_frame)

    @property
    def head_left_cam(self):
        return self._ros.head_left_cam

    @property
    def head_right_cam(self):
        return self._ros.head_right_cam

    @property
    def ee_cam(self):
        return self._ros.ee_cam

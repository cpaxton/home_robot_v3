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
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""ROS2 interface for Innate Mars: joint state, odometry, TF, and cameras."""

from __future__ import annotations

import threading

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from innate_mars_bridge.constants import (
    ARM_STATE_TOPIC,
    EE_IMAGE_TOPIC,
    HEAD_LEFT_CAMERA_INFO_TOPIC,
    HEAD_LEFT_IMAGE_TOPIC,
    HEAD_RIGHT_CAMERA_INFO_TOPIC,
    HEAD_RIGHT_IMAGE_TOPIC,
    ODOM_FRAME,
    ODOM_TOPIC,
)
from innate_mars_bridge.ros.camera import RosCamera, RosCameraNoInfo
from innate_mars_bridge.ros.utils import matrix_from_pose_msg, to_matrix, transform_to_list


class InnateMarsRosInterface(Node):
    """Subscribes to Innate Mars topics: arm state, odom, and head/EE cameras."""

    def __init__(self, verbose: bool = False):
        super().__init__("innate_mars_ros_interface")
        self.verbose = verbose

        self._js_lock = threading.Lock()
        self._joint_positions = np.zeros(6)
        self._joint_velocities = np.zeros(6)
        self._joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

        self._odom_pose = None
        self._odom_lock = threading.Lock()

        self.tf2_buffer = Buffer()
        self.tf2_listener = TransformListener(self.tf2_buffer, self)

        # Subscribers
        self._joint_sub = self.create_subscription(JointState, ARM_STATE_TOPIC, self._joint_callback, 10)
        self._odom_sub = self.create_subscription(Odometry, ODOM_TOPIC, self._odom_callback, 10)

        # Cameras: head left, head right (with camera_info), EE (no camera_info on hardware)
        self.head_left_cam = RosCamera(
            self,
            HEAD_LEFT_IMAGE_TOPIC,
            camera_info_topic=HEAD_LEFT_CAMERA_INFO_TOPIC,
            rotations=0,
            verbose=verbose,
        )
        self.head_right_cam = RosCamera(
            self,
            HEAD_RIGHT_IMAGE_TOPIC,
            camera_info_topic=HEAD_RIGHT_CAMERA_INFO_TOPIC,
            rotations=0,
            verbose=verbose,
        )
        self.ee_cam = RosCameraNoInfo(
            self,
            EE_IMAGE_TOPIC,
            rotations=0,
            image_ext="",
            verbose=verbose,
        )

        self.get_logger().info("InnateMarsRosInterface: waiting for cameras...")
        self.head_left_cam.wait_for_image()
        self.head_right_cam.wait_for_image()
        self.ee_cam.wait_for_image()
        self.get_logger().info("InnateMarsRosInterface: all cameras ready.")

    def _joint_callback(self, msg: JointState):
        with self._js_lock:
            name_to_pos = dict(zip(msg.name, msg.position, strict=False))
            name_to_vel = dict(zip(msg.name, msg.velocity, strict=False)) if msg.velocity else {}
            for i, name in enumerate(self._joint_names):
                if name in name_to_pos:
                    self._joint_positions[i] = name_to_pos[name]
                if name in name_to_vel:
                    self._joint_velocities[i] = name_to_vel[name]

    def _odom_callback(self, msg: Odometry):
        with self._odom_lock:
            self._odom_pose = matrix_from_pose_msg(msg.pose.pose)

    def get_joint_state(self):
        """Returns (positions, velocities) for joint1..joint6."""
        with self._js_lock:
            return self._joint_positions.copy(), self._joint_velocities.copy()

    def get_base_pose_matrix(self):
        """Base pose as 4x4 matrix in odom frame (or None)."""
        with self._odom_lock:
            return self._odom_pose.copy() if self._odom_pose is not None else None

    def get_base_pose_xyt(self):
        """Base pose as (x, y, theta) in odom frame. Theta in radians."""
        mat = self.get_base_pose_matrix()
        if mat is None:
            return np.array([0.0, 0.0, 0.0])
        x, y = mat[0:2, 3]
        # yaw from rotation matrix
        theta = np.arctan2(mat[1, 0], mat[0, 0])
        return np.array([x, y, theta], dtype=np.float64)

    def get_frame_pose(self, frame: str, base_frame: str | None = None, timeout_s: float = 1.0):
        """Look up frame pose in base_frame. Returns 4x4 matrix or None."""
        from rclpy.duration import Duration
        from rclpy.time import Time

        if base_frame is None:
            base_frame = ODOM_FRAME
        try:
            stamped = self.tf2_buffer.lookup_transform(base_frame, frame, Time(), Duration(seconds=timeout_s))
            trans, rot = transform_to_list(stamped)
            return to_matrix(trans, rot)
        except TransformException:
            if self.verbose:
                self.get_logger().warn(f"TF lookup failed: {base_frame} -> {frame}")
            return None

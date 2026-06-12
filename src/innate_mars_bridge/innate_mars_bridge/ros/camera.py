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

"""ROS2 camera subscriber for Innate Mars topics."""

from __future__ import annotations

import threading

import numpy as np
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

from emet.utils.image import Camera
from innate_mars_bridge.ros.msg_numpy import image_to_numpy


def default_K_from_shape(height: int, width: int, fov_deg: float = 60.0) -> np.ndarray:
    """Build 3x3 intrinsics from image shape and approximate FOV (degrees)."""
    fx = width / (2 * np.tan(np.radians(fov_deg) / 2))
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


class RosCamera(Camera):
    """Subscribes to an image topic and optional camera_info; provides get() and get_K()."""

    def __init__(
        self,
        ros_client,
        image_topic: str,
        camera_info_topic: str | None = None,
        rotations: int = 0,
        image_ext: str = "/image_raw",
        verbose: bool = True,
        default_fov_deg: float = 60.0,
    ):
        self._ros_client = ros_client
        self.name = image_topic
        self.rotations = rotations
        self._img = None
        self._t = Time()
        self._lock = threading.Lock()
        self.default_fov_deg = default_fov_deg

        if camera_info_topic is None:
            camera_info_topic = image_topic.replace(image_ext, "/camera_info")

        self._info_sub = ros_client.create_subscription(
            CameraInfo, camera_info_topic, self._cam_info_callback, qos_profile_sensor_data
        )
        self.camera_info = None
        self.height = self.width = None
        self.K = None
        self.frame_id = "unknown"

        topic_name = image_topic if image_topic.endswith(image_ext) else image_topic + image_ext
        self._sub = ros_client.create_subscription(Image, topic_name, self._cb, qos_profile_sensor_data)
        if verbose:
            print("Subscribed to", topic_name, "and", camera_info_topic)

    def _cam_info_callback(self, msg):
        self.camera_info = msg

    def _apply_camera_info(self) -> None:
        info = self.camera_info
        if info is None:
            return
        self.height = info.height
        self.width = info.width
        self.distortion_model = info.distortion_model
        self.D = np.array(info.d)
        self.K = np.array(info.k).reshape(3, 3)
        self.R = np.array(info.r).reshape(3, 3)
        self.P = np.array(info.p).reshape(3, 4)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.px = self.K[0, 2]
        self.py = self.K[1, 2]
        self.near_val = 0.1
        self.far_val = 5.0
        self.frame_id = info.header.frame_id

    def ensure_ready(self, timeout_s: float = 120.0) -> None:
        """Wait for camera_info (if used) and first image. Call before starting rclpy.spin."""
        import rclpy

        t0 = self._ros_client.get_clock().now()
        while rclpy.ok() and self.camera_info is None:
            rclpy.spin_once(self._ros_client, timeout_sec=0.1)
            if timeout_s > 0:
                elapsed = (self._ros_client.get_clock().now() - t0).nanoseconds / 1e9
                if elapsed > timeout_s:
                    raise RuntimeError(f"Timeout waiting for camera_info on {self.name}")
        self._apply_camera_info()
        self.wait_for_image(timeout_s=timeout_s)

    def _cb(self, msg):
        with self._lock:
            img = image_to_numpy(msg)
            if msg.encoding == "16UC1":
                img = img / 1000.0
            self._img = np.rot90(img, k=self.rotations)
            self._t = msg.header.stamp

    def get_time(self):
        return self._t

    def wait_for_image(self, timeout_s: float = 120.0):
        import rclpy

        t0 = self._ros_client.get_clock().now()
        while rclpy.ok():
            rclpy.spin_once(self._ros_client, timeout_sec=0.1)
            with self._lock:
                if self._img is not None:
                    break
            if timeout_s > 0:
                elapsed = (self._ros_client.get_clock().now() - t0).nanoseconds / 1e9
                if elapsed > timeout_s:
                    raise RuntimeError(f"Timeout waiting for image on {self.name}")

    def get(self, device=None):
        with self._lock:
            if self._img is None:
                return None
            img = self._img.copy()
        if device is not None:
            import torch

            img = torch.FloatTensor(img).to(device)
        return img

    def get_frame(self):
        return self.frame_id

    def get_K(self):
        if self.K is None:
            raise RuntimeError(f"Camera {self.name} intrinsics not ready")
        return self.K.copy()


class RosCameraNoInfo(RosCamera):
    """Camera that does not wait for camera_info; uses default K from first image."""

    def __init__(
        self,
        ros_client,
        image_topic: str,
        rotations: int = 0,
        image_ext: str = "/image_raw",
        verbose: bool = True,
        default_fov_deg: float = 60.0,
    ):
        self._ros_client = ros_client
        self.name = image_topic
        self.rotations = rotations
        self._img = None
        self._t = Time()
        self._lock = threading.Lock()
        self.default_fov_deg = default_fov_deg
        self.camera_info = None
        self.frame_id = "unknown"
        self.K = None
        self.height = self.width = None

        topic_name = image_topic if image_topic.endswith(image_ext) else image_topic + image_ext
        self._sub = ros_client.create_subscription(Image, topic_name, self._cb, qos_profile_sensor_data)
        if verbose:
            print("Subscribed to", topic_name, "(no camera_info)")
        # Don't subscribe to camera_info
        self.distortion_model = ""
        self.D = np.zeros(5)
        self.R = np.eye(3)
        self.P = np.eye(3, 4)
        self.fx = self.fy = self.px = self.py = None
        self.near_val = 0.1
        self.far_val = 5.0

    def _cb(self, msg):
        with self._lock:
            img = image_to_numpy(msg)
            if msg.encoding == "16UC1":
                img = img / 1000.0
            self._img = np.rot90(img, k=self.rotations)
            self._t = msg.header.stamp
            if self.K is None and self._img is not None:
                h, w = self._img.shape[:2]
                self.height, self.width = h, w
                self.K = default_K_from_shape(h, w, self.default_fov_deg)
                self.fx = self.K[0, 0]
                self.fy = self.K[1, 1]
                self.px = self.K[0, 2]
                self.py = self.K[1, 2]

    def ensure_ready(self, timeout_s: float = 120.0) -> None:
        self.wait_for_image(timeout_s=timeout_s)
        with self._lock:
            if self.K is None and self._img is not None:
                h, w = self._img.shape[:2]
                self.height, self.width = h, w
                self.K = default_K_from_shape(h, w, self.default_fov_deg)
                self.fx = self.K[0, 0]
                self.fy = self.K[1, 1]
                self.px = self.K[0, 2]
                self.py = self.K[1, 2]

    def get_K(self):
        with self._lock:
            if self.K is None and self._img is not None:
                h, w = self._img.shape[:2]
                self.K = default_K_from_shape(h, w, self.default_fov_deg)
                self.fx = self.K[0, 0]
                self.fy = self.K[1, 1]
                self.px = self.K[0, 2]
                self.py = self.K[1, 2]
            if self.K is None:
                self.K = default_K_from_shape(240, 320, self.default_fov_deg)
            return self.K.copy()

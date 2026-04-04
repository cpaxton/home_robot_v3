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

"""ZMQ server for Innate Mars: exposes pose, proprioception, and all cameras (head left/right, EE)."""

from typing import Any

import click
import cv2
import numpy as np
import rclpy
from overrides import override

import emet.utils.compression as compression
from emet.core.server import BaseZmqServer
from emet.utils.image import scale_camera_matrix
from innate_mars_bridge.remote import InnateMarsClient


class ZmqServer(BaseZmqServer):
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = InnateMarsClient(init_node=False, verbose=self.verbose)
        # Start ROS spin in background (client holds the ROS node)
        import threading

        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self.client._ros,), daemon=True)
        self._spin_thread.start()

    @override
    def is_running(self) -> bool:
        return not self.done and rclpy.ok()

    @override
    def get_control_mode(self) -> str:
        return "manipulation"

    @override
    def get_full_observation_message(self) -> dict[str, Any]:
        q, dq = self.client.get_joint_state()
        base_pose = self.client.base_pose_xyt
        head_left = self.client.head_left_cam.get()
        head_right = self.client.head_right_cam.get()
        ee_img = self.client.ee_cam.get()
        head_left = head_left if head_left is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        head_right = head_right if head_right is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        ee_img = ee_img if ee_img is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        return {
            "base_pose": base_pose,
            "joint": q,
            "joint_velocities": dq,
            "head_cam_left/image": compression.to_jpg(head_left),
            "head_cam_right/image": compression.to_jpg(head_right),
            "ee_cam/image": compression.to_jpg(ee_img),
            "head_cam_left/pose": self.client.head_left_camera_pose,
            "head_cam_right/pose": self.client.head_right_camera_pose,
            "ee_cam/pose": self.client.ee_camera_pose,
            "ee/pose": self.client.ee_pose,
            "step": self._last_step,
            "recv_address": self.recv_address,
        }

    @override
    def get_state_message(self) -> dict[str, Any]:
        q, dq = self.client.get_joint_state()
        base_pose = self.client.base_pose_xyt
        ee_pose = self.client.ee_pose
        return {
            "base_pose": base_pose,
            "ee_pose": ee_pose,
            "joint_positions": q,
            "joint_velocities": dq,
            "control_mode": self.get_control_mode(),
            "step": self._last_step,
        }

    @override
    def handle_action(self, action: dict[str, Any]):
        # Optional: forward actions to /mars/arm/commands or other topics
        pass

    def _rescale_color(self, color_image: np.ndarray, scale: float) -> np.ndarray:
        if scale == 1.0:
            return color_image
        return cv2.resize(color_image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    @override
    def get_servo_message(self) -> dict[str, Any]:
        q, _ = self.client.get_joint_state()

        # Head left
        head_left_img = self.client.head_left_cam.get()
        if head_left_img is None:
            head_left_img = np.zeros((480, 640, 3), dtype=np.uint8)
        head_left_img = self._rescale_color(head_left_img, self.image_scaling)
        head_left_compressed = compression.to_jpg(head_left_img)

        # Head right
        head_right_img = self.client.head_right_cam.get()
        if head_right_img is None:
            head_right_img = np.zeros((480, 640, 3), dtype=np.uint8)
        head_right_img = self._rescale_color(head_right_img, self.image_scaling)
        head_right_compressed = compression.to_jpg(head_right_img)

        # EE camera
        ee_img = self.client.ee_cam.get()
        if ee_img is None:
            ee_img = np.zeros((480, 640, 3), dtype=np.uint8)
        ee_img = self._rescale_color(ee_img, self.ee_image_scaling)
        ee_compressed = compression.to_jpg(ee_img)

        message = {
            "ee/pose": self.client.ee_pose,
            "robot/config": q,
            "step": self._last_step,
            # Head left
            "head_cam_left/color_camera_K": scale_camera_matrix(self.client.head_left_cam.get_K(), self.image_scaling),
            "head_cam_left/color_image": head_left_compressed,
            "head_cam_left/color_image/shape": head_left_img.shape,
            "head_cam_left/image_scaling": self.image_scaling,
            "head_cam_left/pose": self.client.head_left_camera_pose,
            # Head right
            "head_cam_right/color_camera_K": scale_camera_matrix(
                self.client.head_right_cam.get_K(), self.image_scaling
            ),
            "head_cam_right/color_image": head_right_compressed,
            "head_cam_right/color_image/shape": head_right_img.shape,
            "head_cam_right/image_scaling": self.image_scaling,
            "head_cam_right/pose": self.client.head_right_camera_pose,
            # EE cam
            "ee_cam/color_camera_K": scale_camera_matrix(self.client.ee_cam.get_K(), self.ee_image_scaling),
            "ee_cam/color_image": ee_compressed,
            "ee_cam/color_image/shape": ee_img.shape,
            "ee_cam/image_scaling": self.ee_image_scaling,
            "ee_cam/pose": self.client.ee_camera_pose,
        }
        return message


@click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@click.option("--send_port", default=4401, help="Port to send observations to")
@click.option("--recv_port", default=4402, help="Port to receive actions from")
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
def main(
    send_port: int = 4401,
    recv_port: int = 4402,
    local: bool = False,
):
    rclpy.init()
    server = ZmqServer(
        send_port=send_port,
        recv_port=recv_port,
        use_remote_computer=(not local),
    )
    server.start()


if __name__ == "__main__":
    main()

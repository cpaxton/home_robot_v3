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
from emet.core.zmq_protocol import (
    CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
    EMET_ZMQ_ROBOT_ID_KEY,
    EMET_ZMQ_SESSION_KEY,
    EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY,
)
from emet.utils.image import scale_camera_matrix
from innate_mars_bridge.onboard_da3 import create_onboard_da3_from_env, onboard_da3_enabled
from innate_mars_bridge.remote import InnateMarsClient


class ZmqServer(BaseZmqServer):
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import threading

        self.client = InnateMarsClient(init_node=False, verbose=self.verbose)
        self.client._ros.wait_for_cameras()
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self.client._ros,), daemon=True)
        self._spin_thread.start()
        self._warned_ee_camera_black = False
        self._onboard_da3 = create_onboard_da3_from_env()
        self._warned_onboard_da3 = False
        self._onboard_depth_ok = False

    def _emet_session_payload(self) -> dict[str, Any]:
        """Schema v1 session metadata (see docs/zmq_session_metadata.md)."""
        has_depth = self._onboard_depth_ok or onboard_da3_enabled()
        return {
            EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY: CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
            "runtime_kind": "innate_mars_ros2_bridge",
            "is_simulation": False,
            EMET_ZMQ_ROBOT_ID_KEY: "innate_mars",
            "capabilities": {
                "teleport_base": False,
                "depth": has_depth,
                "stereo_head": True,
                "num_cameras": 3,
                "dof": 10,
                "onboard_da3": onboard_da3_enabled(),
            },
            "environment": {"kind": "ros2", "package": "innate_mars_bridge"},
        }

    def _maybe_onboard_depth(
        self,
        head_left: np.ndarray,
        head_right: np.ndarray | None,
        kl: np.ndarray | None,
        kr: np.ndarray | None,
        pose_l: np.ndarray | None,
        pose_r: np.ndarray | None,
    ) -> np.ndarray | None:
        if self._onboard_da3 is None:
            return None
        depth = self._onboard_da3.infer_depth_meters(
            head_left,
            rgb_right=head_right,
            camera_K_left=kl,
            camera_pose_left=pose_l,
            camera_K_right=kr,
            camera_pose_right=pose_r,
        )
        if depth is not None and depth.size > 0 and bool(np.any(np.isfinite(depth) & (depth > 1e-4))):
            self._onboard_depth_ok = True
            return depth
        if not self._warned_onboard_da3:
            self._warned_onboard_da3 = True
            err = self._onboard_da3.load_error or "inference returned empty depth"
            click.echo(f"Warning: onboard DA3 enabled but not producing depth ({err}).", err=True)
        return None

    @override
    def is_running(self) -> bool:
        return not self.done and rclpy.ok()

    @override
    def get_control_mode(self) -> str:
        return "navigation" if not self.client.at_goal() else "manipulation"

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
        bp = np.asarray(base_pose, dtype=np.float64).reshape(-1)
        gps = bp[:2].astype(np.float64)
        compass = bp[2:3].astype(np.float64) if bp.size >= 3 else np.zeros(1, dtype=np.float64)
        kl = self.client.head_left_cam.get_K()
        kr = self.client.head_right_cam.get_K()
        pose_l = self.client.head_left_camera_pose
        pose_r = self.client.head_right_camera_pose
        depth_m = self._maybe_onboard_depth(head_left, head_right, kl, kr, pose_l, pose_r)
        depth_zmq = None
        if depth_m is not None:
            depth_u16 = (np.asarray(depth_m, dtype=np.float32) * 1000.0).astype(np.uint16)
            depth_zmq = compression.to_jp2(depth_u16)
        head_joint = self.client.head_joint_rad
        ee_K = self.client.ee_cam.get_K()
        ee_pose = self.client.ee_camera_pose
        if not self._warned_ee_camera_black and ee_img is not None and int(np.asarray(ee_img).max()) == 0:
            self._warned_ee_camera_black = True
            click.echo(
                "Warning: wrist camera (/mars/arm/image_raw) is all black — no ROS publisher or driver idle. "
                "Head stereo OK; EE Rerun panel will stay blank until maurice_cam publishes the arm stream.",
                err=True,
            )
        return {
            EMET_ZMQ_ROBOT_ID_KEY: "innate_mars",
            EMET_ZMQ_SESSION_KEY: self._emet_session_payload(),
            "base_pose": base_pose,
            "joint": q,
            "joint_velocities": dq,
            "joint_head": head_joint,
            "gps": gps,
            "compass": compass,
            "rgb": compression.to_jpg(head_left),
            "rgb_right": compression.to_jpg(head_right),
            "camera_K": kl,
            "camera_pose": self.client.head_left_camera_pose,
            "camera_K_right": kr,
            "camera_pose_right": pose_r,
            "depth": depth_zmq,
            "head_cam_left/image": compression.to_jpg(head_left),
            "head_cam_right/image": compression.to_jpg(head_right),
            "ee_cam/image": compression.to_jpg(ee_img),
            "head_cam_left/pose": self.client.head_left_camera_pose,
            "head_cam_right/pose": self.client.head_right_camera_pose,
            "ee_cam/pose": ee_pose,
            "rgb_tertiary": compression.to_jpg(ee_img),
            "camera_K_tertiary": ee_K,
            "camera_pose_tertiary": ee_pose,
            "camera_name_tertiary": "camera_arm",
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
            EMET_ZMQ_ROBOT_ID_KEY: "innate_mars",
            EMET_ZMQ_SESSION_KEY: self._emet_session_payload(),
            "base_pose": base_pose,
            "ee_pose": ee_pose,
            "joint_positions": q,
            "joint_velocities": dq,
            "control_mode": self.get_control_mode(),
            "at_goal": self.client.at_goal(),
            "step": self._last_step,
        }

    @override
    def handle_action(self, action: dict[str, Any]):
        if action is None:
            return
        if "xyt" in action:
            # Default non-blocking so ZMQ recv can ack ``step`` while Nav2/Spin runs.
            # Clients that need a finish signal wait on ``at_goal`` (see GenericZmqClient).
            blocking = bool(action.get("nav_blocking", False))
            relative = bool(action.get("nav_relative", False))
            self.client.move_base_to(
                action["xyt"],
                relative=relative,
                blocking=blocking,
                timeout_s=float(action.get("nav_timeout_s", 120.0)),
            )

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
            EMET_ZMQ_ROBOT_ID_KEY: "innate_mars",
            EMET_ZMQ_SESSION_KEY: self._emet_session_payload(),
            "ee/pose": self.client.ee_pose,
            "robot/config": q,
            "joint_positions": q,
            "joint_head": self.client.head_joint_rad,
            "base_pose": self.client.base_pose_xyt,
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

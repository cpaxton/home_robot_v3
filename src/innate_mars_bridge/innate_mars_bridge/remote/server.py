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
from emet.core.zmq_obs_codec import slim_zmq_obs
from emet.core.zmq_protocol import (
    CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
    EMET_ZMQ_ROBOT_ID_KEY,
    EMET_ZMQ_SESSION_KEY,
    EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY,
)
from emet.core.zmq_server_env import (
    resolve_zmq_ee_image_scaling,
    resolve_zmq_image_scaling,
    resolve_zmq_jpeg_quality,
    zmq_h264_enabled,
    zmq_h264_port,
    zmq_obs_include_images,
    zmq_servo_include_images,
    zmq_use_webp_images,
    zmq_video_rtsp_enabled,
)
from emet.utils.image import scale_camera_matrix
from innate_mars_bridge.onboard_da3 import create_onboard_da3_from_env, onboard_da3_enabled
from innate_mars_bridge.onboard_dinov3 import create_onboard_dinov3_from_env, onboard_dinov3_enabled
from innate_mars_bridge.remote import InnateMarsClient
from innate_mars_bridge.video_rtsp import mars_rtsp_capabilities, start_mars_rtsp_subprocess


class ZmqServer(BaseZmqServer):
    @override
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("image_scaling", resolve_zmq_image_scaling(0.5))
        kwargs.setdefault("ee_image_scaling", resolve_zmq_ee_image_scaling(0.5))
        super().__init__(*args, **kwargs)
        import threading

        self._jpeg_quality = resolve_zmq_jpeg_quality()
        self._obs_include_images = zmq_obs_include_images(default=True)
        self._servo_include_images = zmq_servo_include_images(
            default=not self._obs_include_images,
        )
        self._use_webp = zmq_use_webp_images()
        self._h264_enabled = zmq_h264_enabled()
        self._h264_port = zmq_h264_port()
        self._use_remote_computer = True
        self._h264_socket = None
        self._send_h264_thread = None
        self._rtsp_proc = None
        if zmq_video_rtsp_enabled():
            self._rtsp_proc = start_mars_rtsp_subprocess()

        self.client = InnateMarsClient(init_node=False, verbose=self.verbose)
        self.client._ros.wait_for_cameras()
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self.client._ros,), daemon=True)
        self._spin_thread.start()
        self._warned_ee_camera_black = False
        self._onboard_da3 = create_onboard_da3_from_env()
        self._warned_onboard_da3 = False
        self._onboard_depth_ok = False
        self._onboard_dinov3 = create_onboard_dinov3_from_env()
        self._warned_onboard_dinov3 = False
        self._onboard_dinov3_ok = False

    def _emet_session_payload(self) -> dict[str, Any]:
        """Schema v1 session metadata (see docs/zmq_session_metadata.md)."""
        has_depth = self._onboard_depth_ok or onboard_da3_enabled()
        caps: dict[str, Any] = {
            "teleport_base": False,
            "depth": has_depth,
            "stereo_head": True,
            "num_cameras": 3,
            "dof": 10,
            "onboard_da3": onboard_da3_enabled(),
            "onboard_dinov3": onboard_dinov3_enabled(),
            "zmq_obs_slim": True,
            "zmq_lidar_f32": True,
            "zmq_image_scaling": float(self.image_scaling),
            "zmq_ee_image_scaling": float(self.ee_image_scaling),
        }
        if self._use_webp:
            caps["zmq_webp_images"] = True
        if not self._obs_include_images:
            caps["zmq_obs_metadata_only"] = True
            caps["zmq_images_on_port"] = 4404
        video = mars_rtsp_capabilities(self._rtsp_proc)
        if video is not None:
            caps["video_streams"] = video
        if self._h264_enabled:
            caps["zmq_video_h264"] = True
            caps["zmq_h264_port"] = self._h264_port
        return {
            EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY: CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
            "runtime_kind": "innate_mars_ros2_bridge",
            "is_simulation": False,
            EMET_ZMQ_ROBOT_ID_KEY: "innate_mars",
            "capabilities": caps,
            "environment": {"kind": "ros2", "package": "innate_mars_bridge"},
        }

    def _encode_wire_image(self, image: np.ndarray, scale: float) -> bytes:
        scaled = self._rescale_color(image, scale)
        if self._use_webp:
            return compression.to_webp(scaled)
        return compression.to_jpg(scaled, quality=self._jpeg_quality)

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

    def _maybe_onboard_dinov3(self, head_left: np.ndarray) -> list[float] | None:
        if self._onboard_dinov3 is None:
            return None
        emb = self._onboard_dinov3.infer_head_embedding(head_left)
        if emb:
            self._onboard_dinov3_ok = True
            return emb
        if not self._warned_onboard_dinov3:
            self._warned_onboard_dinov3 = True
            err = self._onboard_dinov3.load_error or "inference returned empty embedding"
            click.echo(f"Warning: onboard DINOv3 enabled but not producing embeddings ({err}).", err=True)
        return None

    @override
    def is_running(self) -> bool:
        return not self.done and rclpy.ok()

    def start(self):
        if self._h264_enabled:
            self._h264_socket = self._make_pub_socket(self._h264_port, self._use_remote_computer)
        super().start()
        if self._h264_enabled and self._h264_socket is not None:
            import threading

            self._send_h264_thread = threading.Thread(target=self._spin_send_h264, daemon=True)
            self._send_h264_thread.start()

    def _spin_send_h264(self):
        import timeit

        from emet.core.server import _rate_sleep

        sum_time = 0.0
        steps = 0
        t0 = timeit.default_timer()
        while self.is_running():
            head_left = self.client.head_left_cam.get()
            if head_left is None:
                head_left = np.zeros((480, 640, 3), dtype=np.uint8)
            scaled = self._rescale_color(head_left, self.image_scaling)
            try:
                nal = compression.to_h264(scaled)
            except Exception as exc:
                if steps == 0:
                    click.echo(f"Warning: H.264 encode failed ({exc}); disable EMET_ZMQ_H264.", err=True)
                time.sleep(0.1)
                continue
            msg = {
                EMET_ZMQ_ROBOT_ID_KEY: "innate_mars",
                "h264_nal": nal,
                "is_keyframe": True,
                "camera": "head_left",
                "step": self._last_step,
            }
            self._h264_socket.send_pyobj(msg)
            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            t0 = t1
            _rate_sleep(self._servo_send_period_s, dt, 1e-4)
            t0 = timeit.default_timer()

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
        dinov3_head = self._maybe_onboard_dinov3(head_left)
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
        lidar_points = None
        lidar_timestamp = None
        pts = self.client.lidar.get()
        if pts is not None:
            lidar_points = pts
            lidar_timestamp = int(self.client.lidar.get_time().nanoseconds)
        message = {
            EMET_ZMQ_ROBOT_ID_KEY: "innate_mars",
            EMET_ZMQ_SESSION_KEY: self._emet_session_payload(),
            "base_pose": base_pose,
            "joint": q,
            "joint_velocities": dq,
            "joint_head": head_joint,
            "gps": gps,
            "compass": compass,
            "camera_K": scale_camera_matrix(kl, self.image_scaling),
            "camera_pose": self.client.head_left_camera_pose,
            "camera_K_right": scale_camera_matrix(kr, self.image_scaling),
            "camera_pose_right": pose_r,
            "depth": depth_zmq,
            "dinov3_head": dinov3_head,
            "head_cam_left/pose": self.client.head_left_camera_pose,
            "head_cam_right/pose": self.client.head_right_camera_pose,
            "ee_cam/pose": ee_pose,
            "camera_K_tertiary": scale_camera_matrix(ee_K, self.ee_image_scaling),
            "camera_pose_tertiary": ee_pose,
            "camera_name_tertiary": "camera_arm",
            "ee/pose": self.client.ee_pose,
            "lidar_points": lidar_points,
            "lidar_timestamp": lidar_timestamp,
            "step": self._last_step,
            "recv_address": self.recv_address,
        }
        if self._obs_include_images:
            head_left_wire = self._rescale_color(head_left, self.image_scaling)
            head_right_wire = self._rescale_color(head_right, self.image_scaling)
            ee_wire = self._rescale_color(ee_img, self.ee_image_scaling)
            message["head_cam_left/image"] = self._encode_wire_image(head_left_wire, 1.0)
            message["head_cam_right/image"] = self._encode_wire_image(head_right_wire, 1.0)
            message["ee_cam/image"] = self._encode_wire_image(ee_wire, 1.0)
            message["head_cam_left/image_scaling"] = self.image_scaling
            message["head_cam_right/image_scaling"] = self.image_scaling
            message["ee_cam/image_scaling"] = self.ee_image_scaling
            message["head_cam_left/image/shape"] = head_left_wire.shape
            message["head_cam_right/image/shape"] = head_right_wire.shape
            message["ee_cam/image/shape"] = ee_wire.shape
        slim_zmq_obs(message)
        return message

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

        message = {
            EMET_ZMQ_ROBOT_ID_KEY: "innate_mars",
            EMET_ZMQ_SESSION_KEY: self._emet_session_payload(),
            "ee/pose": self.client.ee_pose,
            "robot/config": q,
            "joint_positions": q,
            "joint_head": self.client.head_joint_rad,
            "base_pose": self.client.base_pose_xyt,
            "step": self._last_step,
            "head_cam_left/pose": self.client.head_left_camera_pose,
            "head_cam_right/pose": self.client.head_right_camera_pose,
        }
        if not self._servo_include_images:
            return message

        # Head left
        head_left_img = self.client.head_left_cam.get()
        if head_left_img is None:
            head_left_img = np.zeros((480, 640, 3), dtype=np.uint8)
        head_left_img = self._rescale_color(head_left_img, self.image_scaling)
        head_left_compressed = self._encode_wire_image(head_left_img, 1.0)

        # Head right
        head_right_img = self.client.head_right_cam.get()
        if head_right_img is None:
            head_right_img = np.zeros((480, 640, 3), dtype=np.uint8)
        head_right_img = self._rescale_color(head_right_img, self.image_scaling)
        head_right_compressed = self._encode_wire_image(head_right_img, 1.0)

        # EE camera
        ee_img = self.client.ee_cam.get()
        if ee_img is None:
            ee_img = np.zeros((480, 640, 3), dtype=np.uint8)
        ee_img = self._rescale_color(ee_img, self.ee_image_scaling)
        ee_compressed = self._encode_wire_image(ee_img, 1.0)

        message.update(
            {
                "head_cam_left/color_camera_K": scale_camera_matrix(
                    self.client.head_left_cam.get_K(), self.image_scaling
                ),
                "head_cam_left/color_image": head_left_compressed,
                "head_cam_left/color_image/shape": head_left_img.shape,
                "head_cam_left/image_scaling": self.image_scaling,
                "head_cam_right/color_camera_K": scale_camera_matrix(
                    self.client.head_right_cam.get_K(), self.image_scaling
                ),
                "head_cam_right/color_image": head_right_compressed,
                "head_cam_right/color_image/shape": head_right_img.shape,
                "head_cam_right/image_scaling": self.image_scaling,
                "ee_cam/color_camera_K": scale_camera_matrix(self.client.ee_cam.get_K(), self.ee_image_scaling),
                "ee_cam/color_image": ee_compressed,
                "ee_cam/color_image/shape": ee_img.shape,
                "ee_cam/image_scaling": self.ee_image_scaling,
                "ee_cam/pose": self.client.ee_camera_pose,
            }
        )
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
    server._use_remote_computer = not local
    server.start()


if __name__ == "__main__":
    main()

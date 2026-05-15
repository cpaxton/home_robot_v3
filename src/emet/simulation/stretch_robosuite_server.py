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
# This source code is licensed under the LICENSE file in the root directory of this source tree.

"""Stretch default-table MuJoCo ZMQ server (merged MJCF + :class:`~emet.simulation.base_mujoco_zmq_server.BaseMujocoZmqServer`)."""

from __future__ import annotations

import threading
import time
from typing import Any

import mujoco
import numpy as np
from overrides import override

import emet.utils.compression as compression
import emet.utils.logger as log
from emet.core.zmq_protocol import EMET_ZMQ_GT_OBJECTS_KEY, EMET_ZMQ_ROBOT_ID_KEY
from emet.motion import HelloStretchIdx
from emet.motion import constants as motion_constants
from emet.simulation.mujoco_ctrl_sync import sync_actuator_ctrl_from_joint_positions
from emet.simulation.robosuite_server import RobosuiteZmqServer
from emet.simulation.stretch_mujoco import config as stretch_cfg
from emet.simulation.stretch_mujoco import utils as stretch_utils
from emet.utils.geometry import pose_global_to_base, xyt_global_to_base
from emet.utils.image import scale_camera_matrix
from emet.utils.observation_layout import rgb_height_width_for_zmq

logger = log.Logger(__name__)

_STRETCH_ACTUATOR_BY_IDX: dict[int, str] = {
    HelloStretchIdx.LIFT: "lift",
    HelloStretchIdx.ARM: "arm",
    HelloStretchIdx.GRIPPER: "gripper",
    HelloStretchIdx.WRIST_ROLL: "wrist_roll",
    HelloStretchIdx.WRIST_PITCH: "wrist_pitch",
    HelloStretchIdx.WRIST_YAW: "wrist_yaw",
    HelloStretchIdx.HEAD_PAN: "head_pan",
    HelloStretchIdx.HEAD_TILT: "head_tilt",
}


class StretchRobosuiteZmqServer(RobosuiteZmqServer):
    """Stretch merged MJCF with the same ZMQ observation shape as the legacy Stretch MuJoCo server."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._manip_xyt: np.ndarray | None = None

    def _zero_stretch_wheel_vel_actuators(self) -> None:
        if self._mjmodel is None or self._mjdata is None:
            return
        for name in ("left_wheel_vel", "right_wheel_vel"):
            aid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                self._mjdata.ctrl[aid] = 0.0

    @override
    def _postprocess_rgb_depth_and_K(
        self,
        camera_name: str,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        *,
        pixel_ops: tuple[str, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Head d435i matches legacy rot90; d405 EE does not (see ``MujocoServerStretch.get_servo_message``)."""
        if pixel_ops is None and (
            camera_name == "d405_rgb" or camera_name == "d405_depth" or camera_name.startswith("d405_")
        ):
            pixel_ops = ()
        return super()._postprocess_rgb_depth_and_K(camera_name, rgb, depth, pixel_ops=pixel_ops)

    @override
    def _sim_loop(self) -> None:
        while self._running:
            with self._mj_lock:
                self._step_base_navigation_drive()
                if self._mjmodel is None or self._mjdata is None:
                    pass
                elif self._is_kinematic:
                    self._mjdata.qvel.fill(0.0)
                    sync_actuator_ctrl_from_joint_positions(self._mjmodel, self._mjdata, self._spec)
                    self._zero_stretch_wheel_vel_actuators()
                    mujoco.mj_forward(self._mjmodel, self._mjdata)
                else:
                    # Velocity wheel actuators are not driven by the ZMQ joint vector; stale ctrl
                    # fights the free joint / navigation qvel hack and can launch the base.
                    self._zero_stretch_wheel_vel_actuators()
                    mujoco.mj_step(self._mjmodel, self._mjdata)
            self._physics_steps_executed += 1
            if self._max_sim_steps is not None and self._physics_steps_executed >= self._max_sim_steps:
                logger.info(f"MuJoCo step limit reached (--steps {self._max_sim_steps}); stopping simulation loop.")
                self._running = False
                break
            time.sleep(1 / self.simulation_rate)

    @override
    def _run_passive_viewer_main_loop(self, show_viewer_ui: bool) -> None:
        import mujoco.viewer

        dt = 1.0 / max(1, int(self.simulation_rate))
        try:
            with mujoco.viewer.launch_passive(
                self._mjmodel,
                self._mjdata,
                show_left_ui=show_viewer_ui,
                show_right_ui=show_viewer_ui,
            ) as viewer:
                self._disable_passive_viewer_rangefinder_visual(viewer)
                logger.info("MuJoCo passive viewer open (close window or Ctrl+C to stop).")
                while self._running and viewer.is_running():
                    with self._mj_lock:
                        self._step_base_navigation_drive()
                        if self._mjmodel is None or self._mjdata is None:
                            pass
                        elif self._is_kinematic:
                            self._mjdata.qvel.fill(0.0)
                            sync_actuator_ctrl_from_joint_positions(self._mjmodel, self._mjdata, self._spec)
                            self._zero_stretch_wheel_vel_actuators()
                            mujoco.mj_forward(self._mjmodel, self._mjdata)
                        else:
                            self._zero_stretch_wheel_vel_actuators()
                            mujoco.mj_step(self._mjmodel, self._mjdata)
                        self._disable_passive_viewer_rangefinder_visual(viewer)
                        viewer.sync()
                        self._disable_passive_viewer_rangefinder_visual(viewer)
                    self._physics_steps_executed += 1
                    if self._max_sim_steps is not None and self._physics_steps_executed >= self._max_sim_steps:
                        logger.info(f"MuJoCo step limit reached (--steps {self._max_sim_steps}); closing viewer loop.")
                        self._running = False
                        break
                    time.sleep(dt)
        except Exception as e:
            logger.warning(
                f"MuJoCo passive viewer failed ({e!r}); falling back to headless background stepping. "
                "Use a desktop session with DISPLAY set, or run with --headless."
            )
            self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
            self._sim_thread.start()
            while self._running:
                time.sleep(dt)
            return
        self._running = False

    def _stretch_apply_joint_targets(self, joint_targets: np.ndarray) -> None:
        if self._mjmodel is None or self._mjdata is None:
            return
        jt = np.asarray(joint_targets, dtype=np.float64).reshape(-1)
        for idx, aname in _STRETCH_ACTUATOR_BY_IDX.items():
            if idx >= jt.size:
                break
            aid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid < 0:
                continue
            val = float(jt[idx])
            if aname == "gripper":
                val = stretch_utils.map_between_ranges(
                    val,
                    stretch_cfg.robot_settings["gripper_min_max"],
                    stretch_cfg.robot_settings["sim_gripper_min_max"],
                )
            self._mjdata.ctrl[aid] = val
        self._zero_stretch_wheel_vel_actuators()
        if self._is_kinematic:
            self._mjdata.qvel.fill(0.0)
            sync_actuator_ctrl_from_joint_positions(self._mjmodel, self._mjdata, self._spec)
            self._zero_stretch_wheel_vel_actuators()
            mujoco.mj_forward(self._mjmodel, self._mjdata)

    @override
    def handle_action(self, action: dict[str, Any]) -> None:
        if "control_mode" in action:
            new_control_mode = action["control_mode"]
            if new_control_mode == "manipulation" and self.control_mode == "navigation":
                self._manip_xyt = self.get_base_pose()
        action_copy = dict(action)
        # Legacy Stretch server applies posture + sets control_mode; base MuJoCo server had no
        # ``posture`` key, so the ZMQ client would time out waiting for navigation after manip.
        posture = action_copy.pop("posture", None)
        if posture is not None:
            p = str(posture).lower()
            if p == "navigation":
                action_copy["control_mode"] = "navigation"
                action_copy["joint"] = np.asarray(motion_constants.STRETCH_NAVIGATION_Q, dtype=np.float64)
            elif p == "manipulation":
                action_copy["control_mode"] = "manipulation"
                action_copy["joint"] = np.asarray(motion_constants.STRETCH_PREGRASP_Q, dtype=np.float64)
        joint_only = action_copy.pop("joint", None)
        super().handle_action(action_copy)
        if joint_only is not None:
            with self._mj_lock:
                if self._mjdata is not None:
                    self._stretch_apply_joint_targets(np.asarray(joint_only, dtype=np.float64))

    @override
    def start(
        self,
        robocasa: bool = False,
        headless: bool = True,
        show_viewer_ui: bool = False,
        **kwargs: Any,
    ) -> None:
        super().start(robocasa=robocasa, headless=headless, show_viewer_ui=show_viewer_ui, **kwargs)
        with self._mj_lock:
            if self._mjmodel is not None and self._mjdata is not None:
                self._zero_stretch_wheel_vel_actuators()
                mujoco.mj_forward(self._mjmodel, self._mjdata)

    @override
    def get_joint_state(self):
        dof = self._spec.dof
        positions = np.zeros(dof)
        velocities = np.zeros(dof)
        efforts = np.zeros(dof)
        if self._mjmodel is None or self._mjdata is None:
            return positions, velocities, efforts
        with self._mj_lock:
            for idx, aname in _STRETCH_ACTUATOR_BY_IDX.items():
                aid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
                if aid < 0:
                    continue
                positions[idx] = float(self._mjdata.actuator(aname).length[0])
                velocities[idx] = float(self._mjdata.actuator(aname).velocity[0])
            xyt = self.get_base_pose()
            if xyt is not None:
                positions[HelloStretchIdx.BASE_X] = float(xyt[0])
                positions[HelloStretchIdx.BASE_Y] = float(xyt[1])
                positions[HelloStretchIdx.BASE_THETA] = float(xyt[2])
            if self.control_mode == "manipulation" and self._manip_xyt is not None and xyt is not None:
                xyt_r = xyt_global_to_base(xyt, self._manip_xyt)
                positions[HelloStretchIdx.BASE_X] = float(xyt_r[0])
            positions[HelloStretchIdx.GRIPPER] = stretch_utils.map_between_ranges(
                positions[HelloStretchIdx.GRIPPER],
                stretch_cfg.robot_settings["sim_gripper_min_max"],
                stretch_cfg.robot_settings["gripper_min_max"],
            )
        return positions, velocities, efforts

    def _link_pose_world(self, body_name: str) -> np.ndarray:
        if self._mjmodel is None or self._mjdata is None:
            return np.eye(4, dtype=np.float64)
        with self._mj_lock:
            bid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                return np.eye(4, dtype=np.float64)
            R = np.asarray(self._mjdata.body(bid).xmat, dtype=np.float64).reshape(3, 3)
            p = np.asarray(self._mjdata.body(bid).xpos, dtype=np.float64).reshape(3)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def get_ee_pose(self) -> np.ndarray | None:
        if self._initial_xyt is None:
            return None
        pose = self._link_pose_world("link_grasp_center")
        return pose_global_to_base(pose, self._initial_xyt)

    def get_ee_camera_pose(self) -> np.ndarray | None:
        if self._initial_xyt is None:
            return None
        pose = self._camera_pose_world("d405_rgb")
        return pose_global_to_base(pose, self._initial_xyt)

    def get_head_camera_pose(self) -> np.ndarray | None:
        if self._initial_xyt is None:
            return None
        pose = np.array(self._camera_pose_world("d435i_camera_rgb"), dtype=np.float64, copy=True)
        pose[:3, :3] = pose[:3, :3] @ np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return pose_global_to_base(pose, self._initial_xyt)

    @override
    def _primary_rgb_and_depth(self, camera_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if camera_name == "d435i_camera_rgb" and self._mjmodel is not None and self._mjdata is not None:
            rgb = self._render_rgb_raw("d435i_camera_rgb")
            # Depth must be rendered from the **same** MuJoCo camera as RGB: MJCF `d435i_camera_depth`
            # / `d405_depth` omit `fovy` and default to 45°, while the RGB cams use 58°. Unprojecting
            # 45° depth with a 58° `K` yields garbage point clouds / voxel noise.
            depth = self._render_depth_raw("d435i_camera_rgb")
            return self._postprocess_rgb_depth_and_K(camera_name, rgb, depth)
        return super()._primary_rgb_and_depth(camera_name)

    def get_full_observation_message(self) -> dict[str, Any] | None:
        if self._mjdata is None:
            return None
        cam_names = self._spec.camera_names
        if not cam_names:
            return None
        primary_cam = cam_names[0]
        try:
            rgb, depth, K = self._primary_rgb_and_depth(primary_cam)
        except Exception:
            return None
        height, width = rgb_height_width_for_zmq(rgb)
        depth_u16 = (depth * 1000).astype(np.uint16)
        positions, _, _ = self.get_joint_state()
        xyt = self.get_base_pose()
        if xyt is None:
            xyt = np.zeros(3)
        # Match legacy ``MujocoServerStretch.get_full_observation_message``: head pose in the same
        # base-centered frame as ``gps``/``compass``, not raw MuJoCo world ``_camera_pose_world``.
        cam_pose = self.get_head_camera_pose()
        if cam_pose is None:
            cam_pose = np.eye(4, dtype=np.float64)
        message = {
            "rgb": compression.to_jpg(rgb),
            "depth": compression.to_jp2(depth_u16),
            "camera_K": K,
            "camera_pose": cam_pose,
            "ee_pose": np.eye(4),
            "joint": positions,
            "gps": xyt[:2],
            "compass": np.array([xyt[2]]),
            "rgb_width": width,
            "rgb_height": height,
            "control_mode": self.get_control_mode(),
            "last_motion_failed": False,
            "recv_address": self.recv_address,
            "step": self._last_step,
            "at_goal": self._at_goal,
            "is_simulation": True,
            "lidar_points": None,
            "lidar_timestamp": None,
            EMET_ZMQ_ROBOT_ID_KEY: self.get_robot_spec().name,
        }
        try:
            with self._mj_lock:
                if self._mjmodel is not None and self._mjdata is not None:
                    from emet.dataset.mujoco_gt import gt_objects_for_zmq_message

                    gt_objs = gt_objects_for_zmq_message(
                        self._mjmodel, self._mjdata, environment=self._environment_descriptor
                    )
                else:
                    gt_objs = []
        except Exception:
            gt_objs = []
        message[EMET_ZMQ_GT_OBJECTS_KEY] = gt_objs
        return self._attach_emet_session(message)

    def get_servo_message(self) -> dict[str, Any] | None:
        """Same ZMQ keys as :meth:`emet.simulation.mujoco_server_stretch.MujocoServerStretch.get_servo_message`."""
        if self._mjdata is None:
            return None
        try:
            head_rgb = self._render_rgb_raw("d435i_camera_rgb")
            head_depth = self._render_depth_raw("d435i_camera_rgb")
            head_rgb, head_depth, K_head = self._postprocess_rgb_depth_and_K("d435i_camera_rgb", head_rgb, head_depth)
            ee_rgb = self._render_rgb_raw("d405_rgb")
            ee_depth = self._render_depth_raw("d405_rgb")
            ee_rgb, ee_depth, K_ee = self._postprocess_rgb_depth_and_K("d405_rgb", ee_rgb, ee_depth)
        except Exception:
            return None

        ee_color_image, ee_depth_image = self._rescale_color_and_depth(ee_rgb, ee_depth, self.ee_image_scaling)
        head_color_image, head_depth_image = self._rescale_color_and_depth(head_rgb, head_depth, self.image_scaling)

        K_ee_servo = scale_camera_matrix(K_ee, self.ee_image_scaling)
        K_head_servo = scale_camera_matrix(K_head, self.image_scaling)

        ee_depth_u16 = (ee_depth_image * 1000).astype(np.uint16)
        head_depth_u16 = (head_depth_image * 1000).astype(np.uint16)

        compressed_ee_depth_image = compression.to_jp2(ee_depth_u16)
        compressed_ee_color_image = compression.to_jpg(ee_color_image)
        compressed_head_depth_image = compression.to_jp2(head_depth_u16)
        compressed_head_color_image = compression.to_jpg(head_color_image)

        positions, _, _ = self.get_joint_state()

        ee_pose = self.get_ee_pose()
        if ee_pose is None:
            ee_pose = np.eye(4, dtype=np.float64)
        ee_cam_pose = self.get_ee_camera_pose()
        if ee_cam_pose is None:
            ee_cam_pose = np.eye(4, dtype=np.float64)
        head_cam_pose = self.get_head_camera_pose()
        if head_cam_pose is None:
            head_cam_pose = np.eye(4, dtype=np.float64)

        message = {
            "ee_cam/color_camera_K": K_ee_servo,
            "ee_cam/depth_camera_K": K_ee_servo,
            "ee_cam/color_image": compressed_ee_color_image,
            "ee_cam/depth_image": compressed_ee_depth_image,
            "ee_cam/color_image/shape": ee_color_image.shape,
            "ee_cam/depth_image/shape": ee_depth_u16.shape,
            "ee_cam/image_scaling": self.ee_image_scaling,
            "ee_cam/depth_scaling": self.ee_depth_scaling,
            "ee_cam/pose": ee_cam_pose,
            "ee/pose": ee_pose,
            "head_cam/color_camera_K": K_head_servo,
            "head_cam/depth_camera_K": K_head_servo,
            "head_cam/color_image": compressed_head_color_image,
            "head_cam/depth_image": compressed_head_depth_image,
            "head_cam/color_image/shape": head_color_image.shape,
            "head_cam/depth_image/shape": head_depth_u16.shape,
            "head_cam/image_scaling": self.image_scaling,
            "head_cam/depth_scaling": self.depth_scaling,
            "head_cam/pose": head_cam_pose,
            "robot/config": positions,
            "is_simulation": True,
            "step": self._last_step,
            EMET_ZMQ_ROBOT_ID_KEY: self.get_robot_spec().name,
        }
        return self._attach_emet_session(message)

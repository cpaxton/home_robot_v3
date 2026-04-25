# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""ZMQ server that wraps a robosuite environment for non-Stretch robots.

For robots natively supported by robosuite (PandaOmron, Tiago, GR1, etc.)
this server keeps the robosuite robot in the scene and exposes the same
ZMQ protocol as MujocoZmqServer.
"""

import contextlib
import threading
import time
from typing import Any, cast

import cv2
import mujoco
import numpy as np
from overrides import override

import emet.utils.compression as compression
import emet.utils.logger as log
from emet.core.server import BaseZmqServer
from emet.core.zmq_protocol import EMET_ZMQ_ROBOT_ID_KEY
from emet.robots.base import RobotSpec
from emet.utils.geometry import xyt_global_to_base

logger = log.Logger(__name__)

# One ``mujoco.Renderer`` / GL context: multiple resolutions (e.g. 640 + 320 wide) each call
# ``mjr_makeContext`` and often hit GL_INVALID_OPERATION (0x502) on EGL. Servo images downsample in CPU.
_PRIMARY_RW, _PRIMARY_RH = 640, 480
_SERVO_RW, _SERVO_RH = 320, 240


class RobosuiteZmqServer(BaseZmqServer):
    """ZMQ server backed by a robosuite environment.

    Unlike MujocoZmqServer (which uses stretch_mujoco), this server
    drives the robosuite env directly via ``env.step(action)``.
    """

    hz = 20

    def __init__(
        self,
        robot_spec: RobotSpec,
        *args,
        scene_xml: str | None = None,
        scene_model: mujoco.MjModel | None = None,
        simulation_rate: int = 80,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._spec = robot_spec
        self._scene_xml = scene_xml
        self._scene_model = scene_model
        self.simulation_rate = simulation_rate

        self._mjmodel: mujoco.MjModel | None = None
        self._mjdata: mujoco.MjData | None = None
        self._initial_xyt: np.ndarray | None = None
        self._running = False
        self.control_mode = "navigation"
        self._at_goal = False
        # mujoco.Renderer / GLFW are not thread-safe; ZMQ send + servo threads render concurrently.
        self._render_lock = threading.Lock()
        # Single Renderer / GL context (see module _PRIMARY_*). Extra resolutions → extra 0x502 on EGL.
        self._primary_renderer: Any | None = None
        # When using the passive viewer, all mj_step / mjdata reads must use the same lock (MuJoCo docs).
        self._mj_data_sync: contextlib.AbstractContextManager[Any] = contextlib.nullcontext()

    @property
    def spec(self) -> RobotSpec:
        return self._spec

    @override
    def is_running(self) -> bool:
        return self._running

    @override
    def get_control_mode(self) -> str:
        return self.control_mode

    def _load_model(self) -> None:
        if self._scene_model is not None:
            self._mjmodel = self._scene_model
        elif self._scene_xml is not None:
            self._mjmodel = mujoco.MjModel.from_xml_string(self._scene_xml)
        else:
            raise ValueError("Either scene_xml or scene_model must be provided")
        self._mjdata = mujoco.MjData(self._mjmodel)
        mujoco.mj_forward(self._mjmodel, self._mjdata)

    def get_scene_summary(self) -> str:
        """Return a short text summary of the scene: robot, position, and notable objects."""
        if self._mjmodel is None or self._mjdata is None:
            return "Scene not loaded."
        lines = [
            "--- Scene summary ---",
            f"Robot: {self._spec.name}",
        ]
        try:
            xyt = self.get_base_xyt()
            lines.append(f"Robot position (x, y, theta): ({xyt[0]:.3f}, {xyt[1]:.3f}, {xyt[2]:.3f})")
            body_id = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_BODY, self._spec.base_link_name)
            if body_id >= 0:
                z = float(self._mjdata.body(body_id).xpos[2])
                lines.append(f"Robot height (z): {z:.3f}")
        except Exception:
            lines.append("Robot position: (unknown)")
        # Describe notable bodies (object1, object2, table, floor)
        for bid in range(self._mjmodel.nbody):
            name = mujoco.mj_id2name(self._mjmodel, mujoco.mjtObj.mjOBJ_BODY, bid)
            if name is None or name == self._spec.base_link_name:
                continue
            xpos = self._mjdata.body(bid).xpos
            if "object1" in (name or ""):
                lines.append(f"  Blue cube (object1): pos ({xpos[0]:.3f}, {xpos[1]:.3f}, {xpos[2]:.3f})")
            elif "object2" in (name or ""):
                lines.append(f"  Red cylinder (object2): pos ({xpos[0]:.3f}, {xpos[1]:.3f}, {xpos[2]:.3f})")
            elif name in ("table", "floor"):
                lines.append(f"  {name}: pos ({xpos[0]:.3f}, {xpos[1]:.3f}, {xpos[2]:.3f})")
        lines.append("-------------------")
        return "\n".join(lines)

    def get_base_xyt(self) -> np.ndarray:
        base_name = self._spec.base_link_name
        try:
            xpos = self._mjdata.body(base_name).xpos
            xmat = self._mjdata.body(base_name).xmat.reshape(3, 3)
            theta = np.arctan2(xmat[1, 0], xmat[0, 0])
            return np.array([xpos[0], xpos[1], theta])
        except Exception:
            return np.zeros(3)

    def get_base_pose(self) -> np.ndarray | None:
        if self._initial_xyt is None:
            return None
        xyt = self.get_base_xyt()
        return xyt_global_to_base(xyt, self._initial_xyt)

    def get_joint_state(self):
        dof = self._spec.dof
        positions = np.zeros(dof)
        velocities = np.zeros(dof)
        efforts = np.zeros(dof)

        if self._mjdata is None:
            return positions, velocities, efforts

        for i, jname in enumerate(self._spec.joint_names):
            try:
                jid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    continue
                qadr = self._mjmodel.jnt_qposadr[jid]
                vadr = self._mjmodel.jnt_dofadr[jid]
                positions[i] = self._mjdata.qpos[qadr]
                velocities[i] = self._mjdata.qvel[vadr]
            except Exception:
                continue

        return positions, velocities, efforts

    def _close_renderers(self) -> None:
        with self._render_lock:
            if self._primary_renderer is not None:
                try:
                    self._primary_renderer.close()
                except Exception:
                    pass
                self._primary_renderer = None

    def _primary_rgb_and_depth(self, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
        """RGB + depth under one ``_render_lock`` and one ``Renderer`` (avoids EGL 0x502 on second context)."""
        with self._render_lock:
            if self._primary_renderer is None:
                self._primary_renderer = mujoco.Renderer(self._mjmodel, _PRIMARY_RH, _PRIMARY_RW)
            renderer = self._primary_renderer
            renderer.update_scene(self._mjdata, camera=camera_name)
            rgb = cast(np.ndarray, renderer.render())
            rgb = np.asarray(rgb, dtype=np.uint8).copy()
            renderer.enable_depth_rendering()
            try:
                renderer.update_scene(self._mjdata, camera=camera_name)
                depth = cast(np.ndarray, renderer.render())
                depth = np.asarray(depth, dtype=np.float32).copy()
            finally:
                renderer.disable_depth_rendering()
            return rgb, depth

    def _get_camera_K(self, camera_name: str, width: int = 640, height: int = 480):
        """Compute intrinsic matrix from MuJoCo camera fovy."""
        cam_id = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            return np.eye(3)
        fovy = self._mjmodel.cam_fovy[cam_id]
        f = 0.5 * height / np.tan(np.radians(fovy) / 2)
        return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])

    @override
    def handle_action(self, action: dict[str, Any]):
        with self._mj_data_sync:
            if "control_mode" in action:
                self.control_mode = action["control_mode"]

            if "joint" in action:
                joint_targets = action["joint"]
                for i, aname in enumerate(self._spec.actuator_names):
                    if i < len(joint_targets):
                        aid = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
                        if aid >= 0:
                            self._mjdata.ctrl[aid] = joint_targets[i]

            if "xyt" in action:
                logger.info(f"Navigation goal received: {action['xyt']} (not yet implemented for robosuite server)")

    @override
    def get_full_observation_message(self) -> dict[str, Any]:
        if self._mjdata is None:
            return None

        cam_names = self._spec.camera_names
        if not cam_names:
            return None

        primary_cam = cam_names[0]
        with self._mj_data_sync:
            try:
                rgb, depth = self._primary_rgb_and_depth(primary_cam)
            except Exception:
                return None

            width, height = rgb.shape[1], rgb.shape[0]
            depth_u16 = (depth * 1000).astype(np.uint16)

            positions, _, _ = self.get_joint_state()
            xyt = self.get_base_pose()
            if xyt is None:
                xyt = np.zeros(3)

            K = self._get_camera_K(primary_cam, width, height)

            message = {
                "rgb": compression.to_jpg(rgb),
                "depth": compression.to_jp2(depth_u16),
                "camera_K": K,
                "camera_pose": np.eye(4),
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
                EMET_ZMQ_ROBOT_ID_KEY: self._spec.name,
            }
            return message

    @override
    def get_state_message(self) -> dict[str, Any]:
        if self._mjdata is None:
            return None
        with self._mj_data_sync:
            q, dq, eff = self.get_joint_state()
            return {
                "base_pose": self.get_base_pose(),
                "ee_pose": np.eye(4),
                "joint_positions": q,
                "joint_velocities": dq,
                "joint_efforts": eff,
                "control_mode": self.get_control_mode(),
                "at_goal": self._at_goal,
                "is_homed": True,
                "is_runstopped": False,
                "step": self._last_step,
                EMET_ZMQ_ROBOT_ID_KEY: self._spec.name,
            }

    @override
    def get_servo_message(self) -> dict[str, Any]:
        if self._mjdata is None:
            return None

        cam_names = self._spec.camera_names
        if not cam_names:
            return None

        primary_cam = cam_names[0]
        with self._mj_data_sync:
            try:
                rgb_full, depth_full = self._primary_rgb_and_depth(primary_cam)
                rgb = cv2.resize(rgb_full, (_SERVO_RW, _SERVO_RH), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth_full, (_SERVO_RW, _SERVO_RH), interpolation=cv2.INTER_NEAREST)
            except Exception:
                return None

            depth_u16 = (depth * 1000).astype(np.uint16)
            q, dq, eff = self.get_joint_state()
            bp = self.get_base_pose()
            xyt = np.zeros(3) if bp is None else np.asarray(bp, dtype=np.float64)

            message = {
                "head_color_image": compression.to_jpg(rgb),
                "head_depth_image": compression.to_jp2(depth_u16),
                "head_camera_K": self._get_camera_K(primary_cam, 320, 240),
                "joint_positions": q,
                "joint_velocities": dq,
                "base_pose": xyt,
                "control_mode": self.get_control_mode(),
                "step": self._last_step,
                "at_goal": self._at_goal,
            }
            return message

    def _sim_loop(self):
        """Step the MuJoCo simulation at the configured rate."""
        while self._running:
            with self._mj_data_sync:
                mujoco.mj_step(self._mjmodel, self._mjdata)
            time.sleep(1 / self.simulation_rate)

    def start(
        self,
        robocasa: bool = False,
        headless: bool = True,
        show_viewer_ui: bool = False,
        **kwargs,
    ) -> None:
        self._load_model()
        self._running = True
        self._initial_xyt = self.get_base_xyt()

        # Print scene summary before any rendering (so it appears in headless / no-DISPLAY runs)
        summary = self.get_scene_summary()
        print(summary, flush=True)
        logger.info("\n" + summary)

        super().start()

        want_viewer = show_viewer_ui and not headless
        viewer_cm = None
        if want_viewer:
            import mujoco.viewer

            try:
                viewer_cm = mujoco.viewer.launch_passive(
                    self._mjmodel,
                    self._mjdata,
                    show_left_ui=show_viewer_ui,
                    show_right_ui=show_viewer_ui,
                )
            except Exception as e:
                logger.warning("MuJoCo passive viewer could not start (%s); continuing without a window.", e)
                want_viewer = False

        if want_viewer and viewer_cm is not None:
            with viewer_cm as viewer:
                self._mj_data_sync = viewer.lock()
                self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
                self._sim_thread.start()

                logger.info(
                    f"RobosuiteZmqServer started for robot '{self._spec.name}' "
                    f"({self._spec.dof} DOF, {len(self._spec.actuator_names)} actuators) with passive viewer"
                )
                print("Server running (MuJoCo viewer open). Close the viewer or press Ctrl+C to stop.", flush=True)

                ui_dt = 1.0 / min(60, max(30, self.simulation_rate))
                while self._running and viewer.is_running():
                    viewer.sync()
                    time.sleep(ui_dt)

                self._running = False
                if getattr(self, "_sim_thread", None) is not None:
                    self._sim_thread.join(timeout=5.0)

            self._mj_data_sync = contextlib.nullcontext()
            self.stop()
            return

        self._mj_data_sync = contextlib.nullcontext()
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()

        logger.info(
            f"RobosuiteZmqServer started for robot '{self._spec.name}' "
            f"({self._spec.dof} DOF, {len(self._spec.actuator_names)} actuators)"
        )
        print("Server running. Press Ctrl+C to stop.", flush=True)

        while self._running:
            time.sleep(1 / self.simulation_rate)

    def stop(self):
        self._close_renderers()
        self._running = False
        self._done = True

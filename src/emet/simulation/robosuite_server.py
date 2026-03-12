"""ZMQ server that wraps a robosuite environment for non-Stretch robots.

For robots natively supported by robosuite (PandaOmron, Tiago, GR1, etc.)
this server keeps the robosuite robot in the scene and exposes the same
ZMQ protocol as MujocoZmqServer.
"""

import threading
import time
import timeit
from typing import Any, Dict, Optional

import mujoco
import numpy as np
from overrides import override

import emet.utils.compression as compression
import emet.utils.logger as log
from emet.core.server import BaseZmqServer
from emet.robots.base import RobotSpec
from emet.utils.geometry import xyt_base_to_global, xyt_global_to_base
from emet.utils.image import scale_camera_matrix

logger = log.Logger(__name__)


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
        scene_xml: Optional[str] = None,
        scene_model: Optional[mujoco.MjModel] = None,
        simulation_rate: int = 80,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._spec = robot_spec
        self._scene_xml = scene_xml
        self._scene_model = scene_model
        self.simulation_rate = simulation_rate

        self._mjmodel: Optional[mujoco.MjModel] = None
        self._mjdata: Optional[mujoco.MjData] = None
        self._initial_xyt: Optional[np.ndarray] = None
        self._running = False
        self.control_mode = "navigation"
        self._at_goal = False

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

    def get_base_xyt(self) -> np.ndarray:
        base_name = self._spec.base_link_name
        try:
            xpos = self._mjdata.body(base_name).xpos
            xmat = self._mjdata.body(base_name).xmat.reshape(3, 3)
            theta = np.arctan2(xmat[1, 0], xmat[0, 0])
            return np.array([xpos[0], xpos[1], theta])
        except Exception:
            return np.zeros(3)

    def get_base_pose(self) -> Optional[np.ndarray]:
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

    def _render_camera(self, camera_name: str, width: int = 640, height: int = 480):
        """Render an RGB image from a named MuJoCo camera."""
        renderer = mujoco.Renderer(self._mjmodel, height, width)
        renderer.update_scene(self._mjdata, camera=camera_name)
        rgb = renderer.render()
        renderer.close()
        return rgb

    def _render_depth(self, camera_name: str, width: int = 640, height: int = 480):
        """Render a depth image from a named MuJoCo camera."""
        renderer = mujoco.Renderer(self._mjmodel, height, width)
        renderer.update_scene(self._mjdata, camera=camera_name)
        renderer.enable_depth_rendering()
        depth = renderer.render()
        renderer.disable_depth_rendering()
        renderer.close()
        return depth

    def _get_camera_K(self, camera_name: str, width: int = 640, height: int = 480):
        """Compute intrinsic matrix from MuJoCo camera fovy."""
        cam_id = mujoco.mj_name2id(self._mjmodel, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            return np.eye(3)
        fovy = self._mjmodel.cam_fovy[cam_id]
        f = 0.5 * height / np.tan(np.radians(fovy) / 2)
        return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])

    @override
    def handle_action(self, action: Dict[str, Any]):
        if "control_mode" in action:
            self.control_mode = action["control_mode"]

        if "joint" in action:
            joint_targets = action["joint"]
            for i, aname in enumerate(self._spec.actuator_names):
                if i < len(joint_targets):
                    aid = mujoco.mj_name2id(
                        self._mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, aname
                    )
                    if aid >= 0:
                        self._mjdata.ctrl[aid] = joint_targets[i]

        if "xyt" in action:
            logger.info(f"Navigation goal received: {action['xyt']} (not yet implemented for robosuite server)")

    @override
    def get_full_observation_message(self) -> Dict[str, Any]:
        if self._mjdata is None:
            return None

        cam_names = self._spec.camera_names
        if not cam_names:
            return None

        primary_cam = cam_names[0]
        try:
            rgb = self._render_camera(primary_cam)
            depth = self._render_depth(primary_cam)
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
        }
        return message

    @override
    def get_state_message(self) -> Dict[str, Any]:
        if self._mjdata is None:
            return None
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
        }

    @override
    def get_servo_message(self) -> Dict[str, Any]:
        if self._mjdata is None:
            return None

        cam_names = self._spec.camera_names
        if not cam_names:
            return None

        primary_cam = cam_names[0]
        try:
            rgb = self._render_camera(primary_cam, 320, 240)
            depth = self._render_depth(primary_cam, 320, 240)
        except Exception:
            return None

        depth_u16 = (depth * 1000).astype(np.uint16)
        q, dq, eff = self.get_joint_state()
        xyt = self.get_base_pose() or np.zeros(3)

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
            mujoco.mj_step(self._mjmodel, self._mjdata)
            time.sleep(1 / self.simulation_rate)

    def start(
        self,
        robocasa: bool = False,
        headless: bool = True,
        **kwargs,
    ) -> None:
        self._load_model()
        self._running = True
        self._initial_xyt = self.get_base_xyt()

        super().start()

        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()

        logger.info(
            f"RobosuiteZmqServer started for robot '{self._spec.name}' "
            f"({self._spec.dof} DOF, {len(self._spec.actuator_names)} actuators)"
        )

        while self._running:
            time.sleep(1 / self.simulation_rate)

    def stop(self):
        self._running = False
        self._done = True

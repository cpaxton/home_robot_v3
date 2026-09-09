#!/usr/bin/env python
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# Stretch-only MuJoCo ZMQ server. Imported only when --robot stretch to avoid loading pinocchio/hppfcl for rby1/galaxea_r1.

import threading
import time
import timeit
from pathlib import Path
from typing import Any

import numpy as np
from overrides import override

from emet.core.zmq_protocol import (
    EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY,
    EMET_ACTION_SIM_SET_BODY_POSE_KEY,
    EMET_ACTION_SIM_SET_JOINT_QPOS_KEY,
)
from emet.motion.constants import STRETCH_CAMERA_FRAME
from emet.simulation.mujoco_ground_truth import (
    mujoco_ground_truth_write_path,
    parse_ground_truth_dump_action_field,
)
from emet.simulation.sim_manipulation import (
    parse_sim_set_body_pose_action,
    parse_sim_set_joint_qpos_action,
)
from emet.simulation.sim_object_placements import (
    apply_navigation_origin_to_session,
    attach_sim_object_placements_to_session,
)
from emet.simulation.stretch_mujoco import StretchMujocoSimulator
from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras

# Robocasa is imported lazily when --use-robocasa is used, to avoid loading robosuite/numba
# on every server start (and to avoid numba init failures when not using Robocasa).
model_generation_wizard = None
_ROBOCASA_IMPORT_FAILED = True

import emet.motion.constants as constants
import emet.utils.compression as compression
import emet.utils.logger as logger
from emet.core.server import BaseZmqServer
from emet.core.zmq_protocol import (
    CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
    EMET_ZMQ_ROBOT_ID_KEY,
    EMET_ZMQ_SESSION_KEY,
    EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY,
    EMET_ZMQ_SIM_TIME_RATIO_KEY,
)
from emet.motion import HelloStretchIdx
from emet.motion.control.goto_controller import GotoVelocityController
from emet.robots.base import RobotSpec
from emet.robots.stretch import StretchBackend
from emet.simulation.molmospaces_env import molmospaces_nav_teleport_enabled
from emet.utils.assets import get_mujoco_models_path
from emet.utils.config import get_control_config
from emet.utils.geometry import pose_global_to_base, xyt_base_to_global, xyt_global_to_base
from emet.utils.image import scale_camera_matrix
from emet.utils.observation_layout import rgb_height_width_for_zmq

default_scene_xml_path = str(get_mujoco_models_path() / "scene.xml")

# Maps HelloStretchIdx to actuators
mujoco_actuators = {
    HelloStretchIdx.BASE_X: "base_x_joint",
    HelloStretchIdx.BASE_Y: "base_y_joint",
    HelloStretchIdx.BASE_THETA: "base_theta_joint",
    HelloStretchIdx.LIFT: "lift",
    HelloStretchIdx.ARM: "arm",
    HelloStretchIdx.GRIPPER: "gripper",
    HelloStretchIdx.WRIST_ROLL: "wrist_roll",
    HelloStretchIdx.WRIST_PITCH: "wrist_pitch",
    HelloStretchIdx.WRIST_YAW: "wrist_yaw",
    HelloStretchIdx.HEAD_PAN: "head_pan",
    HelloStretchIdx.HEAD_TILT: "head_tilt",
}

stretch_dof = constants.stretch_degrees_of_freedom

manip_idx = [
    HelloStretchIdx.BASE_X,
    HelloStretchIdx.LIFT,
    HelloStretchIdx.ARM,
    HelloStretchIdx.WRIST_ROLL,
    HelloStretchIdx.WRIST_PITCH,
    HelloStretchIdx.WRIST_YAW,
]


# Constants for the controller
CONTROL_HZ = 20
VEL_THRESHOlD = 0.001
RVEL_THRESHOLD = 0.005


class MujocoZmqServer(BaseZmqServer):
    """Server for Mujoco simulation with ZMQ communication. This allows us to run the Mujoco simulation in the exact same way as we would run a remote ROS server on the robot, including potentially running it on a different machine or on the cloud. It requires:
    - Mujoco installation
    - Stretch_mujoco installation: https://github.com/hello-robot/stretch_mujoco/
    """

    hz = CONTROL_HZ
    # How long should the controller report done before we're actually confident that we're done?
    done_t = 0.1

    # Print debug messages for control loop
    debug_control_loop = False
    debug_set_goal_pose = False

    def get_robot_spec(self) -> RobotSpec:
        """Which robot does this simulator emulate?"""
        return StretchBackend().get_spec()

    def get_body_xyt(self, body_name: str) -> np.ndarray:
        """Get the se(2) base pose: x, y, and theta"""

        # Get mjdata and mjmodel from simulator
        mjdata = self.robot_sim.mjdata

        xyz = mjdata.body(body_name).xpos
        rotation = mjdata.body("base_link").xmat.reshape(3, 3)
        theta = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.array([xyz[0], xyz[1], theta])

    def list_all_bodies(self):
        """Debug function to list all bodies in the simulation."""
        model = self.robot_sim.mjmodel
        data = self.robot_sim.mjdata

        # Loop over all bodies
        for i in range(model.nbody):
            body_name = model.body(i).name
            body_pos = data.body(i).xpos
            body_quat = data.body(i).xquat

            print(f"Body {i}: {body_name}")
            print(f"  Position: {body_pos}")
            print(f"  Orientation (quaternion): {body_quat}")

    def set_robot_position(self, xyt: np.ndarray, relative: bool = False) -> None:
        """Set the robot position in the simulation

        Args:
            xyt (np.ndarray): The desired pose of the robot in world coordinates (x, y, theta).
            relative (bool): If True, the pose is relative to the current pose of the robot.
        """
        return self.set_body_position(xyt, "base_link", relative=relative)

    def set_body_position(self, xyt: np.ndarray, body_name: str, relative: bool = False) -> None:
        """Set the robot position in the simulation.

        Args:
            xyt (np.ndarray): The desired pose of the robot in world coordinates (x, y, theta).
            body_name (str): The name of the body to set the position of.
            relative (bool): If True, the pose is relative to the current pose of the robot.
        """

        # Get mjdata and mjmodel from simulator
        mjdata = self.robot_sim.mjdata

        # Compute absolute goal
        if relative:
            xyt_base = self.get_body_xyt(body_name)
            xyt_goal = xyt_base_to_global(xyt, xyt_base)
        else:
            xyt_goal = xyt

        from emet.simulation.sim_manipulation import set_free_body_pose

        mjmodel = self.robot_sim.mjmodel
        half = float(xyt_goal[2]) * 0.5
        quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)
        set_free_body_pose(
            mjmodel,
            mjdata,
            body_name,
            [float(xyt_goal[0]), float(xyt_goal[1]), float(mjdata.body(body_name).xpos[2])],
            quat,
        )

    def reset(self):
        """Reset the robot to the initial state."""
        self.robot_sim.reset_state()

    _default_cameras = [
        StretchCameras.cam_d405_rgb,
        StretchCameras.cam_d435i_rgb,
        StretchCameras.cam_d405_depth,
        StretchCameras.cam_d435i_depth,
    ]

    def __init__(
        self,
        *args,
        scene_path: str | None = None,
        scene_model: str | None = None,
        simulation_rate: int = 80,
        camera_hz: int = 15,
        config_name: str = "noplan_velocity_sim",
        objects_info: dict[str, Any] | None = None,
        no_cameras: bool = False,
        environment: dict[str, Any] | None = None,
        scene_source_basename: str | None = None,
        debug_molmospaces_spawn: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        cameras_to_use = [] if no_cameras else self._default_cameras
        self._cameras_enabled = bool(cameras_to_use)
        molmospaces_environment = dict(environment) if environment else None
        # TODO: decide how we want to save scenes, if they should be here in stretch_ai or in stretch_mujoco
        # They should probably stay in stretch mujoco
        if scene_path is None:
            scene_path = default_scene_xml_path
        if scene_model is not None:
            if scene_path is not None:
                logger.warning("Both scene model and scene path provided. Using scene model.")
            self.robot_sim = StretchMujocoSimulator(
                model=scene_model,
                cameras_to_use=cameras_to_use,
                camera_hz=camera_hz,
                molmospaces_environment=molmospaces_environment,
                debug_molmospaces_spawn=debug_molmospaces_spawn,
            )
        else:
            self.robot_sim = StretchMujocoSimulator(
                scene_xml_path=scene_path,
                cameras_to_use=cameras_to_use,
                camera_hz=camera_hz,
                molmospaces_environment=molmospaces_environment,
                debug_molmospaces_spawn=debug_molmospaces_spawn,
            )
        # Get the intrinsic parameters of the d435i rgb camera
        (
            fx,
            _,
            cx,
            _,
            fy,
            cy,
            _,
            _,
            _,
        ) = StretchCameras.cam_d435i_rgb.initial_camera_settings.get_intrinsic_params_k()
        # Rotate the head camera matrix as the head camera realsense is rotated 90 degrees
        self.head_K = np.array([[fy, 0, cy], [0, fx, cx], [0, 0, 1]])

        # Get the intrinsic parameters of the d405 rgb camera
        (
            fx,
            _,
            cx,
            _,
            fy,
            cy,
            _,
            _,
            _,
        ) = StretchCameras.cam_d405_rgb.initial_camera_settings.get_intrinsic_params_k()
        # No need to rotate the ee camera matrix as it is not rotated
        self.ee_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        self.simulation_rate = simulation_rate
        self.objects_info = objects_info
        self._environment_descriptor = dict(environment) if environment else None
        self._scene_xml_path = str(scene_path).strip() if scene_path else None
        if scene_source_basename:
            self._scene_source_basename = scene_source_basename
        elif scene_path:
            self._scene_source_basename = Path(scene_path).name
        else:
            self._scene_source_basename = None

        # Hard coded printout rates
        self.report_steps = 1000
        self.fast_report_steps = 10000
        self.servo_report_steps = 1000

        self._camera_data = None
        self._status = None
        self._initial_xyt = None
        self._manip_xyt = None

        # Controller stuff
        # Is the velocity controller active?
        # TODO: not sure if we want this
        # self.active = False
        # Is it done?
        self.is_done = False
        # Goal set time
        self.goal_set_t: float | None = None
        self.xyt_goal: np.ndarray | None = None
        self._base_controller_at_goal = False
        self.control_mode = "navigation"
        self.controller_finished = True
        self.active = False

        self._emet_session: dict[str, Any] | None = None

        # Control module
        controller_cfg = get_control_config(config_name)
        self.controller = GotoVelocityController(controller_cfg)
        # Update the velocity and acceleration configs from the file
        self.controller.update_velocity_profile(
            controller_cfg.v_max,
            controller_cfg.w_max,
            controller_cfg.acc_lin,
            controller_cfg.acc_ang,
        )

    def set_goal_pose(self, xyt_goal: np.ndarray, relative: bool = False):
        """Set the goal pose for the robot. The controller will then try to reach this goal pose.

        Args:
            xyt_goal (np.ndarray): Goal pose for the robot in world coordinates (x, y, theta).
            relative (bool): If True, the goal is relative to the current pose.
        """
        assert len(xyt_goal) == 3, "Goal pose should be of size 3 (x, y, theta)"

        # Compute absolute goal
        if relative:
            xyt_base = self.get_base_pose()
            xyt_goal = xyt_base_to_global(xyt_goal, xyt_base)
        else:
            xyt_goal = xyt_goal

        if self.debug_control_loop or self.debug_set_goal_pose:
            print("-" * 20)
            print("Control loop callback: ", self.active, self.xyt_goal)
            print("Currently at:", self.get_base_pose())
            print("Setting goal to:", xyt_goal)
            print("Passed goal was: ", xyt_goal)
            print("Relative: ", relative)
            print("-" * 20)

        self.controller.update_goal(xyt_goal)
        self.xyt_goal = self.controller.xyt_goal
        self.active = True

        self.is_done = False
        self.goal_set_t = timeit.default_timer()
        self.controller_finished = False
        self._base_controller_at_goal = False

    def _control_loop_thread(self):
        """Control loop thread for the velocity controller"""
        while self.is_running():
            self.control_loop_callback()
            time.sleep(1 / self.hz)

    def control_loop_callback(self):
        # Serialize controller writes with command cancellation.
        with self._command_lock:
            self._control_loop_callback()

    def _control_loop_callback(self):
        """Actual controller timer callback"""

        if self._status is None:
            vel_odom = [0, 0]
        else:
            vel_odom = self._status["base"].x_vel, self._status["base"].theta_vel

        if self.debug_control_loop:
            print("Control loop callback: ", self.active, self.xyt_goal, vel_odom)

        base_xyt = self.get_base_pose()
        if base_xyt is None:
            return

        self.controller.update_pose_feedback(base_xyt)

        if self.active and self.xyt_goal is not None:
            # Compute control
            self.is_done = False
            v_cmd, w_cmd = self.controller.compute_control()
            done = self.controller.is_done()

            # self.get_logger().info(f"veclocities {v_cmd} and {w_cmd}")
            # Compute timeout
            time_since_goal_set = timeit.default_timer() - self.goal_set_t
            if self.controller.timeout(time_since_goal_set):
                done = True
                v_cmd, w_cmd = 0, 0

            # Check if actually done (velocity = 0)
            if done and vel_odom is not None:
                if abs(vel_odom[0]) < VEL_THRESHOlD and abs(vel_odom[1]) < RVEL_THRESHOLD:
                    if not self.controller_finished:
                        self.controller_finished = True
                        self.done_since = timeit.default_timer()
                    elif self.controller_finished and (timeit.default_timer() - self.done_since) > self.done_t:
                        self.is_done = True
                else:
                    self.controller_finished = False
                    self.done_since = timeit.default_timer()

            # Command robot
            if self.debug_control_loop:
                print(f"Commanding robot with {v_cmd} and {w_cmd}")
            try:
                self.robot_sim.set_base_velocity(v_linear=v_cmd, omega=w_cmd)
            except ConnectionError:
                pass
            self._base_controller_at_goal = self.controller_finished and self.is_done

            if self.is_done:
                self.active = False
                self.xyt_goal = None

    def base_controller_at_goal(self):
        """Check if the base controller is at goal."""
        return self._base_controller_at_goal

    def start_navigation_command(self, action):
        self._contract_navigation_context = None
        self.handle_action(action)
        if self._contract_navigation_context is None:
            raise RuntimeError("simulator did not install navigation goal")
        return self._contract_navigation_context

    def navigation_command_result(self, context):
        from emet.core.navigation_result import measured_arrival

        if not self.base_controller_at_goal():
            return None
        return measured_arrival(context, self.get_base_pose(), xy_tolerance=0.07, yaw_tolerance=0.15)

    def cancel_navigation_command(self):
        self.active = False
        self.xyt_goal = None
        return self.robot_sim.cancel_base_motion()

    def _stretch_sim_publish_ok(self) -> bool:
        """True while the Stretch subprocess can answer pull_* / poses (avoid IPC errors during shutdown)."""
        if self.robot_sim is None:
            return False
        try:
            return bool(self.robot_sim.is_running())
        except (ConnectionError, ConnectionResetError, OSError):
            return False

    def get_joint_state(self):
        """Get the joint state of the robot."""
        status = self._status

        positions = np.zeros(constants.stretch_degrees_of_freedom)
        velocities = np.zeros(constants.stretch_degrees_of_freedom)
        efforts = np.zeros(constants.stretch_degrees_of_freedom)

        if status is None:
            return positions, velocities, efforts

        # Lift joint
        positions[HelloStretchIdx.LIFT] = status["lift"].pos
        velocities[HelloStretchIdx.LIFT] = status["lift"].vel

        # Arm joints
        positions[HelloStretchIdx.ARM] = status["arm"].pos
        velocities[HelloStretchIdx.ARM] = status["arm"].vel

        # Wrist roll joint
        positions[HelloStretchIdx.WRIST_ROLL] = status["wrist_roll"].pos
        velocities[HelloStretchIdx.WRIST_ROLL] = status["wrist_roll"].vel

        # Wrist yaw joint
        positions[HelloStretchIdx.WRIST_YAW] = status["wrist_yaw"].pos
        velocities[HelloStretchIdx.WRIST_YAW] = status["wrist_yaw"].vel

        # Wrist pitch joint
        positions[HelloStretchIdx.WRIST_PITCH] = status["wrist_pitch"].pos
        velocities[HelloStretchIdx.WRIST_PITCH] = status["wrist_pitch"].vel

        # Gripper joint
        positions[HelloStretchIdx.GRIPPER] = status["gripper"].pos
        velocities[HelloStretchIdx.GRIPPER] = status["gripper"].vel

        # Head pan joint
        positions[HelloStretchIdx.HEAD_PAN] = status["head_pan"].pos
        velocities[HelloStretchIdx.HEAD_PAN] = status["head_pan"].vel

        # Head tilt joint
        positions[HelloStretchIdx.HEAD_TILT] = status["head_tilt"].pos
        velocities[HelloStretchIdx.HEAD_TILT] = status["head_tilt"].vel

        # Base SE(2) from status (used by clients to detect "still moving" without wall-clock pose deltas).
        base = status["base"]
        velocities[HelloStretchIdx.BASE_X] = float(getattr(base, "x_vel", 0.0) or 0.0)
        velocities[HelloStretchIdx.BASE_THETA] = float(getattr(base, "theta_vel", 0.0) or 0.0)

        if self.in_manipulation_mode:
            # Get current base xyt
            xyt = self.get_base_pose()
            # Compute the relative xyt
            xyt = xyt_global_to_base(xyt, self._manip_xyt)
            # Set the base x to the x from this xyt
            positions[HelloStretchIdx.BASE_X] = xyt[0]

        return positions, velocities, efforts

    def get_base_pose(self) -> np.ndarray:
        """Base pose is the SE(2) pose of the base in world coords (x, y, theta)"""
        if self._initial_xyt is None:
            return None
        if not self._stretch_sim_publish_ok():
            return None
        try:
            xyt = self.robot_sim.get_base_pose()
        except (ConnectionError, ConnectionResetError, OSError):
            return None
        return xyt_global_to_base(xyt, self._initial_xyt)

    def get_ee_pose(self) -> np.ndarray:
        """EE pose is the 4x4 matrix of the end effector location in world coords"""
        if self._initial_xyt is None:
            return None
        if not self._stretch_sim_publish_ok():
            return None
        try:
            pose = self.robot_sim.get_ee_pose()
        except (ConnectionError, ConnectionResetError, OSError):
            return None
        return pose_global_to_base(pose, self._initial_xyt)

    def _head_camera_opencv_world(self) -> np.ndarray:
        """Head camera 4x4 OpenCV-style transform in MuJoCo world (ZMQ / voxel / Rerun contract)."""
        pose = self.robot_sim.get_link_pose(STRETCH_CAMERA_FRAME)
        pose[:3, :3] = pose[:3, :3] @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        return pose

    def get_head_camera_pose(self) -> np.ndarray:
        """Head camera pose in episode-relative (nav) coords."""
        if self._initial_xyt is None:
            return None
        if not self._stretch_sim_publish_ok():
            return None
        try:
            return pose_global_to_base(self._head_camera_opencv_world(), self._initial_xyt)
        except (ConnectionError, ConnectionResetError, OSError):
            return None

    def get_ee_camera_pose(self) -> np.ndarray:
        """Get the end effector camera pose in world coords"""
        if self._initial_xyt is None:
            return None
        if not self._stretch_sim_publish_ok():
            return None
        try:
            pose = self.robot_sim.get_link_pose("gripper_camera_color_optical_frame")
        except (ConnectionError, ConnectionResetError, OSError):
            return None
        return pose_global_to_base(pose, self._initial_xyt)

    def set_posture(self, posture: str) -> bool:
        """Set the posture of the robot."""

        # Assert posture in ["manipulation", "navigation"]
        if posture not in ["manipulation", "navigation"]:
            logger.error(f"Posture {posture} not supported. Must be in ['manipulation', 'navigation']")
            return False

        # Set the posture
        if posture == "navigation":
            self.manip_to(constants.STRETCH_NAVIGATION_Q, all_joints=True)
        elif posture == "manipulation":
            self.manip_to(constants.STRETCH_PREGRASP_Q, all_joints=True)
        else:
            logger.error(f"Posture {posture} not supported")
            return False
        self.control_mode = posture
        return True

    def manip_to(self, q: np.ndarray, all_joints: bool = False, skip_gripper: bool = False) -> None:
        """Move the robot to a given joint configuration. q should be of size 11.

        Args:
            q (np.ndarray): Joint configuration to move the robot to.
        """

        # Check size

        if all_joints:
            assert len(q) == stretch_dof, f"q should be of size {stretch_dof}"
            # Move the robot to the given joint configuration
            for idx in range(3, stretch_dof):
                if idx == HelloStretchIdx.GRIPPER and skip_gripper:
                    continue
                self.robot_sim.move_to(mujoco_actuators[idx], q[idx])
        else:
            assert len(q) == len(manip_idx), f"q should be of size {len(manip_idx)}"
            # Just read the manipulator joints
            for i, idx in enumerate(manip_idx):
                if idx == HelloStretchIdx.BASE_X:
                    if not self.in_manipulation_mode:
                        logger.warning("Cannot move base by base_x alone in navigation mode")
                        continue
                    elif self._manip_xyt is None:
                        logger.error("Manipulation mode not set up correctly")
                    # Send an xyt goal: x, y, theta
                    # This is computed based on self._manip_xyt
                    xyt_delta = [q[i], 0, 0]
                    xyt_goal = xyt_base_to_global(xyt_delta, self._manip_xyt)
                    print("Manip xyt =", self._manip_xyt)
                    print("delta =", xyt_delta)
                    print("Setting base to", xyt_goal)
                    print("Current base is", self.get_base_pose())
                    self.set_goal_pose(xyt_goal, relative=False)
                else:
                    self.robot_sim.move_to(mujoco_actuators[idx], q[i])

    def __del__(self):
        self.stop()

    def stop(self):
        """Stop the server and the robot. Sets _done first so spin threads exit cleanly."""
        self._done = True
        time.sleep(0.3)
        if hasattr(self, "_control_thread") and self._control_thread is not None:
            self._control_thread.join(timeout=2.0)
        for name in ("_send_thread", "_recv_thread", "_send_state_thread", "_send_servo_thread"):
            t = getattr(self, name, None)
            if t is not None and t.is_alive():
                t.join(timeout=1.0)
        if hasattr(self, "robot_sim") and self.robot_sim is not None:
            self.robot_sim.stop()

    @override
    def get_control_mode(self) -> str:
        """Get the control mode of the robot."""
        return self.control_mode

    @override
    def start(
        self,
        show_viewer_ui: bool = False,
        robocasa: bool = False,
        headless: bool = False,
        use_glx: bool = False,
    ) -> None:
        self.robot_sim.start(
            show_viewer_ui=show_viewer_ui, headless=headless, use_glx=use_glx
        )  # This will start the simulation and open Mujoco-Viewer window
        if not self.robot_sim.is_running():
            raise RuntimeError(
                "MuJoCo simulator did not start. See above for errors; on WSL try DISPLAY=:99 and --use-glx, or --no-cameras."
            )
        self._emet_session = self._build_emet_session_stretch(robocasa=robocasa)
        if self._is_molmospaces_session() and not molmospaces_nav_teleport_enabled():
            logger.info("MolmoSpaces navigation: wheel/goal drive (EMET_MOLMOSPACES_NAV_TELEPORT=0)")
        super().start()

        # Create a thread for the control loop
        self._control_thread = threading.Thread(target=self._control_loop_thread)
        self._control_thread.start()

        self._initial_xyt = self.robot_sim.get_base_pose()
        if self._emet_session is not None:
            apply_navigation_origin_to_session(self._emet_session, self._initial_xyt)

        while self.is_running():
            try:
                self._camera_data = self.robot_sim.pull_camera_data()
                self._status = self.robot_sim.pull_status()
            except ConnectionError:
                break
            time.sleep(1 / self.simulation_rate)

    def _build_emet_session_stretch(self, *, robocasa: bool) -> dict[str, Any]:
        spec = self.get_robot_spec()
        if self._environment_descriptor:
            env = dict(self._environment_descriptor)
        elif robocasa:
            env = {"kind": "robocasa"}
        else:
            env = {"kind": "stretch_default_scene"}
        caps: dict[str, Any] = {
            "teleport_base": self._is_molmospaces_session() and molmospaces_nav_teleport_enabled(),
            "depth": self._cameras_enabled,
            "num_cameras": 2 if self._cameras_enabled else 0,
            "dof": int(spec.dof),
            "sim_set_body_pose": True,
            "sim_set_joint_qpos": True,
        }
        session: dict[str, Any] = {
            EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY: CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
            "runtime_kind": "stretch_mujoco_sim",
            "is_simulation": True,
            EMET_ZMQ_ROBOT_ID_KEY: spec.name,
            "capabilities": caps,
            "environment": env,
        }
        if self._scene_source_basename:
            session["scene_source_basename"] = self._scene_source_basename
        if self._environment_descriptor and self._environment_descriptor.get("spawn_floor_map") is not None:
            session["spawn_floor_map"] = self._environment_descriptor["spawn_floor_map"]
        env_kind = env.get("kind") if isinstance(env, dict) else None
        gt_model, gt_data = self._gt_model_data_for_session()
        attach_sim_object_placements_to_session(
            session,
            objects_info=self.objects_info,
            environment_kind=str(env_kind) if env_kind else None,
            model=gt_model,
            data=gt_data,
            robot_root_name="base_link",
        )
        return session

    def _gt_model_data_for_session(self) -> tuple[Any, Any]:
        """Resolve MuJoCo model/data for GT scan (subprocess sim may only expose MJCF on disk)."""
        import mujoco

        from emet.simulation.sim_object_placements import _mj_forward, mujoco_model_data_for_gt_scan

        model, data = mujoco_model_data_for_gt_scan(self.robot_sim)
        if model is not None:
            return model, data
        scene_path = getattr(self, "_scene_xml_path", None)
        if scene_path and Path(str(scene_path)).is_file():
            model = mujoco.MjModel.from_xml_path(str(scene_path))
            data = mujoco.MjData(model)
            _mj_forward(model, data)
            return model, data
        return None, None

    def _patch_emet_session_body_pos(
        self,
        body: str,
        pos: list[float],
        quat: list[float] | None = None,
    ) -> None:
        """Update one body entry in cached session GT (Stretch sim runs MuJoCo in a subprocess)."""
        if self._emet_session is None:
            return
        placements = self._emet_session.get("sim_object_placements")
        if not isinstance(placements, dict) or body not in placements:
            return
        entry = placements[body]
        if not isinstance(entry, dict):
            return
        entry["pos"] = [float(x) for x in pos[:3]]
        if quat is not None:
            entry["quat"] = [float(x) for x in quat[:4]]

    def _is_molmospaces_session(self) -> bool:
        env = self._environment_descriptor
        if isinstance(env, dict) and env.get("kind") == "molmospaces":
            return True
        bn = self._scene_source_basename or ""
        return bn.startswith("molmospaces_merged")

    def _use_molmospaces_nav_teleport(self) -> bool:
        return self._is_molmospaces_session() and molmospaces_nav_teleport_enabled()

    def _teleport_base_world(self, world_xyt: np.ndarray, *, timeout: float = 2.0) -> bool:
        """Apply free-joint teleport in the MuJoCo subprocess and wait until pose matches."""
        goal = np.asarray(world_xyt, dtype=np.float64).reshape(-1)[:3]
        # Do not call set_base_velocity here: StatusCommand.set_base_velocity clears teleport_base.trigger.
        ok = self.robot_sim.teleport_base_xyt(
            float(goal[0]), float(goal[1]), float(goal[2]), wait=True, timeout=timeout
        )
        if not ok:
            cur = self.robot_sim.get_base_pose()
            logger.warning(
                f"MolmoSpaces teleport did not reach goal within {timeout:.2f}s "
                f"(goal={goal.tolist()!r}, current={None if cur is None else list(cur)!r})"
            )
        return ok

    def _xyt_action_to_world(self, xyt: np.ndarray, *, relative: bool) -> np.ndarray:
        """Map client ``xyt`` (spawn-relative) to world coordinates for MuJoCo free joint."""
        raw = np.asarray(xyt, dtype=np.float64).reshape(-1)[:3]
        init = self._initial_xyt
        if init is None:
            init = np.zeros(3, dtype=np.float64)
        if relative:
            cur = self.get_base_pose()
            if cur is None:
                cur = np.zeros(3, dtype=np.float64)
            rel = xyt_base_to_global(raw, cur)
        else:
            rel = raw
        return xyt_base_to_global(rel, init)

    def _world_to_base_xyt(self, world: np.ndarray) -> np.ndarray:
        """Map a MuJoCo **world** xyt to the episode-relative frame the velocity controller uses."""
        raw = np.asarray(world, dtype=np.float64).reshape(-1)[:3]
        init = self._initial_xyt
        if init is None:
            init = np.zeros(3, dtype=np.float64)
        return xyt_global_to_base(raw, init)

    def _attach_emet_session(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._emet_session is not None:
            message[EMET_ZMQ_SESSION_KEY] = self._emet_session
        return message

    @override
    def handle_action(self, action: dict[str, Any]):
        """Handle the action received from the client."""
        if EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY in action:
            path_gt, exclude_robot, as_json = parse_ground_truth_dump_action_field(
                action[EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY]
            )
            if path_gt:
                hdr = None
                envd = self._environment_descriptor
                if isinstance(envd, dict):
                    hdr = {k: envd[k] for k in sorted(envd.keys()) if k in ("kind", "task", "style", "layout")}

                try:
                    model = getattr(self.robot_sim, "mjmodel", None)
                    data = getattr(self.robot_sim, "mjdata", None)
                    if model is None or data is None:
                        logger.warning(
                            "mujoco_ground_truth_dump: no mjmodel/mjdata; cannot write %r",
                            path_gt,
                        )
                    else:
                        out = mujoco_ground_truth_write_path(
                            model,
                            data,
                            dest=path_gt,
                            exclude_robot=exclude_robot,
                            robot_base_body_name=str(self.get_robot_spec().base_link_name),
                            json=as_json,
                            extras=hdr,
                        )
                        logger.info(f"Wrote MuJoCo ground-truth snapshot -> {out}")
                except Exception as e:
                    logger.error("mujoco_ground_truth_dump failed for %r: %s", path_gt, e)

        if EMET_ACTION_SIM_SET_BODY_POSE_KEY in action:
            body, pos, quat = parse_sim_set_body_pose_action(action[EMET_ACTION_SIM_SET_BODY_POSE_KEY])
            if body and pos is not None:
                pos_arr = np.asarray(pos, dtype=np.float64).reshape(3)
                quat_arr = np.asarray(quat, dtype=np.float64).reshape(4) if quat is not None else None
                ok = self.robot_sim.teleport_body_pose(
                    body,
                    pos_arr,
                    quat_arr,
                    wait=True,
                    timeout=2.0,
                )
                if ok:
                    self._patch_emet_session_body_pos(body, pos, quat)
                else:
                    logger.warning("sim_set_body_pose failed for body %r", body)

        if EMET_ACTION_SIM_SET_JOINT_QPOS_KEY in action:
            joint, value = parse_sim_set_joint_qpos_action(action[EMET_ACTION_SIM_SET_JOINT_QPOS_KEY])
            if joint and value is not None:
                measured = self.robot_sim.set_joint_qpos(joint, value, wait=True, timeout=2.0)
                if measured is not None:
                    logger.info(f"sim_set_joint_qpos: {joint!r} requested={value:.4f} measured={measured:.4f}")
                else:
                    logger.warning(f"sim_set_joint_qpos failed for joint {joint!r}")

        if "control_mode" in action:
            new_control_mode = action["control_mode"]
            if new_control_mode not in ["navigation", "manipulation"]:
                logger.error(f"Control mode {new_control_mode} not supported")
            # If we are switching to manipulation mode, recort base xyt
            if new_control_mode == "manipulation" and self.get_control_mode() == "navigation":
                self._manip_xyt = self.get_base_pose()
                logger.info(
                    "Switching to manipulation mode, recording initial base pose for manipulation: "
                    + str(self._manip_xyt)
                )

            self.control_mode = new_control_mode

        if "posture" in action:
            self.set_posture(action["posture"])
        if "gripper" in action:
            # Get current gripper pose
            positions, _, _ = self.get_joint_state()
            current_gripper_pos = positions[HelloStretchIdx.GRIPPER]
            target_gripper_pos = action["gripper"]
            step = 0.01
            t0 = timeit.default_timer()
            if current_gripper_pos < target_gripper_pos:
                while current_gripper_pos < target_gripper_pos:
                    current_gripper_pos += step
                    positions, _, _ = self.get_joint_state()
                    # TODO: remove debug print
                    # print(current_gripper_pos, positions[HelloStretchIdx.GRIPPER])
                    self.robot_sim.move_to("gripper", current_gripper_pos)
                    time.sleep(0.01)
                    dt = timeit.default_timer() - t0
                    if dt > 5:
                        logger.error("Gripper move took too long")
                        break
            else:
                while current_gripper_pos > target_gripper_pos:
                    current_gripper_pos -= step
                    positions, _, _ = self.get_joint_state()
                    # TODO: remove debug print
                    # print(current_gripper_pos, positions[HelloStretchIdx.GRIPPER])
                    self.robot_sim.move_to("gripper", current_gripper_pos)
                    time.sleep(0.02)
                    dt = timeit.default_timer() - t0
                    if dt > 5:
                        logger.error("Gripper move took too long")
                        break
        elif "say" in action:
            pass
        if "joint" in action:
            # Move the robot to the given joint configuration
            # Only send the manipulator joints, not gripper or head
            print("[ROBOT] Moving to joint configuration", action["joint"])
            self.manip_to(action["joint"], all_joints=False)
        if "head_to" in action:
            self.robot_sim.move_to("head_pan", action["head_to"][0])
            self.robot_sim.move_to("head_tilt", action["head_to"][1])
        if "base_velocity" in action:
            self.robot_sim.set_base_velocity(v_linear=action["base_velocity"]["v"], omega=action["base_velocity"]["w"])
        elif "xyt" in action:
            relative_motion = bool(action.get("nav_relative", False))
            nav_world = bool(action.get("nav_world", False))
            nav_teleport = bool(action.get("nav_teleport", False)) or self._use_molmospaces_nav_teleport()
            if not nav_teleport:
                from emet.simulation.env_flags import env_sim_nav_teleport

                if env_sim_nav_teleport():
                    nav_teleport = True
            raw = np.asarray(action["xyt"], dtype=np.float64).reshape(-1)[:3]
            if nav_world and not relative_motion:
                world = raw
            else:
                world = self._xyt_action_to_world(raw, relative=relative_motion)
            # The velocity controller tracks pose in **episode-relative** frame (get_base_pose is
            # xyt_global_to_base(_initial_xyt)). nav_world goals arrive in MuJoCo world frame, so
            # convert back to episode-relative before set_goal_pose, or the goal is offset by the
            # spawn position and the base circles forever. Non-world xyt is already episode-relative.
            goal_episode = self._world_to_base_xyt(world)
            self._contract_navigation_context = {
                "resolved_goal": goal_episode.tolist(),
                "frame": "episode",
                "motion_mode": "teleport" if nav_teleport else "velocity_drive",
            }
            if nav_teleport:
                if self._teleport_base_world(world):
                    self.active = False
                    self.xyt_goal = None
                    self._base_controller_at_goal = True
                    self.is_done = True
                    logger.info(
                        f"MolmoSpaces teleport nav: base at x={world[0]:.3f} y={world[1]:.3f} theta={world[2]:.3f}"
                    )
                else:
                    raise RuntimeError("requested teleport could not be applied")
            else:
                self.set_goal_pose(goal_episode, relative=False)

    @override
    def get_full_observation_message(self) -> dict[str, Any]:
        """Get the full observation message for the robot. This includes the full state of the robot, including images and depth images."""
        if not self._stretch_sim_publish_ok():
            return None
        cam_data = self._camera_data
        if cam_data is None:
            return None

        # Rotate the camera matrix and images
        rgb = cam_data.cam_d435i_rgb
        depth = cam_data.cam_d435i_depth
        if not isinstance(rgb, np.ndarray) or not isinstance(depth, np.ndarray) or rgb.ndim < 2 or depth.ndim < 2:
            # Happens when cameras are disabled (e.g. --no-cameras) or not initialized yet.
            return None
        rgb = np.rot90(rgb, k=-1)
        depth = np.rot90(depth, k=-1)
        rgb_height, rgb_width = rgb_height_width_for_zmq(rgb)

        # Convert depth into int format
        depth = (depth * 1000).astype(np.uint16)

        # Get the joint state
        positions, _, _ = self.get_joint_state()

        # Make both into jpegs
        rgb = compression.to_jpg(rgb)
        depth = compression.to_jp2(depth)

        if self._initial_xyt is None or not self._stretch_sim_publish_ok():
            return None
        xyt = self.get_base_pose()
        if xyt is None:
            return None
        try:
            # ZMQ contract: camera_pose is MuJoCo world; gps/compass are episode-relative.
            # Frame contract: src/test/simulation/test_zmq_observation_frame_contract.py
            head_cam_world = self._head_camera_opencv_world()
            ee_world = self.robot_sim.get_ee_pose()
        except (ConnectionError, ConnectionResetError, OSError):
            return None

        # Get the other fields from an observation
        message = {
            "rgb": rgb,
            "depth": depth,
            "camera_K": self.head_K,
            "camera_pose": head_cam_world,
            "ee_pose": ee_world,
            "joint": positions,
            "gps": xyt[:2],
            "compass": np.array([xyt[2]]),
            "rgb_width": rgb_width,
            "rgb_height": rgb_height,
            "control_mode": self.get_control_mode(),
            "last_motion_failed": False,
            "recv_address": self.recv_address,
            "step": self._last_step,
            "at_goal": self.base_controller_at_goal(),
            "is_simulation": True,
            "lidar_points": None,
            "lidar_timestamp": None,
            EMET_ZMQ_ROBOT_ID_KEY: self.get_robot_spec().name,
        }
        return self._attach_emet_session(message)

    @override
    def get_state_message(self) -> dict[str, Any]:
        """Get the state message for the robot. This is a smalll message that includes floating point information and booleans like if the robot is homed."""
        if not self._stretch_sim_publish_ok():
            return None
        q, dq, eff = self.get_joint_state()
        base_pose = self.get_base_pose()
        ee_pose = self.get_ee_pose()
        if base_pose is None or ee_pose is None:
            return None
        message = {
            "base_pose": base_pose,
            "ee_pose": ee_pose,
            "joint_positions": q,
            "joint_velocities": dq,
            "joint_efforts": eff,
            "control_mode": self.get_control_mode(),
            "at_goal": self.base_controller_at_goal(),
            "is_homed": True,
            "is_runstopped": False,
            "step": self._last_step,
            EMET_ZMQ_SIM_TIME_RATIO_KEY: getattr(self._status, "sim_to_real_ratio", None),
            EMET_ZMQ_ROBOT_ID_KEY: self.get_robot_spec().name,
        }
        return self._attach_emet_session(message)

    @override
    def get_servo_message(self) -> dict[str, Any]:
        """Get messages for e2e policy learning and visual servoing. These are images and depth images, but lower resolution than the large full state observations, and they include the end effector camera."""
        if not self._stretch_sim_publish_ok():
            return None

        cam_data = self._camera_data
        if cam_data is None:
            return None

        head_color_image = cam_data.cam_d435i_rgb
        head_depth_image = cam_data.cam_d435i_depth
        if (
            not isinstance(head_color_image, np.ndarray)
            or not isinstance(head_depth_image, np.ndarray)
            or head_color_image.ndim < 2
            or head_depth_image.ndim < 2
        ):
            return None
        head_color_image = np.rot90(head_color_image, k=-1)
        head_depth_image = np.rot90(head_depth_image, k=-1)
        ee_color_image = cam_data.cam_d405_rgb
        ee_depth_image = cam_data.cam_d405_depth
        if (
            not isinstance(ee_color_image, np.ndarray)
            or not isinstance(ee_depth_image, np.ndarray)
            or ee_color_image.ndim < 2
            or ee_depth_image.ndim < 2
        ):
            return None

        # Adapt color so we can use higher shutter speed
        # TODO: do we need this? Probably not.
        # ee_color_image = adjust_gamma(ee_color_image, 2.5)

        ee_color_image, ee_depth_image = self._rescale_color_and_depth(
            ee_color_image, ee_depth_image, self.ee_image_scaling
        )
        head_color_image, head_depth_image = self._rescale_color_and_depth(
            head_color_image, head_depth_image, self.image_scaling
        )

        # Conversion
        ee_depth_image = (ee_depth_image * 1000).astype(np.uint16)
        head_depth_image = (head_depth_image * 1000).astype(np.uint16)

        # Compress the images
        compressed_ee_depth_image = compression.to_jp2(ee_depth_image)
        compressed_ee_color_image = compression.to_jpg(ee_color_image)
        compressed_head_depth_image = compression.to_jp2(head_depth_image)
        compressed_head_color_image = compression.to_jpg(head_color_image)

        # Get position info
        positions, _, _ = self.get_joint_state()
        if self._initial_xyt is None:
            return None
        try:
            ee_pose_cam = self.robot_sim.get_link_pose("gripper_camera_color_optical_frame")
            head_pose_cam = self._head_camera_opencv_world()
            ee_pose_mat = self.robot_sim.get_ee_pose()
        except (ConnectionError, ConnectionResetError, OSError):
            return None

        message = {
            "ee_cam/color_camera_K": scale_camera_matrix(self.ee_K, self.ee_image_scaling),
            "ee_cam/depth_camera_K": scale_camera_matrix(self.ee_K, self.ee_image_scaling),
            "ee_cam/color_image": compressed_ee_color_image,
            "ee_cam/depth_image": compressed_ee_depth_image,
            "ee_cam/color_image/shape": ee_color_image.shape,
            "ee_cam/depth_image/shape": ee_depth_image.shape,
            "ee_cam/image_scaling": self.ee_image_scaling,
            "ee_cam/depth_scaling": self.ee_depth_scaling,
            "ee_cam/pose": ee_pose_cam,
            "ee/pose": ee_pose_mat,
            "head_cam/color_camera_K": scale_camera_matrix(self.head_K, self.image_scaling),
            "head_cam/depth_camera_K": scale_camera_matrix(self.head_K, self.image_scaling),
            "head_cam/color_image": compressed_head_color_image,
            "head_cam/depth_image": compressed_head_depth_image,
            "head_cam/color_image/shape": head_color_image.shape,
            "head_cam/depth_image/shape": head_depth_image.shape,
            "head_cam/image_scaling": self.image_scaling,
            "head_cam/depth_scaling": self.depth_scaling,
            "head_cam/pose": head_pose_cam,
            "robot/config": positions,
            "is_simulation": True,
            "step": self._last_step,
            EMET_ZMQ_ROBOT_ID_KEY: self.get_robot_spec().name,
        }
        return self._attach_emet_session(message)

    @override
    def is_running(self) -> bool:
        """Check if the server is running. Will be used to make sure inner loops terminate.

        Returns:
            bool: True if the server is running, False otherwise."""
        return self.running and self.robot_sim is not None and self.robot_sim.is_running()

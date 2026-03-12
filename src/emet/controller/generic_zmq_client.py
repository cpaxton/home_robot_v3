# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Robot-agnostic ZMQ client driven by RobotSpec.

This client communicates with a MuJoCo ZMQ server using the same protocol as
StretchZmqClient, but derives joint indexing and DOF from a RobotSpec rather
than hardcoding Stretch-specific constants.
"""

import sys
import threading
import time
import timeit
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import zmq

import emet.utils.compression as compression
from emet.core.interfaces import ContinuousNavigationAction, Observations
from emet.core.parameters import Parameters, get_parameters
from emet.core.robot import AbstractRobotClient, ControlMode
from emet.robots.base import RobotSpec
from emet.utils.geometry import (
    angle_difference,
    xyt_base_to_global,
)
from emet.utils.image import Camera
from emet.utils.logger import Logger
from emet.utils.memory import lookup_address

logger = Logger(__name__)


class GenericZmqClient(AbstractRobotClient):
    """ZMQ client parameterised by a RobotSpec.

    Provides the same ZMQ protocol as StretchZmqClient but without
    Stretch-specific joint indices, kinematics, or camera assumptions.
    """

    num_state_report_steps: int = 10000

    def __init__(
        self,
        robot_spec: RobotSpec,
        robot_ip: str = "",
        recv_port: int = 4401,
        send_port: int = 4402,
        recv_state_port: int = 4403,
        recv_servo_port: int = 4404,
        port_offset: int = 0,
        parameters: Optional[Parameters] = None,
        use_remote_computer: bool = True,
        start_immediately: bool = True,
    ):
        super().__init__()
        if port_offset:
            recv_port += port_offset
            send_port += port_offset
            recv_state_port += port_offset
            recv_servo_port += port_offset
        self._spec = robot_spec
        self.recv_port = recv_port
        self.send_port = send_port

        self._joint_index: Dict[str, int] = {
            name: i for i, name in enumerate(robot_spec.joint_names)
        }

        if parameters is None:
            parameters = get_parameters("default_planner.yaml")
        self._parameters = parameters

        self._iter = -1
        self._seq_id = 0
        self._started = False
        self._finish = False

        self._obs: Optional[Dict[str, Any]] = None
        self._state: Optional[Dict[str, Any]] = None
        self._servo: Optional[Dict[str, Any]] = None
        self._last_step = -1

        self._obs_lock = Lock()
        self._act_lock = Lock()

        self._base_xyt = np.zeros(3)

        # ZMQ sockets
        self.context = zmq.Context()
        self.recv_socket = self._create_recv_socket(
            recv_port, robot_ip, use_remote_computer, "observations"
        )
        self.recv_state_socket = self._create_recv_socket(
            recv_state_port, robot_ip, use_remote_computer, "low level state"
        )
        self.recv_servo_socket = self._create_recv_socket(
            recv_servo_port, robot_ip, use_remote_computer, "visual servoing data"
        )

        ip_address = lookup_address(robot_ip, use_remote_computer)
        send_address = ip_address + ":" + str(send_port)
        logger.debug(f"Connecting to {send_address} to send action messages...")
        self.send_socket = self.context.socket(zmq.PUB)
        self.send_socket.setsockopt(zmq.SNDHWM, 1)
        self.send_socket.setsockopt(zmq.RCVHWM, 1)
        self.send_socket.connect(send_address)
        logger.debug("...connected.")

        if start_immediately:
            self.start()

    @property
    def spec(self) -> RobotSpec:
        return self._spec

    @property
    def parameters(self) -> Parameters:
        return self._parameters

    # -- Socket helpers -------------------------------------------------------

    def _create_recv_socket(
        self, port: int, robot_ip: str, use_remote: bool, message_type: str = ""
    ) -> zmq.Socket:
        sock = self.context.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.SNDHWM, 1)
        sock.setsockopt(zmq.RCVHWM, 1)
        sock.setsockopt(zmq.CONFLATE, 1)

        ip_address = lookup_address(robot_ip, use_remote)
        if ip_address is None:
            logger.error("No robot IP address found.")
            sys.exit(1)
        addr = f"{ip_address}:{port}"
        logger.debug(f"Connecting to {addr} to receive {message_type}...")
        sock.connect(addr)
        return sock

    def send_message(self, message: Dict[str, Any]) -> None:
        self.send_socket.send_pyobj(message)

    # -- Lifecycle ------------------------------------------------------------

    def start(self) -> bool:
        if self._started:
            return False
        self._started = True
        self._finish = False
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._state_thread.start()
        self._servo_thread = threading.Thread(target=self._servo_loop, daemon=True)
        self._servo_thread.start()
        return True

    def stop(self) -> None:
        self._finish = True

    def reset(self) -> None:
        self._obs = None
        self._state = None
        self._servo = None
        self._base_xyt = np.zeros(3)
        self._base_control_mode = ControlMode.IDLE

    # -- Receive loops --------------------------------------------------------

    def _recv_loop(self) -> None:
        while not self._finish:
            try:
                output = self.recv_socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.01)
                continue
            if output is None:
                continue
            self._seq_id += 1
            output["rgb"] = compression.from_jpg(output["rgb"])
            depth = compression.from_jp2(output["depth"]) / 1000
            output["depth"] = depth
            with self._obs_lock:
                self._obs = output
                if "step" in output:
                    self._last_step = output["step"]
                if "gps" in output and "compass" in output:
                    self._base_xyt = np.array(
                        [output["gps"][0], output["gps"][1], output["compass"][0]]
                    )

    def _state_loop(self) -> None:
        while not self._finish:
            try:
                msg = self.recv_state_socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.01)
                continue
            if msg is None:
                continue
            with self._obs_lock:
                self._state = msg
                if "step" in msg:
                    self._last_step = msg["step"]

    def _servo_loop(self) -> None:
        while not self._finish:
            try:
                msg = self.recv_servo_socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.01)
                continue
            if msg is None:
                continue
            with self._obs_lock:
                self._servo = msg

    # -- Observations ---------------------------------------------------------

    def wait_for_obs(self, timeout: float = 10.0) -> bool:
        t0 = timeit.default_timer()
        while self._obs is None:
            time.sleep(0.05)
            if timeit.default_timer() - t0 > timeout:
                logger.error("Timeout waiting for observations.")
                return False
        return True

    def get_joint_positions(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        state = self._state
        if state is not None and "joint_positions" in state:
            return np.array(state["joint_positions"])
        obs = self._obs
        if obs is not None and "joint" in obs:
            return np.array(obs["joint"])
        return None

    def get_joint_velocities(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        state = self._state
        if state is not None and "joint_velocities" in state:
            return np.array(state["joint_velocities"])
        return None

    def get_base_pose(self, timeout: float = 5.0) -> np.ndarray:
        state = self._state
        if state is not None and "base_pose" in state:
            bp = state["base_pose"]
            if bp is not None:
                return np.array(bp)
        return self._base_xyt.copy()

    def at_goal(self) -> bool:
        state = self._state
        if state is not None:
            return state.get("at_goal", False)
        obs = self._obs
        if obs is not None:
            return obs.get("at_goal", False)
        return False

    def get_observation(self, max_iter: int = 5) -> Optional[Observations]:
        """Get the latest observation from the server."""
        with self._obs_lock:
            obs = self._obs
        if obs is None:
            return None

        rgb = obs.get("rgb")
        depth = obs.get("depth")
        if rgb is None or depth is None:
            return None

        camera_K = obs.get("camera_K")
        camera_pose = obs.get("camera_pose")
        ee_pose = obs.get("ee_pose")
        joint = obs.get("joint")
        gps = obs.get("gps", np.zeros(2))
        compass = obs.get("compass", np.zeros(1))

        return Observations(
            rgb=rgb,
            depth=depth,
            camera_K=camera_K,
            camera_pose=camera_pose,
            ee_pose=ee_pose,
            joint=joint,
            gps=gps,
            compass=compass,
        )

    # -- Actions --------------------------------------------------------------

    def send_action(
        self,
        action: Dict[str, Any],
        timeout: float = 5.0,
        reliable: bool = True,
    ) -> Dict[str, Any]:
        with self._act_lock:
            block_id = max(self._iter, self._last_step + 1)
            action["step"] = block_id
            self._iter = block_id + 1
            self.send_message(action)
            while reliable and self._last_step < block_id:
                self.send_message(action)
                time.sleep(0.01)
        return action

    def move_base_to(
        self,
        xyt,
        relative=False,
        blocking=False,
        verbose: bool = False,
        timeout: Optional[float] = None,
    ) -> bool:
        if isinstance(xyt, ContinuousNavigationAction):
            xyt = xyt.xyt
        xyt = np.array(xyt, dtype=float)
        action = {"xyt": xyt.tolist(), "nav_relative": relative}
        self.send_action(action)
        if blocking:
            self._wait_at_goal(timeout=timeout or 30.0)
        return True

    def _wait_at_goal(self, timeout: float = 30.0) -> bool:
        t0 = timeit.default_timer()
        while not self.at_goal():
            time.sleep(0.05)
            if timeit.default_timer() - t0 > timeout:
                logger.warning("Timeout waiting to reach goal.")
                return False
        return True

    def set_joint_positions(self, positions: Dict[str, float]) -> None:
        """Send a joint position command using actuator names."""
        action = {"joint": list(positions.values())}
        self.send_action(action)

    def open_gripper(self, gripper_name: str = "left_gripper", amount: float = 0.05) -> None:
        action = {"gripper": amount}
        self.send_action(action)

    def close_gripper(self, gripper_name: str = "left_gripper") -> None:
        action = {"gripper": -0.1}
        self.send_action(action)

    def switch_to_navigation_mode(self) -> None:
        action = {"control_mode": "navigation"}
        self.send_action(action)
        self._base_control_mode = ControlMode.NAVIGATION

    def switch_to_manipulation_mode(self, verbose: bool = False) -> None:
        action = {"control_mode": "manipulation"}
        self.send_action(action)
        self._base_control_mode = ControlMode.MANIPULATION

    def move_to_nav_posture(self) -> None:
        action = {"posture": "navigation"}
        self.send_action(action)

    def move_to_manip_posture(self) -> None:
        action = {"posture": "manipulation"}
        self.send_action(action)

    def get_robot_model(self):
        return None

    def get_pose_graph(self) -> np.ndarray:
        return np.zeros((0, 3))

    def execute_trajectory(
        self,
        trajectory,
        pos_err_threshold: float = 0.2,
        rot_err_threshold: float = 0.75,
        spin_rate: int = 10,
        verbose: bool = False,
        per_waypoint_timeout: float = 10.0,
        relative: bool = False,
        final_timeout: float = 60.0,
        blocking: bool = True,
    ) -> bool:
        for waypoint in trajectory:
            self.move_base_to(waypoint, relative=relative, blocking=blocking, timeout=per_waypoint_timeout)
        return True

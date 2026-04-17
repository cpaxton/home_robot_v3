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

"""Robot-agnostic ZMQ client driven by RobotSpec.

This client communicates with a MuJoCo ZMQ server using the same protocol as
StretchZmqClient, but derives joint indexing and DOF from a RobotSpec rather
than hardcoding Stretch-specific constants.
"""

import sys
import threading
import time
import timeit
from threading import Lock
from typing import Any

import numpy as np
import zmq

import emet.utils.compression as compression
from emet.core.interfaces import ContinuousNavigationAction, Observations
from emet.core.parameters import Parameters, get_parameters
from emet.core.robot import AbstractRobotClient, ControlMode
from emet.core.zmq_protocol import EMET_ZMQ_ROBOT_ID_KEY, read_emet_robot_id, robot_ids_match
from emet.robots.base import RobotSpec
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
        parameters: Parameters | None = None,
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

        self._joint_index: dict[str, int] = {name: i for i, name in enumerate(robot_spec.joint_names)}

        if parameters is None:
            parameters = get_parameters("default_planner.yaml")
        self._parameters = parameters

        self._iter = -1
        self._seq_id = 0
        self._started = False
        self._finish = False

        self._obs: dict[str, Any] | None = None
        self._state: dict[str, Any] | None = None
        self._servo: dict[str, Any] | None = None
        self._last_step = -1

        self._obs_lock = Lock()
        self._act_lock = Lock()

        self._base_xyt = np.zeros(3)

        # ZMQ sockets
        self.context = zmq.Context()
        self.recv_socket = self._create_recv_socket(recv_port, robot_ip, use_remote_computer, "observations")
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

        self._recv_threads_started = False

        if start_immediately:
            self.start()

    @property
    def spec(self) -> RobotSpec:
        return self._spec

    @property
    def parameters(self) -> Parameters:
        return self._parameters

    # -- Socket helpers -------------------------------------------------------

    def _create_recv_socket(self, port: int, robot_ip: str, use_remote: bool, message_type: str = "") -> zmq.Socket:
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

    def send_message(self, message: dict[str, Any]) -> None:
        self.send_socket.send_pyobj(message)

    # -- Lifecycle ------------------------------------------------------------

    def _wait_for_zmq_ready(self, timeout: float = 10.0) -> bool:
        """Wait until at least one observation and one state message arrived."""
        t0 = timeit.default_timer()
        while True:
            with self._obs_lock:
                ready = self._obs is not None and self._state is not None
            if ready:
                return True
            if timeit.default_timer() - t0 > timeout:
                return False
            time.sleep(0.05)

    def _verify_emet_robot_id(self) -> bool:
        """Ensure ``emet_robot_id`` from the server matches this client's RobotSpec (if present)."""
        with self._obs_lock:
            msg = self._obs if self._obs is not None else self._state
        rid = read_emet_robot_id(msg)
        if rid is None:
            return True
        expected = self._spec.name
        if robot_ids_match(rid, expected):
            return True
        logger.error(
            f"Robot ID mismatch: server reports {EMET_ZMQ_ROBOT_ID_KEY}={rid!r} but this client expects "
            f"{expected!r} (same as `emet serve mujoco --robot`). "
            "Use matching `--robot` on the agent, e.g. `--robot stretch` for the Stretch sim."
        )
        return False

    def start(self) -> bool:
        if self._started:
            return True
        if not self._recv_threads_started:
            self._recv_threads_started = True
            self._finish = False
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()
            self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
            self._state_thread.start()
            self._servo_thread = threading.Thread(target=self._servo_loop, daemon=True)
            self._servo_thread.start()
        if not self._wait_for_zmq_ready(timeout=10.0):
            logger.error(
                "Timeout waiting for observations/state from ZMQ server. "
                "Start `emet serve mujoco` with the same `--robot` and check IP / `--port-offset`."
            )
            return False
        if not self._verify_emet_robot_id():
            return False
        self._started = True
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
                    self._base_xyt = np.array([output["gps"][0], output["gps"][1], output["compass"][0]])

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

    def get_joint_positions(self, timeout: float = 5.0) -> np.ndarray | None:
        state = self._state
        if state is not None and "joint_positions" in state:
            return np.array(state["joint_positions"])
        obs = self._obs
        if obs is not None and "joint" in obs:
            return np.array(obs["joint"])
        return None

    def get_joint_velocities(self, timeout: float = 5.0) -> np.ndarray | None:
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

    def get_observation(self, max_iter: int = 5) -> Observations | None:
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
        action: dict[str, Any],
        timeout: float = 5.0,
        reliable: bool = True,
    ) -> dict[str, Any]:
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
        timeout: float | None = None,
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

    def set_joint_positions(self, positions: dict[str, float]) -> None:
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

    # -- Stretch / DynaMem API shims (this client is spec-driven, not Stretch-indexed) ------------

    def get_joint_state(
        self, timeout: float = 5.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        """Joint positions, velocities, and efforts from the state stream (Stretch-compatible shape)."""
        t0 = timeit.default_timer()
        while timeit.default_timer() - t0 < timeout:
            with self._obs_lock:
                st = self._state
            if st is not None and "joint_positions" in st:
                q = np.asarray(st["joint_positions"], dtype=float)
                dq = np.asarray(st.get("joint_velocities", np.zeros_like(q)), dtype=float)
                tau = np.asarray(st.get("joint_efforts", np.zeros_like(q)), dtype=float)
                return q, dq, tau
            time.sleep(0.01)
        return None, None, None

    def get_six_joints(self, _timeout: float = 5.0) -> np.ndarray:
        """Placeholder for DynaMem's Stretch 6-DOF slice; non-Stretch robots have no universal mapping."""
        return np.zeros(6, dtype=float)

    def get_pan_tilt(self) -> tuple[float, float]:
        """Stretch head pan/tilt; mobile manipulators without that head return zeros."""
        return (0.0, 0.0)

    def get_gripper_position(self) -> float:
        q, _, _ = self.get_joint_state(timeout=2.0)
        if q is None:
            return 0.5
        for i, name in enumerate(self._spec.joint_names):
            if "gripper" in name and "finger" in name:
                return float(q[i])
        return 0.5

    def arm_to(self, joint_angles=None, gripper=None, head=None, blocking=True, **kwargs) -> bool:
        if not getattr(self, "_logged_arm_to_nonstretch", False):
            logger.warning(
                "arm_to() targets Stretch's 6-DOF arm; this generic robot does not map those commands (no-op). "
                "DynaMem manipulation paths remain Stretch-oriented."
            )
            self._logged_arm_to_nonstretch = True
        return True

    def head_to(self, head_pan: float, head_tilt: float, blocking: bool = False, **kwargs) -> None:
        if not getattr(self, "_logged_head_to_nonstretch", False):
            logger.warning("head_to() is Stretch-specific; ignored for this robot.")
            self._logged_head_to_nonstretch = True

    def navigate_to(self, xyt, relative: bool = False, blocking: bool = True, **kwargs) -> None:
        xyt_a = np.asarray(xyt, dtype=float).reshape(-1)
        if xyt_a.size != 3:
            logger.error("navigate_to expects a length-3 xyt vector")
            return
        self.move_base_to(xyt_a, relative=relative, blocking=blocking, timeout=kwargs.get("timeout"))

    def gripper_to(self, target: float, blocking: bool = True, reliable: bool = True) -> None:
        """Stretch-compatible gripper command (absolute opening target)."""
        action: dict[str, Any] = {"gripper": float(target)}
        if blocking:
            action["gripper_blocking"] = True
        self.send_action(action, reliable=reliable)
        if blocking:
            time.sleep(0.05)

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

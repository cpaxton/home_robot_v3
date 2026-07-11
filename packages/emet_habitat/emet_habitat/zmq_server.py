# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""ZMQ server exposing Habitat-Sim as a Stretch-compatible sim backend."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import zmq

import emet.utils.compression as compression
from emet.core.server import BaseZmqServer, _action_recv_log_line
from emet.core.zmq_protocol import (
    CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
    EMET_ZMQ_ROBOT_ID_KEY,
    EMET_ZMQ_SESSION_KEY,
    EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY,
)
from emet.motion.constants import STRETCH_HOME_Q
from emet.simulation.sim_object_placements import apply_navigation_origin_to_session
from emet.utils.geometry import xyt_base_to_global, xyt_global_to_base
from emet.utils.observation_layout import rgb_height_width_for_zmq
from emet_habitat.habitat_serve_session import HabitatServeConfig, open_habitat_robot_for_serve
from emet_habitat.robot_client import HabitatRobotClient


class HabitatZmqServer(BaseZmqServer):
    """Publish Stretch-shaped ZMQ observations backed by Habitat-Sim."""

    def __init__(
        self,
        robot: HabitatRobotClient,
        *,
        scene_id: str,
        port_offset: int = 0,
        verbose: bool = False,
        image_scaling: float = 0.5,
        depth_scaling: float = 0.001,
    ):
        from emet.utils.port_utils import get_ports

        ports = get_ports(int(port_offset))
        super().__init__(
            send_port=int(ports.send),
            recv_port=int(ports.recv),
            send_state_port=int(ports.state),
            send_servo_port=int(ports.servo),
            use_remote_computer=False,
            verbose=verbose,
            image_scaling=image_scaling,
            depth_scaling=depth_scaling,
            ee_image_scaling=image_scaling,
            ee_depth_scaling=depth_scaling,
        )
        self._robot = robot
        self._scene_id = str(scene_id)
        self._running = True
        self._at_goal = True
        self._last_step = 0
        self._initial_xyt: np.ndarray | None = None
        self._emet_session: dict[str, Any] | None = None
        self._stretch_joint_count = int(len(STRETCH_HOME_Q))
        self.control_mode = "navigation"
        self._sync_navigation_origin()

    @classmethod
    def from_serve_config(
        cls,
        cfg: HabitatServeConfig,
        *,
        port_offset: int = 0,
        verbose: bool = False,
    ) -> HabitatZmqServer:
        robot, _sim = open_habitat_robot_for_serve(cfg)
        return cls(robot, scene_id=cfg.scene_id, port_offset=port_offset, verbose=verbose)

    def close(self) -> None:
        self._running = False
        sim = getattr(self._robot, "_sim", None)
        if sim is not None and hasattr(sim, "close"):
            sim.close()

    def _sync_navigation_origin(self) -> None:
        self._robot._sync_pose_from_sim()
        self._initial_xyt = np.asarray(self._robot.get_base_pose(), dtype=np.float64).reshape(-1)[:3].copy()
        session: dict[str, Any] = {
            EMET_ZMQ_SESSION_SCHEMA_VERSION_KEY: CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
            "runtime_kind": "habitat_hmeqa",
            "scene_id": self._scene_id,
            "capabilities": {"navigation": True, "manipulation": False},
        }
        apply_navigation_origin_to_session(session, self._initial_xyt)
        self._emet_session = session
        self._robot.set_emet_session(session)

    def _episode_xyt(self) -> np.ndarray:
        world = np.asarray(self._robot.get_base_pose(), dtype=np.float64).reshape(-1)[:3]
        init = self._initial_xyt
        if init is None:
            return world.copy()
        return xyt_global_to_base(world, init)

    def _xyt_action_to_world(self, xyt: np.ndarray, *, relative: bool) -> np.ndarray:
        raw = np.asarray(xyt, dtype=np.float64).reshape(-1)[:3]
        init = self._initial_xyt
        if init is None:
            init = np.zeros(3, dtype=np.float64)
        if relative:
            cur = self._episode_xyt()
            rel = xyt_base_to_global(raw, cur)
        else:
            rel = raw
        return xyt_base_to_global(rel, init)

    def _attach_emet_session(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._emet_session is not None:
            message[EMET_ZMQ_SESSION_KEY] = self._emet_session
        return message

    def get_control_mode(self) -> str:
        mode = str(getattr(self, "control_mode", "navigation") or "navigation")
        if mode in ("navigation", "manipulation"):
            return mode
        return "navigation"

    def handle_action(self, action: dict[str, Any]) -> None:
        if "control_mode" in action:
            self.control_mode = str(action["control_mode"])
        if "posture" in action:
            p = str(action["posture"])
            if p in ("navigation", "manipulation"):
                self.control_mode = p
            self._at_goal = True
        if "xyt" not in action:
            return
        self._at_goal = False
        raw = np.asarray(action["xyt"], dtype=np.float64).reshape(-1)[:3]
        relative = bool(action.get("nav_relative", False))
        nav_world = bool(action.get("nav_world", False))
        if nav_world and not relative:
            world_goal = raw
        else:
            world_goal = self._xyt_action_to_world(raw, relative=relative)
        self._robot.move_base_to(world_goal, relative=False, world_frame=True)
        self._last_step += 1
        self._at_goal = True

    def _observation_message(self) -> dict[str, Any] | None:
        obs = self._robot.get_observation()
        if obs is None or obs.rgb is None or obs.depth is None:
            return None
        rgb = np.asarray(obs.rgb)
        depth = np.asarray(obs.depth, dtype=np.float32)
        if rgb.ndim < 2 or depth.ndim < 2:
            return None
        rgb, depth = self._rescale_color_and_depth(rgb, depth, self.image_scaling)
        rgb_height, rgb_width = rgb_height_width_for_zmq(rgb)
        depth_u16 = (depth * 1000.0).astype(np.uint16)
        rgb_jpg = compression.to_jpg(rgb)
        depth_jp2 = compression.to_jp2(depth_u16)
        if self._initial_xyt is None:
            return None
        episode = self._episode_xyt()
        joint = np.asarray(STRETCH_HOME_Q, dtype=np.float64).copy()
        joint[0] = float(episode[0])
        joint[1] = float(episode[1])
        joint[2] = float(episode[2])
        return {
            "rgb": rgb_jpg,
            "depth": depth_jp2,
            "camera_K": np.asarray(obs.camera_K, dtype=np.float64),
            "camera_pose": np.asarray(obs.camera_pose, dtype=np.float64),
            "ee_pose": np.eye(4, dtype=np.float64),
            "joint": joint,
            "gps": np.asarray(episode[:2], dtype=np.float64),
            "compass": np.array([float(episode[2])], dtype=np.float64),
            "rgb_width": rgb_width,
            "rgb_height": rgb_height,
            "control_mode": self.get_control_mode(),
            "last_motion_failed": False,
            "recv_address": self.recv_address,
            "step": self._last_step,
            "at_goal": bool(self._at_goal),
            "is_simulation": True,
            "lidar_points": None,
            "lidar_timestamp": None,
            EMET_ZMQ_ROBOT_ID_KEY: "stretch",
        }

    def get_full_observation_message(self) -> dict[str, Any] | None:
        message = self._observation_message()
        if message is None:
            return None
        return self._attach_emet_session(message)

    def get_state_message(self) -> dict[str, Any] | None:
        if self._initial_xyt is None:
            return None
        episode = self._episode_xyt()
        q = np.asarray(STRETCH_HOME_Q, dtype=np.float64).copy()
        q[0], q[1], q[2] = float(episode[0]), float(episode[1]), float(episode[2])
        dq = np.zeros(self._stretch_joint_count, dtype=np.float64)
        eff = np.zeros(self._stretch_joint_count, dtype=np.float64)
        message = {
            "base_pose": episode.copy(),
            "ee_pose": np.eye(4, dtype=np.float64),
            "joint_positions": q,
            "joint_velocities": dq,
            "joint_efforts": eff,
            "control_mode": self.get_control_mode(),
            "at_goal": bool(self._at_goal),
            "is_homed": True,
            "is_runstopped": False,
            "step": self._last_step,
            EMET_ZMQ_ROBOT_ID_KEY: "stretch",
        }
        return self._attach_emet_session(message)

    def get_servo_message(self) -> dict[str, Any] | None:
        obs = self._robot.get_observation()
        if obs is None or obs.rgb is None or obs.depth is None:
            return None
        rgb = np.asarray(obs.rgb)
        depth = np.asarray(obs.depth, dtype=np.float32)
        if rgb.ndim < 2 or depth.ndim < 2:
            return None
        rgb, depth = self._rescale_color_and_depth(rgb, depth, self.ee_image_scaling)
        depth_u16 = (depth * 1000.0).astype(np.uint16)
        if self._initial_xyt is None:
            return None
        episode = self._episode_xyt()
        joint = np.asarray(STRETCH_HOME_Q, dtype=np.float64).copy()
        joint[0] = float(episode[0])
        joint[1] = float(episode[1])
        joint[2] = float(episode[2])
        # StretchZmqClient expects robosuite/MuJoCo servo keys (head_color_image), not full-obs rgb/depth.
        return self._attach_emet_session(
            {
                "head_color_image": compression.to_jpg(rgb),
                "head_depth_image": compression.to_jp2(depth_u16),
                "head_camera_K": np.asarray(obs.camera_K, dtype=np.float64),
                "camera_pose": np.asarray(obs.camera_pose, dtype=np.float64),
                "joint_positions": joint,
                "base_pose": episode.copy(),
                "control_mode": self.get_control_mode(),
                "at_goal": bool(self._at_goal),
                "is_simulation": True,
                "step": self._last_step,
                EMET_ZMQ_ROBOT_ID_KEY: "stretch",
            }
        )

    def is_running(self) -> bool:
        return self._running

    def _poll_and_handle_action(self) -> None:
        """Non-blocking recv + duplicate-step filtering (same rules as BaseZmqServer.spin_recv)."""
        from emet.core.zmq_protocol import zmq_meta_action_should_bypass_duplicate_step
        from emet.simulation.env_flags import env_sim_nav_debug
        from emet.utils.logger import Logger

        logger = Logger(__name__)
        try:
            action = self.recv_socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            return
        if action is None:
            return
        action_step = action.get("step", -1)
        if (
            self.skip_duplicate_steps
            and action_step <= self._last_step
            and "xyt" not in action
            and "posture" not in action
            and "control_mode" not in action
            and "joint" not in action
            and "head_to" not in action
            and not zmq_meta_action_should_bypass_duplicate_step(action)
        ):
            logger.warning(f"Skipping duplicate action {action_step}, last step = {self._last_step}")
            return
        self.handle_action(action)
        self._last_step = max(action_step, self._last_step)
        line = _action_recv_log_line(action, self._last_step)
        if "xyt" in action and env_sim_nav_debug():
            logger.warning(line)
        else:
            logger.info(line)

    def start(self) -> None:
        """Run ZMQ I/O on the main thread — Habitat GL context is not thread-safe."""
        print("Habitat ZMQ server: single-threaded loop (GL context must stay on main thread)")
        last_full = 0.0
        last_state = 0.0
        last_servo = 0.0
        full_period = 0.05
        state_period = 0.02
        servo_period = 0.01
        while self.is_running():
            now = time.time()
            self._poll_and_handle_action()
            if now - last_servo >= servo_period:
                msg = self.get_servo_message()
                if msg is not None:
                    self.send_servo_socket.send_pyobj(msg)
                last_servo = now
            if now - last_state >= state_period:
                msg = self.get_state_message()
                if msg is not None:
                    self.send_state_socket.send_pyobj(msg)
                last_state = now
            if now - last_full >= full_period:
                msg = self.get_full_observation_message()
                if msg is not None:
                    self.send_socket.send_pyobj(msg)
                last_full = now
            time.sleep(1e-4)


def run_habitat_zmq_server(
    cfg: HabitatServeConfig,
    *,
    port_offset: int = 0,
    verbose: bool = False,
) -> None:
    """Block until KeyboardInterrupt; publishes Habitat observations on ZMQ."""
    server = HabitatZmqServer.from_serve_config(cfg, port_offset=port_offset, verbose=verbose)
    print(f"Habitat ZMQ server ready (scene={cfg.scene_id!r}, robot=stretch, port_offset={port_offset}).")
    print("Connect with: emet run dynagraph --no-rerun   or   emet run agent -c \"describe the scene\"")
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down Habitat ZMQ server…")
    finally:
        server.close()

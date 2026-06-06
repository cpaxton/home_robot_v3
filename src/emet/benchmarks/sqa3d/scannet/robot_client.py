# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""``AbstractRobotClient`` shim over :class:`ScanNetEQASimulator`."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from emet.benchmarks.sqa3d.scannet.observations import scannet_rgb_depth_to_observations
from emet.benchmarks.sqa3d.scannet.simulator import ScanNetEQASimulator
from emet.core.interfaces import Observations
from emet.core.robot import AbstractRobotClient, ControlMode
from emet.motion import Footprint, RobotModel
from emet.utils.geometry import xyt_base_to_global


class ScanNetRobotClient(AbstractRobotClient, RobotModel):
    """In-process ScanNet mesh agent for GraphEQA / Dynagraph on SQA3D."""

    def __init__(self, simulator: ScanNetEQASimulator):
        super().__init__()
        self._sim = simulator
        self._xyt = np.zeros(3, dtype=np.float64)
        self._v = 0.3
        self._w = 0.4
        self._base_control_mode = ControlMode.NAVIGATION
        self.dof = 3
        self._sync_pose_from_sim()

    def _sync_pose_from_sim(self) -> None:
        frame = self._sim.get_frame()
        obs = self._frame_to_obs(frame)
        self._xyt = np.array([obs.gps[0], obs.gps[1], float(obs.compass[0])], dtype=np.float64)

    def _frame_to_obs(self, frame) -> Observations:
        return scannet_rgb_depth_to_observations(
            rgb=frame.rgb,
            depth=frame.depth,
            position=frame.position,
            quat_xyzw=frame.quat_xyzw,
            intrinsics=frame.intrinsics,
            sensor_height=self._sim.sensor_height,
            camera_tilt_deg=self._sim.camera_tilt_deg,
        )

    def get_observation(self, max_iter: int = 5) -> Observations | None:
        return self._frame_to_obs(self._sim.get_frame())

    def get_base_pose(self, timeout: float = 5.0) -> np.ndarray:
        self._sync_pose_from_sim()
        return self._xyt.copy()

    def _greedy_to_xy(self, goal_x: float, goal_y: float, max_steps: int = 40) -> None:
        for _ in range(max_steps):
            self._sync_pose_from_sim()
            dx = goal_x - self._xyt[0]
            dy = goal_y - self._xyt[1]
            dist = math.hypot(dx, dy)
            if dist < 0.12:
                break
            target_heading = math.atan2(dy, dx)
            dtheta = (target_heading - self._xyt[2] + math.pi) % (2 * math.pi) - math.pi
            if abs(dtheta) > 0.12:
                self._sim.step("turn_left" if dtheta > 0 else "turn_right")
            else:
                self._sim.step("move_forward")
            self._sync_pose_from_sim()

    def move_base_to(
        self,
        xyt: Iterable[float] | object,
        relative: bool = False,
        blocking: bool = False,
        verbose: bool = False,
        timeout: float | None = None,
        world_frame: bool = False,
        **kwargs: Any,
    ):
        goal = np.asarray(list(xyt)[:3], dtype=np.float64)
        if relative:
            goal = xyt_base_to_global(goal, self._xyt)
        self._greedy_to_xy(float(goal[0]), float(goal[1]), max_steps=80)
        if len(goal) >= 3:
            for _ in range(18):
                self._sync_pose_from_sim()
                dtheta = (float(goal[2]) - self._xyt[2] + math.pi) % (2 * math.pi) - math.pi
                if abs(dtheta) < 0.1:
                    break
                self._sim.step("turn_left" if dtheta > 0 else "turn_right")
        self._sync_pose_from_sim()

    def reset(self):
        self._base_control_mode = ControlMode.NAVIGATION

    def start(self) -> bool:
        return True

    def switch_to_navigation_mode(self):
        self._base_control_mode = ControlMode.NAVIGATION

    def switch_to_manipulation_mode(self):
        self._base_control_mode = ControlMode.MANIPULATION

    def open_gripper(self) -> None:
        return None

    def move_to_nav_posture(self):
        return True

    def move_to_manip_posture(self):
        return True

    def get_robot_model(self) -> RobotModel:
        return self

    def execute_trajectory(
        self,
        trajectory: list[np.ndarray],
        pos_err_threshold: float = 0.2,
        rot_err_threshold: float = 0.75,
        spin_rate: int = 10,
        verbose: bool = False,
        per_waypoint_timeout: float = 10.0,
        relative: bool = False,
        final_timeout: float = 60.0,
        blocking: bool = True,
        world_frame: bool = False,
        **kwargs: Any,
    ):
        for wp in trajectory:
            self.move_base_to(wp, relative=relative, blocking=blocking)

    def get_pose_graph(self) -> np.ndarray:
        return np.zeros((0, 3), dtype=np.float64)

    def at_goal(self) -> bool:
        return True

    def get_footprint(self) -> Footprint:
        return Footprint(width=0.34, length=0.33, width_offset=0.0, length_offset=-0.1)

    def get_dof(self):
        return self.dof

    def set_config(self, q) -> None:
        return None

    def get_config(self):
        self._sync_pose_from_sim()
        return self._xyt.copy()

    def set_velocity(self, v: float, w: float):
        self._v = float(v)
        self._w = float(w)

    def say(self, text: str):
        return None

    def get_pan_tilt(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def get_six_joints(self, timeout: float = 5.0) -> np.ndarray:
        return np.zeros(6, dtype=np.float64)

    def navigate_to(self, xyt, relative: bool = False, blocking: bool = True, **kwargs) -> bool:
        xyt_a = np.asarray(xyt, dtype=np.float64).reshape(-1)
        if xyt_a.size != 3:
            return False
        self.move_base_to(xyt_a, relative=relative, blocking=blocking, timeout=kwargs.get("timeout"))
        return True

    def arm_to(self, joint_angles=None, gripper=None, head=None, blocking=True, **kwargs) -> bool:
        return True

    def head_to(self, head_pan: float, head_tilt: float, blocking: bool = False, **kwargs) -> None:
        return None

    def look_front(self, blocking: bool = True, timeout: float = 10.0) -> None:
        return None

    def gripper_to(self, target: float, blocking: bool = True, reliable: bool = True) -> None:
        return None

    def get_gripper_position(self) -> float:
        return 0.5

    def get_emet_session(self) -> dict[str, Any] | None:
        return None

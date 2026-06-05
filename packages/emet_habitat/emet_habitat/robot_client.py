# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""``AbstractRobotClient`` shim over :class:`HabitatEQASimulator`."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from emet.core.interfaces import Observations
from emet.core.robot import AbstractRobotClient, ControlMode
from emet.motion import Footprint, RobotModel
from emet.utils.geometry import xyt_base_to_global

from emet_habitat.observations import habitat_rgb_depth_to_observations
from emet_habitat.simulator import HabitatEQASimulator


class HabitatRobotClient(AbstractRobotClient, RobotModel):
    """In-process Habitat agent for GraphEQA / Dynagraph controllers."""

    def __init__(self, simulator: HabitatEQASimulator):
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
        obs = habitat_rgb_depth_to_observations(
            rgb=frame.rgb,
            depth=frame.depth,
            agent_state=frame.agent_state,
            intrinsics=frame.intrinsics,
        )
        self._xyt = np.array([obs.gps[0], obs.gps[1], float(obs.compass[0])], dtype=np.float64)

    def get_observation(self, max_iter: int = 5) -> Observations | None:
        frame = self._sim.get_frame()
        return habitat_rgb_depth_to_observations(
            rgb=frame.rgb,
            depth=frame.depth,
            agent_state=frame.agent_state,
            intrinsics=frame.intrinsics,
        )

    def get_base_pose(self, timeout: float = 5.0) -> np.ndarray:
        self._sync_pose_from_sim()
        return self._xyt.copy()

    def move_base_to(
        self,
        xyt: Iterable[float] | object,
        relative: bool = False,
        blocking: bool = False,
        verbose: bool = False,
        timeout: float | None = None,
    ):
        goal = np.asarray(list(xyt)[:3], dtype=np.float64)
        if relative:
            goal = xyt_base_to_global(goal, self._xyt)
        max_steps = 40
        for _ in range(max_steps):
            self._sync_pose_from_sim()
            dx = goal[0] - self._xyt[0]
            dy = goal[1] - self._xyt[1]
            dist = math.hypot(dx, dy)
            if dist < 0.15:
                break
            target_heading = math.atan2(dy, dx)
            dtheta = (target_heading - self._xyt[2] + math.pi) % (2 * math.pi) - math.pi
            if abs(dtheta) > 0.15:
                self._sim.step("turn_left" if dtheta > 0 else "turn_right")
            else:
                self._sim.step("move_forward")
        self._sync_pose_from_sim()

    def reset(self):
        self._base_control_mode = ControlMode.NAVIGATION

    def switch_to_navigation_mode(self):
        self._base_control_mode = ControlMode.NAVIGATION

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

    def set_velocity(self, v: float, w: float):
        self._v = float(v)
        self._w = float(w)

    def say(self, text: str):
        return None

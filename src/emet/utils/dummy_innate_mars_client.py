# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

"""Offline / test double for Innate Mars (`AbstractRobotClient` + `RobotModel`), mirroring `DummyStretchClient`."""

from __future__ import annotations

import numpy as np

from emet.core.robot import AbstractRobotClient, ControlMode
from emet.motion import Footprint, RobotModel
from emet.robots.innate_mars import INNATE_MARS_JOINT_NAMES, InnateMarsBackend
from emet.utils.geometry import xyt_base_to_global


class DummyInnateMarsClient(AbstractRobotClient, RobotModel):
    """Minimal innate Mars client for tests and tooling without ZMQ or ROS."""

    def __init__(self) -> None:
        super().__init__()
        self._robot_model = self
        spec = InnateMarsBackend().get_spec()
        self._footprint = spec.footprint
        self.dof = len(INNATE_MARS_JOINT_NAMES)
        self.xyt = np.zeros(3, dtype=np.float64)
        self._q = np.zeros(self.dof, dtype=np.float64)
        self._base_control_mode = ControlMode.NAVIGATION

    # --- AbstractRobotClient ---

    def move_base_to(
        self,
        xyt,
        relative: bool = False,
        blocking: bool = True,
        verbose: bool = False,
        timeout: float | None = None,
    ):
        xyt_a = np.asarray(xyt, dtype=float).reshape(-1)
        if relative:
            xyt_goal = xyt_base_to_global(xyt_a[:3], self.xyt)
        else:
            xyt_goal = xyt_a[:3]
        self.xyt[:] = xyt_goal

    def reset(self) -> None:
        self.xyt[:] = 0.0
        self._q[:] = 0.0
        self._base_control_mode = ControlMode.IDLE

    def switch_to_navigation_mode(self) -> bool:
        self._base_control_mode = ControlMode.NAVIGATION
        return True

    def switch_to_manipulation_mode(self) -> bool:
        self._base_control_mode = ControlMode.MANIPULATION
        return True

    def get_robot_model(self) -> RobotModel:
        return self._robot_model

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
        raise NotImplementedError("DummyInnateMarsClient does not simulate trajectory execution.")

    def move_to_nav_posture(self) -> bool:
        return True

    def move_to_manip_posture(self) -> bool:
        return True

    def get_base_pose(self) -> np.ndarray:
        return np.asarray(self.xyt, dtype=np.float64)

    def get_pose_graph(self) -> np.ndarray:
        return np.empty((0, 3))

    def at_goal(self) -> bool:
        return True

    # --- joint commands (for innate emotes / tests) ---

    def get_joint_state(self):
        """Return (q, dq, tau) like ROS / ZMQ clients."""
        dq = np.zeros_like(self._q)
        tau = np.zeros_like(self._q)
        return self._q.copy(), dq, tau

    def set_joint_positions(self, positions: dict[str, float]) -> None:
        name_to_i = {n: i for i, n in enumerate(INNATE_MARS_JOINT_NAMES)}
        for name, val in positions.items():
            if name in name_to_i:
                self._q[name_to_i[name]] = float(val)

    # --- RobotModel ---

    def get_dof(self) -> int:
        return self.dof

    def set_config(self, q):
        q = np.asarray(q, dtype=float).reshape(-1)
        n = min(len(q), self.dof)
        self._q[:n] = q[:n]

    def get_config(self):
        return self._q.copy()

    def get_footprint(self) -> Footprint:
        return self._footprint

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

"""ROS 2 Innate Mars client as :class:`emet.core.robot.AbstractRobotClient`.

This composes ``innate_mars_bridge.remote.api.InnateMarsClient`` (observations, cameras). Base ``xyt`` navigation is wired via the bridge Nav2 client; arm trajectories are not implemented yet.

The Hello Stretch equivalent is **not** in emet: see ``stretch_ros2_bridge.remote.api.StretchClient``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np

from emet.core.interfaces import ContinuousNavigationAction
from emet.core.robot import AbstractRobotClient, ControlMode

logger = logging.getLogger(__name__)


class InnateMarsRosRobotClient(AbstractRobotClient):
    """``AbstractRobotClient`` over ``InnateMarsClient`` (subscribe + TF + cameras)."""

    def __init__(self, init_node: bool = True, verbose: bool = False) -> None:
        super().__init__()
        try:
            from innate_mars_bridge.remote.api import InnateMarsClient
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "InnateMarsRosRobotClient requires the innate_mars_bridge package (colcon / install)."
            ) from e
        self._client: Any = InnateMarsClient(init_node=init_node, verbose=verbose)
        self._base_control_mode = ControlMode.NAVIGATION

    @property
    def ros(self) -> Any:
        return self._client

    def move_base_to(
        self,
        xyt: Iterable[float] | ContinuousNavigationAction,
        relative: bool = False,
        blocking: bool = True,
        verbose: bool = False,
        timeout: float | None = None,
    ):
        arr = np.asarray(list(xyt), dtype=np.float64).reshape(-1)
        if arr.size < 3:
            raise ValueError("xyt must have at least 3 elements (x, y, theta)")
        return self._client.move_base_to(
            arr[:3],
            relative=relative,
            blocking=blocking,
            timeout_s=float(timeout or 120.0),
        )

    def reset(self) -> None:
        self._base_control_mode = ControlMode.IDLE

    def switch_to_navigation_mode(self) -> bool:
        self._base_control_mode = ControlMode.NAVIGATION
        return True

    def switch_to_manipulation_mode(self) -> bool:
        self._base_control_mode = ControlMode.MANIPULATION
        return True

    def get_robot_model(self):
        return None

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
        raise NotImplementedError(
            "InnateMarsRosRobotClient: trajectory execution is not implemented for Innate Mars ROS yet."
        )

    def move_to_nav_posture(self) -> bool:
        logger.warning("InnateMarsRosRobotClient.move_to_nav_posture: no-op (not wired).")
        return True

    def move_to_manip_posture(self) -> bool:
        logger.warning("InnateMarsRosRobotClient.move_to_manip_posture: no-op (not wired).")
        return True

    def get_base_pose(self) -> np.ndarray:
        return np.asarray(self._client.base_pose_xyt, dtype=np.float64).reshape(-1)

    def get_pose_graph(self) -> np.ndarray:
        return np.empty((0, 3))

    def at_goal(self) -> bool:
        return bool(self._client.at_goal())

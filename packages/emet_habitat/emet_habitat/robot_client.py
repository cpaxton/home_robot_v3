# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""``AbstractRobotClient`` shim over :class:`~emet_habitat.simulator.HabitatEQASimulator`.

This client lets GraphEQA / Dynagraph controllers run unchanged against Habitat-Sim
instead of a ZMQ MuJoCo server. Navigation uses Habitat discrete actions
(``move_forward``, ``turn_left``, ``turn_right``) with optional navmesh path following.

Typical construction (inside ``emet-habitat`` runner)::

    sim = HabitatEQASimulator(scene_glb, scene_id=question.scene)
    robot = HabitatRobotClient(sim)
    agent = DynagraphController(robot, parameters, ...)
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from emet.core.interfaces import ContinuousNavigationAction, Observations
from emet.core.robot import AbstractRobotClient, ControlMode
from emet.motion import Footprint, RobotModel
from emet.utils.geometry import xyt_base_to_global
from emet_habitat.observations import habitat_rgb_depth_to_observations
from emet_habitat.simulator import HabitatEQASimulator


class HabitatRobotClient(AbstractRobotClient, RobotModel):
    """In-process Habitat agent implementing the emet robot client interface.

    Args:
        simulator: Open :class:`~emet_habitat.simulator.HabitatEQASimulator` for one HM3D scene.

    Attributes:
        dof: Planar DoF count (3: x, y, yaw) for navigation planners.
        _xyt: Cached nav pose ``(x, z, heading)`` in Habitat world coordinates.
        _v, _w: Nominal linear / angular velocity hints (used by some planners).
    """

    def __init__(self, simulator: HabitatEQASimulator):
        super().__init__()
        self._sim = simulator
        self._emet_session: dict | None = None
        self._xyt = np.zeros(3, dtype=np.float64)
        self._v = 0.3
        self._w = 0.4
        self._base_control_mode = ControlMode.NAVIGATION
        self.dof = 3
        self._sync_pose_from_sim()

    def _sync_pose_from_sim(self) -> None:
        """Refresh cached ``_xyt`` from the latest simulator agent state."""
        frame = self._sim.get_frame()
        obs = habitat_rgb_depth_to_observations(
            rgb=frame.rgb,
            depth=frame.depth,
            agent_state=frame.agent_state,
            intrinsics=frame.intrinsics,
            semantic=frame.semantic,
        )
        self._xyt = np.array([obs.gps[0], obs.gps[1], float(obs.compass[0])], dtype=np.float64)

    @property
    def hm3d_semantic_labeler(self):
        """Optional :class:`~emet.habitat.hm3d_semantics.Hm3dSemanticLabeler` when semantics enabled."""
        return getattr(self._sim, "semantic_labeler", None)

    @property
    def uses_hm3d_semantics(self) -> bool:
        """True when the simulator was constructed with HM3D semantic sensors."""
        return bool(getattr(self._sim, "uses_hm3d_semantics", False))

    def get_observation(self, max_iter: int = 5) -> Observations | None:
        """Return the current head RGB-D (and optional semantic) observation.

        Args:
            max_iter: Unused; kept for API compatibility with ZMQ clients.

        Returns:
            Latest :class:`~emet.core.interfaces.Observations` from Habitat-Sim.
        """
        frame = self._sim.get_frame()
        return habitat_rgb_depth_to_observations(
            rgb=frame.rgb,
            depth=frame.depth,
            agent_state=frame.agent_state,
            intrinsics=frame.intrinsics,
            semantic=frame.semantic,
        )

    def get_base_pose(self, timeout: float = 5.0) -> np.ndarray:
        """Return planar base pose ``(x, z, heading)`` in Habitat world coordinates.

        Args:
            timeout: Unused; pose is read synchronously from the simulator.
        """
        self._sync_pose_from_sim()
        return self._xyt.copy()

    def _greedy_to_habitat_point(self, habitat_xyz: np.ndarray, max_steps: int = 40) -> None:
        """Greedy move/turn toward a Habitat world XYZ target (uses X/Z plane)."""
        goal_x = float(habitat_xyz[0])
        goal_z = float(habitat_xyz[2])
        for _ in range(max_steps):
            self._sync_pose_from_sim()
            dx = goal_x - self._xyt[0]
            dz = goal_z - self._xyt[1]
            dist = math.hypot(dx, dz)
            if dist < 0.12:
                break
            target_heading = math.atan2(dz, dx)
            dtheta = (target_heading - self._xyt[2] + math.pi) % (2 * math.pi) - math.pi
            if abs(dtheta) > 0.12:
                self._sim.step("turn_left" if dtheta > 0 else "turn_right")
            else:
                self._sim.step("move_forward")

    def move_base_to(
        self,
        xyt: Iterable[float] | ContinuousNavigationAction,
        relative: bool = False,
        blocking: bool = False,
        verbose: bool = False,
        timeout: float | None = None,
        world_frame: bool = False,
        **kwargs: Any,
    ):
        """Navigate to ``(x, z[, yaw])`` using navmesh path following when available.

        Args:
            xyt: Goal pose in Habitat world coordinates (x, z, optional yaw).
            relative: When True, goal is interpreted relative to current ``_xyt``.
            blocking: Ignored (Habitat steps are synchronous).
            verbose: Ignored.
            timeout: Ignored.
        """
        goal = np.asarray(xyt, dtype=np.float64).reshape(-1)[:3]
        if relative:
            goal = xyt_base_to_global(goal, self._xyt)
        goal_theta = float(goal[2]) if len(goal) >= 3 else None
        if not relative:
            path_pts = self._sim.find_path_to_xy(float(goal[0]), float(goal[1]))
            if path_pts is not None:
                for pt in path_pts[1:]:
                    self._greedy_to_habitat_point(pt)
                if goal_theta is not None:
                    for _ in range(18):
                        self._sync_pose_from_sim()
                        dtheta = (goal_theta - self._xyt[2] + math.pi) % (2 * math.pi) - math.pi
                        if abs(dtheta) < 0.1:
                            break
                        self._sim.step("turn_left" if dtheta > 0 else "turn_right")
                self._sync_pose_from_sim()
                return
        self._greedy_to_habitat_point(
            np.array([goal[0], 0.0, goal[1]], dtype=np.float64),
            max_steps=80,
        )
        if goal_theta is not None:
            for _ in range(18):
                self._sync_pose_from_sim()
                dtheta = (goal_theta - self._xyt[2] + math.pi) % (2 * math.pi) - math.pi
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
            self.move_base_to(
                wp,
                relative=relative,
                blocking=blocking,
                world_frame=world_frame,
            )

    def get_pose_graph(self) -> np.ndarray:
        return np.zeros((0, 3), dtype=np.float64)

    def at_goal(self) -> bool:
        return True

    def get_footprint(self) -> Footprint:
        return Footprint(width=0.34, length=0.33, width_offset=0.0, length_offset=-0.1)

    def get_dof(self):
        return self.dof

    def set_config(self, q) -> None:
        """No-op: Habitat agent pose is owned by the simulator."""

    def get_config(self):
        self._sync_pose_from_sim()
        return self._xyt.copy()

    def set_velocity(self, v: float, w: float):
        self._v = float(v)
        self._w = float(w)

    def say(self, text: str):
        return None

    # Stretch-shaped stubs for DynaMem ManipulationWrapper (EQA uses navigation only).
    def get_pan_tilt(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def get_six_joints(self, timeout: float = 5.0) -> np.ndarray:
        return np.zeros(6, dtype=np.float64)

    def navigate_to(self, xyt, relative: bool = False, blocking: bool = True, **kwargs) -> bool:
        xyt_a = np.asarray(xyt, dtype=np.float64).reshape(-1)
        if xyt_a.size != 3:
            return False
        self.move_base_to(
            xyt_a,
            relative=relative,
            blocking=blocking,
            timeout=kwargs.get("timeout"),
            world_frame=bool(kwargs.get("world_frame", False)),
        )
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
        """Optional session metadata (e.g. injected HM3D GT placements for find-phase)."""
        return self._emet_session

    def set_emet_session(self, session: dict | None) -> None:
        """Attach session dict for GT graph refresh (Habitat find-phase harness)."""
        self._emet_session = dict(session) if session is not None else None

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""``AbstractRobotClient`` shim over ScanNet replay simulators.

Drives :class:`~emet.benchmarks.sqa3d.scannet.simulator.ScanNetReplaySimulator`
(posed ``.sens`` RGB-D with mesh fallback) or mesh-only
:class:`~emet.benchmarks.sqa3d.scannet.simulator.ScanNetEQASimulator` for SQA3D
embodied Dynagraph / DynaMem evaluation.

Typical construction::

    sim = create_scannet_simulator(scene_id, replay_mode="auto")
    sim.set_sqa3d_pose(question)
    robot = ScanNetRobotClient(sim)
    agent = DynagraphController(robot, parameters, ...)
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from emet.benchmarks.sqa3d.scannet.observations import scannet_rgb_depth_to_observations
from emet.benchmarks.sqa3d.scannet.simulator import ScanNetEQASimulator, ScanNetReplaySimulator
from emet.core.interfaces import Observations
from emet.core.robot import AbstractRobotClient, ControlMode
from emet.motion import Footprint, RobotModel
from emet.utils.geometry import xyt_base_to_global


class ScanNetRobotClient(AbstractRobotClient, RobotModel):
    """In-process ScanNet replay agent for GraphEQA / Dynagraph on SQA3D.

    Goals are expressed in the ScanNet / SQA3D scene coordinate frame. The
    ``world_frame`` argument on navigation methods is accepted for API parity with
    :class:`~emet.controller.generic_zmq_client.GenericZmqClient` but **ignored**
    (there is no separate episode-relative frame).

    Args:
        simulator: Open :class:`ScanNetReplaySimulator` or mesh-only
            :class:`ScanNetEQASimulator`.

    Attributes:
        dof: Planar DoF count (3: x, y, heading).
        _sim: Underlying ScanNet simulator (replay or mesh render).
        _xyt: Cached nav pose ``(x, y, heading)`` in scene coordinates.
        _v: Nominal linear velocity hint (planner metadata).
        _w: Nominal angular velocity hint (planner metadata).
        _base_control_mode: Current :class:`~emet.core.robot.ControlMode` stub.
    """

    def __init__(self, simulator: ScanNetEQASimulator | ScanNetReplaySimulator):
        super().__init__()
        self._sim = simulator
        self._xyt = np.zeros(3, dtype=np.float64)
        self._v = 0.3
        self._w = 0.4
        self._base_control_mode = ControlMode.NAVIGATION
        self.dof = 3
        self._sync_pose_from_sim()

    def _sync_pose_from_sim(self) -> None:
        """Refresh cached ``_xyt`` from the latest simulator frame."""
        frame = self._sim.get_frame()
        obs = self._frame_to_obs(frame)
        self._xyt = np.array([obs.gps[0], obs.gps[1], float(obs.compass[0])], dtype=np.float64)

    def _frame_to_obs(self, frame) -> Observations:
        """Convert a :class:`~emet.benchmarks.sqa3d.scannet.simulator.ScanNetFrame` to Observations."""
        return scannet_rgb_depth_to_observations(
            rgb=frame.rgb,
            depth=frame.depth,
            position=frame.position,
            quat_xyzw=frame.quat_xyzw,
            intrinsics=frame.intrinsics,
            sensor_height=self._sim.sensor_height,
            camera_tilt_deg=self._sim.camera_tilt_deg,
            camera_to_world=getattr(frame, "camera_to_world", None),
        )

    def get_observation(self, max_iter: int = 5) -> Observations | None:
        """Return the current RGB-D observation from the replay simulator.

        Args:
            max_iter: Unused; kept for ZMQ client API compatibility.
        """
        return self._frame_to_obs(self._sim.get_frame())

    def get_base_pose(self, timeout: float = 5.0) -> np.ndarray:
        """Return planar base pose ``(x, y, heading)`` in scene coordinates.

        Args:
            timeout: Unused; pose is read synchronously from the simulator.
        """
        self._sync_pose_from_sim()
        return self._xyt.copy()

    def _greedy_to_xy(self, goal_x: float, goal_y: float, max_steps: int = 40) -> None:
        """Greedy turn/move toward a scene XY target using discrete sim steps."""
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
        """Navigate to ``(x, y[, yaw])`` via greedy discrete steps on the replay sim.

        Args:
            xyt: Goal pose in ScanNet scene coordinates.
            relative: When True, goal is interpreted relative to current ``_xyt``.
            blocking: Ignored (steps are synchronous).
            verbose: Ignored.
            timeout: Ignored.
            world_frame: Ignored; goals are already scene/world coordinates.
            **kwargs: Ignored (ZMQ client compatibility).
        """
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
        """Reset control mode to navigation (simulator pose is unchanged)."""
        self._base_control_mode = ControlMode.NAVIGATION

    def start(self) -> bool:
        """No-op startup hook; returns True so controllers proceed to update."""
        return True

    def switch_to_navigation_mode(self):
        """Mark client as in navigation mode."""
        self._base_control_mode = ControlMode.NAVIGATION

    def switch_to_manipulation_mode(self):
        """Mark client as in manipulation mode (EQA harness does not move an arm)."""
        self._base_control_mode = ControlMode.MANIPULATION

    def open_gripper(self) -> None:
        """No-op gripper stub."""
        return None

    def move_to_nav_posture(self):
        """No-op posture stub."""
        return True

    def move_to_manip_posture(self):
        """No-op posture stub."""
        return True

    def get_robot_model(self) -> RobotModel:
        """Return this client as the planning robot model."""
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
        """Visit each waypoint via :meth:`move_base_to` (open-loop).

        Args:
            trajectory: List of ``(x, y[, yaw])`` goals in scene coordinates.
            pos_err_threshold: Ignored.
            rot_err_threshold: Ignored.
            spin_rate: Ignored.
            verbose: Ignored.
            per_waypoint_timeout: Ignored.
            relative: Passed to each :meth:`move_base_to` call.
            final_timeout: Ignored.
            blocking: Passed to each :meth:`move_base_to` call.
            world_frame: Ignored; see :meth:`move_base_to`.
            **kwargs: Ignored.
        """
        for wp in trajectory:
            self.move_base_to(
                wp,
                relative=relative,
                blocking=blocking,
                world_frame=world_frame,
            )

    def get_pose_graph(self) -> np.ndarray:
        """Return an empty pose graph (no SLAM in SQA3D replay harness)."""
        return np.zeros((0, 3), dtype=np.float64)

    def at_goal(self) -> bool:
        """Always True; greedy navigation does not expose goal-reached state."""
        return True

    def get_footprint(self) -> Footprint:
        """Stretch-shaped footprint for planner compatibility."""
        return Footprint(width=0.34, length=0.33, width_offset=0.0, length_offset=-0.1)

    def get_dof(self):
        """Return planar DoF count (3)."""
        return self.dof

    def set_config(self, q) -> None:
        """No-op: agent pose is owned by the simulator."""
        return None

    def get_config(self):
        """Return current planar pose ``(x, y, heading)``."""
        self._sync_pose_from_sim()
        return self._xyt.copy()

    def set_velocity(self, v: float, w: float):
        """Store nominal velocity hints for planner metadata."""
        self._v = float(v)
        self._w = float(w)

    def say(self, text: str):
        """No-op speech stub."""
        return None

    def get_pan_tilt(self) -> tuple[float, float]:
        """Return fixed head pan/tilt stub ``(0, 0)``."""
        return (0.0, 0.0)

    def get_six_joints(self, timeout: float = 5.0) -> np.ndarray:
        """Return zero arm joint vector (no manipulator in replay harness)."""
        return np.zeros(6, dtype=np.float64)

    def navigate_to(self, xyt, relative: bool = False, blocking: bool = True, **kwargs) -> bool:
        """Navigate to a goal; wrapper around :meth:`move_base_to`.

        Args:
            xyt: ``(x, y, yaw)`` goal in scene coordinates.
            relative: When True, goal is relative to current pose.
            blocking: Passed to :meth:`move_base_to`.
            **kwargs: Optional ``timeout`` and ``world_frame`` (see :meth:`move_base_to`).

        Returns:
            False when ``xyt`` does not have exactly three elements; otherwise True.
        """
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
        """No-op arm stub; returns True."""
        return True

    def head_to(self, head_pan: float, head_tilt: float, blocking: bool = False, **kwargs) -> None:
        """No-op head stub."""
        return None

    def look_front(self, blocking: bool = True, timeout: float = 10.0) -> None:
        """No-op look-front stub."""
        return None

    def gripper_to(self, target: float, blocking: bool = True, reliable: bool = True) -> None:
        """No-op gripper stub."""
        return None

    def get_gripper_position(self) -> float:
        """Return fixed half-open gripper stub ``0.5``."""
        return 0.5

    def get_emet_session(self) -> dict[str, Any] | None:
        """Always ``None``; SQA3D replay has no ZMQ ``emet_session`` block."""
        return None

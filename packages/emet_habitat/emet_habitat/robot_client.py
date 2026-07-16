# Copyright (c) Chris Paxton 2026
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

Session metadata (optional)::

    robot.set_emet_session({"sim_object_placements": {...}})
    placements = robot.get_emet_session()
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
from emet.utils.logger import Logger
from emet_habitat.observations import habitat_rgb_depth_to_observations
from emet_habitat.simulator import HabitatEQASimulator

logger = Logger(__name__)


class HabitatRobotClient(AbstractRobotClient, RobotModel):
    """In-process Habitat agent implementing the emet robot client interface.

    Wraps :class:`~emet_habitat.simulator.HabitatEQASimulator` so DynaMem / Dynagraph
    controllers can call the same :class:`~emet.core.robot.AbstractRobotClient` API
    used against ZMQ sims.

    Navigation goals are always in Habitat **world** coordinates (X/Z plane + yaw).
    The ``world_frame`` argument on :meth:`move_base_to`, :meth:`navigate_to`, and
    :meth:`execute_trajectory` is accepted for API parity with
    :class:`~emet.controller.generic_zmq_client.GenericZmqClient` but **ignored**
    (there is no separate episode-relative frame in this client).

    Args:
        simulator: Open Habitat EQA simulator for one HM3D scene.

    Attributes:
        dof: Planar DoF count (3: x, z, heading) for navigation planners.
        _sim: Underlying :class:`~emet_habitat.simulator.HabitatEQASimulator`.
        _xyt: Cached nav pose ``(x, z, heading)`` in Habitat world coordinates.
        _v: Nominal linear velocity hint (planner metadata; not sent to Habitat-Sim).
        _w: Nominal angular velocity hint (planner metadata; not sent to Habitat-Sim).
        _base_control_mode: Current :class:`~emet.core.robot.ControlMode` stub.
        _emet_session: Optional session dict (GT placements, capabilities) for find-phase harnesses.
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
        self._post_step_hooks: list[Any] = []
        self._head_pan = 0.0
        self._head_tilt = float(np.deg2rad(float(getattr(simulator, "camera_tilt_deg", -30.0))))
        self._pending_nav: dict[str, Any] | None = None
        self._sync_pose_from_sim()

    def add_post_step_hook(self, hook: Any) -> None:
        """Register ``hook(robot, frame)`` after each Habitat discrete action."""
        if hook not in self._post_step_hooks:
            self._post_step_hooks.append(hook)

    def remove_post_step_hook(self, hook: Any) -> None:
        if hook in self._post_step_hooks:
            self._post_step_hooks.remove(hook)

    def _sim_step(self, action: str):
        """Run one Habitat discrete action and notify post-step hooks."""
        frame = self._sim.step(action)
        self._sync_pose_from_sim()
        for hook in self._post_step_hooks:
            try:
                hook(self, frame)
            except Exception as exc:
                logger.warning(f"Habitat post-step hook failed: {exc}")
        return frame

    def _observation_kwargs(self) -> dict:
        return {
            "floor_y": self._sim.floor_y,
            "sensor_height": self._sim.sensor_height,
            "camera_tilt_deg": self._sim.camera_tilt_deg,
        }

    def _sync_pose_from_sim(self) -> None:
        """Refresh cached ``_xyt`` from the latest simulator agent state."""
        frame = self._sim.get_frame()
        obs = habitat_rgb_depth_to_observations(
            rgb=frame.rgb,
            depth=frame.depth,
            agent_state=frame.agent_state,
            intrinsics=frame.intrinsics,
            semantic=frame.semantic,
            **self._observation_kwargs(),
        )
        self._xyt = np.array([obs.gps[0], obs.gps[1], float(obs.compass[0])], dtype=np.float64)

    @property
    def hm3d_semantic_labeler(self):
        """Optional HM3D semantic labeler when the simulator was built with semantics."""
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
            **self._observation_kwargs(),
        )

    def get_base_pose(self, timeout: float = 5.0) -> np.ndarray:
        """Return planar base pose ``(x, z, heading)`` in Habitat world coordinates.

        Args:
            timeout: Unused; pose is read synchronously from the simulator.
        """
        self._sync_pose_from_sim()
        return self._xyt.copy()

    def _greedy_to_habitat_point(
        self,
        habitat_xyz: np.ndarray,
        max_steps: int | None = None,
        *,
        stop_radius_m: float = 0.12,
    ) -> bool:
        """Greedy move/turn toward a Habitat world XYZ target (uses X/Z plane).

        Returns True when the robot ends within ``stop_radius_m`` of the target.
        """
        self.begin_nav_to_point(habitat_xyz, goal_theta=None, max_steps_hint=max_steps)
        while True:
            status = self.nav_tick(max_sim_steps=1, stop_radius_m=stop_radius_m)
            if status != "running":
                return status == "done"

    def begin_nav_to(
        self,
        xyt: Iterable[float] | ContinuousNavigationAction,
        *,
        relative: bool = False,
        **_kwargs: Any,
    ) -> None:
        """Start non-blocking navigation toward a world (or relative) goal.

        Call :meth:`nav_tick` repeatedly until it returns ``done`` or ``failed``.
        """
        goal = np.asarray(xyt, dtype=np.float64).reshape(-1)[:3]
        if relative:
            goal = xyt_base_to_global(goal, self._xyt)
        goal_theta = float(goal[2]) if len(goal) >= 3 else None
        waypoints: list[np.ndarray] = []
        if not relative:
            find_path = getattr(self._sim, "find_path_to_xy", None)
            if callable(find_path):
                path_pts = find_path(float(goal[0]), float(goal[1]))
                if path_pts is not None:
                    for pt in path_pts[1:]:
                        waypoints.append(np.asarray(pt, dtype=np.float64).reshape(-1))
        if not waypoints:
            waypoints.append(np.array([goal[0], 0.0, goal[1]], dtype=np.float64))
        dist0 = 0.0
        self._sync_pose_from_sim()
        if waypoints:
            pt0 = waypoints[0]
            dist0 = math.hypot(float(pt0[0]) - self._xyt[0], float(pt0[2]) - self._xyt[1])
        self._pending_nav = {
            "waypoints": waypoints,
            "goal_theta": goal_theta,
            "wp_i": 0,
            "steps_on_wp": 0,
            "max_steps_per_wp": max(40, int(math.ceil(dist0 / 0.2) * 4)),
            "yaw_steps": 0,
            "phase": "path",
            "failed": False,
        }

    def begin_nav_to_point(
        self,
        habitat_xyz: np.ndarray,
        *,
        goal_theta: float | None = None,
        max_steps_hint: int | None = None,
    ) -> None:
        """Start non-blocking greedy navigation to a single Habitat XYZ point."""
        pt = np.asarray(habitat_xyz, dtype=np.float64).reshape(-1)
        self._sync_pose_from_sim()
        dist0 = math.hypot(float(pt[0]) - self._xyt[0], float(pt[2]) - self._xyt[1])
        max_steps = max_steps_hint
        if max_steps is None:
            max_steps = max(40, int(math.ceil(dist0 / 0.2) * 4))
        self._pending_nav = {
            "waypoints": [pt],
            "goal_theta": goal_theta,
            "wp_i": 0,
            "steps_on_wp": 0,
            "max_steps_per_wp": int(max_steps),
            "yaw_steps": 0,
            "phase": "path",
            "failed": False,
        }

    def nav_tick(self, max_sim_steps: int = 2, *, stop_radius_m: float = 0.12) -> str:
        """Advance pending navigation by up to ``max_sim_steps`` Habitat actions.

        Returns:
            ``running`` while the goal is not finished, ``done`` on success, ``failed``
            when a waypoint budget is exhausted before reaching the target.
        """
        nav = getattr(self, "_pending_nav", None)
        if not nav or nav.get("phase") == "done":
            if nav and nav.get("failed"):
                return "failed"
            return "done"

        for _ in range(max(1, int(max_sim_steps))):
            if nav["phase"] == "path":
                if nav["wp_i"] >= len(nav["waypoints"]):
                    if nav["goal_theta"] is not None:
                        nav["phase"] = "yaw"
                        nav["yaw_steps"] = 0
                    else:
                        nav["phase"] = "done"
                        self._sync_pose_from_sim()
                        return "done"
                    continue
                habitat_xyz = nav["waypoints"][nav["wp_i"]]
                goal_x = float(habitat_xyz[0])
                goal_z = float(habitat_xyz[2])
                self._sync_pose_from_sim()
                dx = goal_x - self._xyt[0]
                dz = goal_z - self._xyt[1]
                dist = math.hypot(dx, dz)
                if dist < stop_radius_m:
                    nav["wp_i"] += 1
                    nav["steps_on_wp"] = 0
                    continue
                target_heading = math.atan2(dz, dx)
                dtheta = (target_heading - self._xyt[2] + math.pi) % (2 * math.pi) - math.pi
                if abs(dtheta) > 0.12:
                    self._sim_step("turn_right" if dtheta > 0 else "turn_left")
                else:
                    self._sim_step("move_forward")
                nav["steps_on_wp"] += 1
                if nav["steps_on_wp"] > int(nav["max_steps_per_wp"]):
                    self._sync_pose_from_sim()
                    dist = math.hypot(goal_x - self._xyt[0], goal_z - self._xyt[1])
                    if dist < stop_radius_m * 1.5:
                        nav["wp_i"] += 1
                        nav["steps_on_wp"] = 0
                        continue
                    nav["failed"] = True
                    nav["phase"] = "done"
                    logger.warning(f"Habitat nav: stopped before reaching path waypoint ({goal_x:.2f}, {goal_z:.2f})")
                    return "failed"
            elif nav["phase"] == "yaw":
                goal_theta = float(nav["goal_theta"])
                self._sync_pose_from_sim()
                dtheta = (goal_theta - self._xyt[2] + math.pi) % (2 * math.pi) - math.pi
                if abs(dtheta) < 0.1 or nav["yaw_steps"] >= 18:
                    nav["phase"] = "done"
                    self._sync_pose_from_sim()
                    return "done"
                self._sim_step("turn_right" if dtheta > 0 else "turn_left")
                nav["yaw_steps"] += 1
            else:
                return "failed" if nav.get("failed") else "done"
        return "running"

    def move_base_to(
        self,
        xyt: Iterable[float] | ContinuousNavigationAction,
        relative: bool = False,
        blocking: bool = False,
        verbose: bool = False,
        timeout: float | None = None,
        world_frame: bool | None = None,
        **kwargs: Any,
    ) -> bool:
        """Navigate to ``(x, z[, yaw])`` using navmesh path following when available.

        Args:
            xyt: Goal pose in Habitat world coordinates (x, z, optional yaw).
            relative: When True, goal is interpreted relative to current ``_xyt``.
            blocking: Ignored (Habitat steps are synchronous); always runs to completion.
            verbose: Ignored.
            timeout: Ignored.
            world_frame: Ignored; goals are already Habitat world coordinates. Kept for
                API parity with ZMQ clients that distinguish episode vs nav-world frames.
            **kwargs: Ignored (ZMQ client compatibility).

        Returns:
            True when the final pose is within the stop radius (and yaw if requested).
        """
        self.begin_nav_to(xyt, relative=relative, world_frame=world_frame, **kwargs)
        while True:
            status = self.nav_tick(max_sim_steps=1)
            if status != "running":
                return status == "done"

    def reset(self):
        """Reset control mode to navigation (simulator pose is unchanged)."""
        self._base_control_mode = ControlMode.NAVIGATION
        self._pending_nav = None

    def start(self) -> bool:
        """No-op startup hook; returns True so controllers proceed to update."""
        return True

    def switch_to_navigation_mode(self):
        """Mark client as in navigation mode (no Habitat action)."""
        self._base_control_mode = ControlMode.NAVIGATION

    def switch_to_manipulation_mode(self):
        """Mark client as in manipulation mode (EQA harness does not move the arm)."""
        self._base_control_mode = ControlMode.MANIPULATION

    def open_gripper(self) -> None:
        """No-op gripper stub."""
        return None

    def move_to_nav_posture(self):
        """No-op posture stub; Habitat agent has no Stretch arm."""
        return True

    def move_to_manip_posture(self):
        """No-op posture stub; Habitat agent has no Stretch arm."""
        return True

    def get_robot_model(self) -> RobotModel:
        """Return this client as the planning robot model (planar DoF only)."""
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
        world_frame: bool | None = None,
        **kwargs: Any,
    ) -> bool:
        """Visit each waypoint via :meth:`move_base_to` (open-loop).

        Args:
            trajectory: List of ``(x, z[, yaw])`` goals in Habitat world coordinates.
            pos_err_threshold: Ignored (greedy controller has fixed stop distance).
            rot_err_threshold: Ignored.
            spin_rate: Ignored.
            verbose: Ignored.
            per_waypoint_timeout: Ignored.
            relative: Passed to :meth:`move_base_to` for every waypoint.
            final_timeout: Ignored.
            blocking: Passed to :meth:`move_base_to` (synchronous regardless).
            world_frame: Ignored; see :meth:`move_base_to`.
            **kwargs: Ignored.

        Returns:
            True if every waypoint succeeded.
        """
        ok = True
        for wp in trajectory:
            ok = (
                self.move_base_to(
                    wp,
                    relative=relative,
                    blocking=blocking,
                    world_frame=world_frame,
                )
                and ok
            )
        return ok

    def get_pose_graph(self) -> np.ndarray:
        """Return an empty pose graph (no SLAM in Habitat EQA harness)."""
        return np.zeros((0, 3), dtype=np.float64)

    def at_goal(self) -> bool:
        """True when no pending nav is active (or last nav finished successfully)."""
        nav = getattr(self, "_pending_nav", None)
        if not nav:
            return True
        return nav.get("phase") == "done" and not nav.get("failed")

    def get_footprint(self) -> Footprint:
        """Stretch-shaped footprint for planner compatibility."""
        return Footprint(width=0.34, length=0.33, width_offset=0.0, length_offset=-0.1)

    def get_dof(self):
        """Return planar DoF count (3)."""
        return self.dof

    def set_config(self, q) -> None:
        """No-op: Habitat agent pose is owned by the simulator."""
        return None

    def get_config(self):
        """Return current planar pose ``(x, z, heading)`` (alias of cached ``_xyt``)."""
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
        """Return current head pan/tilt in radians."""
        return (float(self._head_pan), float(self._head_tilt))

    def get_joint_state(self, timeout: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stretch-shaped joint vector with base XYT + head pan/tilt filled in."""
        from emet.motion.constants import STRETCH_HOME_Q
        from emet.motion.kinematics import HelloStretchIdx

        self._sync_pose_from_sim()
        q = np.asarray(STRETCH_HOME_Q, dtype=np.float64).copy()
        q[0] = float(self._xyt[0])
        q[1] = float(self._xyt[1])
        q[2] = float(self._xyt[2])
        q[HelloStretchIdx.HEAD_PAN] = float(self._head_pan)
        q[HelloStretchIdx.HEAD_TILT] = float(self._head_tilt)
        dq = np.zeros_like(q)
        tau = np.zeros_like(q)
        return q, dq, tau

    def get_six_joints(self, timeout: float = 5.0) -> np.ndarray:
        """Return zero arm joint vector (no manipulator in Habitat EQA harness)."""
        return np.zeros(6, dtype=np.float64)

    def navigate_to(self, xyt, relative: bool = False, blocking: bool = True, **kwargs) -> bool:
        """Navigate to a goal; wrapper around :meth:`move_base_to`.

        Args:
            xyt: ``(x, z, yaw)`` goal.
            relative: When True, goal is relative to current pose.
            blocking: Passed to :meth:`move_base_to`.
            **kwargs: Optional ``timeout`` and ``world_frame`` (both forwarded/ignored per
                :meth:`move_base_to`).

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
        return self.at_goal()

    def arm_to(self, joint_angles=None, gripper=None, head=None, blocking=True, **kwargs) -> bool:
        """No-op arm stub; returns True."""
        return True

    def head_to(self, head_pan: float, head_tilt: float, blocking: bool = False, **kwargs) -> None:
        """Aim the Habitat head camera (pan/tilt radians); updates RGB/depth on next frame."""
        pan = float(head_pan)
        tilt = float(head_tilt)
        self._head_pan = pan
        self._head_tilt = tilt
        set_look = getattr(self._sim, "set_camera_look", None)
        if callable(set_look):
            set_look(pan, tilt)

    def look_front(self, blocking: bool = True, timeout: float = 10.0) -> None:
        """Reset head to Stretch ``look_front`` pan/tilt."""
        from emet.motion.constants import look_front

        self.head_to(float(look_front[0]), float(look_front[1]), blocking=blocking)

    def gripper_to(self, target: float, blocking: bool = True, reliable: bool = True) -> None:
        """No-op gripper stub."""
        return None

    def get_gripper_position(self) -> float:
        """Return fixed half-open gripper stub ``0.5``."""
        return 0.5

    def get_emet_session(self) -> dict[str, Any] | None:
        """Optional session metadata injected by Habitat find-phase / GT harnesses.

        Typical keys include ``sim_object_placements`` for ground-truth graph refresh.
        Unlike ZMQ clients, this is set locally via :meth:`set_emet_session` rather than
        streamed from a sim server.
        """
        return self._emet_session

    def set_emet_session(self, session: dict | None) -> None:
        """Attach or clear session dict for GT graph refresh (Habitat find-phase harness).

        Args:
            session: Shallow-copied dict stored on the client, or ``None`` to clear.
        """
        self._emet_session = dict(session) if session is not None else None

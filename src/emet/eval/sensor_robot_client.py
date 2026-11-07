# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Production robot interface over the qualified RGB-D evaluation backend.

This client has no simulator handle, world-pose reader, semantic sensor or
navmesh API. Planning coordinates are the odometry frame. The body has no arm
or independent head pan; unsupported commands fail rather than pretending to
move hardware. This is an EQA/search body, not a manipulation benchmark robot.
"""

import numpy as np

from emet.core.interfaces import Observations
from emet.core.robot import AbstractRobotClient, ControlMode
from emet.eval.grapheqa_navigation import (
    ROTATION_STEP_DEG,
    CollectionBudgetExceeded,
    HabitatMotionExecutor,
    base_xyt,
)
from emet.utils.geometry import xyt_base_to_global


class SensorRobotClient(AbstractRobotClient):
    """Adapt measured observations and checked motion, without an alternate agent."""

    def __init__(self, sensor, navigation):
        super().__init__()
        self.sensor, self.navigation = sensor, navigation
        self.executor = HabitatMotionExecutor(sensor, navigation)
        self._seq_id = sensor.last["sequence"]
        self.last_navigation_result = None
        self.switch_to_navigation_mode()

    def get_observation(self, **kwargs):
        frame = self.sensor.capture()
        self.navigation.update(frame)
        self._seq_id = frame["sequence"]
        pose = self.get_base_pose()
        return Observations(
            gps=pose[:2],
            compass=pose[2:],
            rgb=frame["rgb"],
            depth=frame["depth"],
            camera_K=frame["camera_K"],
            camera_pose=frame["camera_pose"],
            seq_id=self._seq_id,
            image_timing={
                "head_cam_left": {
                    "timestamp_ns": frame["timestamp_ns"],
                    "clock_domain": "habitat_monotonic",
                    "available": True,
                }
            },
        )

    def get_base_pose(self, **kwargs):
        pose = self.sensor.last["camera_pose"]
        if pose is None:
            raise RuntimeError("RGB-D tracking lost; no measured base pose")
        return base_xyt(pose)

    def wait_for_fresh_observation(self, *, timeout):
        """Synchronous renderers acquire on demand rather than in a recv thread."""
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("Positive observation timeout required")
        return self.get_observation()

    def move_base_to(self, xyt, relative=False, blocking=True, **kwargs):
        goal = np.asarray(getattr(xyt, "xyt", xyt), dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("Finite local-world XYT required")
        if relative:
            goal = xyt_base_to_global(goal, self.get_base_pose())
        return self.execute_trajectory([goal], blocking=blocking)

    def navigate_to(self, xyt, **kwargs):
        return self.move_base_to(xyt, **kwargs)

    def execute_trajectory(self, trajectory, *, relative=False, blocking=True, **kwargs):
        if relative or not blocking:
            raise ValueError("Only blocking local-world trajectories are supported")
        self.last_navigation_result = self.executor.execute(trajectory)
        self._seq_id = self.sensor.last["sequence"]
        return self.last_navigation_result["success"]

    def get_pan_tilt(self):
        pose = self.sensor.last["camera_pose"]
        if pose is None:
            raise RuntimeError("Cannot measure head pose after tracking loss")
        return 0.0, float(np.arcsin(np.clip(pose[2, 2], -1, 1)))

    def get_head_capability(self):
        from emet.robots.head_capability import HeadCapability

        return HeadCapability((0.0, 0.0), (np.deg2rad(-65), np.deg2rad(-15)), "qualified_sensor_body")

    def head_to(self, head_pan, head_tilt, blocking=True, **kwargs):
        if not np.isfinite([head_pan, head_tilt]).all():
            return False
        if abs(head_pan) > 1e-6 or not np.deg2rad(-65) <= head_tilt <= np.deg2rad(-15):
            return False
        while abs(delta := head_tilt - self.get_pan_tilt()[1]) > np.deg2rad(0.1):
            try:
                frame = self.sensor.step(
                    "look_up" if delta > 0 else "look_down", min(ROTATION_STEP_DEG, abs(np.rad2deg(delta)))
                )
            except CollectionBudgetExceeded:
                return False
            self.navigation.update(frame)
            self._seq_id = frame["sequence"]
        return True

    def look_front(self, **kwargs):
        return self.head_to(0.0, np.deg2rad(-30))

    def get_robot_model(self):
        return self

    def get_footprint(self):
        return self.navigation.footprint

    def get_dof(self):
        return 3

    def get_config(self):
        return self.get_base_pose()

    def get_pose_graph(self):
        return np.empty((0, 3))

    def at_goal(self):
        return bool(self.last_navigation_result and self.last_navigation_result["success"])

    def last_motion_failed(self):
        return bool(self.last_navigation_result and not self.last_navigation_result["success"])

    def start(self):
        return True

    def reset(self):
        # Episode reset belongs to the runner: never teleport the renderer.
        self.last_navigation_result = None
        self.switch_to_navigation_mode()

    def switch_to_navigation_mode(self):
        self._base_control_mode = ControlMode.NAVIGATION

    def move_to_nav_posture(self):
        return True  # This body has no manipulator to stow.

    def move_to_manip_posture(self):
        raise NotImplementedError("The sensor-only search body has no manipulator")

# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Nav2 base motion for Innate Mars (ZMQ ``xyt`` → ``NavigateToPose``)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

from emet.utils.geometry import angle_difference, xyt_base_to_global
from innate_mars_bridge.constants import MAP_FRAME, NAVIGATE_TO_POSE_ACTION, ODOM_FRAME
from innate_mars_bridge.remote.modules.nav_geometry import xyt_to_pose_stamped

if TYPE_CHECKING:
    from innate_mars_bridge.remote.ros import InnateMarsRosInterface


class MarsNavigationClient:
    """Send planar goals to innate-os Nav2 (``navigate_to_pose`` action)."""

    def __init__(self, ros: InnateMarsRosInterface, *, nav_frame: str | None = None):
        self._ros = ros
        self._nav_frame = nav_frame or MAP_FRAME
        self._action_client = None
        self._goal_lock = threading.Lock()
        self._at_goal = True
        self._active_goal_handle = None
        self._pending_goal_xyt: np.ndarray | None = None

    def _ensure_action_client(self):
        if self._action_client is not None:
            return
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient

        self._action_client = ActionClient(self._ros, NavigateToPose, NAVIGATE_TO_POSE_ACTION)
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self._ros.get_logger().warning(
                f"Nav2 action {NAVIGATE_TO_POSE_ACTION!r} not available; "
                "ensure maurice_nav is in navigation mode."
            )

    def at_goal(self) -> bool:
        with self._goal_lock:
            return bool(self._at_goal)

    def _resolve_goal_xyt(self, xyt: list[float] | np.ndarray, relative: bool) -> np.ndarray:
        goal = np.asarray(xyt, dtype=np.float64).reshape(3)
        if relative:
            base = self._ros.get_base_pose_xyt()
            return xyt_base_to_global(goal, base)
        return goal

    def _pick_nav_frame(self) -> str:
        """Prefer ``map`` when TF is available; fall back to ``odom``."""
        if self._ros.get_frame_pose(self._nav_frame, base_frame=ODOM_FRAME, timeout_s=0.2) is not None:
            return self._nav_frame
        if self._ros.get_frame_pose(ODOM_FRAME, base_frame=self._nav_frame, timeout_s=0.2) is not None:
            return self._nav_frame
        return ODOM_FRAME

    def move_base_to(
        self,
        xyt: list[float] | np.ndarray,
        *,
        relative: bool = False,
        blocking: bool = True,
        timeout_s: float = 120.0,
    ) -> bool:
        """Send ``NavigateToPose`` goal; optional blocking wait."""
        from nav2_msgs.action import NavigateToPose

        self._ensure_action_client()
        if self._action_client is None or not self._action_client.server_is_ready():
            self._ros.get_logger().error("Nav2 navigate_to_pose server not ready")
            return False

        goal_xyt = self._resolve_goal_xyt(xyt, relative)
        frame = self._pick_nav_frame()
        pose = xyt_to_pose_stamped(goal_xyt, frame)
        pose.header.stamp = self._ros.get_clock().now().to_msg()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        with self._goal_lock:
            self._at_goal = False
            self._pending_goal_xyt = goal_xyt.copy()

        if self._active_goal_handle is not None:
            try:
                self._active_goal_handle.cancel_goal_async()
            except Exception:
                pass

        send_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._on_feedback,
        )
        send_future.add_done_callback(self._on_goal_response)

        if blocking:
            return self._wait_for_goal(timeout_s=timeout_s, target_xyt=goal_xyt)
        return True

    def _on_goal_response(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._ros.get_logger().warning("NavigateToPose goal rejected")
            with self._goal_lock:
                self._at_goal = True
            return
        self._active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_feedback(self, feedback_msg) -> None:
        del feedback_msg

    def _on_result(self, future) -> None:
        del future
        with self._goal_lock:
            self._at_goal = True
            self._pending_goal_xyt = None
            self._active_goal_handle = None

    def _wait_for_goal(
        self,
        *,
        timeout_s: float,
        target_xyt: np.ndarray,
        pos_tol: float = 0.15,
        rot_tol: float = 0.35,
    ) -> bool:
        import time

        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self.at_goal():
                curr = self._ros.get_base_pose_xyt()
                pos_err = float(np.linalg.norm(curr[:2] - target_xyt[:2]))
                rot_err = abs(angle_difference(curr[2], target_xyt[2]))
                return pos_err < pos_tol and rot_err < rot_tol
            time.sleep(0.1)
        self._ros.get_logger().warning(f"NavigateToPose timed out after {timeout_s}s")
        return False

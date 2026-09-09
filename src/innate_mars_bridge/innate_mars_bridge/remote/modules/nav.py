# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Nav2 base motion for Innate Mars (ZMQ ``xyt`` → ``NavigateToPose`` / ``Spin``)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

from emet.utils.geometry import angle_difference, xyt_base_to_global
from innate_mars_bridge.constants import MAP_FRAME, NAVIGATE_TO_POSE_ACTION, ODOM_FRAME, SPIN_ACTION
from innate_mars_bridge.nav_helpers import is_yaw_only_relative

if TYPE_CHECKING:
    from innate_mars_bridge.remote.ros import InnateMarsRosInterface


class MarsNavigationClient:
    """Send planar goals to innate-os Nav2 (``navigate_to_pose`` / ``spin``)."""

    def __init__(self, ros: InnateMarsRosInterface, *, nav_frame: str | None = None):
        self._ros = ros
        self._nav_frame = nav_frame or MAP_FRAME
        self._action_client = None
        self._spin_client = None
        self._goal_lock = threading.Lock()
        self._at_goal = True
        self._active_goal_handle = None
        self._pending_goal_xyt: np.ndarray | None = None
        self._generation = 0
        self._terminal_status = None
        self._result_event = threading.Event()
        self._cancel_requested = False
        self.command_context = None

    def _begin_goal(self, goal, mode):
        with self._goal_lock:
            if not self._at_goal or self._terminal_status == "unknown":
                raise RuntimeError("navigation busy")
            self._generation += 1
            self._at_goal = False
            self._terminal_status = None
            self._result_event.clear()
            self._cancel_requested = False
            self._pending_goal_xyt = goal.copy()
            self.command_context = {"resolved_goal": goal.tolist(), "frame": ODOM_FRAME, "motion_mode": mode}
            return self._generation

    def terminal_status(self):
        with self._goal_lock:
            return self._terminal_status

    def cancel_navigation(self, timeout_s=2.0):
        with self._goal_lock:
            self._cancel_requested = True
            handle = self._active_goal_handle
        if handle is not None:
            handle.cancel_goal_async()
        if not self._result_event.wait(timeout_s):
            return False
        return self.terminal_status() in {"succeeded", "cancelled", "aborted", "rejected"}

    def _ensure_action_client(self):
        if self._action_client is not None:
            return
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient

        self._action_client = ActionClient(self._ros, NavigateToPose, NAVIGATE_TO_POSE_ACTION)
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self._ros.get_logger().warning(
                f"Nav2 action {NAVIGATE_TO_POSE_ACTION!r} not available; ensure maurice_nav is in navigation mode."
            )

    def _ensure_spin_client(self):
        if self._spin_client is not None:
            return
        from nav2_msgs.action import Spin
        from rclpy.action import ActionClient

        self._spin_client = ActionClient(self._ros, Spin, SPIN_ACTION)
        if not self._spin_client.wait_for_server(timeout_sec=2.0):
            self._ros.get_logger().warning(
                f"Nav2 action {SPIN_ACTION!r} not available; yaw-only goals will fall back to NavigateToPose."
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

    def _map_tf_available(self) -> bool:
        return self._ros.get_frame_pose(MAP_FRAME, base_frame=ODOM_FRAME, timeout_s=0.2) is not None

    def _pick_nav_frame(self) -> str:
        """Prefer ``map`` when TF is available; fall back to ``odom``."""
        if self._map_tf_available():
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
        """Send Nav2 goal; yaw-only relative uses ``Spin`` (no map TF required)."""
        goal_in = np.asarray(xyt, dtype=np.float64).reshape(3)
        if relative and is_yaw_only_relative(goal_in):
            ok = self._spin_in_place(float(goal_in[2]), blocking=blocking, timeout_s=timeout_s)
            if ok or self._spin_client is not None and self._spin_client.server_is_ready():
                return ok
            self._ros.get_logger().warning(
                "Spin failed or unavailable; falling back to NavigateToPose for yaw-only goal"
            )

        return self._navigate_to_pose(goal_in, relative=relative, blocking=blocking, timeout_s=timeout_s)

    def _spin_in_place(self, yaw_rad: float, *, blocking: bool, timeout_s: float) -> bool:
        """In-place rotate via Nav2 ``Spin`` (works when ``map`` TF is missing)."""
        from nav2_msgs.action import Spin

        self._ensure_spin_client()
        if self._spin_client is None or not self._spin_client.server_is_ready():
            return False

        base = self._ros.get_base_pose_xyt()
        target_xyt = np.asarray(
            [float(base[0]), float(base[1]), float(base[2]) + float(yaw_rad)],
            dtype=np.float64,
        )

        goal_msg = Spin.Goal()
        goal_msg.target_yaw = float(yaw_rad)
        goal_msg.time_allowance.sec = max(1, int(timeout_s))
        goal_msg.time_allowance.nanosec = 0

        generation = self._begin_goal(target_xyt, "nav2_spin")

        self._ros.get_logger().info(f"Spin in place by {yaw_rad:.3f} rad ({np.degrees(yaw_rad):.1f}°)")
        send_future = self._spin_client.send_goal_async(goal_msg)
        send_future.add_done_callback(lambda future: self._on_goal_response(future, generation))

        if blocking:
            return self._wait_for_goal(timeout_s=timeout_s, target_xyt=target_xyt)
        return True

    def _navigate_to_pose(
        self,
        xyt: np.ndarray,
        *,
        relative: bool,
        blocking: bool,
        timeout_s: float,
    ) -> bool:
        """Send ``NavigateToPose`` goal; optional blocking wait."""
        from nav2_msgs.action import NavigateToPose

        from innate_mars_bridge.remote.modules.nav_geometry import xyt_to_pose_stamped

        self._ensure_action_client()
        if self._action_client is None or not self._action_client.server_is_ready():
            self._ros.get_logger().error("Nav2 navigate_to_pose server not ready")
            return False

        if not self._map_tf_available():
            self._ros.get_logger().warning(
                "TF frame 'map' is missing (slam/localization inactive). "
                "NavigateToPose often hangs; prefer Spin for yaw-only, or restore map→odom TF."
            )

        goal_xyt = self._resolve_goal_xyt(xyt, relative)
        # Goals and base feedback are odometry-frame coordinates. Never merely relabel
        # an odometry pose as map; Nav2 owns any required TF conversion.
        pose = xyt_to_pose_stamped(goal_xyt, ODOM_FRAME)
        pose.header.stamp = self._ros.get_clock().now().to_msg()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        generation = self._begin_goal(goal_xyt, "nav2_navigate_to_pose")

        send_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._on_feedback,
        )
        send_future.add_done_callback(lambda future: self._on_goal_response(future, generation))

        if blocking:
            return self._wait_for_goal(timeout_s=timeout_s, target_xyt=goal_xyt)
        return True

    def _on_goal_response(self, future, generation) -> None:
        try:
            goal_handle = future.result()
            with self._goal_lock:
                if generation != self._generation:
                    if goal_handle.accepted:
                        goal_handle.cancel_goal_async()
                    return
                if not goal_handle.accepted:
                    self._terminal_status = "rejected"
                    self._at_goal = True
                    self._result_event.set()
                    return
                self._active_goal_handle = goal_handle
                cancel = self._cancel_requested
            if cancel:
                goal_handle.cancel_goal_async()
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(lambda result: self._on_result(result, generation))
        except Exception:
            self._record_result(generation, "unknown")

    def _on_feedback(self, feedback_msg) -> None:
        del feedback_msg

    def _on_result(self, future, generation) -> None:
        try:
            # action_msgs/GoalStatus: SUCCEEDED=4, CANCELED=5, ABORTED=6.
            status = {4: "succeeded", 5: "cancelled", 6: "aborted"}.get(future.result().status, "unknown")
        except Exception:
            status = "unknown"
        self._record_result(generation, status)

    def _record_result(self, generation, status):
        with self._goal_lock:
            if generation != self._generation:
                return
            self._terminal_status = status
            self._at_goal = True
            self._pending_goal_xyt = None
            self._active_goal_handle = None
            self._result_event.set()

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
                if self.terminal_status() != "succeeded":
                    return False
                curr = self._ros.get_base_pose_xyt()
                pos_err = float(np.linalg.norm(curr[:2] - target_xyt[:2]))
                rot_err = abs(angle_difference(curr[2], target_xyt[2]))
                return bool(pos_err < pos_tol and rot_err < rot_tol)
            time.sleep(0.1)
        self._ros.get_logger().warning(f"Nav2 goal timed out after {timeout_s}s")
        if not self.cancel_navigation():
            self._ros.get_logger().error("Nav2 cancellation not confirmed; do not issue another goal")
        return False

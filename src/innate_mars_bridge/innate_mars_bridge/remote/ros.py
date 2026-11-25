# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""ROS2 interface for Innate Mars: joint state, odometry, TF, and cameras."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.action.graph import get_action_server_names_and_types_by_node
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from innate_mars_bridge.constants import (
    ARM_STATE_TOPIC,
    EE_IMAGE_TOPIC,
    HEAD_BODY_FRAME,
    HEAD_LEFT_CAMERA_INFO_TOPIC,
    HEAD_LEFT_IMAGE_TOPIC,
    HEAD_POSITION_TOPIC,
    HEAD_RIGHT_CAMERA_INFO_TOPIC,
    HEAD_RIGHT_IMAGE_TOPIC,
    LIDAR_TOPIC,
    MAP_FRAME,
    ODOM_FRAME,
    ODOM_TOPIC,
)
from innate_mars_bridge.joint_layout import pack_innate_mars_joint_positions, pack_innate_mars_joint_velocities
from innate_mars_bridge.remote.arm_motion import published_arm_limits
from innate_mars_bridge.remote.joint_recovery import ACTION_NAME
from innate_mars_bridge.remote.modules.nav import MarsNavigationClient
from innate_mars_bridge.remote.native_arm import PREFIX
from innate_mars_bridge.ros.camera import RosCamera, RosCameraNoInfo
from innate_mars_bridge.ros.lidar import RosLidar
from innate_mars_bridge.ros.utils import matrix_from_pose_msg, to_matrix, transform_to_list

try:
    from mars_msgs.msg import ArmStatus
except ModuleNotFoundError as exc:
    if exc.name != "mars_msgs":
        raise
    from maurice_msgs.msg import ArmStatus

# Prefer odom for mapping; fall back when TF trees are split across bringup nodes.
ARM_COMMAND_TOPIC = "/mars/arm/commands"
ARM_STATUS_TOPIC = "/mars/arm/status"

# Innate's manipulation server subscribes to the arm command topic to cache
# commands for LeRobot readback. It is the one ordinary receiver permitted during
# isolated recovery; any other subscriber blocks recovery admission. Verified
# against Innate 0.8.0; if a future release renames this node, update both names.
LE_ROBOT_READBACK_NODE = ("manipulation_server", "/")

_TF_BASE_FRAMES = (ODOM_FRAME, MAP_FRAME, "base_footprint", "base_link")


class InnateMarsRosInterface(Node):
    """Subscribes to Innate Mars topics: arm state, odom, and head/EE cameras."""

    def __init__(self, verbose: bool = False):
        super().__init__("innate_mars_ros_interface")
        self.verbose = verbose

        self._js_lock = threading.Lock()
        self._joint_positions = np.zeros(6)
        self._joint_velocities = np.zeros(6)
        self._arm_feedback_time = None
        self._arm_status = None
        self._published_arm_limits = None
        self._arm_applied = None
        self._warned_arm_status_fields = False
        self._joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

        self._odom_pose = None
        self._odom_lock = threading.Lock()

        self._head_lock = threading.Lock()
        self._head_joint_rad = 0.0
        self._head_topic_updated = False

        self.tf2_buffer = Buffer()
        self.tf2_listener = TransformListener(self.tf2_buffer, self)

        # Subscribers
        self._robot_description_sub = self.create_subscription(
            String,
            "/robot_description",
            self._robot_description_callback,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._joint_sub = self.create_subscription(JointState, ARM_STATE_TOPIC, self._joint_callback, 10)
        self._arm_applied_sub = self.create_subscription(
            JointState, "/mars/arm/command_state", self._arm_applied_callback, 10
        )
        self._arm_status_sub = self.create_subscription(ArmStatus, ARM_STATUS_TOPIC, self._arm_status_callback, 10)
        self._arm_command_pub = self.create_publisher(
            Float64MultiArray, ARM_COMMAND_TOPIC, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        self._odom_sub = self.create_subscription(Odometry, ODOM_TOPIC, self._odom_callback, 10)
        self._head_sub = self.create_subscription(String, HEAD_POSITION_TOPIC, self._head_callback, 10)

        # Cameras: head left, head right (with camera_info), EE (no camera_info on hardware)
        self.head_left_cam = RosCamera(
            self,
            HEAD_LEFT_IMAGE_TOPIC,
            camera_info_topic=HEAD_LEFT_CAMERA_INFO_TOPIC,
            rotations=0,
            verbose=verbose,
        )
        self.head_right_cam = RosCamera(
            self,
            HEAD_RIGHT_IMAGE_TOPIC,
            camera_info_topic=HEAD_RIGHT_CAMERA_INFO_TOPIC,
            rotations=0,
            verbose=verbose,
        )
        self.ee_cam = RosCameraNoInfo(
            self,
            EE_IMAGE_TOPIC,
            rotations=0,
            image_ext="",
            verbose=verbose,
        )
        self._lidar = RosLidar(self, LIDAR_TOPIC, verbose=verbose)

        self.nav = None

    def wait_for_cameras(self) -> None:
        """Block until head stereo and EE cameras have published at least one frame."""
        self.get_logger().info("InnateMarsRosInterface: waiting for cameras...")
        self.head_left_cam.ensure_ready()
        self.head_right_cam.ensure_ready()
        try:
            self.ee_cam.ensure_ready(timeout_s=15.0)
        except RuntimeError as exc:
            self.get_logger().warning(f"EE camera not ready at startup ({exc}); continuing with head stereo.")
        self.get_logger().info("InnateMarsRosInterface: all cameras ready.")
        if self.nav is None:
            self.nav = MarsNavigationClient(self)

    def _joint_callback(self, msg: JointState):
        with self._js_lock:
            name_to_pos = dict(zip(msg.name, msg.position, strict=False))
            name_to_vel = dict(zip(msg.name, msg.velocity, strict=False)) if msg.velocity else {}
            for i, name in enumerate(self._joint_names):
                if name in name_to_pos:
                    self._joint_positions[i] = name_to_pos[name]
                if name in name_to_vel:
                    self._joint_velocities[i] = name_to_vel[name]

            if all(name in name_to_pos and name in name_to_vel for name in self._joint_names):
                self._arm_feedback_time = time.monotonic()
            else:
                self._arm_feedback_time = None

    def _robot_description_callback(self, msg: String):
        limits = published_arm_limits(msg.data)
        with self._js_lock:
            self._published_arm_limits = limits

    def get_published_arm_limits(self):
        with self._js_lock:
            return self._published_arm_limits

    def _arm_status_callback(self, msg: ArmStatus):
        # mars_msgs (Innate 0.8.0) and maurice_msgs (0.5.1) both expose is_ok and
        # is_torque_enabled. Fail closed if either field is missing rather than
        # assuming health from an unrecognized message shape.
        ok = getattr(msg, "is_ok", None)
        torque_enabled = getattr(msg, "is_torque_enabled", None)
        if ok is None or torque_enabled is None:
            if not self._warned_arm_status_fields:
                self._warned_arm_status_fields = True
                self.get_logger().warning(
                    "ArmStatus message is missing is_ok/is_torque_enabled; arm health treated as unavailable"
                )
            healthy = False
        else:
            healthy = bool(ok) and bool(torque_enabled)
        with self._js_lock:
            self._arm_status = (time.monotonic(), healthy)

    def _arm_applied_callback(self, msg: JointState):
        positions = dict(zip(msg.name, msg.position, strict=False))
        with self._js_lock:
            self._arm_applied = (
                time.monotonic(),
                np.array([positions.get(name, np.nan) for name in self._joint_names]),
            )

    def arm_target_applied(self, target, since, *, tolerance=0.005):
        # Innate 0.8 publishes an echo only when the hardware setpoint changes.
        # Fresh measured motion and exclusive ownership are checked separately.
        with self._js_lock:
            applied = self._arm_applied
            return applied is not None and applied[0] >= since and np.max(np.abs(applied[1] - target)) <= tolerance

    def arm_motion_feedback(self, *, require_health=True):
        with self._js_lock:
            now = time.monotonic()
            if self._arm_feedback_time is None or now - self._arm_feedback_time > 0.25:
                raise RuntimeError("Mars arm feedback is stale or incomplete")
            # The Maurice driver publishes health at 0.2 Hz.
            if require_health and (
                self._arm_status is None or now - self._arm_status[0] > 7.5 or not self._arm_status[1]
            ):
                raise RuntimeError("Mars arm torque or servo status is unavailable or unhealthy")
            q, dq = self._joint_positions.copy(), self._joint_velocities.copy()
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            raise RuntimeError("Mars arm feedback is non-finite")
        return q, dq

    def verify_arm_command_path(self):
        publishers = self.get_publishers_info_by_topic(ARM_COMMAND_TOPIC)
        if len(publishers) != 1 or publishers[0].node_name != self.get_name():
            raise RuntimeError("Mars arm command topic is not exclusively assigned to Emet")
        if not self._arm_command_pub.get_subscription_count():
            raise RuntimeError("Mars arm command topic has no matched subscriber")

    def verify_arm_recovery_path(self, *, native=False):
        subscribers = self.get_subscriptions_info_by_topic(ARM_COMMAND_TOPIC)
        # Innate's manipulation server caches commands for LeRobot readback.
        if any((sub.node_name, sub.node_namespace) != LE_ROBOT_READBACK_NODE for sub in subscribers):
            raise RuntimeError("An ordinary Mars arm command receiver is still active")
        states = self.get_publishers_info_by_topic(ARM_STATE_TOPIC)
        statuses = self.get_publishers_info_by_topic(ARM_STATUS_TOPIC)
        if len(states) != 1 or len(statuses) != 1:
            raise RuntimeError("Mars recovery requires one arm feedback provider")
        driver = (states[0].node_name, states[0].node_namespace)
        if driver != (statuses[0].node_name, statuses[0].node_namespace):
            raise RuntimeError("Mars arm state and health have different providers")
        if native:
            services = dict(self.get_service_names_and_types_by_node(*driver))
            if services.get(PREFIX + "/goto_js_v2") != ["mars_msgs/srv/GotoJS"] or any(
                services.get(PREFIX + suffix) != ["std_srvs/srv/Trigger"]
                for suffix in ("/acquire", "/release")
            ):
                raise RuntimeError("Mars arm feedback provider does not own the native arm session")
            return
        actions = dict(get_action_server_names_and_types_by_node(self, *driver))
        if actions.get(ACTION_NAME) != ["control_msgs/action/FollowJointTrajectory"]:
            raise RuntimeError("Mars arm feedback provider does not own the isolated recovery action")

    def publish_arm_targets(self, q):
        self._arm_command_pub.publish(Float64MultiArray(data=q.tolist()))

    def _odom_callback(self, msg: Odometry):
        with self._odom_lock:
            self._odom_pose = matrix_from_pose_msg(msg.pose.pose)

    def _parse_head_position_message(self, msg: String) -> float | None:
        raw = (msg.data or "").strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                for key in ("current_position", "position", "deg", "degrees"):
                    if key in data:
                        return float(data[key])
                return None
            if isinstance(data, (int, float)):
                return float(data)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        try:
            return float(raw)
        except ValueError:
            return None

    def _head_callback(self, msg: String):
        deg = self._parse_head_position_message(msg)
        if deg is None:
            return
        with self._head_lock:
            self._head_joint_rad = float(-np.deg2rad(deg))
            self._head_topic_updated = True

    def _head_joint_from_tf(self) -> float | None:
        """Fallback when ``/mars/head/current_position`` is idle: ``base_link`` → ``head`` TF."""
        t = self.get_frame_pose(HEAD_BODY_FRAME, base_frame="base_link", timeout_s=0.05)
        if t is None:
            return None
        r = np.asarray(t, dtype=np.float64).reshape(4, 4)[:3, :3]
        theta_urdf = float(np.arctan2(-r[0, 2], r[0, 0]))
        return float(-theta_urdf)

    def get_head_joint_rad(self) -> float:
        """Prefer ``base_link``→``head`` TF (tracks live nod); topic is fallback only."""
        tf_rad = self._head_joint_from_tf()
        if tf_rad is not None:
            return float(tf_rad)
        with self._head_lock:
            return float(self._head_joint_rad)

    def get_arm_joint_state(self):
        """Returns (positions, velocities) for joint1..joint6."""
        with self._js_lock:
            return self._joint_positions.copy(), self._joint_velocities.copy()

    def get_joint_state(self):
        """Returns 10-DoF (positions, velocities) for Emet ``RobotSpec`` layout."""
        arm_q, arm_dq = self.get_arm_joint_state()
        base = self.get_base_pose_xyt()
        return (
            pack_innate_mars_joint_positions(arm_q, base_xyt=base),
            pack_innate_mars_joint_velocities(arm_dq),
        )

    def at_goal(self) -> bool:
        if self.nav is None:
            return True
        return self.nav.at_goal()

    def get_base_pose_matrix(self):
        """Base pose as 4x4 matrix in odom frame (or None)."""
        with self._odom_lock:
            return self._odom_pose.copy() if self._odom_pose is not None else None

    def get_base_pose_xyt(self):
        """Base pose as (x, y, theta) in odom frame. Theta in radians."""
        mat = self.get_base_pose_matrix()
        if mat is None:
            return np.array([0.0, 0.0, 0.0])
        x, y = mat[0:2, 3]
        # yaw from rotation matrix
        theta = np.arctan2(mat[1, 0], mat[0, 0])
        return np.array([x, y, theta], dtype=np.float64)

    def get_frame_pose(self, frame: str, base_frame: str | None = None, timeout_s: float = 1.0):
        """Look up ``frame`` pose in ``base_frame``. Returns 4x4 cam-to-base or None."""
        from rclpy.duration import Duration
        from rclpy.time import Time

        bases = (base_frame,) if base_frame else _TF_BASE_FRAMES
        last_exc: TransformException | None = None
        for bf in bases:
            try:
                stamped = self.tf2_buffer.lookup_transform(bf, frame, Time(), Duration(seconds=timeout_s))
                trans, rot = transform_to_list(stamped)
                return to_matrix(trans, rot)
            except TransformException as exc:
                last_exc = exc
                continue
        if self.verbose and last_exc is not None:
            self.get_logger().warn(f"TF lookup failed for {frame!r} in {list(bases)!r}: {last_exc}")
        return None

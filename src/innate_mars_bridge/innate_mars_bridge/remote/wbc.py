"""Bind the robot-local watchdog transport to Innate's ROS topics."""

import time

from geometry_msgs.msg import Twist
from mars_msgs.msg import ArmStatus
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from innate_mars_bridge.wbc_transport import MarsWatchdogTransport

ARM_TOPIC = "/mars/arm/commands"
BASE_TOPIC = "/cmd_vel_teleop"


def create_wbc_transport(node, *, driver_verification):
    """Install the timer and publishers on an already spinning robot ROS node.

    Driver verification must come from an independently qualified native
    feedback path. The bridge's structural checks alone cannot establish it.
    """
    arm = node.create_publisher(Float64MultiArray, ARM_TOPIC, qos_profile_sensor_data)
    # The native cmd_vel_mux requests reliable delivery. Keep only the
    # newest setpoint so a slow subscriber cannot accumulate a velocity queue.
    base = node.create_publisher(Twist, BASE_TOPIC, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
    last_status = None
    last_grip = None

    def health(message):
        nonlocal last_status
        last_status = (time.monotonic(), bool(message.is_ok and message.is_torque_enabled))

    def echo(message):
        nonlocal last_grip
        if "joint6" in message.name and len(message.name) == len(message.position):
            last_grip = float(message.position[message.name.index("joint6")])

    node.create_subscription(ArmStatus, "/mars/arm/status", health, qos_profile_sensor_data)
    node.create_subscription(
        JointState,
        "/mars/arm/command_state",
        echo,
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )

    def owns():
        expected = [(node.get_name(), node.get_namespace())]
        paths = ((ARM_TOPIC, arm, "mars_arm"), (BASE_TOPIC, base, "cmd_vel_mux"))
        return all(
            [(info.node_name, info.node_namespace) for info in node.get_publishers_info_by_topic(topic)] == expected
            and publisher.get_subscription_count() > 0
            and any(info.node_name == consumer for info in node.get_subscriptions_info_by_topic(topic))
            for topic, publisher, consumer in paths
        ) and [info.node_name for info in node.get_publishers_info_by_topic("/cmd_vel")] == ["cmd_vel_mux"]

    def feedback():
        s = node.get_feedback_snapshot()
        now = time.monotonic()
        joints = s["samples"].get("arm", {})
        odom = s["samples"].get("odom", {})
        verified = driver_verification()
        efforts = joints.get("effort_proxy")
        return {
            "valid": s["complete_and_fresh"] and last_grip is not None,
            "arm_healthy": last_status is not None and last_status[1] and now - last_status[0] <= 5.5,
            "joint_position": joints.get("position", [])[:5],
            "joint_velocity": joints.get("velocity", [])[:5],
            "elbow_load_percent": efforts[2] if efforts is not None else None,
            "arm_captured_monotonic": now - max(joints.get("arrival_age_s", 1), joints.get("source_age_s", 1)),
            "odom_captured_monotonic": now - max(odom.get("arrival_age_s", 1), odom.get("source_age_s", 1)),
            "base_pose": odom.get("base_pose", []),
            "base_twist": odom.get("twist_child", []),
            "gripper_target_rad": last_grip,
            "gripper_state": [joints["position"][5], joints["velocity"][5]] if joints.get("position") else [],
            "feedback_si_units_verified": verified.get("feedback_si_units_verified") is True,
            "motor_read_validity_verified": verified.get("motor_read_validity_verified") is True,
            "elbow_load_percent_verified": verified.get("elbow_load_percent_verified") is True,
        }

    def wheels(values):
        twist = Twist()
        twist.linear.x, twist.angular.z = map(float, values)
        base.publish(twist)

    transport = MarsWatchdogTransport(
        feedback=feedback,
        owns_topics=owns,
        publish_arm=lambda q: arm.publish(Float64MultiArray(data=q.tolist())),
        publish_base=wheels,
    )
    node.create_timer(0.02, transport.tick)
    return transport

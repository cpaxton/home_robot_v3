#!/usr/bin/env python3
"""Exercise the production ROS WBC binding against Innate's headless MJCF simulator."""

import argparse
import importlib.util
import json
import os
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Twist
from innate_mars_bridge.remote.ros import InnateMarsRosInterface
from innate_mars_bridge.remote.wbc import ARM_TOPIC, BASE_TOPIC
from mars_msgs.msg import ArmStatus
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from emet.core.pose_stream import FORMAT, pose_stream_goal
from emet.rl.manipulation.tracking import TaskFrameFK
from emet.rl.manipulation.whole_body_observation import pose_rotation
from emet.robots.innate_mars.startup_recovery import JOINT_NAMES
from emet.robots.innate_mars.wbc_commissioning import VERIFICATION_KEYS, create_executor


def create_plant(package):
    spec = importlib.util.spec_from_file_location("innate_upstream_sim", package / "maurice_sim/sim.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)

    class Plant(upstream.MauriceBotNode):
        def __init__(self):
            # Keep upstream physics, control and sample generation, omit the
            # upstream viewer/camera thread in this headless test.
            Node.__init__(self, "mars_arm")
            model = ET.parse(package / "mjcf/maurice.mjcf").getroot()
            model.find("compiler").set("meshdir", str(package / "meshes"))
            self.model = mujoco.MjModel.from_xml_string(ET.tostring(model, encoding="unicode"))
            self.data = mujoco.MjData(self.model)
            self.twist_cmd = Twist()
            self.arm_commands = []
            self.received_targets = 0
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            self.odom = self.create_publisher(Odometry, "/odom", 10)
            self.arm = self.create_publisher(JointState, "/mars/arm/state", 10)
            self.status = self.create_publisher(ArmStatus, "/mars/arm/status", 10)
            self.echo = self.create_publisher(
                JointState, "/mars/arm/command_state", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            )
            self.odom_pub = SimpleNamespace(publish=self.publish_child_twist)
            self.joint_state_pub = SimpleNamespace(publish=self.publish_full_arm)
            self.create_subscription(Twist, "/cmd_vel", self.set_twist, 10)
            self.create_subscription(Float64MultiArray, ARM_TOPIC, self.set_arm, qos_profile_sensor_data)
            self.create_timer(0.002, self.simulation_timer_callback)
            self.create_timer(1 / 30, self.publish_timer_callback)
            self.create_timer(0.1, self.publish_status)
            self.publish_echo(self.data.qpos[3:9].tolist())

        def set_twist(self, message):
            self.twist_cmd = message

        def set_arm(self, message):
            self.arm_commands = message.data.tolist()
            self.received_targets += 1
            self.publish_echo(self.arm_commands)

        def publish_echo(self, target):
            message = JointState(name=list(JOINT_NAMES), position=target)
            message.header.stamp = self.get_clock().now().to_msg()
            self.echo.publish(message)

        def publish_status(self):
            valid = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
            self.status.publish(ArmStatus(is_ok=valid, is_torque_enabled=valid))

        def publish_full_arm(self, message):
            message.velocity = self.data.qvel[3:9].tolist()
            message.effort = [0.0] * 6
            self.arm.publish(message)

        def publish_child_twist(self, message):
            yaw = self.data.qpos[2]
            x, y = message.twist.twist.linear.x, message.twist.twist.linear.y
            message.twist.twist.linear.x = float(np.cos(yaw) * x + np.sin(yaw) * y)
            message.twist.twist.linear.y = float(-np.sin(yaw) * x + np.cos(yaw) * y)
            self.odom.publish(message)

    return Plant()


def run(package):
    plant = create_plant(package)
    mux = Node("cmd_vel_mux")
    wheels = mux.create_publisher(Twist, "/cmd_vel", 10)
    mux.create_subscription(Twist, BASE_TOPIC, wheels.publish, 10)
    bridge = InnateMarsRosInterface()
    executor = create_executor(bridge)
    ros = SingleThreadedExecutor()
    for node in (plant, mux, bridge):
        ros.add_node(node)
    thread = threading.Thread(target=ros.spin, daemon=True)
    thread.start()
    try:

        def await_readiness(expected):
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if executor.available() == expected:
                    return
                time.sleep(0.05)
            raise RuntimeError(f"expected WBC readiness {expected}: {executor.wbc.driver.capabilities()}")

        await_readiness(True)
        competing = plant.create_publisher(Float64MultiArray, ARM_TOPIC, qos_profile_sensor_data)
        try:
            await_readiness(False)
        finally:
            plant.destroy_publisher(competing)
        await_readiness(True)
        executor._measure()
        control = executor.control
        hand = control.ee_pose().copy()
        base = control.base_pose().copy()
        start = control.positions().copy()
        target = start.copy()
        target[3] += 0.05
        fk = TaskFrameFK(control.model, control.data, control)
        waypoint = fk(base, target)

        def goal(pose):
            quat = Rotation.from_matrix(pose_rotation(pose)).as_quat()
            return pose_stream_goal(
                {"arms": {"arm": [*pose[:3], *quat]}},
                duration_s=0.8,
                frame="odom",
                free_base=False,
                capabilities=executor.stream.capabilities(),
            )

        executor.start(goal(hand))
        time.sleep(0.35)
        executor.update(goal(waypoint))
        executor.update({"pose_stream": {"format": FORMAT, "finish": True}})
        if not executor._done.wait(8):
            raise RuntimeError("sim WBC stream timed out")
        stopped = executor.stop()
        feedback = executor.wbc.driver.feedback()
        motion = float(np.max(np.abs(np.asarray(feedback["joint_position"]) - start)))
        result = executor.result()
        report = {
            "result": result,
            "stop_confirmed": stopped,
            "competing_publisher_blocked_readiness": True,
            "arm_setpoints_received": plant.received_targets,
            "max_final_joint_displacement_rad": motion,
            "base_displacement_m": float(np.linalg.norm(np.asarray(feedback["base_pose"][:2]) - base[:2])),
            "hardware": False,
        }
        print(json.dumps(report, indent=2))
        return 0 if result[0] == "succeeded" and stopped and motion > 0.005 else 1
    finally:
        if executor._worker is not None:
            executor.stop()
        ros.shutdown()
        thread.join()
        for node in (bridge, mux, plant):
            node.destroy_node()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--innate", type=Path, required=True, help="Path to local Innate OS checkout")
    args = parser.parse_args()
    domain = os.environ.get("ROS_DOMAIN_ID", "")
    if os.environ.get("ROS_LOCALHOST_ONLY") != "1" or not domain.isdecimal() or not 80 <= int(domain) <= 99:
        parser.error("set ROS_LOCALHOST_ONLY=1 and use a dedicated ROS_DOMAIN_ID from 80 to 99")
    config = json.loads(Path(os.environ["EMET_MARS_WBC_CONFIG"]).read_text())
    if not all(config.get(key) is True for key in VERIFICATION_KEYS):
        parser.error("use a dedicated simulator commissioning record")
    package = args.innate.resolve() / "ros2_ws/src/maurice_bot/maurice_sim"
    rclpy.init(args=[])
    try:
        return run(package)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

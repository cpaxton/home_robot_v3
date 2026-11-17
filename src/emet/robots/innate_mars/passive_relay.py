"""Standalone ROS subscriber used over SSH by the operator console.

Only subscriptions and TF lookups. No actuator publishers, service calls,
package installation, driver starts or persistent files on the robot.
"""

import argparse
import base64
import json
import time
import uuid

PREFIX = "EMET_PASSIVE "


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--fps", type=float, default=5)
    args = parser.parse_args()
    if not 0 < args.seconds <= 21600 or not 0 < args.fps <= 10:
        parser.error("bounded duration <=6h and camera rate <=10 Hz required")
    import cv2
    import numpy as np
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import CompressedImage, Image, JointState
    from tf2_ros import Buffer, TransformException, TransformListener

    rclpy.init()
    node = rclpy.create_node("emet_passive_operator_view")
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    session = uuid.uuid4().hex
    latest, image_times, subscriptions = {}, {}, []
    sequence = 0

    def emit(kind, **values):
        nonlocal sequence
        sequence += 1
        print(
            PREFIX
            + json.dumps(
                {"session": session, "sequence": sequence, "kind": kind, **values},
                allow_nan=False,
                separators=(",", ":"),
            ),
            flush=True,
        )

    def timing(header):
        stamp = header.stamp.sec * 10**9 + header.stamp.nanosec
        return {
            "source_stamp_ns": stamp,
            "source_age_s": (node.get_clock().now().nanoseconds - stamp) / 1e9 if stamp > 0 else None,
        }

    def arm(msg):
        latest["arm"] = {
            "names": list(msg.name),
            "position": list(msg.position),
            "velocity": list(msg.velocity),
            "effort_proxy": list(msg.effort),
            **timing(msg.header),
        }

    def odom(msg):
        p, q, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
        latest["odom"] = {
            "position": [p.x, p.y, p.z],
            "quaternion_xyzw": [q.x, q.y, q.z, q.w],
            "twist_child": [v.linear.x, v.linear.y, v.linear.z, v.angular.x, v.angular.y, v.angular.z],
            "frame_id": msg.header.frame_id,
            "child_frame_id": msg.child_frame_id,
            **timing(msg.header),
        }

    def image(name, msg, compressed):
        now = time.monotonic()
        if now - image_times.get(name, -1e9) < 1 / args.fps:
            return
        image_times[name] = now
        try:
            if compressed:
                frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("undecodable image")
            else:
                channels = {"rgb8": 3, "bgr8": 3, "mono8": 1}.get(msg.encoding)
                if channels is None or not 0 < msg.width <= 4096 or not 0 < msg.height <= 4096:
                    raise ValueError("unsupported image encoding/size")
                rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
                frame = rows[:, : msg.width * channels].reshape(msg.height, msg.width, channels).copy()
                if msg.encoding == "rgb8":
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif channels == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.shape[1] > 640:
                frame = cv2.resize(frame, (640, round(640 * frame.shape[0] / frame.shape[1])))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                raise ValueError("JPEG encode failed")
            emit(
                "camera",
                name=name,
                jpeg=base64.b64encode(encoded).decode(),
                width=frame.shape[1],
                height=frame.shape[0],
                **timing(msg.header),
            )
        except (ValueError, cv2.error) as exc:
            emit("camera_error", name=name, error=str(exc)[:200])

    for topic, kind, callback in (("/mars/arm/state", JointState, arm), ("/odom", Odometry, odom)):
        subscriptions.append(node.create_subscription(kind, topic, callback, qos_profile_sensor_data))
    for name, topic, compressed in (
        ("head_left", "/mars/main_camera/left/image_rect_color/compressed", True),
        ("head_right", "/mars/main_camera/right/image_raw", False),
        ("wrist", "/mars/arm/image_raw", False),
    ):
        subscriptions.append(
            node.create_subscription(
                CompressedImage if compressed else Image,
                topic,
                lambda msg, n=name, c=compressed: image(n, msg, c),
                qos_profile_sensor_data,
            )
        )
    deadline, last_state = time.monotonic() + args.seconds, 0
    emit("hello", mode="passive", command_publishers_created=False, robot_motion_enabled=False)
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.monotonic()
            if now - last_state < 0.1:
                continue
            last_state = now
            try:
                tf = buffer.lookup_transform("odom", "ee_link", Time())
                p, q = tf.transform.translation, tf.transform.rotation
                latest["hand"] = {
                    "position": [p.x, p.y, p.z],
                    "quaternion_xyzw": [q.x, q.y, q.z, q.w],
                    "frame_id": "odom",
                    **timing(tf.header),
                }
            except TransformException:
                latest.pop("hand", None)
            # Compute ages against the original ROS capture, never emission time.
            stamp = node.get_clock().now().nanoseconds
            state = {
                k: {**v, "source_age_s": (stamp - v["source_stamp_ns"]) / 1e9 if v["source_stamp_ns"] > 0 else None}
                for k, v in latest.items()
            }
            emit("state", samples=state, units_verified=False, motor_read_validity_verified=False)
    finally:
        del listener
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

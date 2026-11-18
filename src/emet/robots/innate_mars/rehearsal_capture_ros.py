"""Bounded standalone ROS capture for offline Herman rehearsal, subscriptions only."""

import argparse
import base64
import collections
import json
import time
import zlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=8)
    args = parser.parse_args()
    if not 3 <= args.seconds <= 20:
        parser.error("capture duration must be 3..20 seconds")
    import cv2
    import numpy as np
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState, LaserScan
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener

    rclpy.init()
    node = rclpy.create_node("emet_passive_rehearsal_capture")
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    latest, counts, subscriptions = {}, collections.Counter(), []
    arms = collections.deque(maxlen=1600)
    images = collections.defaultdict(lambda: collections.deque(maxlen=16))

    def header(msg):
        return {"stamp_ns": msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec, "frame_id": msg.header.frame_id}

    def joint(name, msg):
        counts[name] += 1
        item = {
            **header(msg),
            "names": list(msg.name),
            "position": list(msg.position),
            "velocity": list(msg.velocity),
            "effort_proxy": list(msg.effort),
        }
        latest[name] = item
        if name == "arm":
            arms.append(item)

    def odom(msg):
        counts["odom"] += 1
        p, q, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
        latest["odom"] = {
            **header(msg),
            "child_frame_id": msg.child_frame_id,
            "position": [p.x, p.y, p.z],
            "quaternion_xyzw": [q.x, q.y, q.z, q.w],
            "twist_child": [v.linear.x, v.linear.y, v.linear.z, v.angular.x, v.angular.y, v.angular.z],
        }

    def camera_info(name, msg):
        counts[name] += 1
        latest[name] = {
            **header(msg),
            "width": msg.width,
            "height": msg.height,
            "K": list(msg.k),
            "P": list(msg.p),
            "R": list(msg.r),
            "D": list(msg.d),
            "distortion_model": msg.distortion_model,
            "binning_x": msg.binning_x,
            "binning_y": msg.binning_y,
            "roi": {k: getattr(msg.roi, k) for k in ("x_offset", "y_offset", "width", "height", "do_rectify")},
        }

    def collect_image(name, msg):
        counts[name] += 1
        images[name].append(msg)

    def encode_image(name, msg):
        if isinstance(msg, CompressedImage):
            latest[name] = {**header(msg), "encoding": msg.format, "jpeg_base64": base64.b64encode(msg.data).decode()}
            return
        if not 0 < msg.height <= 4096 or not 0 < msg.width <= 4096:
            return
        if name == "depth":
            # Preserve signed/unknown depth without guessing units. The offline
            # decoder requires explicit source provenance before using it.
            latest[name] = {
                **header(msg),
                "width": msg.width,
                "height": msg.height,
                "step": msg.step,
                "encoding": msg.encoding,
                "bigendian": bool(msg.is_bigendian),
                "data_zlib_base64": base64.b64encode(zlib.compress(bytes(msg.data), 1)).decode(),
            }
        else:
            channels = {"rgb8": 3, "bgr8": 3, "mono8": 1}.get(msg.encoding)
            if channels is None:
                return
            a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * channels]
            a = a.reshape(msg.height, msg.width, channels).copy()
            if msg.encoding == "rgb8":
                a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
            elif channels == 1:
                a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
            ok, encoded = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                latest[name] = {
                    **header(msg),
                    "width": msg.width,
                    "height": msg.height,
                    "encoding": "jpeg",
                    "jpeg_base64": base64.b64encode(encoded).decode(),
                }

    def scan(msg):
        counts["scan"] += 1
        latest["scan"] = {
            **header(msg),
            "angle_min": msg.angle_min,
            "angle_increment": msg.angle_increment,
            "range_min": msg.range_min,
            "range_max": msg.range_max,
            "count": len(msg.ranges),
            "ranges_zlib_base64": base64.b64encode(
                zlib.compress(np.asarray(msg.ranges, dtype="<f4").tobytes(), 1)
            ).decode(),
        }

    def description(msg):
        counts["robot_description"] += 1
        latest["robot_description"] = msg.data

    for topic, kind, callback in (
        ("/mars/arm/state", JointState, lambda m: joint("arm", m)),
        ("/mars/arm/command_state", JointState, lambda m: joint("commanded_arm", m)),
        ("/odom", Odometry, odom),
        ("/mars/main_camera/left/camera_info", CameraInfo, lambda m: camera_info("head_left_info", m)),
        ("/mars/main_camera/right/camera_info", CameraInfo, lambda m: camera_info("head_right_info", m)),
        (
            "/mars/main_camera/left/image_rect_color/compressed",
            CompressedImage,
            lambda m: collect_image("head_left", m),
        ),
        ("/mars/main_camera/right/image_raw", Image, lambda m: collect_image("head_right", m)),
        ("/mars/arm/image_raw", Image, lambda m: collect_image("wrist", m)),
        ("/mars/main_camera/depth/image_rect_raw", Image, lambda m: collect_image("depth", m)),
        ("/scan", LaserScan, scan),
    ):
        subscriptions.append(node.create_subscription(kind, topic, callback, qos_profile_sensor_data))
    subscriptions.append(
        node.create_subscription(
            String,
            "/robot_description",
            description,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
    )
    deadline = time.monotonic() + args.seconds
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        now = node.get_clock().now().nanoseconds
        # Pair arm with interpolated TF ~150 ms in the past, not an unrelated
        # latest transform. Keep the time choice and any unavailable transform.
        valid = [a for a in arms if 0 < a["stamp_ns"] <= now - 150_000_000]
        chosen = valid[-1] if valid else latest.get("arm")
        if chosen:
            latest["arm"] = chosen
        query_ns = chosen["stamp_ns"] if chosen else 0
        for name, messages in images.items():
            selected = min(messages, key=lambda msg: abs(header(msg)["stamp_ns"] - query_ns))
            encode_image(name, selected)
        transforms = {}
        frames = ["ee_link", "head", "camera_optical_frame", "right_camera_optical_frame", "arm_camera_optical_frame"]
        if "scan" in latest:
            frames.append(latest["scan"]["frame_id"])
        if "depth" in latest:
            frames.append(latest["depth"]["frame_id"])
        for child in dict.fromkeys(frames):
            if not child:
                continue
            try:
                tf = buffer.lookup_transform("base_link", child, Time(nanoseconds=query_ns))
                p, q = tf.transform.translation, tf.transform.rotation
                transforms[child] = {
                    "parent": "base_link",
                    "child": child,
                    "query_stamp_ns": query_ns,
                    **header(tf),
                    "position": [p.x, p.y, p.z],
                    "quaternion_xyzw": [q.x, q.y, q.z, q.w],
                }
            except TransformException as exc:
                transforms[child] = {
                    "error": str(exc)[:300],
                    "parent": "base_link",
                    "child": child,
                    "query_stamp_ns": query_ns,
                }
        image_transforms = {}
        for name in images:
            sample = latest.get(name)
            if not sample:
                continue
            try:
                tf = buffer.lookup_transform("base_link", sample["frame_id"], Time(nanoseconds=sample["stamp_ns"]))
                p, q = tf.transform.translation, tf.transform.rotation
                image_transforms[name] = {
                    "parent": "base_link",
                    "child": sample["frame_id"],
                    **header(tf),
                    "query_stamp_ns": sample["stamp_ns"],
                    "position": [p.x, p.y, p.z],
                    "quaternion_xyzw": [q.x, q.y, q.z, q.w],
                }
            except TransformException as exc:
                image_transforms[name] = {"error": str(exc)[:300]}
        result = {
            "format": "herman-rehearsal-capture-v1",
            "captured_ros_ns": now,
            "capture_duration_s": args.seconds,
            "counts": dict(counts),
            "samples": latest,
            "transforms": transforms,
            "image_transforms": image_transforms,
            "arm_position_history": [{k: a[k] for k in ("stamp_ns", "names", "position")} for a in list(arms)[::20]],
            "hardware_motion_commands_sent": 0,
            "actuator_publishers_created": False,
            "velocity_units_verified": False,
            "motor_read_validity_verified": False,
        }
        print("EMET_REHEARSAL " + json.dumps(result, allow_nan=False, separators=(",", ":")), flush=True)
    finally:
        del listener
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

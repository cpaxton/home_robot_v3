"""Bounded named-joint motion over the vendor Mars arm driver."""

import math
import time
import xml.etree.ElementTree as ET

import numpy as np

# Maurice description limits, in joint1..joint6 order, in radians.
JOINT_LIMITS = {
    "joint1": (-1.5708, 1.5708),
    "joint2": (-1.5708, 1.22),
    "joint3": (-1.5708, 1.7453),
    "joint4": (-1.9199, 1.7453),
    "joint5": (-1.5708, 1.5708),
    "joint6": (-0.8727, 0.3491),
}
MAX_SPEED = 0.15
POSITION_TOLERANCE = 0.025
VELOCITY_TOLERANCE = 0.04
SETTLE_SECONDS = 0.25
ENCODER_STEP_RAD = math.tau / 4096


def outside_joint_limits(positions, limits=JOINT_LIMITS):
    return [name for name, (lower, upper) in limits.items() if not lower <= positions[name] <= upper]


def published_arm_limits(description):
    """Return a complete named arm range from the vendor's robot description."""
    try:
        limits = {}
        for joint in ET.fromstring(description).findall("joint"):
            name = joint.get("name")
            if name not in JOINT_LIMITS:
                continue
            limit = joint.find("limit")
            if limit is None or name in limits:
                return None
            lower, upper = (float(limit.attrib[key]) for key in ("lower", "upper"))
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                return None
            limits[name] = (lower, upper)
    except (ET.ParseError, KeyError, ValueError):
        return None
    return limits if limits.keys() == JOINT_LIMITS.keys() else None


def startup_joint_positions(names, values):
    if len(names) != len(values) or len(set(names)) != len(names):
        return None
    by_name = dict(zip(names, values, strict=True))
    if any(name not in by_name or not math.isfinite(by_name[name]) for name in JOINT_LIMITS):
        return None
    return {name: float(by_name[name]) for name in JOINT_LIMITS}


class MarsArmMotion:
    def __init__(self, ros):
        self.ros = ros
        self.target = self.command = None

    @staticmethod
    def validate(goal, *, limits=JOINT_LIMITS):
        if not isinstance(goal, dict) or set(goal) - {"arm", "frame"} or goal.get("frame", "world") != "world":
            raise ValueError("Mars arm motion requires a standalone named arm goal")
        arm = goal.get("arm")
        if not isinstance(arm, dict) or not arm or set(arm) - JOINT_LIMITS.keys():
            raise ValueError("Mars arm targets must name joints joint1 through joint6")
        for name, value in arm.items():
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not (limits[name][0] <= value <= limits[name][1])
            ):
                raise ValueError(f"Mars arm target outside limits: {name}")

    def start(self, goal):
        self.ros.verify_arm_command_path()
        q, dq = self.ros.arm_motion_feedback()
        if np.max(np.abs(dq)) > VELOCITY_TOLERANCE:
            raise RuntimeError("Mars arm must be stationary before accepting a goal")
        target = np.array([goal["arm"].get(name, value) for name, value in zip(JOINT_LIMITS, q, strict=True)])
        limits = self.ros.get_published_arm_limits()
        if limits is None:
            raise RuntimeError("Mars arm joint limits from robot_description are unavailable")
        outside = outside_joint_limits(dict(zip(JOINT_LIMITS, q, strict=True)), limits)
        if outside:
            raise RuntimeError(f"Measured Mars joint outside published limits: {outside[0]}")
        outside = outside_joint_limits(dict(zip(JOINT_LIMITS, target, strict=True)), limits)
        if outside:
            raise ValueError(f"Mars arm target outside published limits: {outside[0]}")
        self.command, self.target = q, target
        self.started_at = self.last_step = time.monotonic()
        self.settling_since = None
        return {"arm": goal["arm"]}

    def poll(self):
        now = time.monotonic()
        elapsed = now - self.last_step
        if elapsed < 0.02:
            return None
        self.ros.verify_arm_command_path()
        q, dq = self.ros.arm_motion_feedback()
        if np.max(np.abs(self.command - q)) > 0.09:
            raise RuntimeError("Mars arm tracking error exceeded 0.09 rad")
        step = MAX_SPEED * min(elapsed, 0.05)
        self.command += np.clip(self.target - self.command, -step, step)
        self.ros.publish_arm_targets(self.command)
        self.last_step = now
        settled = np.array_equal(self.command, self.target) and (
            np.max(np.abs(q - self.target)) <= POSITION_TOLERANCE and np.max(np.abs(dq)) <= VELOCITY_TOLERANCE
        )
        if not settled or not self.ros.arm_target_applied(self.target, self.started_at):
            self.settling_since = None
        elif self.settling_since is None:
            self.settling_since = now
        elif now - self.settling_since >= SETTLE_SECONDS:
            return "succeeded", {"measured_arm": dict(zip(JOINT_LIMITS, q.tolist(), strict=True))}
        return None

    def stop(self):
        if self.command is None:
            return True
        self.ros.verify_arm_command_path()
        hold, _ = self.ros.arm_motion_feedback(require_health=False)
        started_at = time.monotonic()
        # Reasserting a hold that already matches the applied encoder target
        # need not generate a second echo. Use the receipt from this command.
        echo_since = (
            self.started_at if self.ros.arm_target_applied(hold, self.started_at, tolerance=0.002) else started_at
        )
        deadline = started_at + 0.75
        settling_since = None
        while time.monotonic() < deadline:
            self.ros.publish_arm_targets(hold)
            time.sleep(0.02)
            q, dq = self.ros.arm_motion_feedback(require_health=False)
            if (
                np.max(np.abs(q - hold)) > POSITION_TOLERANCE
                or np.max(np.abs(dq)) > VELOCITY_TOLERANCE
                or not self.ros.arm_target_applied(hold, echo_since)
            ):
                settling_since = None
            elif settling_since is None:
                settling_since = time.monotonic()
            elif time.monotonic() - settling_since >= SETTLE_SECONDS:
                self.ros.verify_arm_command_path()
                self.target = self.command = None
                return True
        return False


class MarsJointRecovery:
    """Run a pose-bound, single-joint request on an isolated driver endpoint."""

    def __init__(self, ros, driver, *, native=False):
        self.ros = ros
        self.driver = driver
        self.pending = None
        self.native = native

    @staticmethod
    def validate(goal, *, limits=JOINT_LIMITS):
        if not isinstance(goal, dict) or set(goal) - {"arm_recovery", "frame"} or goal.get("frame", "world") != "world":
            raise ValueError("Mars joint recovery requires a standalone goal")
        recovery = goal.get("arm_recovery")
        required = {"joint", "target_rad", "start_arm_rad"}
        if (
            not isinstance(recovery, dict)
            or not required <= recovery.keys() <= required | {"stage_only"}
            or type(recovery.get("stage_only", False)) is not bool
        ):
            raise ValueError("Mars joint recovery requires a joint, target, and observed start")
        joint, target, start = (recovery[key] for key in ("joint", "target_rad", "start_arm_rad"))
        if not isinstance(joint, str) or joint not in JOINT_LIMITS:
            raise ValueError("Unknown Mars recovery joint")
        if type(target) not in (int, float) or not limits[joint][0] <= target <= limits[joint][1]:
            raise ValueError("Mars recovery target must be inside the named joint limits")
        if (
            not isinstance(start, dict)
            or set(start) != JOINT_LIMITS.keys()
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in start.values())
        ):
            raise ValueError("Mars recovery requires six finite starting positions")

    def start(self, goal):
        if not self.driver.ready():
            raise RuntimeError("Mars driver has no isolated joint recovery endpoint")
        self.ros.verify_arm_recovery_path(native=self.native)
        q, dq = self.ros.arm_motion_feedback()
        if np.max(np.abs(dq)) > VELOCITY_TOLERANCE:
            raise RuntimeError("Mars arm must be stationary before recovery")
        limits = self.ros.get_published_arm_limits()
        if limits is None:
            raise RuntimeError("Mars arm joint limits from robot_description are unavailable")
        self.validate(goal, limits=limits)
        recovery = goal["arm_recovery"]
        joint = recovery["joint"]
        index = tuple(JOINT_LIMITS).index(joint)
        expected = np.array([recovery["start_arm_rad"][name] for name in JOINT_LIMITS])
        if np.max(np.abs(q - expected)) > 0.005:
            raise RuntimeError("Mars arm has moved since the recovery start was observed")
        measured = dict(zip(JOINT_LIMITS, q, strict=True))
        outside = outside_joint_limits(measured, limits)
        target = float(recovery["target_rad"])
        lower, upper = limits[joint]
        if recovery.get("stage_only", False):
            # The native service is bound to one reviewed start and target.
            if joint in outside or (not outside and not self.native):
                raise RuntimeError("Recovery staging requires an in-range joint and another joint outside limits")
        elif joint not in outside:
            # A fresh sample can land one encoder count inward of the reviewed
            # overrun. The same recovery is still inward from that sample.
            near_reviewed_overrun = abs(q[index] - expected[index]) <= ENCODER_STEP_RAD + 1e-9
            inward = expected[index] < lower and target > q[index] or expected[index] > upper and target < q[index]
            if not (near_reviewed_overrun and inward):
                raise RuntimeError("Mars recovery joint is already inside limits")
        self.start_pose = q.copy()
        self.index = index
        self.target = target
        self.target_limits = limits[joint]
        self.motion_limits = np.asarray([limits[name] for name in JOINT_LIMITS])
        self.other = np.arange(len(q)) != index
        self.settling_since = None
        driver_options = {"start_arm": expected.tolist()} if self.native else {}
        self.pending = self.driver.start(joint, float(q[index]), target, abs(target - q[index]) / MAX_SPEED, **driver_options)
        return {"arm_recovery": {"joint": joint, "target_rad": target}}

    def poll(self):
        self.ros.verify_arm_recovery_path(native=self.native)
        q, dq = self.ros.arm_motion_feedback()
        tolerance = ENCODER_STEP_RAD + 1e-9 if self.native else POSITION_TOLERANCE
        if np.max(np.abs(q[self.other] - self.start_pose[self.other])) > tolerance:
            raise RuntimeError("An unselected Mars joint moved during isolated recovery")
        low, high = sorted((self.start_pose[self.index], self.target))
        if not low - POSITION_TOLERANCE <= q[self.index] <= high + POSITION_TOLERANCE:
            raise RuntimeError("Mars joint left the isolated recovery interval")
        if self.native:
            lower, upper = self.motion_limits.T
            initial_overrun = np.maximum(0, np.maximum(lower - self.start_pose, self.start_pose - upper))
            overrun = np.maximum(0, np.maximum(lower - q, q - upper))
            if np.any(overrun > initial_overrun + 1e-9):
                raise RuntimeError("Mars joint moved farther beyond the published limit")
        outcome = self.pending.outcome()
        if outcome is None:
            return None
        if outcome != "succeeded":
            return "failed", {"driver_recovery": outcome}
        within = self.target_bounds_hold(q, dq)
        if self.native:
            now = time.monotonic()
            if not within:
                self.settling_since = None
                return None
            if self.settling_since is None:
                self.settling_since = now
            if now - self.settling_since < SETTLE_SECONDS:
                return None
        return ("succeeded" if within else "failed"), {"measured_arm": dict(zip(JOINT_LIMITS, q.tolist(), strict=True))}

    def target_bounds_hold(self, q, dq):
        lower, upper = self.target_limits
        return (
            lower <= q[self.index] <= upper
            and abs(q[self.index] - self.target) <= (2 * ENCODER_STEP_RAD if self.native else POSITION_TOLERANCE)
            and np.max(np.abs(dq)) <= VELOCITY_TOLERANCE
        )

    def stop(self):
        if self.pending is None:
            return True
        self.pending.cancel()
        deadline = time.monotonic() + 0.75
        settling_since = None
        while time.monotonic() < deadline:
            terminal = self.pending.outcome()
            if terminal == "stop_unconfirmed":
                return False
            if terminal == "rejected":
                self.pending = None
                return True
            if terminal in {"succeeded", "cancelled", "aborted", "failed"}:
                _, dq = self.ros.arm_motion_feedback(require_health=False)
                if np.max(np.abs(dq)) > VELOCITY_TOLERANCE:
                    settling_since = None
                elif settling_since is None:
                    settling_since = time.monotonic()
                elif time.monotonic() - settling_since >= SETTLE_SECONDS:
                    self.pending = None
                    return True
            time.sleep(0.02)
        return False


def main(*, release_native=False):
    """Report arm and command ownership readiness without creating a publisher."""
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    from innate_mars_bridge.constants import ARM_STATE_TOPIC
    from innate_mars_bridge.remote.native_arm import NativeArmDriver
    from innate_mars_bridge.remote.ros import ARM_COMMAND_TOPIC

    positions = None
    limits = None

    def receive(msg):
        nonlocal positions
        positions = startup_joint_positions(msg.name, msg.position)

    def receive_description(msg):
        nonlocal limits
        limits = published_arm_limits(msg.data)

    rclpy.init()
    node = rclpy.create_node("emet_mars_arm_startup_check")
    try:
        driver = NativeArmDriver(node)
        if release_native:
            if driver.trajectory is None:
                return 0
            present = driver.release.wait_for_service(timeout_sec=1.0)
            if not present:
                if driver.trajectory.service_is_ready():
                    print("Mars native arm release unavailable.")
                    return 1
                return 0
            future = driver.release.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)
            if not future.done() or not future.result().success:
                print("Mars native arm hold and release unconfirmed.")
                return 1
            print("Mars native arm hold and release confirmed.")
            return 0
        node.create_subscription(JointState, ARM_STATE_TOPIC, receive, qos_profile_sensor_data)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String, "/robot_description", receive_description, qos)
        deadline = time.monotonic() + 5
        while (positions is None or limits is None) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        unavailable = positions is None or limits is None
        if unavailable:
            print("Mars arm position readiness unavailable: incomplete joint state or robot_description.")
            outside = []
        else:
            outside = outside_joint_limits(positions, limits)
            for name in outside:
                lower, upper = limits[name]
                print(
                    f"Mars arm range blocker: {name} = {positions[name]:+.5f} rad (published {lower:+.5f} to {upper:+.5f})."
                )
            if not outside:
                print("Mars arm position: all six joints are within published limits.")
        publishers = [info.node_name for info in node.get_publishers_info_by_topic(ARM_COMMAND_TOPIC)]
        subscribers = node.get_subscriptions_info_by_topic(ARM_COMMAND_TOPIC)
        exclusive = publishers == ["innate_mars_zmq_server"] and any(
            info.node_name in {"maurice_arm", "mars_arm"} for info in subscribers
        )
        if not unavailable and driver.ready():
            print("Mars native recovery session available. An explicit reviewed goal is required.")
            return 3
        if not exclusive:
            print("Mars arm ownership blocker: " + (", ".join(sorted(publishers)) or "no command publishers"))
        if not unavailable and not outside and exclusive:
            print("Mars arm preflight: position and command ownership checks passed.")
        return 1 if unavailable else 2 if outside or not exclusive else 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the patched Mars ROS arm node against an emulated Dynamixel bus."""

import argparse
import errno
import math
import os
import pty
import select
import struct
import subprocess
import threading
import time
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.action.graph import get_action_server_names_and_types_by_node
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

POSITION = 132
GOAL = 116
PROFILE = 112
SELECTED = 3


def crc(data):
    value = 0
    for octet in data:
        value ^= octet << 8
        for _ in range(8):
            value = ((value << 1) ^ (0x8005 if value & 0x8000 else 0)) & 0xFFFF
    return value


def status(servo, data=b"", error=0):
    payload = bytes((0x55, error)) + data
    packet = b"\xff\xff\xfd\x00" + bytes((servo,)) + struct.pack("<H", len(payload) + 2) + payload
    return packet + struct.pack("<H", crc(packet))


class Bus:
    def __init__(self):
        self.start = [3026, 3117, 1334, 1436, 2050, 1875, 2048]
        self.master, self.slave = pty.openpty()
        self.name = os.ttyname(self.slave)
        self.active = True
        self.mode = "normal"
        self.writes = []
        self.goals = []
        self.lock = threading.Lock()
        self.group_writes = 0
        self.reset()
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def reset(self):
        with self.lock:
            self.reset_registers()

    def reset_registers(self):
        self.reg = {servo: bytearray(256) for servo in range(1, 8)}
        for servo, position in enumerate(self.start, 1):
            registers = self.reg[servo]
            registers[11] = 3
            registers[64] = 1
            registers[146] = 25
            for addr, value in ((48, 4095), (52, 0), (POSITION, position), (GOAL, position), (PROFILE, 40)):
                struct.pack_into("<I", registers, addr, value)
        self.writes.clear()
        self.goals.clear()

    def serve(self):
        buf = bytearray()
        while self.active:
            if not select.select([self.master], [], [], 0.05)[0]:
                continue
            try:
                buf.extend(os.read(self.master, 65536))
            except OSError as exc:
                if exc.errno == errno.EIO:
                    return
                raise
            while len(buf) >= 7:
                start = buf.find(b"\xff\xff\xfd\x00")
                if start == -1:
                    del buf[:-3]
                    break
                if start:
                    del buf[:start]
                if len(buf) < 7:
                    break
                size = struct.unpack_from("<H", buf, 5)[0] + 7
                if len(buf) < size:
                    break
                packet = bytes(buf[:size])
                del buf[:size]
                if crc(packet[:-2]) != struct.unpack_from("<H", packet, size - 2)[0]:
                    continue
                with self.lock:
                    self.handle(packet[4], packet[7], packet[8:-2])

    def handle(self, servo, instruction, args):
        if instruction == 0x82:
            address, length = struct.unpack_from("<HH", args)
            os.write(self.master, b"".join(status(i, self.reg[i][address : address + length]) for i in args[4:]))
        elif instruction == 2:
            address, length = struct.unpack("<HH", args)
            os.write(self.master, status(servo, self.reg[servo][address : address + length]))
        elif instruction == 3:
            address = struct.unpack_from("<H", args)[0]
            payload = args[2:]
            self.writes.append((servo, address))
            if address == GOAL:
                target = struct.unpack("<I", payload)[0]
                lower = struct.unpack_from("<I", self.reg[servo], 52)[0]
                upper = struct.unpack_from("<I", self.reg[servo], 48)[0]
                if self.mode == "write_error" or not lower <= target <= upper:
                    os.write(self.master, status(servo, error=4))
                    return
                self.goals.append((servo, target))
                if self.mode != "stuck":
                    self.reg[servo][POSITION : POSITION + 4] = payload
            self.reg[servo][address : address + len(payload)] = payload
            os.write(self.master, status(servo))
        elif instruction == 0x83:
            self.group_writes += 1

    def shutdown(self):
        self.active = False
        self.thread.join()
        os.close(self.master)
        os.close(self.slave)


def spin_until(node, pred, timeout=8.0):
    deadline = time.monotonic() + timeout
    while not pred() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    assert pred(), "timed out waiting for the ROS action"


def goal(target, start):
    request = FollowJointTrajectory.Goal()
    request.trajectory.joint_names = ["joint3"]
    request.trajectory.points = [
        JointTrajectoryPoint(positions=[start]),
        JointTrajectoryPoint(positions=[target], time_from_start=Duration(sec=2)),
    ]
    return request


def test_native_session(node, bus, start, target):
    from mars_msgs.srv import GotoJS

    prefix = "/emet/mars/arm/"
    services = {
        name: node.create_client(service, prefix + name)
        for name, service in (("acquire", Trigger), ("release", Trigger), ("goto_js_v2", GotoJS))
    }
    assert all(client.wait_for_service(timeout_sec=4) for client in services.values())

    def call(name, request):
        future = services[name].call_async(request)
        spin_until(node, future.done)
        return future.result().success

    request = GotoJS.Request()
    request.data.data = start.copy()
    request.data.data[2] = target
    request.time = 1.0
    assert not call("goto_js_v2", request)
    assert not bus.writes and not bus.group_writes
    assert call("acquire", Trigger.Request())
    assert not call("acquire", Trigger.Request())
    request.data.data[0] += 0.1
    assert not call("goto_js_v2", request)
    assert not bus.writes and not bus.group_writes
    request.data.data[0] = start[0]
    moving = services["goto_js_v2"].call_async(request)
    spin_until(node, lambda: len(bus.goals) >= 2)
    assert call("release", Trigger.Request())
    spin_until(node, moving.done)
    assert not moving.result().success
    held = bus.goals.copy()
    time.sleep(0.1)
    assert bus.goals == held
    assert all(servo == SELECTED for servo, _ in held)

    bus.reset()
    assert call("acquire", Trigger.Request())
    assert call("goto_js_v2", request)
    assert call("release", Trigger.Request())
    assert bus.goals and all(servo == SELECTED for servo, _ in bus.goals)
    encoders = [value for _, value in bus.goals]
    assert encoders == sorted(encoders)
    assert bus.start[SELECTED - 1] <= encoders[0] < encoders[-1] == 1031
    assert set(bus.writes) == {(SELECTED, GOAL), (SELECTED, PROFILE)}
    assert bus.group_writes == 0, "adopted staging must not rewrite gains or other servo settings"
    print("native adoption: read-only startup, exclusive lease, inward elbow stroke and verified hold", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["ROS_DOMAIN_ID"] = str(40 + os.getpid() % 60)
    bus = Bus()
    if args.native:
        # Herman's captured Innate 0.8.0 fold in raw servo order.
        bus.start = [3072, 2842, 1023, 2244, 2045, 1875, 2048]
        bus.reset()
    start = [
        (encoder - 2048) * 2 * math.pi / 4096 * (-1 if i in (1, 2, 3, 5) else 1)
        for i, encoder in enumerate(bus.start[:6])
    ]
    target = 1.56 if args.native else 0.95
    prefix = "adopt_recovery" if args.native else "isolated_recovery"
    mode = ["-p", "adopt_existing_holds:=true", "-p", "auto_rest:=false"] if args.native else []
    log = args.driver_log.open("w")
    proc = subprocess.Popen(
        [
            str(args.arm_binary.resolve()),
            "--ros-args",
            "--params-file",
            str(args.config),
            "-p",
            f"{prefix}_joint:=joint3",
            "-p",
            f"isolated_recovery_device:={bus.name}",
            "-p",
            f"{prefix}_start_rad:={start}",
            "-p",
            f"{prefix}_target_rad:={target}",
            *mode,
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    rclpy.init()
    node = rclpy.create_node("emet_recovery_smoke")
    node.create_publisher(Float64MultiArray, "/mars/arm/commands", 1)
    driver = ActionClient(node, FollowJointTrajectory, "/mars/arm/recover_joint")
    try:
        if args.native:
            test_native_session(node, bus, start, target)
            return
        assert driver.wait_for_server(timeout_sec=8)
        spin_until(
            node,
            lambda: any(
                endpoint.node_name == "mars_arm" and endpoint.node_namespace == "/"
                for endpoint in node.get_publishers_info_by_topic("/mars/arm/state")
            ),
        )
        arm_state, = node.get_publishers_info_by_topic("/mars/arm/state")
        actions = dict(get_action_server_names_and_types_by_node(node, arm_state.node_name, arm_state.node_namespace))
        assert actions["/mars/arm/recover_joint"] == ["control_msgs/action/FollowJointTrajectory"]
        assert node.count_subscribers("/mars/arm/commands") == 0

        def send(target_rad, start_rad=start[2]):
            handle = driver.send_goal_async(goal(target_rad, start_rad))
            spin_until(node, handle.done)
            return handle.result()

        rejected = send(target + 0.02)
        assert not rejected.accepted and not bus.writes

        ok = send(target)
        assert ok.accepted
        result = ok.get_result_async()
        spin_until(node, result.done)
        assert result.result().status == GoalStatus.STATUS_SUCCEEDED, result.result()
        assert len(bus.writes) > 2 and set(bus.writes) == {(SELECTED, GOAL), (SELECTED, PROFILE)}
        print(f"approved goal: {len(bus.writes)} writes to servo {SELECTED} only", flush=True)

        bus.reset()
        cancelling = send(target)
        result = cancelling.get_result_async()
        spin_until(node, lambda: len(bus.writes) > 4)
        cancellation = cancelling.cancel_goal_async()
        spin_until(node, cancellation.done)
        spin_until(node, result.done)
        assert result.result().status == GoalStatus.STATUS_CANCELED, result.result()
        assert set(bus.writes) == {(SELECTED, GOAL), (SELECTED, PROFILE)}
        print("cancel: selected servo held and action confirmed cancellation", flush=True)

        bus.reset()
        boundary = bus.start[SELECTED - 1] + 1
        with bus.lock:
            struct.pack_into("<I", bus.reg[SELECTED], 52, boundary)
        one_count = send(target)
        result = one_count.get_result_async()
        spin_until(node, result.done)
        assert result.result().status == GoalStatus.STATUS_SUCCEEDED, result.result()
        assert bus.goals[0] == (SELECTED, boundary)
        assert all(servo == SELECTED and encoder >= boundary for servo, encoder in bus.goals)
        print("one-count overrun: first write was the legal boundary, then selected servo only", flush=True)

        bus.reset()
        # Pick a native lower bound above the measured encoder. The driver
        # must decline even this configured path before touching a register.
        with bus.lock:
            struct.pack_into("<I", bus.reg[SELECTED], 52, boundary + 1)
        invalid = send(target)
        result = invalid.get_result_async()
        spin_until(node, result.done)
        assert result.result().status == GoalStatus.STATUS_ABORTED and not bus.writes, result.result()
        print("outside native range: zero writes", flush=True)

        bus.reset()
        bus.mode = "stuck"
        stuck = send(target)
        result = stuck.get_result_async()
        spin_until(node, result.done)
        assert result.result().status == GoalStatus.STATUS_ABORTED, result.result()
        assert bus.goals[-1] == (SELECTED, bus.start[SELECTED - 1])
        assert set(bus.writes) == {(SELECTED, GOAL), (SELECTED, PROFILE)}
        print("stuck motor: tracking guard halted with a measured hold", flush=True)

        bus.reset()
        bus.mode = "write_error"
        error = send(target)
        result = error.get_result_async()
        spin_until(node, result.done)
        assert result.result().result.error_code == -100, result.result()
        assert not send(target).accepted
        assert bus.group_writes == 0
        print("servo rejection: stop unconfirmed, further recovery latched off, zero group writes", flush=True)
    finally:
        driver.destroy()
        node.destroy_node()
        rclpy.shutdown()
        proc.terminate()
        proc.wait(timeout=5)
        bus.shutdown()
        log.close()


if __name__ == "__main__":
    main()

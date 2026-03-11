#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Slim ZMQ viewer: connect to a bridge (Innate Mars or Stretch) and display images + state. No Stretch-specific logic."""

import time

import click
import cv2
import numpy as np
import zmq

from emet.utils.connection import get_host_from_connection
from emet.utils.memory import lookup_address


def _decode_jpg(buf: bytes) -> np.ndarray | None:
    if not buf:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _robot_host(robot_ip: str) -> str:
    """Resolve robot host for ZMQ (no tcp://). Use connection or robot_ip.txt when empty."""
    if robot_ip:
        return robot_ip.strip()
    addr = lookup_address("", use_remote_computer=True, update=False)
    if addr and addr.startswith("tcp://"):
        return addr[6:].strip()
    host = get_host_from_connection()
    return host or "127.0.0.1"


@click.command()
@click.option(
    "--robot-ip",
    "--robot_ip",
    "robot_ip",
    default="",
    help="Robot IP or host (default: active connection or ~/.stretch/robot_ip.txt)",
)
@click.option("--recv-port", default=4404, help="Servo port (default 4404)")
@click.option("--state-port", default=4403, help="State port (default 4403)")
def main(robot_ip: str, recv_port: int, state_port: int) -> None:
    """View images and state from a robot bridge (Innate Mars or Stretch).

    Connects to the ZMQ servo port (4404 by default) and displays head_cam_left,
    head_cam_right, ee_cam (and head_cam if present). Prints step and joint state.
    Press 'q' in a window to quit.

    Examples:
      emet view-bridge
      emet view-bridge --robot-ip 192.168.1.43
    """
    host = _robot_host(robot_ip)
    recv_addr = f"tcp://{host}:{recv_port}"
    state_addr = f"tcp://{host}:{state_port}"

    ctx = zmq.Context()
    recv_socket = ctx.socket(zmq.SUB)
    recv_socket.setsockopt(zmq.SUBSCRIBE, b"")
    recv_socket.setsockopt(zmq.CONFLATE, 1)
    recv_socket.connect(recv_addr)

    state_socket = ctx.socket(zmq.SUB)
    state_socket.setsockopt(zmq.SUBSCRIBE, b"")
    state_socket.setsockopt(zmq.CONFLATE, 1)
    state_socket.connect(state_addr)

    print(f"Connected to {recv_addr} (servo) and {state_addr} (state). Press 'q' in a window to quit.")
    poller = zmq.Poller()
    poller.register(recv_socket, zmq.POLLIN)
    poller.register(state_socket, zmq.POLLIN)

    last_state = None
    last_servo = None
    step_printed = -1

    while True:
        socks = dict(poller.poll(timeout=100))
        if recv_socket in socks:
            try:
                last_servo = recv_socket.recv_pyobj()
            except Exception:
                pass
        if state_socket in socks:
            try:
                last_state = state_socket.recv_pyobj()
            except Exception:
                pass

        if last_servo is None:
            time.sleep(0.02)
            continue

        # Decode and show images (Innate Mars: head_cam_left, head_cam_right, ee_cam; Stretch: head_cam, ee_cam)
        def show(key: str, title: str) -> None:
            if key not in last_servo:
                return
            buf = last_servo[key]
            img = _decode_jpg(buf)
            if img is not None:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow(title, bgr)

        show("head_cam_left/color_image", "head_cam_left")
        show("head_cam_right/color_image", "head_cam_right")
        show("ee_cam/color_image", "ee_cam")
        show("head_cam/color_image", "head_cam")

        step = last_servo.get("step", -1)
        if step != step_printed and (step >= 0 or last_state):
            step_printed = step
            if last_state:
                base = last_state.get("base_pose", "")
                joints = last_state.get("joint_positions", [])
                jpre = str(joints)[:60] + "..." if len(str(joints)) > 60 else str(joints)
                print(f"step={step} base_pose={base} joint_positions={jpre}")
            else:
                cfg = last_servo.get("robot/config", [])
                print(f"step={step} robot/config len={len(cfg)}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    recv_socket.close()
    state_socket.close()
    ctx.term()


if __name__ == "__main__":
    main()

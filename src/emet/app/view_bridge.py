#!/usr/bin/env python3
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

"""Slim ZMQ viewer: connect to a bridge (Innate Mars or Stretch) and display images + state. No Stretch-specific logic."""

import time
from typing import Any

import click
import cv2
import numpy as np
import zmq

from emet.utils.connection import get_host_from_connection
from emet.utils.memory import lookup_address

# (servo JPEG key, full-obs slim key, legacy alias, window title)
_VIEW_IMAGE_KEYS: tuple[tuple[str, ...], str] = (
    (("head_cam_left/color_image", "head_cam_left/image", "rgb"), "head_cam_left"),
    (("head_cam_right/color_image", "head_cam_right/image", "rgb_right"), "head_cam_right"),
    (("ee_cam/color_image", "ee_cam/image", "rgb_tertiary"), "ee_cam"),
    (("head_cam/color_image", "head_cam/image",), "head_cam"),
)


def _decode_jpg(buf: bytes) -> np.ndarray | None:
    if not buf:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _decode_image(val: Any) -> np.ndarray | None:
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        arr = np.asarray(val)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            return np.ascontiguousarray(arr[..., :3])
        if arr.ndim == 1 or (arr.ndim == 2 and arr.size > 0):
            return _decode_jpg(arr.tobytes() if arr.ndim == 1 else bytes(arr))
        return None
    if isinstance(val, (bytes, bytearray)):
        return _decode_jpg(bytes(val))
    return None


def _find_image_blob(*messages: dict[str, Any] | None, keys: tuple[str, ...]) -> Any:
    for msg in messages:
        if not msg:
            continue
        for key in keys:
            blob = msg.get(key)
            if blob is not None:
                return blob
    return None


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
@click.option("--obs-port", default=4401, show_default=True, help="Full observation port (Mars JPEGs when servo is pose-only)")
@click.option("--recv-port", default=4404, show_default=True, help="Servo port")
@click.option("--state-port", default=4403, show_default=True, help="State port")
def main(robot_ip: str, obs_port: int, recv_port: int, state_port: int) -> None:
    """View images and state from a robot bridge (Innate Mars or Stretch).

    Subscribes to full obs (4401) and servo (4404). Default Mars launch puts JPEGs on
    4401 with pose-only 4404; ``--metadata-only-obs`` inverts that (images on 4404).
    Press 'q' in a window to quit.

    Examples:
      emet view-bridge
      emet view-bridge --robot-ip 192.168.1.43
    """
    host = _robot_host(robot_ip)
    obs_addr = f"tcp://{host}:{obs_port}"
    recv_addr = f"tcp://{host}:{recv_port}"
    state_addr = f"tcp://{host}:{state_port}"

    ctx = zmq.Context()
    obs_socket = ctx.socket(zmq.SUB)
    obs_socket.setsockopt(zmq.SUBSCRIBE, b"")
    obs_socket.setsockopt(zmq.CONFLATE, 1)
    obs_socket.connect(obs_addr)

    recv_socket = ctx.socket(zmq.SUB)
    recv_socket.setsockopt(zmq.SUBSCRIBE, b"")
    recv_socket.setsockopt(zmq.CONFLATE, 1)
    recv_socket.connect(recv_addr)

    state_socket = ctx.socket(zmq.SUB)
    state_socket.setsockopt(zmq.SUBSCRIBE, b"")
    state_socket.setsockopt(zmq.CONFLATE, 1)
    state_socket.connect(state_addr)

    print(
        f"Connected to {obs_addr} (obs), {recv_addr} (servo), {state_addr} (state). "
        "Press 'q' in a window to quit."
    )
    poller = zmq.Poller()
    poller.register(obs_socket, zmq.POLLIN)
    poller.register(recv_socket, zmq.POLLIN)
    poller.register(state_socket, zmq.POLLIN)

    last_obs = None
    last_state = None
    last_servo = None
    step_printed = -1

    while True:
        socks = dict(poller.poll(timeout=100))
        if obs_socket in socks:
            try:
                last_obs = obs_socket.recv_pyobj()
            except Exception:
                pass
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

        if last_obs is None and last_servo is None:
            time.sleep(0.02)
            continue

        for keys, title in _VIEW_IMAGE_KEYS:
            blob = _find_image_blob(last_obs, last_servo, keys=keys)
            img = _decode_image(blob)
            if img is not None:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow(title, bgr)

        step_src = last_obs if last_obs is not None else last_servo
        step = step_src.get("step", -1) if step_src else -1
        if step != step_printed and (step >= 0 or last_state):
            step_printed = step
            if last_state:
                base = last_state.get("base_pose", "")
                joints = last_state.get("joint_positions", [])
                jpre = str(joints)[:60] + "..." if len(str(joints)) > 60 else str(joints)
                print(f"step={step} base_pose={base} joint_positions={jpre}")
            elif last_servo:
                cfg = last_servo.get("robot/config", [])
                print(f"step={step} robot/config len={len(cfg)}")
            else:
                print(f"step={step}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    obs_socket.close()
    recv_socket.close()
    state_socket.close()
    ctx.term()


if __name__ == "__main__":
    main()

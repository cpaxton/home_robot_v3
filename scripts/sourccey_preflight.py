#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Check the real Sourccey host's observation stream without opening a command socket.

Run in the vendor's Python 3.12 environment with lerobot-robot-sourccey installed.
Stop other clients first: the host PUSH socket splits observations among receivers.
This validates packet contents, not servo calibration, odometry, or motion readiness.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

CAMERAS = ("front_left", "front_right", "bottom", "wrist_left", "wrist_right")
ROLES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
STATE_KEYS = tuple(f"{side}_{role}.pos" for side in ("left", "right") for role in ROLES) + (
    "z.pos",
    "x.vel",
    "y.vel",
    "theta.vel",
)


def validate_observation(observation: dict) -> dict:
    errors = []
    for name in STATE_KEYS:
        value = observation.get(name)
        if not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
            errors.append(f"missing or non-finite state: {name}")
    cameras = {}
    for name in CAMERAS:
        frame = observation.get(name)
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            errors.append(f"missing or invalid camera: {name}")
        elif not np.isfinite(frame).all() or float(frame.std()) < 1.5:
            errors.append(f"blank or invalid camera: {name}")
        else:
            cameras[name] = list(frame.shape)
    if errors:
        raise ValueError("; ".join(errors))
    return {"camera_shapes": cameras, "state": {key: float(observation[key]) for key in STATE_KEYS}}


def decode_packet(payload, message_type, converter, protocol_version):
    packet = message_type()
    packet.ParseFromString(payload)
    if packet.protocol_version != protocol_version:
        raise ValueError(f"Host protocol {packet.protocol_version}; installed client requires {protocol_version}")
    # Proto3 absent submessages otherwise silently decode as all-zero state.
    for field in ("left_arm_joints", "right_arm_joints", "base_position", "base_velocity"):
        if not packet.HasField(field):
            raise ValueError(f"Host omitted {field}")
    return converter.protobuf_to_observation(packet)


def check_host(ip: str, *, port: int = 5556, seconds: float = 5.0, timeout: float = 10.0) -> dict:
    import zmq
    from lerobot_robot_sourccey.robots.protobuf.generated import sourccey_pb2
    from lerobot_robot_sourccey.robots.protobuf.sourccey_protobuf import PROTOCOL_VERSION, SourcceyProtobuf

    if not ip.strip() or not 1 <= port <= 65535 or not all(math.isfinite(v) and v > 0 for v in (seconds, timeout)):
        raise ValueError("IP, port, duration and timeout must be valid and positive")
    converter = SourcceyProtobuf()
    count, started, report = 0, None, None
    context = zmq.Context()
    socket = context.socket(zmq.PULL)  # never create PUSH / command / SDK sockets
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.CONFLATE, 1)
    previous, changed = {}, set()
    try:
        socket.connect(f"tcp://{ip}:{port}")
        while started is None or time.monotonic() - started < seconds:
            if not socket.poll(int(timeout * 1000)):
                raise TimeoutError(f"No fresh observation from {ip}:{port} for {timeout}s")
            observation = decode_packet(socket.recv(), sourccey_pb2.SourcceyRobotState, converter, PROTOCOL_VERSION)
            report = validate_observation(observation)
            for name in CAMERAS:
                frame = observation[name]
                if name in previous and not np.array_equal(previous[name], frame):
                    changed.add(name)
                previous[name] = frame.copy()
            count += 1
            if started is None:
                started = time.monotonic()
        if count < 2 or changed != set(CAMERAS):
            raise ValueError(
                f"Insufficient changing camera frames; unchanged cameras: {sorted(set(CAMERAS) - changed)}"
            )
        return {
            "status": "observation_check_passed",
            "packets": count,
            "protocol_version": PROTOCOL_VERSION,
            "units": "vendor native; not converted to simulation SI coordinates",
            **report,
        }
    finally:
        socket.close(0)
        context.term()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    try:
        print(json.dumps(check_host(args.ip, port=args.port, seconds=args.seconds, timeout=args.timeout), indent=2))
    except (ImportError, ValueError, TimeoutError) as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

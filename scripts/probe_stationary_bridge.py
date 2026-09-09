#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Receive-only hardware camera audit. Never creates a command socket or robot client."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import zmq
from PIL import Image

from emet.utils import compression

CAMERAS = {
    "head_left": ("rgb", "head_cam_left/image"),
    "head_right": ("rgb_right", "head_cam_right/image"),
    "wrist": ("rgb_tertiary", "ee_cam/image"),
}


def decode_cameras(message):
    result = {}
    for name, keys in CAMERAS.items():
        value = next((message[k] for k in keys if message.get(k) is not None), None)
        if value is None:
            continue
        rgb = value if isinstance(value, np.ndarray) and value.ndim == 3 else compression.from_jpg(value)
        if rgb is not None and rgb.ndim == 3 and rgb.shape[2] == 3:
            result[name] = rgb
    return result


def run(host, port, output, seconds):
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "host": host,
        "port": port,
        "mode": "receive_only",
        "commands_sent": 0,
        "acquisition_freshness": "not verified: transport delivery is not a sensor timestamp",
        "messages": 0,
        "cameras": {},
        "depth_frames": 0,
    }
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect(f"tcp://{host}:{port}")
    deadline = time.monotonic() + seconds
    hashes = {name: set() for name in CAMERAS}
    try:
        while time.monotonic() < deadline:
            if not socket.poll(500):
                continue
            message = socket.recv_pyobj()
            if not isinstance(message, dict) or message.get("emet_robot_id") != "innate_mars":
                raise ValueError("observation is not identified as innate_mars")
            report["messages"] += 1
            report["session"] = message.get("emet_session")
            report["command_protocol"] = message.get("command_protocol")
            cameras = decode_cameras(message)
            arrays = dict(cameras)
            for name, rgb in cameras.items():
                hashes[name].add(hashlib.sha256(rgb.tobytes()).hexdigest())
                report["cameras"][name] = {
                    "shape": list(rgb.shape),
                    "nonblack": bool(np.max(rgb) > 1),
                    "distinct_images": len(hashes[name]),
                }
                Image.fromarray(rgb).save(output / f"{name}.png")
            for key in ("camera_K", "camera_K_right", "camera_pose", "camera_pose_right", "base_pose", "depth"):
                value = message.get(key)
                if isinstance(value, np.ndarray):
                    arrays[key] = value
            depth = arrays.get("depth")
            if depth is not None and np.any(np.isfinite(depth) & (depth > 0)):
                report["depth_frames"] += 1
            if cameras:
                np.savez_compressed(output / "latest_frame.npz", **arrays)
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        socket.close(0)
        context.term()
    report["head_rgb_received"] = all(
        report["cameras"].get(name, {}).get("nonblack", False) for name in ("head_left", "head_right")
    )
    report["rgbd_ready"] = report["head_rgb_received"] and report["depth_frames"] > 0
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["head_rgb_received"] and "error" not in report else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=4401)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    raise SystemExit(run(args.host, args.port, args.output_dir, args.seconds))

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

from emet.core.zmq_obs_codec import decode_zmq_obs_depth_inplace, read_image_timing
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


def decode_geometry(message, head):
    """Return current sensor arrays and primary RGB-D pairing status."""
    decode_zmq_obs_depth_inplace(message)
    arrays = {}
    for key in ("camera_K", "camera_K_right", "camera_pose", "camera_pose_right", "base_pose", "depth", "gps", "compass"):
        value = message.get(key)
        if isinstance(value, np.ndarray):
            arrays[key] = value
    depth, k, pose = (arrays.get(key) for key in ("depth", "camera_K", "camera_pose"))
    pose_ok = bool(pose is not None and pose.shape == (4, 4) and np.isfinite(pose).all()
                   and np.allclose(pose[3], [0., 0., 0., 1.])
                   and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
                   and np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-4))
    k_ok = bool(k is not None and k.shape == (3, 3) and np.isfinite(k).all()
                and min(k[0, 0], k[1, 1]) > 0 and np.allclose(k[2], [0., 0., 1.]))
    depth_ok = bool(depth is not None and depth.ndim == 2 and np.issubdtype(depth.dtype, np.floating)
                    and np.any(np.isfinite(depth) & (depth > 0)))
    return arrays, {
        "depth": depth_ok,
        "paired_metric_rgbd": bool(depth_ok and k_ok and pose_ok and head is not None
                                   and head.shape[:2] == depth.shape),
        "intrinsics": k_ok,
        "camera_pose": pose_ok,
    }


def run(host, port, output, seconds, *, robot="innate_mars"):
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "host": host,
        "port": port,
        "mode": "receive_only",
        "robot": robot,
        "commands_sent": 0,
        "acquisition_freshness": "not verified: transport delivery is not a sensor timestamp",
        "messages": 0,
        "cameras": {},
        "depth_frames": 0,
        "paired_metric_rgbd_frames": 0,
        "intrinsics_frames": 0,
        "camera_pose_frames": 0,
    }
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect(f"tcp://{host}:{port}")
    started = time.monotonic()
    deadline = started + seconds
    hashes = {name: set() for name in CAMERAS}
    arrivals, latest_arrays, latest_timing = [], {}, {}
    stamps = {key: [] for key in ("head_cam_left", "head_cam_right", "ee_cam")}
    try:
        while time.monotonic() < deadline:
            if not socket.poll(500):
                continue
            message = socket.recv_pyobj()
            if not isinstance(message, dict) or message.get("emet_robot_id") != robot:
                raise ValueError(f"observation is not identified as {robot}")
            arrivals.append(time.monotonic() - started)
            report["messages"] += 1
            report["session"] = message.get("emet_session")
            report["command_protocol"] = message.get("command_protocol")
            cameras = decode_cameras(message)
            geometry, checks = decode_geometry(message, cameras.get("head_left"))
            timing = read_image_timing(message)
            for name, value in timing.items():
                stamp = value.get("timestamp_ns") if isinstance(value, dict) else None
                if type(stamp) is int and stamp > 0:
                    stamps[name].append(stamp)
            for key, ok in checks.items():
                report[key + "_frames"] += int(ok)
            for name, rgb in cameras.items():
                hashes[name].add(hashlib.sha256(rgb.tobytes()).hexdigest())
                report["cameras"][name] = {
                    "shape": list(rgb.shape),
                    "nonblack": bool(np.max(rgb) > 1),
                    "distinct_images": len(hashes[name]),
                    "frames": report["cameras"].get(name, {}).get("frames", 0) + 1,
                }
            if cameras:
                latest_arrays, latest_timing = {**geometry, **cameras}, timing
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        socket.close(0)
        context.term()
    receive_end = time.monotonic() - started
    if latest_arrays:
        for name in CAMERAS.keys() & latest_arrays.keys():
            Image.fromarray(latest_arrays[name]).save(output / f"{name}.png")
        np.savez_compressed(output / "latest_frame.npz", **latest_arrays)
        (output / "latest_frame_timing.json").write_text(json.dumps(latest_timing, indent=2) + "\n")
    report["max_receive_gap_s"] = float(max(np.diff([0., *arrivals, receive_end])))
    report["timing"] = {
        name: {"packets_with_timestamp": len(times),
               "advancing": sum(b > a for a, b in zip(times, times[1:])),
               "regressions": sum(b < a for a, b in zip(times, times[1:]))}
        for name, times in stamps.items()
    }
    for camera in report["cameras"].values():
        camera["packets_missing"] = report["messages"] - camera["frames"]
    required = ("head_left", "head_right") if robot == "innate_mars" else ("head_left",)
    report["head_rgb_received"] = all(report["cameras"].get(name, {}).get("nonblack", False) for name in required)
    report["rgbd_ready"] = report["head_rgb_received"] and report["paired_metric_rgbd_frames"] > 0
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["head_rgb_received"] and "error" not in report else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=4401)
    parser.add_argument("--robot", choices=("stretch", "innate_mars"), default="innate_mars")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    raise SystemExit(run(args.host, args.port, args.output_dir, args.seconds, robot=args.robot))

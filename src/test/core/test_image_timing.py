# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Timestamp transport tests, including the ROS method without ROS dependencies."""

import ast
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emet.core.zmq_obs_codec import merge_servo_images_into_full_obs, read_image_timing


def snapshot_method():
    path = Path(__file__).resolve().parents[2] / "innate_mars_bridge/innate_mars_bridge/ros/camera.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RosCamera")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_snapshot")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["get_snapshot"]


def test_snapshot_keeps_capture_stamp_on_republication_and_copies_pixels():
    camera = SimpleNamespace(_lock=threading.Lock(), _img=np.ones((2, 2, 3)), _t=SimpleNamespace(sec=12, nanosec=34))
    snapshot = snapshot_method()
    image, timing = snapshot(camera)
    assert timing["timestamp_ns"] == 12_000_000_034
    assert timing["clock_domain"] == "ros"
    image[:] = 0
    assert camera._img.all()
    assert snapshot(camera)[1] == timing
    camera._t = SimpleNamespace(sec=0, nanosec=0)
    assert snapshot(camera)[1]["timestamp_ns"] is None
    camera._img = None
    camera._t = SimpleNamespace(sec=99, nanosec=0)
    assert snapshot(camera)[1]["available"] is False
    assert snapshot(camera)[1]["timestamp_ns"] is None


def test_merge_timing_follows_pixels_not_newer_metadata():
    timing = {"timestamp_ns": 123, "clock_domain": "ros"}
    full = {"head_cam_left/image_timing": {"timestamp_ns": 999}}
    servo = {"head_cam_left/color_image": b"pixels", "head_cam_left/image_timing": timing}
    assert merge_servo_images_into_full_obs(full, servo)
    assert read_image_timing(full)["head_cam_left"] == timing
    servo["head_cam_left/image_timing"] = {"timestamp_ns": 456}
    assert not merge_servo_images_into_full_obs(full, servo)
    assert read_image_timing(full)["head_cam_left"] == timing


def test_legacy_image_does_not_inherit_timestamp():
    full = {"head_cam_left/image_timing": {"timestamp_ns": 999}}
    merge_servo_images_into_full_obs(full, {"head_cam_left/color_image": b"pixels"})
    assert read_image_timing(full) == {}

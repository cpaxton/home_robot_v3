# Copyright (c) Chris Paxton 2026
"""Acquisition time follows cached pixels, never publication time."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from emet.simulation.stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
from emet.simulation.stretch_mujoco.mujoco_server_camera_manager import (
    MujocoServerCameraManagerSync,
    MujocoServerCameraManagerThreaded,
)


def test_camera_timing_roundtrip_and_untimed_replacement():
    frame = StatusStretchCameras.from_dict({"time": 0, "fps": 0})
    head = StretchCameras.cam_d435i_rgb
    wrist = StretchCameras.cam_d405_rgb
    assert frame.image_timing_for(head) is None
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    frame.set_camera_data(head, pixels, timestamp_ns=100)
    frame.set_camera_data(wrist, pixels, timestamp_ns=200)
    restored = StatusStretchCameras.from_dict(frame.to_dict())
    assert restored.image_timing_for(head)["timestamp_ns"] == 100
    assert restored.image_timing_for(wrist)["timestamp_ns"] == 200
    restored.set_camera_data(head, pixels)
    assert restored.image_timing_for(head) is None
    assert restored.image_timing_for(wrist)["timestamp_ns"] == 200


def test_render_timestamp_precedes_scene_capture():
    manager = MujocoServerCameraManagerSync.__new__(MujocoServerCameraManagerSync)
    manager.camera_lock = threading.Lock()
    manager.mujoco_server = SimpleNamespace(mjdata=object())
    events = []
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    renderer = MagicMock()
    renderer.update_scene.side_effect = lambda **kwargs: events.append("scene")
    renderer.render.side_effect = lambda: (events.append("pixels"), pixels)[1]
    camera = SimpleNamespace(camera_name_in_mjcf="head", post_processing_callback=None)
    with patch(
        "emet.simulation.stretch_mujoco.mujoco_server_camera_manager.time.monotonic_ns",
        side_effect=lambda: (events.append("timestamp"), 123456)[1],
    ):
        captured_camera, captured_pixels, stamp = manager._render_camera(renderer, camera)
    assert events == ["timestamp", "scene", "pixels"]
    assert captured_camera is camera
    assert captured_pixels is pixels
    assert stamp == 123456


@pytest.mark.parametrize("threaded", [False, True])
def test_camera_batch_retains_per_camera_timing(threaded):
    cls = MujocoServerCameraManagerThreaded if threaded else MujocoServerCameraManagerSync
    manager = cls.__new__(cls)
    proxy = MagicMock()
    manager.mujoco_server = SimpleNamespace(mjdata=SimpleNamespace(time=1), data_proxies=proxy)
    manager.camera_fps_counter = SimpleNamespace(fps=20)
    head, wrist = StretchCameras.cam_d435i_rgb, StretchCameras.cam_d405_rgb
    manager.camera_renderers = {head: 100, wrist: 200}
    manager._render_camera = lambda stamp, camera: (camera, np.zeros((2, 2, 3)), stamp)
    manager.get_camera_params = lambda camera: np.eye(3)
    with ThreadPoolExecutor(max_workers=1) as pool:
        manager.cameras_rendering_thread_pool = pool
        if threaded:
            manager._pull_camera_data_threadpool()
        else:
            manager._pull_camera_data()
    frame = proxy.set_cameras.call_args.args[0]
    assert frame.image_timing_for(head)["timestamp_ns"] == 100
    assert frame.image_timing_for(wrist)["timestamp_ns"] == 200


@pytest.mark.parametrize("timed", [False, True])
def test_stretch_full_client_preserves_capture_metadata(timed):
    from emet.controller.zmq_client import StretchZmqClient

    client = StretchZmqClient.__new__(StretchZmqClient)
    client._obs_lock = threading.Lock()
    client._seq_id = 1
    client.is_up_to_date = lambda **kwargs: True
    client._obs = {
        "gps": np.zeros(2),
        "compass": np.zeros(1),
        "rgb": np.zeros((2, 2, 3)),
        "depth": np.ones((2, 2)),
        "xyz": None,
        "lidar_points": None,
        "lidar_timestamp": None,
    }
    timing = {
        "timestamp_ns": 100,
        "clock_domain": "mujoco_render_monotonic",
        "source": "renderer_scene_capture",
        "available": True,
    }
    if timed:
        client._obs["head_cam_left/image_timing"] = timing
    with patch("emet.controller.zmq_client.time.sleep"):
        observation = client.get_observation()
    assert observation.image_timing == ({"head_cam_left": timing} if timed else {})

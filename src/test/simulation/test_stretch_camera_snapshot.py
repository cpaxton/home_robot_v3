# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
from emet.simulation.stretch_mujoco.mujoco_server_camera_manager import (
    MujocoServerCameraManagerSync,
    MujocoServerCameraManagerThreaded,
)


@pytest.mark.parametrize("yaw", [0.0, 0.8])
def test_captured_robot_geometry_matches_rendered_model_and_is_immutable(yaw):
    assets = Path(__file__).resolve().parents[2] / "emet/assets/robot"
    model = mujoco.MjModel.from_xml_path(str(assets / "scene.xml"))
    data = mujoco.MjData(model)
    base = model.body("base_link")
    addr = model.jnt_qposadr[base.jntadr[0]]
    data.qpos[addr : addr + 2] += [2.0, -1.0]
    data.qpos[addr + 3 : addr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    data.qpos[model.joint("joint_wrist_yaw").qposadr] = 0.3
    data.time = 12.5
    mujoco.mj_forward(model, data)
    manager = MujocoServerCameraManagerSync.__new__(MujocoServerCameraManagerSync)
    manager.mujoco_server = SimpleNamespace(mjmodel=model, mjdata=data)
    manager.camera_lock = Lock()
    manager.camera_fps_counter = SimpleNamespace(fps=15.0)
    snapshot, imagery = manager._capture_state()
    assert imagery.time == 12.5
    assert imagery.image_timing == {
        "timestamp_ns": 12_500_000_000,
        "clock_domain": "mujoco_sim",
        "source": "render_state_snapshot",
        "available": True,
    }
    for name, pose, image_rotation in (
        ("d405_rgb", imagery.cam_d405_pose, np.eye(3)),
        ("d435i_camera_rgb", imagery.cam_d435i_pose, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])),
    ):
        np.testing.assert_allclose(pose[:3, 3], data.camera(name).xpos)
        np.testing.assert_allclose(
            pose[:3, :3], data.camera(name).xmat.reshape(3, 3) @ np.diag([1, -1, -1]) @ image_rotation
        )
    np.testing.assert_allclose(imagery.ee_pose[:3, 3], data.body("link_grasp_center").xpos)
    np.testing.assert_allclose(imagery.ee_pose[:3, :3], data.body("link_grasp_center").xmat.reshape(3, 3))
    expected_qpos = snapshot.qpos.copy()
    expected_pose = imagery.ee_pose.copy()
    data.qpos[addr] += 1.0
    data.time += 1.0
    mujoco.mj_forward(model, data)
    renderer = Mock()
    renderer.render.return_value = np.zeros((2, 2, 3))
    manager._render_camera(renderer, StretchCameras.cam_d405_rgb, snapshot)
    renderer.update_scene.assert_called_once_with(data=snapshot, camera="d405_rgb")
    np.testing.assert_array_equal(snapshot.qpos, expected_qpos)
    np.testing.assert_array_equal(imagery.ee_pose, expected_pose)
    assert snapshot.time == 12.5
    assert imagery.image_timing["timestamp_ns"] == 12_500_000_000


@pytest.mark.parametrize("threaded", [False, True])
def test_rgb_and_depth_render_one_snapshot_even_if_physics_advances(threaded):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
        <body name="link_grasp_center"><joint type="slide"/><geom size=".1"/>
        <camera name="d405_rgb"/><camera name="d435i_camera_rgb"/></body>
        </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    manager = MujocoServerCameraManagerThreaded.__new__(MujocoServerCameraManagerThreaded)
    manager.mujoco_server = SimpleNamespace(mjmodel=model, mjdata=data, data_proxies=Mock())
    manager.camera_lock = Lock()
    manager.camera_fps_counter = SimpleNamespace(fps=15.0)
    manager.get_camera_params = Mock(return_value=np.eye(3))
    manager.camera_renderers = {
        StretchCameras.cam_d405_rgb: Mock(),
        StretchCameras.cam_d405_depth: Mock(),
    }

    def advance_physics():
        data.qpos[0] += 0.1
        data.time += 0.1
        mujoco.mj_forward(model, data)
        return np.ones((2, 2))

    for renderer in manager.camera_renderers.values():
        renderer.render.side_effect = advance_physics
    if threaded:
        with ThreadPoolExecutor(max_workers=1) as pool:
            manager.cameras_rendering_thread_pool = pool
            manager._pull_camera_data_threadpool()
    else:
        manager._pull_camera_data()
    snapshots = [r.update_scene.call_args.kwargs["data"] for r in manager.camera_renderers.values()]
    assert snapshots[0] is snapshots[1]
    assert snapshots[0].qpos[0] == 0
    assert data.qpos[0] == 0.2
    imagery = manager.mujoco_server.data_proxies.set_cameras.call_args.args[0]
    assert imagery.time == 0
    np.testing.assert_allclose(imagery.ee_pose[:3, 3], [0, 0, 0])

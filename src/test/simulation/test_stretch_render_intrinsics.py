# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Actual Stretch model: render frustum, depth registration and published head K."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import pytest

from emet.core.server import BaseZmqServer
from emet.simulation.mujoco_server_stretch import MujocoZmqServer
from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
from emet.simulation.stretch_mujoco.mujoco_server_camera_manager import MujocoServerCameraManagerSync
from emet.utils.assets import get_mujoco_models_path


@pytest.mark.parametrize("scale", ["1", "2"])
def test_actual_stretch_registration_and_published_k_match_render_projection(monkeypatch, scale):
    monkeypatch.setenv("EMET_STRETCH_HEAD_CAMERA_SCALE", scale)
    model = mujoco.MjModel.from_xml_path(str(get_mujoco_models_path() / "stretch.xml"))
    manager = MujocoServerCameraManagerSync.__new__(MujocoServerCameraManagerSync)
    manager.mujoco_server = SimpleNamespace(mjmodel=model)
    for camera in (
        StretchCameras.cam_d435i_rgb,
        StretchCameras.cam_d435i_depth,
        StretchCameras.cam_d405_rgb,
        StretchCameras.cam_d405_depth,
    ):
        manager.set_camera_params(camera)
    settings = StretchCameras.cam_d435i_rgb.initial_camera_settings
    cid = model.camera("d435i_camera_rgb").id
    assert model.cam_fovy[cid] == settings.field_of_view_vertical_in_degrees
    assert np.all(model.cam_sensorsize[cid] == 0)  # Pinhole renderer uses fovy.
    imagery = SimpleNamespace(
        cam_d435i_rgb=np.zeros((settings.height, settings.width, 3), np.uint8),
        cam_d435i_depth=np.full((settings.height, settings.width), 1.25),
        cam_d435i_K=None,
    )
    manager._register_head_depth(imagery)
    raw_k = imagery.cam_d435i_K
    # Exercise the real constructor's head_K assignment without sockets, physics
    # processes or rendering. Only external startup is replaced.
    with (
        patch.object(BaseZmqServer, "__init__", return_value=None),
        patch(
            "emet.simulation.mujoco_server_stretch.StretchMujocoSimulator",
            return_value=SimpleNamespace(mjmodel=model, stop=Mock()),
        ),
    ):
        server = MujocoZmqServer()
    expected = np.array(
        [[raw_k[1, 1], 0.0, settings.height - 1 - raw_k[1, 2]], [0.0, raw_k[0, 0], raw_k[0, 2]], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(server.head_K, expected, atol=1e-10)
    np.testing.assert_allclose(manager.get_camera_params(StretchCameras.cam_d435i_rgb), raw_k)
    # Independently ask MuJoCo for the actual GL camera frustum (no GL context).
    data, scene = mujoco.MjData(model), mujoco.MjvScene(model, maxgeom=2000)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type, camera.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, cid
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    frustum = scene.camera[0]
    focal = settings.height * frustum.frustum_near / (frustum.frustum_top - frustum.frustum_bottom)
    np.testing.assert_allclose(raw_k[[0, 1], [0, 1]], focal, atol=1e-4)
    points = np.array([[0.03, 0.01, 1.0], [-0.05, 0.1, 0.8]])
    pixels = points @ raw_k.T
    pixels = pixels[:, :2] / pixels[:, 2, None]
    upright_points = points @ np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    published = upright_points @ server.head_K.T
    published = published[:, :2] / published[:, 2, None]
    np.testing.assert_allclose(published, np.column_stack((settings.height - 1 - pixels[:, 1], pixels[:, 0])))
    # Registration keeps a missing strip from the physical camera baseline.
    assert np.any(imagery.cam_d435i_depth == 0)
    valid = imagery.cam_d435i_depth > 0
    depth_id = model.camera("d435i_camera_depth").id
    np.testing.assert_allclose(data.cam_xmat[cid], data.cam_xmat[depth_id], atol=1e-12)
    # The model's approximate Euler angles give the baseline a small axial
    # component; preserve it instead of assuming depth is numerically unchanged.
    forward = -data.cam_xmat[cid].reshape(3, 3)[:, 2]
    axial_offset = forward @ (data.cam_xpos[depth_id] - data.cam_xpos[cid])
    np.testing.assert_allclose(imagery.cam_d435i_depth[valid], 1.25 + axial_offset, atol=2e-7)
    # Wrist frames use the actual 480x270 render, not the 1280x720 sensor
    # metadata or the physical 242.56/242.34 focal calibration.
    wrist = StretchCameras.cam_d405_rgb
    wk = manager.get_camera_params(wrist)
    np.testing.assert_allclose(server.ee_K, wk)
    camera.fixedcamid = model.camera(wrist.camera_name_in_mjcf).id
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    frustum = scene.camera[0]
    expected_focal = (
        wrist.initial_camera_settings.height * frustum.frustum_near / (frustum.frustum_top - frustum.frustum_bottom)
    )
    np.testing.assert_allclose(wk[[0, 1], [0, 1]], expected_focal, atol=1e-4)
    np.testing.assert_array_equal(
        wk[:2, 2], [(wrist.initial_camera_settings.width - 1) / 2, (wrist.initial_camera_settings.height - 1) / 2]
    )

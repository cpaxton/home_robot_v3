# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras


def fixture():
    root = Path(__file__).resolve().parent
    capture = json.loads((root / "fixtures/stretch_preclosure_cylinder.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(root.parents[1] / "emet/assets/robot" / capture["scene"]))
    data = mujoco.MjData(model)
    data.qpos[:] = capture["qpos"]
    for camera in (StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth):
        model.cam_fovy[model.camera(camera.camera_name_in_mjcf).id] = (
            camera.initial_camera_settings.field_of_view_vertical_in_degrees
        )
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize("names", [("d435i_camera_rgb", "d435i_camera_depth"), ("d405_rgb", "d405_depth")])
def test_published_registered_rgbd_cameras_share_optical_frame(names):
    model, data = fixture()
    color, depth = [model.camera(name).id for name in names]
    np.testing.assert_allclose(data.cam_xpos[color], data.cam_xpos[depth], atol=1e-12)
    np.testing.assert_allclose(data.cam_xmat[color], data.cam_xmat[depth], atol=1e-12)
    assert model.cam_fovy[color] == model.cam_fovy[depth]


@pytest.mark.parametrize("registered", [False, True])
def test_rendered_object_depth_agrees_with_color_pixel_rays(registered):
    # Ground-truth segmentation is evaluator-only, never supplied to the agent.
    model, data = fixture()
    if not registered:
        model.cam_pos[model.camera("d435i_camera_depth").id] = [0, 0, 0]
        mujoco.mj_forward(model, data)
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 0, 0, 0, 0]
    renderer = mujoco.Renderer(model, height=240, width=424)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="d435i_camera_rgb", scene_option=option)
        segmentation = renderer.render().copy()
        ids = np.flatnonzero(model.geom_bodyid == model.body("object2").id)
        mask = np.isin(segmentation[..., 0], ids) & (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        assert mask.sum() > 25
        renderer.disable_segmentation_rendering()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="d435i_camera_rgb", scene_option=option)
        expected = renderer.render().copy()
        renderer.update_scene(data, camera="d435i_camera_depth", scene_option=option)
        actual = renderer.render().copy()
        error = np.abs(actual[mask] - expected[mask])
        if registered:
            assert np.max(error) < 1e-5
        else:
            assert np.max(error) > 0.03
            assert np.mean(error > 0.01) > 0.05
    finally:
        renderer.close()

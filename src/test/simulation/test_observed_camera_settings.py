# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import mujoco
import numpy as np
import pytest

from emet.simulation.depth_registration import rendered_camera_intrinsics
from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
from emet.simulation.stretch_mujoco.mujoco_server_camera_manager import head_camera_geomgroup_mask


def test_head_resolution_changes_both_sensors_and_preserves_projection(monkeypatch):
    monkeypatch.setenv('EMET_STRETCH_HEAD_CAMERA_SCALE', '2')
    rgb = StretchCameras.cam_d435i_rgb.initial_camera_settings
    depth = StretchCameras.cam_d435i_depth.initial_camera_settings
    assert (rgb.width, rgb.height) == (848, 480) == (depth.width, depth.height)
    k = rendered_camera_intrinsics(rgb.field_of_view_vertical_in_degrees, rgb.width, rgb.height)
    monkeypatch.setenv('EMET_STRETCH_HEAD_CAMERA_SCALE', '1')
    small = StretchCameras.cam_d435i_rgb.initial_camera_settings
    original = rendered_camera_intrinsics(small.field_of_view_vertical_in_degrees, small.width, small.height)
    assert k[0, 0] == pytest.approx(2*original[0, 0])
    np.testing.assert_allclose(k[:2, 2] + .5, 2*(original[:2, 2]+.5))
    monkeypatch.setenv('EMET_STRETCH_HEAD_CAMERA_SCALE', '3')
    with pytest.raises(ValueError):
        _ = StretchCameras.cam_d435i_rgb.initial_camera_settings


def test_realism_mode_keeps_robot_and_hides_collision_proxy_groups(monkeypatch):
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="base_link"><geom group="2" size=".1"/></body></worldbody></mujoco>')
    monkeypatch.setenv('EMET_SIM_HEAD_RENDER_ROBOT', '1')
    assert head_camera_geomgroup_mask(model).tolist() == [1, 1, 1, 0, 0, 0]
    monkeypatch.delenv('EMET_SIM_HEAD_RENDER_ROBOT')
    assert head_camera_geomgroup_mask(model).tolist() == [1, 1, 0, 0, 0, 0]

# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import numpy as np
import pytest

from emet.simulation.depth_registration import register_depth, rendered_camera_intrinsics


def test_identity_preserves_only_measured_depth():
    depth = np.array([[1.0, np.nan, 2.0], [0.0, 3.0, 4.0]])
    result = register_depth(depth, np.eye(3), np.eye(3), np.eye(4), depth.shape)
    np.testing.assert_array_equal(result, [[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]])


def test_baseline_moves_measured_pixels_and_leaves_holes():
    depth = np.ones((3, 4))
    transform = np.eye(4)
    transform[0, 3] = 1.0
    result = register_depth(depth, np.eye(3), np.eye(3), transform, depth.shape)
    np.testing.assert_array_equal(result[:, 0], 0.0)
    np.testing.assert_array_equal(result[:, 1:], 1.0)


def test_nearest_surface_wins_when_samples_project_to_same_pixel():
    depth = np.array([[1.0, 2.0]])
    k = np.diag([0.1, 1.0, 1.0])
    result = register_depth(depth, np.eye(3), k, np.eye(4), (1, 2))
    np.testing.assert_array_equal(result, [[1.0, 0.0]])


def test_behind_camera_and_invalid_transform_reject():
    transform = np.eye(4)
    transform[2, 3] = -2.0
    assert not register_depth(np.ones((2, 2)), np.eye(3), np.eye(3), transform, (2, 2)).any()
    transform[0, 0] = 2.0
    with pytest.raises(ValueError, match="calibration"):
        register_depth(np.ones((2, 2)), np.eye(3), np.eye(3), transform, (2, 2))


def test_rendered_intrinsics_use_fov_and_pixel_centers():
    k = rendered_camera_intrinsics(90.0, 4, 2)
    np.testing.assert_allclose(k, [[1.0, 0.0, 1.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])


def test_head_camera_manager_registers_rigid_pair_instead_of_rendering_new_depth():
    from types import SimpleNamespace

    import mujoco

    from emet.simulation.stretch_mujoco.mujoco_server_camera_manager import MujocoServerCameraManagerSync

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body>"
        '<camera name="d435i_camera_rgb" fovy="90"/>'
        '<camera name="d435i_camera_depth" pos="1 0 0" fovy="90"/>'
        "</body></worldbody></mujoco>"
    )
    manager = SimpleNamespace(mujoco_server=SimpleNamespace(mjmodel=model))
    imagery = SimpleNamespace(
        cam_d435i_rgb=np.zeros((4, 6, 3), dtype=np.uint8), cam_d435i_depth=np.ones((4, 6)), cam_d435i_K=None
    )
    MujocoServerCameraManagerSync._register_head_depth(manager, imagery)
    np.testing.assert_array_equal(imagery.cam_d435i_depth[:, :2], 0.0)
    np.testing.assert_array_equal(imagery.cam_d435i_depth[:, 2:], 1.0)
    np.testing.assert_allclose(imagery.cam_d435i_K, rendered_camera_intrinsics(90.0, 6, 4))

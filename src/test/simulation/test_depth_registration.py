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


def test_provenance_preserves_original_pixel_and_continuous_projection():
    depth = np.array([[1.0, np.nan, 2.0], [0.0, 3.0, 4.0]])
    transform = np.eye(4)
    transform[0, 3] = 0.2
    aligned, evidence = register_depth(depth, np.eye(3), np.eye(3), transform, depth.shape, return_provenance=True)
    np.testing.assert_array_equal(aligned, register_depth(depth, np.eye(3), np.eye(3), transform, depth.shape))
    np.testing.assert_array_equal(evidence["source_pixel_index"], [[0, -1, 2], [-1, 4, 5]])
    np.testing.assert_allclose(evidence["projected_uv"][0, 0], [0.2, 0])
    np.testing.assert_allclose(evidence["projected_uv"][0, 2], [2.1, 0])
    assert np.isnan(evidence["projected_uv"][0, 1]).all()
    assert np.isnan(evidence["projected_uv"][1, 0]).all()


@pytest.mark.parametrize("depth, expected_source", [([[2.0, 1.0]], 1), ([[1.0, 1.0]], 0)])
def test_provenance_selects_nearest_surface_with_deterministic_ties(depth, expected_source):
    aligned, evidence = register_depth(
        depth, np.eye(3), np.diag([0.1, 1.0, 1.0]), np.eye(4), (1, 2), return_provenance=True
    )
    np.testing.assert_array_equal(aligned, [[1.0, 0.0]])
    np.testing.assert_array_equal(evidence["source_pixel_index"], [[expected_source, -1]])
    np.testing.assert_allclose(evidence["projected_uv"][0, 0], [0.1 * expected_source, 0])


def test_empty_registration_has_no_provenance():
    aligned, evidence = register_depth(
        np.zeros((2, 3)), np.eye(3), np.eye(3), np.eye(4), (4, 5), return_provenance=True
    )
    assert not aligned.any()
    assert (evidence["source_pixel_index"] == -1).all()
    assert np.isnan(evidence["projected_uv"]).all()


def test_every_winning_source_reconstructs_its_saved_depth_and_uv():
    depth = np.random.default_rng(52).uniform(0.4, 2.0, (8, 12))
    depth[2, 3], depth[4, 8] = 0.0, np.nan
    kd = rendered_camera_intrinsics(60.0, 12, 8)
    kr = rendered_camera_intrinsics(50.0, 10, 6)
    transform = np.eye(4)
    transform[:3, 3] = [0.03, -0.01, 0.05]
    aligned, evidence = register_depth(depth, kd, kr, transform, (6, 10), return_provenance=True)
    np.testing.assert_array_equal(aligned, register_depth(depth, kd, kr, transform, (6, 10)))
    for y, x in np.argwhere(evidence["source_pixel_index"] >= 0):
        sy, sx = divmod(int(evidence["source_pixel_index"][y, x]), depth.shape[1])
        p = transform[:3, :3] @ (np.linalg.inv(kd) @ [sx, sy, 1.0] * depth[sy, sx]) + transform[:3, 3]
        uv = kr @ p
        np.testing.assert_allclose(evidence["projected_uv"][y, x], uv[:2] / uv[2])
        np.testing.assert_array_equal(np.rint(uv[:2] / uv[2]), [x, y])
        assert aligned[y, x] == np.float32(p[2])


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


def test_original_registration_packet_matches_timed_camera_sample_and_clears_on_replacement(monkeypatch):
    from types import SimpleNamespace

    import mujoco

    from emet.core.rgbd_registration import decode_registration
    from emet.simulation.stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
    from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
    from emet.simulation.stretch_mujoco.mujoco_server_camera_manager import MujocoServerCameraManagerSync

    monkeypatch.setenv("EMET_SIM_RGBD_PROVENANCE", "1")
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body>"
        '<camera name="d435i_camera_rgb" fovy="90"/>'
        '<camera name="d435i_camera_depth" pos="1 0 0" fovy="90"/>'
        "</body></worldbody></mujoco>"
    )
    frame = StatusStretchCameras.default()
    rgb, depth = np.zeros((4, 6, 3), np.uint8), np.ones((4, 6), np.float32)
    frame.set_camera_data(StretchCameras.cam_d435i_rgb, rgb, timestamp_ns=100, capture_end_ns=110)
    frame.set_camera_data(StretchCameras.cam_d435i_depth, depth, timestamp_ns=100, capture_end_ns=110)
    MujocoServerCameraManagerSync._register_head_depth(
        SimpleNamespace(mujoco_server=SimpleNamespace(mjmodel=model)), frame
    )
    packet = frame.head_depth_registration
    decoded = decode_registration(packet)
    np.testing.assert_array_equal(decoded["raw_depth"], depth)
    np.testing.assert_array_equal(decoded["registered_depth"], frame.cam_d435i_depth)
    assert packet["rgb_interval_ns"] == packet["depth_interval_ns"] == [100, 110]
    restored = StatusStretchCameras.from_dict(frame.to_dict())
    assert restored.head_depth_registration == packet
    restored.set_camera_data(StretchCameras.cam_d435i_depth, depth, timestamp_ns=120, capture_end_ns=130)
    assert restored.head_depth_registration is None

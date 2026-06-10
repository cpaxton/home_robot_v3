# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import numpy as np
import open3d as o3d
import pytest
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from emet.benchmarks.sqa3d.scannet.config import (
    scene_replay_assets_present,
    scene_sens_present,
)
from unittest.mock import MagicMock

from emet.benchmarks.sqa3d.scannet.sens import (
    ScanNetSensLoader,
    SensFrameMatch,
    _yaw_distance_rad,
    scannet_camera_to_opencv_camera_to_world,
    scannet_camera_to_opencv_world_to_camera,
    target_camera_forward_xy,
)
from emet.benchmarks.sqa3d.scannet.config import default_scannet_root, scene_sens_path
from emet.benchmarks.sqa3d.scannet.simulator import ScanNetReplaySimulator

BOX_SCENE_ID = "scene_fixture_00"


def _write_box_scene(scannet_root: Path) -> Path:
    scan_dir = scannet_root / "scans" / BOX_SCENE_ID
    scan_dir.mkdir(parents=True, exist_ok=True)
    mesh = o3d.geometry.TriangleMesh.create_box(4.0, 4.0, 2.5)
    mesh.translate([-2.0, -2.0, 0.0])
    mesh.paint_uniform_color([0.7, 0.6, 0.5])
    mesh.compute_vertex_normals()
    ply = scan_dir / f"{BOX_SCENE_ID}_vh_clean_2.ply"
    o3d.io.write_triangle_mesh(str(ply), mesh)
    return ply


def test_yaw_distance_symmetric():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert _yaw_distance_rad(a, b) == pytest.approx(np.pi / 2, abs=1e-6)
    assert _yaw_distance_rad(b, a) == pytest.approx(np.pi / 2, abs=1e-6)


def test_target_camera_forward_xy_turns_with_agent():
    pos = np.zeros(3)
    quat_fwd = np.array([0.0, 0.0, 0.0, 1.0])
    quat_left = R.from_euler("z", np.pi / 2).as_quat()
    fwd0 = target_camera_forward_xy(pos, quat_fwd, sensor_height=1.5, camera_tilt_deg=-15.0)
    fwd1 = target_camera_forward_xy(pos, quat_left, sensor_height=1.5, camera_tilt_deg=-15.0)
    assert fwd0[0] == pytest.approx(1.0, abs=0.05)
    assert fwd1[1] == pytest.approx(1.0, abs=0.05)


def test_scannet_camera_to_opencv_pose_roundtrip():
    c2w = np.eye(4)
    c2w[:3, :3] = R.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = scannet_camera_to_opencv_world_to_camera(c2w)
    c2w_cv = scannet_camera_to_opencv_camera_to_world(c2w)
    assert w2c.shape == (4, 4)
    gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    assert np.allclose(w2c, gl_to_cv @ np.linalg.inv(c2w), atol=1e-5)
    assert np.allclose(c2w_cv, c2w @ gl_to_cv, atol=1e-5)
    assert np.allclose(c2w_cv, np.linalg.inv(w2c), atol=1e-5)


def test_nearest_frame_skips_invalid_sens_pose():
    sens = scene_sens_path("scene0553_00", default_scannet_root())
    if not sens.is_file():
        pytest.skip("scene0553_00.sens not on disk")
    loader = ScanNetSensLoader(sens)
    assert not loader._valid_mask[-1]
    idx = loader.nearest_frame_index(
        np.zeros(3),
        np.array([0.0, 0.0, 0.0, 1.0]),
        sensor_height=1.5,
        camera_tilt_deg=-15.0,
    )
    assert loader._valid_mask[idx]
    frame = loader._data.frames[idx]
    assert np.isfinite(frame.camera_to_world).all()


def test_scene_replay_assets_present(tmp_path):
    scene = "scene_test_00"
    scan = tmp_path / "scans" / scene
    scan.mkdir(parents=True)
    (scan / f"{scene}_vh_clean_2.ply").write_text("ply")
    assert scene_replay_assets_present(scene, tmp_path, replay_mode="auto")
    assert scene_replay_assets_present(scene, tmp_path, replay_mode="mesh")
    assert not scene_replay_assets_present(scene, tmp_path, replay_mode="sens")
    (scan / f"{scene}.sens").write_bytes(b"\x00")
    assert scene_sens_present(scene, tmp_path)
    assert scene_replay_assets_present(scene, tmp_path, replay_mode="sens")


def test_replay_simulator_mesh_fallback_on_poor_sens_match(tmp_path):
    scannet_root = tmp_path / "scannet"
    _write_box_scene(scannet_root)
    sim = ScanNetReplaySimulator(
        BOX_SCENE_ID,
        scannet_root=scannet_root,
        replay_mode="auto",
        image_width=160,
        image_height=120,
        sens_match_max_xy_m=0.75,
    )
    mock_sens = MagicMock()
    mock_sens.nearest_frame_match.return_value = SensFrameMatch(frame_index=3, xy_m=2.5, pos_3d_m=2.6)
    sim._sens = mock_sens
    try:
        sim.set_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        sim._anchor_xy = np.array([0.0, 0.0])
        sim._update_anchor_replay_info()
        assert sim.replay_backend == "mesh"
        assert sim.anchor_sens_match_xy_m == pytest.approx(2.5)
        assert sim._prefer_sens() is False
        frame = sim.get_frame()
        assert frame.replay_source == "mesh"
    finally:
        sim.close()


def test_replay_simulator_mesh_fallback_without_sens(tmp_path):
    scannet_root = tmp_path / "scannet"
    _write_box_scene(scannet_root)
    sim = ScanNetReplaySimulator(
        BOX_SCENE_ID,
        scannet_root=scannet_root,
        replay_mode="auto",
        image_width=160,
        image_height=120,
    )
    try:
        sim.set_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        assert sim.replay_backend == "mesh"
        frame = sim.get_frame()
        assert frame.replay_source == "mesh"
        assert frame.rgb.shape == (120, 160, 3)
    finally:
        sim.close()


def test_replay_simulator_sens_mode_requires_file(tmp_path):
    scannet_root = tmp_path / "scannet"
    _write_box_scene(scannet_root)
    with pytest.raises(FileNotFoundError, match="\\.sens"):
        ScanNetReplaySimulator(BOX_SCENE_ID, scannet_root=scannet_root, replay_mode="sens")

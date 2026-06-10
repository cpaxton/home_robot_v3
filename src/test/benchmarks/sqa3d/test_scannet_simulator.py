# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import os
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from emet.benchmarks.sqa3d.datasets import load_sqa3d_questions
from emet.benchmarks.sqa3d.scannet.observations import scannet_rgb_depth_to_observations
from emet.benchmarks.sqa3d.scannet.pose import apply_forward, apply_turn, planar_heading_rad, quat_xyzw_to_rotation
from emet.benchmarks.sqa3d.scannet.simulator import ScanNetEQASimulator

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SCENE_ID = "scene_fixture_00"


def _write_box_scene(scannet_root: Path) -> Path:
    scan_dir = scannet_root / "scans" / SCENE_ID
    scan_dir.mkdir(parents=True, exist_ok=True)
    mesh = o3d.geometry.TriangleMesh.create_box(4.0, 4.0, 2.5)
    mesh.translate([-2.0, -2.0, 0.0])
    mesh.paint_uniform_color([0.7, 0.6, 0.5])
    mesh.compute_vertex_normals()
    ply = scan_dir / f"{SCENE_ID}_vh_clean_2.ply"
    o3d.io.write_triangle_mesh(str(ply), mesh)
    return ply


def test_pose_turn_and_forward():
    pos = np.array([0.0, 0.0, 0.0])
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    quat = apply_turn(quat, np.deg2rad(90))
    pos2 = apply_forward(pos, quat, 1.0)
    assert pos2[0] == pytest.approx(0.0, abs=0.05)
    assert pos2[1] == pytest.approx(1.0, abs=0.05)
    assert planar_heading_rad(quat_xyzw_to_rotation(quat)) == pytest.approx(np.pi / 2, abs=0.05)


def test_scannet_simulator_renders(tmp_path: Path):
    scannet_root = tmp_path / "scannet"
    _write_box_scene(scannet_root)
    sim = ScanNetEQASimulator(SCENE_ID, scannet_root=scannet_root, image_width=160, image_height=120)
    try:
        sim.set_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        frame = sim.get_frame()
        assert frame.rgb.shape == (120, 160, 3)
        assert frame.depth.shape == (120, 160)
        obs = scannet_rgb_depth_to_observations(
            rgb=frame.rgb,
            depth=frame.depth,
            position=frame.position,
            quat_xyzw=frame.quat_xyzw,
            intrinsics=frame.intrinsics,
            sensor_height=sim.sensor_height,
            camera_tilt_deg=sim.camera_tilt_deg,
        )
        assert obs.rgb.shape[:2] == (120, 160)
        assert obs.gps.shape == (2,)
        sim.step("turn_left")
        frame2 = sim.get_frame()
        assert not np.allclose(frame2.quat_xyzw, frame.quat_xyzw)
    finally:
        sim.close()


def test_scannet_embodied_smoke():
    from emet.benchmarks.sqa3d.scannet.config import default_scannet_root, scene_assets_present
    from emet.benchmarks.sqa3d.scannet.runner import run_sqa3d_episode

    scannet_root = Path(os.environ.get("SCANNET_ROOT", str(default_scannet_root())))
    target_scene = os.environ.get("SQA3D_SCANNET_TEST_SCENE", "scene0380_00")
    if os.environ.get("RUN_SQA3D_SCANNET_TESTS", "") != "1" and not scene_assets_present(
        target_scene, scannet_root
    ):
        pytest.skip(
            f"ScanNet mesh missing for {target_scene}; "
            "run: uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00"
        )
    q = None
    chosen_split = "val"
    for split in ("train", "val", "test"):
        for candidate in load_sqa3d_questions(split):
            if candidate.scene_id == target_scene:
                q = candidate
                chosen_split = split
                break
        if q is not None:
            break
    if q is None:
        pytest.skip(f"No SQA3D question for scene {target_scene}")
    if not scene_assets_present(q.scene_id, scannet_root):
        pytest.skip(f"ScanNet mesh missing for {q.scene_id} under {scannet_root}")

    row = run_sqa3d_episode(
        question_id=q.question_id,
        split=chosen_split,
        mock_llm=True,
        max_planning_steps=3,
        scannet_root=scannet_root,
    )
    assert row.em is True

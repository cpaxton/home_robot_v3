# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from emet.benchmarks.sqa3d.datasets import SQA3DQuestion
from emet.benchmarks.sqa3d.scannet.runner import _is_infra_failure_text, _sanitize_prediction_text, _score_episode
from emet.benchmarks.sqa3d.scannet.simulator import ScanNetEQASimulator

SCENE_ID = "scene_runner_00"


def _write_box_scene(scannet_root: Path) -> None:
    scan_dir = scannet_root / "scans" / SCENE_ID
    scan_dir.mkdir(parents=True, exist_ok=True)
    mesh = o3d.geometry.TriangleMesh.create_box(4.0, 4.0, 2.5)
    mesh.translate([-2.0, -2.0, 0.0])
    mesh.paint_uniform_color([0.2, 0.5, 0.8])
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(scan_dir / f"{SCENE_ID}_vh_clean_2.ply"), mesh)


def test_sanitize_prediction_text():
    assert _sanitize_prediction_text("white") == "white"
    assert _sanitize_prediction_text("CUDA out of memory") == ""
    assert _sanitize_prediction_text("unknown") == "unknown"
    assert _sanitize_prediction_text("Caption:\nfoo") == ""
    assert _sanitize_prediction_text("The shelf is at approximately (2.48, 0.21, 3.06) m.") == ""


def test_is_infra_failure_text():
    assert _is_infra_failure_text("CUDA out of memory")
    assert _is_infra_failure_text("", "ERROR: model load failed")
    assert not _is_infra_failure_text("white", "")


def test_score_episode_marks_infra_from_raw_eqa():
    q = SQA3DQuestion(
        question_id=1,
        scene_id="scene0000_00",
        question="What color?",
        situation="I am standing by the door.",
        answers=("white",),
        position=(0.0, 0.0, 0.0),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    row = _score_episode(
        q,
        method="dynagraph",
        predicted="",
        raw_eqa="ERROR: CUDA out of memory",
        model_confident=False,
        planning_steps=0,
        replay_backend="sens",
        sens_frame_index=12,
        sens_match_xy_m=0.2,
    )
    assert row.infra_failure is True
    assert row.replay_backend == "sens"
    assert row.sens_frame_index == 12
    assert row.sens_match_xy_m == pytest.approx(0.2)


@pytest.mark.sim
@pytest.mark.timeout(120)
def test_capture_rotate_views_returns_to_start(tmp_path: Path, open3d_offscreen: None):
    scannet_root = tmp_path / "scannet"
    _write_box_scene(scannet_root)
    sim = ScanNetEQASimulator(SCENE_ID, scannet_root=scannet_root, image_width=160, image_height=120)
    try:
        sim.set_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        q0 = sim._quat_xyzw.copy()
        views = sim.capture_rotate_views(n_views=8)
        assert len(views) == 8
        assert np.allclose(sim._quat_xyzw, q0)
    finally:
        sim.close()

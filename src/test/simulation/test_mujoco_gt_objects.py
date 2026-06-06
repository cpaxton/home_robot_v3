# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from emet.memory.graph_eqa.graph_object_fusion.fusion import bounds_3d_iou
from emet.simulation.mujoco_gt_objects import bbox_xyxy_from_bounds


def test_project_synthetic_box_centered():
    K = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    bounds = {"min": [-0.1, -0.1, 1.0], "max": [0.1, 0.1, 1.2]}
    bb = bbox_xyxy_from_bounds(bounds, T_cam_world=T, K=K, image_hw=(80, 100))
    assert bb is not None
    assert bb[0] < bb[2] and bb[1] < bb[3]


def test_bounds_3d_iou_half_overlap():
    a = {"min": [0, 0, 0], "max": [1, 1, 1]}
    b = {"min": [0.5, 0, 0], "max": [1.5, 1, 1]}
    iou = bounds_3d_iou(a, b)
    assert 0.1 < iou < 0.4


def test_fixture_gt_loads():
    p = Path(__file__).resolve().parents[1] / "fixtures" / "gt_robocasa_seed0_snippet.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert len(data["objects"]) == 2


@pytest.mark.skipif(
    not Path(__file__).resolve().parents[3].joinpath("third_party/robocasa").is_dir(),
    reason="robocasa third_party missing",
)
@pytest.mark.timeout(120)
def test_export_robocasa_gt_smoke():
    pytest.importorskip("robocasa")
    import tempfile

    from emet.simulation.mujoco_gt_objects import export_robocasa_gt_scene

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "gt.json"
        export_robocasa_gt_scene(
            robot="innate_mars",
            seed=0,
            layout=1,
            style=1,
            out_path=out,
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["objects"]
        assert "bounds_3d" in data["objects"][0]

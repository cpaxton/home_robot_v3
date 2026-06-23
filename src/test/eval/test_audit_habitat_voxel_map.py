# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for scripts/audit_habitat_voxel_map.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_audit_bundle_with_history_and_grids(tmp_path: Path) -> None:
    explored = np.zeros((10, 10), dtype=bool)
    explored[2:5, 2:5] = True
    explored[8, 8] = True
    np.save(tmp_path / "explored_2d.npy", explored)
    np.save(tmp_path / "obstacles_2d.npy", np.zeros_like(explored))

    header = {
        "type": "header",
        "n_observations": 1,
        "grid_resolution": 0.1,
        "grid_origin_xy": [0.0, 0.0],
        "shape_hw": [10, 10],
    }
    obs = {
        "type": "observation",
        "obs_idx": 0,
        "gps_grid_ij": [3, 3],
        "camera_grid_ij": [8, 8],
        "camera_grid_ij_xz": [3, 3],
        "gps_camera_grid_delta_ij": [5, 5],
        "base_pose_xyt": [0.3, 0.3, 0.0],
        "pcd_centroid_xz": [0.3, 0.3],
    }
    (tmp_path / "observations_history.jsonl").write_text(
        json.dumps(header) + "\n" + json.dumps(obs) + "\n",
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[3] / "scripts" / "audit_habitat_voxel_map.py"
    out = subprocess.check_output(
        [sys.executable, str(script), str(tmp_path), "--json"],
        text=True,
    )
    report = json.loads(out)
    assert report["explored_cells"] == 10
    assert len(report["explored_components"]) >= 2
    assert len(report["gps_camera_grid_mismatches"]) == 1
    assert report["satellite_blob_hypothesis"]["best_matching_obs_idx"] == 0

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_habitat_camera_projection import analyze_bundle


def test_analyze_bundle_flags_camera_gps_mismatch(tmp_path: Path) -> None:
    header = {
        "type": "header",
        "n_observations": 2,
        "grid_resolution": 0.1,
        "grid_origin_xy": [512.0, 512.0],
        "shape_hw": [1024, 1024],
    }
    obs = {
        "type": "observation",
        "obs_idx": 0,
        "base_pose_xyt": [1.0, 2.0, 0.0],
        "pcd_centroid_xy": [3.0, 0.0],
        "gps_camera_grid_delta_ij": [0, -7],
    }
    obs2 = dict(obs)
    obs2["obs_idx"] = 1
    (tmp_path / "observations_history.jsonl").write_text(
        json.dumps(header) + "\n" + json.dumps(obs) + "\n" + json.dumps(obs2) + "\n",
        encoding="utf-8",
    )
    report = analyze_bundle(tmp_path)
    assert report["gps_camera_grid_mismatch_count"] == 2
    assert report["gps_camera_grid_delta_mean_ij"][1] == -7.0

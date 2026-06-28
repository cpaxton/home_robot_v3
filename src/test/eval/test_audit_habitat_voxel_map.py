# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for scripts/audit_habitat_voxel_map.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "audit_habitat_voxel_map.py"
HAB_CLI = REPO / ".venv-habitat" / "bin" / "emet-habitat"


def _run_audit(bundle_dir: Path) -> dict:
    out = subprocess.check_output(
        [sys.executable, str(SCRIPT), str(bundle_dir), "--json"],
        text=True,
    )
    return json.loads(out)


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

    report = _run_audit(tmp_path)
    assert report["explored_cells"] == 10
    assert len(report["explored_components"]) >= 2
    assert len(report["gps_camera_grid_mismatches"]) == 1
    assert report["satellite_blob_hypothesis"]["best_matching_obs_idx"] == 0


def test_audit_frame_aligned_bundle_passes_offset_thresholds(tmp_path: Path) -> None:
    """Aligned gps / PCD history should not flag large centroid offsets."""
    explored = np.ones((10, 10), dtype=bool)
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
        "gps_grid_ij": [5, 5],
        "camera_grid_ij": [5, 5],
        "camera_grid_ij_xz": [5, 5],
        "gps_camera_grid_delta_ij": [0, 0],
        "base_pose_xyt": [-4.55, 3.30, 0.0],
        "pcd_centroid_xz": [-4.52, 3.28],
    }
    (tmp_path / "observations_history.jsonl").write_text(
        json.dumps(header) + "\n" + json.dumps(obs) + "\n",
        encoding="utf-8",
    )

    report = _run_audit(tmp_path)
    assert report["large_pcd_centroid_offsets_m"] == []
    assert report.get("pcd_planar_x_mismatches", []) == []
    assert report["gps_camera_grid_mismatches"] == []


@pytest.mark.skipif(
    os.environ.get("RUN_HABITAT_FRAME_TESTS", "").strip().lower() not in ("1", "true", "yes"),
    reason="set RUN_HABITAT_FRAME_TESTS=1 to audit live Habitat episode bundles",
)
@pytest.mark.skipif(not HAB_CLI.is_file(), reason="habitat venv not installed")
@pytest.mark.parametrize("question_id", [6, 25, 57])
def test_live_habitat_frame_audit_smoke(tmp_path: Path, question_id: int) -> None:
    """One-frame Habitat episode: gps vs PCD offsets and obstacle frac sanity."""
    tag = f"frame_audit_q{question_id:04d}"
    out_jsonl = tmp_path / f"{tag}.jsonl"
    env = os.environ.copy()
    env["EMET_EVAL_EXPORT_VOXEL_HISTORY"] = "1"
    env["EMET_EVAL_EXPORT_MAP"] = "1"
    cmd = [
        str(HAB_CLI),
        "run-episode",
        "--question-id",
        str(question_id),
        "--mock-llm",
        "--method",
        "dynagraph",
        "--max-planning-steps",
        "1",
        "--max-movement-step",
        "0",
        "--no-rotate-in-place",
        "--output",
        str(out_jsonl),
        "--export-map",
    ]
    subprocess.run(cmd, cwd=str(REPO), env=env, check=True, timeout=600)

    episodes_root = Path.home() / ".cache" / "habitat_eqa" / "episodes"
    bundles = sorted(episodes_root.glob(f"**/*q{question_id:04d}*"))
    assert bundles, f"no episode bundle under {episodes_root} for question {question_id}"
    report = _run_audit(bundles[-1])

    for row in report.get("pcd_planar_x_mismatches", []):
        assert row["offset_x_m"] < 3.0, row
        assert row["base_x"] * row["pcd_x"] >= 0, row

    frac = report.get("explored_obstacle_frac")
    if frac is not None:
        assert frac < 0.85, f"question {question_id}: explored_obstacle_frac={frac} suggests frame bug"

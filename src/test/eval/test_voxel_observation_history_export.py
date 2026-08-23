# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for voxel observation history export."""

from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

from emet.eval.episode_diagnostics import export_voxel_observation_history

Frame = namedtuple(
    "Frame",
    [
        "camera_pose",
        "camera_K",
        "xyz",
        "rgb",
        "feats",
        "depth",
        "instance",
        "instance_classes",
        "instance_scores",
        "base_pose",
        "info",
        "obs",
        "full_world_xyz",
        "xyz_frame",
    ],
)


class _FakeVoxelMap:
    def __init__(self) -> None:
        self.grid_resolution = 0.1
        self.grid_origin = torch.tensor([512.0, 512.0], dtype=torch.float32)
        cam_pose = torch.eye(4, dtype=torch.float32)
        cam_pose[0, 3] = 1.0
        cam_pose[1, 3] = 1.5
        cam_pose[2, 3] = -3.0
        depth = np.ones((4, 4), dtype=np.float32)
        depth[0, 0] = 0.0
        pcd = np.array(
            [
                [1.0, 1.4, -3.0],
                [1.1, 1.5, -3.1],
                [0.9, 1.6, -2.9],
            ],
            dtype=np.float32,
        )
        self.observations = [
            Frame(
                camera_pose=cam_pose,
                camera_K=torch.eye(3),
                xyz=None,
                rgb=torch.zeros(3, 3),
                feats=None,
                depth=depth,
                instance=None,
                instance_classes=None,
                instance_scores=None,
                base_pose=torch.tensor([1.31, -3.47, 0.0], dtype=torch.float32),
                info={},
                obs=None,
                full_world_xyz=torch.from_numpy(pcd),
                xyz_frame="world",
            )
        ]

    def get_2d_map(self):
        obstacles = np.zeros((20, 20), dtype=bool)
        explored = np.zeros((20, 20), dtype=bool)
        explored[5:15, 5:15] = True
        return obstacles, explored


class _FakeAgent:
    def __init__(self) -> None:
        self.voxel_map = _FakeVoxelMap()


def test_export_voxel_observation_history_writes_jsonl(tmp_path: Path) -> None:
    agent = _FakeAgent()
    spawn = {"init_pose_csv": {"x": 1.0, "y": 0.0, "z": -3.0, "heading": 0.0}}
    agent._habitat_spawn_record = spawn
    out = export_voxel_observation_history(agent, tmp_path, spawn_record=spawn)
    assert "observations_history" in out
    lines = (tmp_path / "observations_history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    header = json.loads(lines[0])
    row = json.loads(lines[1])
    assert header["type"] == "header"
    assert header["spawn_record"] == spawn
    assert row["type"] == "observation"
    assert row["obs_idx"] == 0
    assert row["gps_grid_ij"] is not None
    assert row["camera_grid_ij"] is not None
    assert row["pcd_centroid_xz"] is not None
    assert row["n_world_points"] == 3

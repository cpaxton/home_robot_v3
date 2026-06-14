# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for :mod:`emet.eval.episode_diagnostics`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emet.eval.episode_diagnostics import (
    DIAGNOSTICS_MANIFEST,
    EpisodeDiagnosticsConfig,
    EpisodeDiagnosticsRecorder,
    flush_episode_diagnostics,
)


class _FakeVoxelMap:
    def __init__(self) -> None:
        self.grid_resolution = 0.1
        self.grid_origin = np.array([0.0, 0.0], dtype=np.float64)

    def get_2d_map(self):
        obstacles = np.zeros((20, 20), dtype=bool)
        explored = np.zeros((20, 20), dtype=bool)
        explored[5:15, 5:15] = True
        obstacles[10, 10] = True
        return obstacles, explored


class _FakeRobot:
    def get_base_pose(self):
        return np.array([1.0, 0.5, 0.0], dtype=np.float64)

    def get_observation(self):
        return None


class _FakeAgent:
    def __init__(self) -> None:
        self.robot = _FakeRobot()
        self.voxel_map = _FakeVoxelMap()
        self.graph_memory = None


def test_flush_writes_map_and_manifest(tmp_path: Path) -> None:
    agent = _FakeAgent()
    rec = EpisodeDiagnosticsRecorder(
        cfg=EpisodeDiagnosticsConfig(
            export_map=True,
            export_obstacle_grids=True,
            export_trajectory=True,
            export_rgb_frames=False,
            export_video=False,
            export_object_crops=False,
        )
    )
    rec.record_step(rgb=None, pose=(1.0, 0.5, 0.0), agent=agent, step_idx=0)
    manifest = flush_episode_diagnostics(tmp_path, agent, rec)
    assert (tmp_path / "topdown_map.png").is_file()
    assert (tmp_path / "obstacles_2d.npy").is_file()
    assert (tmp_path / "explored_2d.npy").is_file()
    assert (tmp_path / "grid_meta.json").is_file()
    assert (tmp_path / "trajectory.jsonl").is_file()
    assert (tmp_path / DIAGNOSTICS_MANIFEST).is_file()
    assert manifest.get("topdown_map")


def test_config_from_env_defaults_on() -> None:
    cfg = EpisodeDiagnosticsConfig.from_env()
    assert cfg.export_map is True
    assert cfg.export_video is True

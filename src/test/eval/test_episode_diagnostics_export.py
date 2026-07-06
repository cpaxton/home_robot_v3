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
    bind_diagnostics_recorder,
    flush_episode_diagnostics,
    unbind_diagnostics_recorder,
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


def test_flush_writes_map_without_dark_unknown_padding(tmp_path: Path) -> None:
    agent = _FakeAgent()
    rec = EpisodeDiagnosticsRecorder(
        cfg=EpisodeDiagnosticsConfig(
            export_map=True,
            export_obstacle_grids=False,
            export_trajectory=False,
            export_rgb_frames=False,
            export_video=False,
            export_object_crops=False,
        )
    )
    flush_episode_diagnostics(tmp_path, agent, rec)
    from PIL import Image

    img = np.asarray(Image.open(tmp_path / "topdown_map.png"))
    dark = np.all(img < np.uint8([40, 40, 40]), axis=-1)
    assert int(dark.sum()) == 0
    assert np.any(np.all(img == np.uint8([50, 160, 80]), axis=-1))


def test_flush_writes_map_video_when_stride_snapshots(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    agent = _FakeAgent()
    rec = EpisodeDiagnosticsRecorder(
        cfg=EpisodeDiagnosticsConfig(
            export_map=True,
            export_map_video=True,
            export_map_stride=1,
            export_map_overlay=True,
            export_obstacle_grids=False,
            export_trajectory=True,
            export_rgb_frames=False,
            export_video=False,
            export_object_crops=False,
            min_map_side=64,
            max_map_side=128,
        )
    )
    rec.record_step(rgb=None, pose=(1.0, 0.5, 0.0), agent=agent, step_idx=0)
    rec.record_step(rgb=None, pose=(1.2, 0.6, 0.1), agent=agent, step_idx=1)
    manifest = flush_episode_diagnostics(tmp_path, agent, rec)
    assert (tmp_path / "maps" / "step_0000.png").is_file()
    assert (tmp_path / "topdown_exploration.mp4").is_file()
    assert manifest.get("topdown_exploration_mp4")


def test_config_from_env_defaults_on() -> None:
    cfg = EpisodeDiagnosticsConfig.from_env()
    assert cfg.export_map is True
    assert cfg.export_video is True
    assert cfg.export_map_video is True


def test_bind_step_callback_records_from_agent() -> None:
    agent = _FakeAgent()
    rec = EpisodeDiagnosticsRecorder(
        cfg=EpisodeDiagnosticsConfig(
            export_map=False,
            export_trajectory=True,
            export_rgb_frames=False,
            export_video=False,
        )
    )
    bind_diagnostics_recorder(agent, rec, spawn_record={"init_pose_csv": {"x": 0.0}})
    assert rec.spawn_record == {"init_pose_csv": {"x": 0.0}}
    assert rec.record_from_agent in agent._on_step_callbacks
    for cb in agent._on_step_callbacks:
        cb(agent)
    assert len(rec._frames) == 1
    unbind_diagnostics_recorder(agent, rec)
    assert rec.record_from_agent not in getattr(agent, "_on_step_callbacks", [])


def test_map_stride_writes_under_episode_dir(tmp_path: Path) -> None:
    agent = _FakeAgent()
    rec = EpisodeDiagnosticsRecorder(
        cfg=EpisodeDiagnosticsConfig(
            export_map=True,
            export_map_stride=1,
            export_obstacle_grids=False,
            export_trajectory=False,
            export_rgb_frames=False,
            export_video=False,
            export_object_crops=False,
        )
    )
    rec.record_step(rgb=None, pose=(1.0, 0.5, 0.0), agent=agent, step_idx=0)
    rec.record_step(rgb=None, pose=(1.1, 0.5, 0.0), agent=agent, step_idx=1)
    manifest = flush_episode_diagnostics(tmp_path, agent, rec)
    assert (tmp_path / "maps" / "step_0000.png").is_file()
    assert (tmp_path / "maps" / "step_0001.png").is_file()
    assert manifest.get("map_stride_dir")


def test_record_habitat_substep_dedupes_identical_frames() -> None:
    rec = EpisodeDiagnosticsRecorder(
        cfg=EpisodeDiagnosticsConfig(
            export_map=False,
            export_trajectory=True,
            export_rgb_frames=True,
            export_video=False,
        )
    )
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    pose = (1.0, 2.0, 0.5)
    rec.record_habitat_substep(rgb=rgb, pose=pose)
    rec.record_habitat_substep(rgb=rgb.copy(), pose=pose)
    assert len(rec._frames) == 1

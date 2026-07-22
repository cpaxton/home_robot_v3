# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for scene map cache keying / completeness / agent wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from emet.config.sim_launch_config import SimLaunchMolmospaces, SimLaunchRobocasa
from emet.eval.scene_map_cache import (
    BUILD_MODE_GT,
    ensure_cached_map,
    has_cached_map,
    scene_cache_dir,
    scene_cache_key,
    scene_map_cache_enabled,
    write_cache_metadata,
)
from emet.memory.format import MANIFEST_FILENAME, VOXEL_PICKLE_FILENAME


def test_scene_cache_key_robocasa_deterministic():
    cfg = SimLaunchRobocasa(
        robot="stretch",
        robocasa_task="PickPlaceCounterToCabinet",
        robocasa_style=1,
        robocasa_layout=1,
        seed=0,
    )
    key = scene_cache_key(cfg)
    assert key == "robocasa_pickplacecountertocabinet_s1_l1_seed0_stretch_gt"
    assert scene_cache_key(cfg, build_mode=BUILD_MODE_GT) == key


def test_scene_cache_key_molmo_deterministic():
    cfg = SimLaunchMolmospaces(robot="stretch", scene="ithor", split="train", index=0, seed=0)
    key = scene_cache_key(cfg)
    assert key == "molmo_ithor_train_idx0_stretch_gt"


def test_has_cached_map_requires_manifest_and_voxel(tmp_path: Path):
    d = tmp_path / "empty"
    d.mkdir()
    assert not has_cached_map(d)
    (d / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    assert not has_cached_map(d)
    (d / VOXEL_PICKLE_FILENAME).write_bytes(b"pkl")
    assert has_cached_map(d)


def test_ensure_cached_map_local_hit(tmp_path: Path):
    key = "robocasa_pickplacecountertocabinet_s1_l1_seed0_stretch_gt"
    d = scene_cache_dir(key, root=tmp_path)
    d.mkdir(parents=True)
    (d / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    (d / VOXEL_PICKLE_FILENAME).write_bytes(b"pkl")
    hit = ensure_cached_map(key, root=tmp_path, try_download=False)
    assert hit == d


def test_ensure_cached_map_miss(tmp_path: Path):
    assert ensure_cached_map("missing_key_xyz", root=tmp_path, try_download=False) is None


def test_write_cache_metadata(tmp_path: Path):
    cfg = SimLaunchRobocasa(robot="stretch", seed=0)
    key = scene_cache_key(cfg)
    d = tmp_path / key
    d.mkdir()
    path = write_cache_metadata(d, cfg, key=key, build_params={"explore_steps": 8})
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert key in text
    assert "explore_steps" in text


def test_scene_map_cache_enabled_env(monkeypatch):
    monkeypatch.delenv("EMET_USE_SCENE_MAP_CACHE", raising=False)
    assert scene_map_cache_enabled() is True
    monkeypatch.setenv("EMET_USE_SCENE_MAP_CACHE", "0")
    assert scene_map_cache_enabled() is False
    assert scene_map_cache_enabled(override=True) is True


@patch("emet.controller.controller_dynagraph.DynagraphController")
def test_create_find_phase_agent_forwards_graph_memory_input_path(mock_cls):
    from emet.eval.ovmm_find_phase import create_find_phase_agent

    mock_agent = MagicMock()
    mock_cls.return_value = mock_agent
    create_find_phase_agent(
        MagicMock(),
        {},
        "dynagraph",
        graph_memory_input_path="/tmp/fake_cache",
    )
    assert mock_cls.call_args.kwargs["graph_memory_input_path"] == "/tmp/fake_cache"
    mock_agent.start.assert_called_once()

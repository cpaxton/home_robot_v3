# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from emet.config.eval_config import (
    load_eval_diagnostics_from_parameters,
    resolve_episode_diagnostics_config,
)
from emet.config.loader import default_config_path, load_config
from emet.core import get_parameters
from emet.eval.episode_diagnostics import EpisodeDiagnosticsConfig


def test_defaults_compose_eval_section():
    cfg = load_config(default_config_path())
    ev = cfg.raw.get("eval")
    assert isinstance(ev, dict)
    assert ev.get("export_map_video") is True
    assert ev.get("map_video_stride") == 5


def test_load_eval_from_parameters():
    cfg = load_eval_diagnostics_from_parameters({"eval": {"export_map_video": False, "map_video_stride": 3}})
    assert cfg.export_map_video is False
    assert cfg.map_video_stride == 3


def test_resolve_yaml_overrides_defaults():
    cfg = resolve_episode_diagnostics_config({"eval": {"export_video": False}})
    assert cfg.export_video is False
    assert cfg.export_map is True


def test_resolve_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("EMET_EVAL_EXPORT_MAP_VIDEO", "0")
    cfg = resolve_episode_diagnostics_config({"eval": {"export_map_video": True}})
    assert cfg.export_map_video is False


def test_resolve_compact_memory_env(monkeypatch):
    monkeypatch.setenv("EMET_EVAL_EXPORT_COMPACT_MEMORY", "1")
    monkeypatch.setenv("EMET_EVAL_EXPORT_WORLD_EVIDENCE_RGB", "0")

    cfg = resolve_episode_diagnostics_config()

    assert cfg.export_compact_memory is True
    assert cfg.export_world_evidence_rgb is False


def test_resolve_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("EMET_EVAL_EXPORT_VIDEO", "0")
    cfg = resolve_episode_diagnostics_config({"eval": {"export_video": False}}, export_video=True)
    assert cfg.export_video is True


def test_get_parameters_includes_eval_block():
    params = get_parameters("dynav_config.yaml")
    ev = params.get("eval")
    assert isinstance(ev, dict)
    assert "export_map_video" in ev


def test_dot_override_eval_export_map_video():
    params = get_parameters("dynav_config.yaml", overrides=["eval.export_map_video=false"])
    cfg = resolve_episode_diagnostics_config(params)
    assert cfg.export_map_video is False


def test_from_env_accepts_parameters():
    cfg = EpisodeDiagnosticsConfig.from_env({"eval": {"map_video_stride": 2}})
    assert cfg.map_video_stride == 2


def test_resolve_video_motion_paced_from_yaml():
    cfg = resolve_episode_diagnostics_config({"eval": {"video_motion_paced": False, "export_video_substeps": False}})
    assert cfg.video_motion_paced is False
    assert cfg.export_video_substeps is False
    assert cfg.video_meters_per_frame == 0.25

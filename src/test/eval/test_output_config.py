# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import pytest

from emet.config.eval_config import resolve_episode_diagnostics_config
from emet.eval.output_config import (
    EvalOutputConfig,
    eval_log_eqa_prep,
    eval_log_vl_progress,
    habitat_voxel_history_default,
    normalize_eval_output_profile,
    resolve_eval_output_profile,
)


def test_normalize_aliases():
    assert normalize_eval_output_profile("compact") == "lean"
    assert normalize_eval_output_profile("minimal") == "metrics"
    assert normalize_eval_output_profile("") == "full"


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="Unknown eval output profile"):
        normalize_eval_output_profile("verbose")


def test_lean_profile_disables_heavy_exports(monkeypatch):
    monkeypatch.setenv("EMET_EVAL_OUTPUT_PROFILE", "lean")
    cfg = resolve_episode_diagnostics_config()
    assert cfg.export_map is True
    assert cfg.export_trajectory is True
    assert cfg.export_rgb_frames is False
    assert cfg.export_video is False
    assert cfg.export_video_substeps is False
    assert cfg.export_map_video is False
    assert cfg.export_object_crops is False
    assert cfg.export_world_evidence_rgb is False
    assert cfg.export_voxel_history is False
    assert cfg.export_frontier_picks is False


def test_lean_per_flag_env_still_wins(monkeypatch):
    monkeypatch.setenv("EMET_EVAL_OUTPUT_PROFILE", "lean")
    monkeypatch.setenv("EMET_EVAL_EXPORT_FRAMES", "1")
    cfg = resolve_episode_diagnostics_config()
    assert cfg.export_rgb_frames is True
    assert cfg.export_video is False


def test_metrics_profile_drops_maps(monkeypatch):
    monkeypatch.setenv("EMET_EVAL_OUTPUT_PROFILE", "metrics")
    cfg = resolve_episode_diagnostics_config()
    assert cfg.export_map is False
    assert cfg.export_trajectory is False
    assert cfg.export_rgb_frames is False


def test_yaml_profile_lean():
    cfg = resolve_episode_diagnostics_config({"eval": {"profile": "lean", "export_map_video": True}})
    assert cfg.export_rgb_frames is False
    assert cfg.export_map_video is False


def test_output_config_logs_follow_profile(monkeypatch):
    monkeypatch.setenv("EMET_EVAL_OUTPUT_PROFILE", "lean")
    out = EvalOutputConfig.from_env()
    assert out.profile == "lean"
    assert out.log_vl_progress is False
    assert out.log_eqa_prep is False
    assert eval_log_vl_progress() is False
    assert eval_log_eqa_prep() is False


def test_log_vl_env_overrides_lean(monkeypatch):
    monkeypatch.setenv("EMET_EVAL_OUTPUT_PROFILE", "lean")
    monkeypatch.setenv("EMET_EVAL_LOG_VL", "1")
    assert eval_log_vl_progress() is True


def test_habitat_voxel_history_follows_lean(monkeypatch):
    monkeypatch.delenv("EMET_EVAL_EXPORT_VOXEL_HISTORY", raising=False)
    monkeypatch.setenv("EMET_EVAL_OUTPUT_PROFILE", "lean")
    assert habitat_voxel_history_default() is False
    monkeypatch.delenv("EMET_EVAL_OUTPUT_PROFILE", raising=False)
    assert habitat_voxel_history_default() is True


def test_resolve_profile_from_env(monkeypatch):
    monkeypatch.setenv("EMET_EVAL_OUTPUT_PROFILE", "metrics")
    assert resolve_eval_output_profile({"eval": {"profile": "lean"}}) == "metrics"

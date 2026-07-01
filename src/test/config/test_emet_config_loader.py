# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for unified nested config loader."""

from __future__ import annotations

from emet.config.loader import (
    apply_dot_overrides,
    default_config_path,
    load_config,
    merge_robot_overlay,
    normalize_legacy_yaml,
    parse_dot_override,
)


def test_default_config_path_points_at_repo_default():
    path = default_config_path()
    assert path.endswith("configs/emet/default.yaml")


def test_defaults_compose_mapping_and_agent():
    cfg = load_config(default_config_path())
    assert cfg.mapping_dict.get("voxel_size") == 0.1
    agent = cfg.agent_section()
    assert agent.llm == "qwen3-vl-eqa"


def test_legacy_flat_dynav_wraps_mapping():
    raw = {"voxel_size": 0.2, "depth_source": "sensor"}
    norm = normalize_legacy_yaml(raw)
    assert norm["mapping"]["voxel_size"] == 0.2
    assert "voxel_size" not in norm


def test_robot_overlay_merges_innate_mars_mapping():
    cfg = load_config(default_config_path())
    merged = merge_robot_overlay(cfg.raw, "innate_mars")
    assert merged["mapping"]["depth_source"] == "auto"
    assert merged["mapping"]["local_radius"] == 0.85
    filters = merged["mapping"]["filters"]
    assert filters["depth_speckle_open_kernel"] == 3
    assert filters["voxel_pcd_dbscan_min_samples"] == 8


def test_dot_override_wins_over_file():
    cfg = load_config(default_config_path(), overrides=["mapping.voxel_size=0.08"])
    assert cfg.mapping_dict["voxel_size"] == 0.08


def test_parse_dot_override_coerces_types():
    assert parse_dot_override("agent.eqa=true")[1] is True
    assert parse_dot_override("mapping.voxel_size=8")[1] == 8
    assert parse_dot_override('mapping.encoder="siglip"')[1] == "siglip"


def test_legacy_dynav_config_still_loads():
    cfg = load_config("dynav_config.yaml")
    assert cfg.mapping_dict.get("encoder") == "siglip"


def test_load_embodied_agent_overlay_resolves_extends():
    from emet.config.embodied_agent_config import load_embodied_agent_overlay

    cfg = load_embodied_agent_overlay("configs/agent_innate_mars.yaml")
    assert cfg.graph_eqa_memory.enabled is True
    assert cfg.open_vocab_scene_graph.enabled is True


def test_extends_agent_preset():
    cfg = load_config("configs/agent_innate_mars.yaml")
    assert cfg.robot == "innate_mars"
    assert cfg.embodied_agent().graph_eqa_memory.enabled is True


def test_apply_dot_overrides_nested():
    base = {"mapping": {"eqa": {"vl_family": "qwen3_vl"}}}
    out = apply_dot_overrides(base, ["mapping.eqa.vl_family=gemma4"])
    assert out["mapping"]["eqa"]["vl_family"] == "gemma4"

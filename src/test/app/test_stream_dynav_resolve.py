# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Dynav / config preset resolution for emet stream/capture."""

from emet.app.stream_agent_factory import (
    load_stream_parameters,
    resolve_stream_config_path,
    resolve_stream_dynav_config,
)
from emet.config.loader import default_config_path
from emet.robots import DEFAULT_DYNAV_CONFIG_YAML


def test_default_dynav_resolves_to_unified_config():
    resolved = resolve_stream_config_path(DEFAULT_DYNAV_CONFIG_YAML, dynav_from_default=True)
    assert resolved == default_config_path()


def test_innate_mars_remote_host_uses_auto_depth_via_overlay():
    config_path = resolve_stream_dynav_config(
        "innate_mars",
        "herman",
        DEFAULT_DYNAV_CONFIG_YAML,
        dynav_from_default=True,
    )
    params, _ = load_stream_parameters("innate_mars", "herman", config_path)
    assert config_path == default_config_path()
    assert str(params.get("depth_source", "")).lower() in ("auto", "da3")
    filters = params.get("filters") or {}
    assert int(filters.get("depth_speckle_open_kernel", 0) or 0) == 3
    assert int(filters.get("voxel_pcd_dbscan_min_samples", 0) or 0) == 8


def test_innate_mars_localhost_keeps_sensor_depth_when_sim():
    config_path = resolve_stream_dynav_config(
        "innate_mars",
        "127.0.0.1",
        DEFAULT_DYNAV_CONFIG_YAML,
        dynav_from_default=True,
    )
    params, _ = load_stream_parameters("innate_mars", "127.0.0.1", config_path)
    assert config_path == default_config_path()
    assert str(params.get("depth_source", "")).lower() in ("auto", "sensor")


def test_explicit_dynav_not_overridden():
    resolved = resolve_stream_dynav_config(
        "innate_mars",
        "herman",
        "dynav_config.yaml",
        dynav_from_default=False,
    )
    assert resolved == "dynav_config.yaml"


def test_legacy_alias_resolve_stream_dynav_compat():
    resolved = resolve_stream_dynav_config(
        "innate_mars",
        "127.0.0.1",
        DEFAULT_DYNAV_CONFIG_YAML,
        dynav_from_default=True,
    )
    assert resolved == default_config_path()

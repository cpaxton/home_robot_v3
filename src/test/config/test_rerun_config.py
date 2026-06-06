# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Tests for Draccus-based ``rerun`` YAML overlay."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import draccus

from emet.config.rerun_config import (
    RerunAgentConfig,
    build_rerun_visualizer_kwargs,
    load_rerun_agent_overlay,
    load_rerun_config_from_parameters,
)


def test_load_overlay_missing_key_defaults():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("voxel_size: 0.1\n")
        path = f.name
    try:
        cfg = load_rerun_agent_overlay(path)
        assert cfg.dynagraph.log_crops is False
        assert cfg.mjcf_show_visual_mesh is True
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_overlay_dynagraph_flags():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("rerun:\n  dynagraph:\n    log_edges: true\n")
        path = f.name
    try:
        cfg = load_rerun_agent_overlay(path)
        assert cfg.dynagraph.log_edges is True
        assert cfg.dynagraph.log_crops is False
    finally:
        Path(path).unlink(missing_ok=True)


def test_build_kwargs_yaml_overrides_env(monkeypatch):
    monkeypatch.setenv("EMET_DYNAGRAPH_RERUN_CROPS", "1")
    params = {"rerun": {"dynagraph": {"log_crops": False, "log_edges": True}}}
    kw = build_rerun_visualizer_kwargs(params)
    assert kw["dynagraph_rerun_crops"] is False
    assert kw["dynagraph_rerun_edges"] is True


def test_build_kwargs_cli_headless_wins():
    params = {"rerun": {"headless": False}}
    kw = build_rerun_visualizer_kwargs(params, cli_headless=True)
    assert kw["headless"] is True


def test_build_kwargs_yaml_headless_without_cli(monkeypatch):
    monkeypatch.delenv("RERUN_HEADLESS", raising=False)
    params = {"rerun": {"headless": True}}
    kw = build_rerun_visualizer_kwargs(params, cli_headless=False)
    assert kw["headless"] is True


def test_decode_empty_dict_all_defaults():
    cfg = draccus.decode(RerunAgentConfig, {})
    assert isinstance(cfg, RerunAgentConfig)
    assert cfg.server_memory_limit == "4GB"
    assert cfg.dynagraph.log_edges is False


def test_load_from_parameters_dict():
    cfg = load_rerun_config_from_parameters({"rerun": {"bind_all": True}})
    assert cfg.bind_all is True


def test_spatial3d_view_world_fixed_frame():
    from emet.visualization.rerun import (
        RERUN_SPATIAL3D_CONTENTS_WORLD,
        RERUN_SPATIAL3D_ORIGIN_WORLD,
        spatial3d_view_robot,
        spatial3d_view_world,
    )

    view = spatial3d_view_world()
    assert view.origin == RERUN_SPATIAL3D_ORIGIN_WORLD
    assert view.contents == RERUN_SPATIAL3D_CONTENTS_WORLD
    assert spatial3d_view_robot().origin == RERUN_SPATIAL3D_ORIGIN_WORLD


def test_build_kwargs_live_stride_defaults():
    kw = build_rerun_visualizer_kwargs({})
    assert kw["show_camera_point_clouds"] is False
    assert kw["voxel_map_stride"] == 2
    assert kw["mjcf_mesh_stride"] == 3


def test_build_kwargs_keys_match_rerun_visualizer_init():
    """Every key from build_rerun_visualizer_kwargs must be accepted by RerunVisualizer."""
    import inspect

    from emet.core import get_parameters
    from emet.visualization.rerun import RerunVisualizer

    kw = build_rerun_visualizer_kwargs(get_parameters("dynav_innate_mars.yaml"))
    params = inspect.signature(RerunVisualizer.__init__).parameters
    extra = set(kw) - set(params)
    assert not extra, f"unexpected RerunVisualizer kwargs: {extra}"

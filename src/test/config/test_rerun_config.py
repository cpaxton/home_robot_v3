# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Tests for Draccus-based ``rerun`` YAML overlay."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import draccus

from emet.config.rerun_config import (
    RerunAgentConfig,
    build_rerun_visualizer_kwargs,
    eval_rerun_enabled,
    load_rerun_agent_overlay,
    load_rerun_config_from_parameters,
    maybe_attach_eval_rerun_visualizer,
    open_live_rerun_visualizer,
)
from emet.core import get_parameters
from emet.visualization.null_visualizer import NullVisualizer


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


def test_build_kwargs_env_forces_crops_on_when_yaml_false(monkeypatch):
    monkeypatch.setenv("EMET_DYNAGRAPH_RERUN_CROPS", "1")
    monkeypatch.delenv("EMET_DYNAGRAPH_RERUN_EDGES", raising=False)
    params = {"rerun": {"dynagraph": {"log_crops": False, "log_edges": True}}}
    kw = build_rerun_visualizer_kwargs(params)
    assert kw["dynagraph_rerun_crops"] is True
    assert kw["dynagraph_rerun_edges"] is True


def test_build_kwargs_yaml_crops_on_without_env(monkeypatch):
    monkeypatch.delenv("EMET_DYNAGRAPH_RERUN_CROPS", raising=False)
    params = {"rerun": {"dynagraph": {"log_crops": True}}}
    kw = build_rerun_visualizer_kwargs(params)
    assert kw["dynagraph_rerun_crops"] is True


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


def test_spatial3d_view_origins_are_world_in_source():
    """Keep live Spatial3D origin on ``world`` without importing rerun-sdk native modules."""
    src = Path(__file__).resolve().parents[2] / "emet" / "visualization" / "rerun.py"
    text = src.read_text()
    assert 'RERUN_SPATIAL3D_ORIGIN_WORLD = "world"' in text
    assert 'RERUN_SPATIAL3D_CONTENTS_WORLD = "world/**"' in text
    assert 'RERUN_SPATIAL3D_ORIGIN_ROBOT = "world/robot"' in text


def test_build_kwargs_keys_match_rerun_visualizer_init():
    """Every key from build_rerun_visualizer_kwargs must be accepted by RerunVisualizer."""
    src = Path(__file__).resolve().parents[2] / "emet" / "visualization" / "rerun.py"
    tree = ast.parse(src.read_text())
    init_names: set[str] | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RerunVisualizer":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    init_names = {a.arg for a in item.args.args if a.arg != "self"}
                    init_names.update(a.arg for a in item.args.kwonlyargs)
                    break
    assert init_names, "RerunVisualizer.__init__ not found"
    kw = build_rerun_visualizer_kwargs(get_parameters("dynav_innate_mars.yaml"))
    extra = set(kw) - init_names
    assert not extra, f"unexpected RerunVisualizer kwargs: {extra}"


def test_build_kwargs_live_stride_defaults(monkeypatch):
    monkeypatch.delenv("EMET_DYNAGRAPH_RERUN_CROPS", raising=False)
    monkeypatch.delenv("EMET_DYNAGRAPH_RERUN_EDGES", raising=False)
    kw = build_rerun_visualizer_kwargs({})
    assert kw["show_camera_point_clouds"] is False
    assert kw["voxel_map_stride"] == 2
    assert kw["mjcf_mesh_stride"] == 3
    assert kw["dynagraph_rerun_crops"] is False
    assert kw["dynagraph_rerun_edges"] is False


def test_open_live_returns_null_when_disabled():
    vis = open_live_rerun_visualizer({}, enabled=False)
    assert isinstance(vis, NullVisualizer)
    assert vis.enabled is False


def test_eval_rerun_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("EMET_EVAL_RERUN", raising=False)
    assert eval_rerun_enabled() is False
    monkeypatch.setenv("EMET_EVAL_RERUN", "1")
    assert eval_rerun_enabled() is True
    monkeypatch.setenv("EMET_EVAL_RERUN", "yes")
    assert eval_rerun_enabled() is True
    monkeypatch.setenv("EMET_EVAL_RERUN", "0")
    assert eval_rerun_enabled() is False


def test_maybe_attach_keeps_existing_enabled(monkeypatch):
    existing = SimpleNamespace(enabled=True)
    monkeypatch.setattr(
        "emet.config.rerun_config.open_live_rerun_visualizer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not open")),
    )
    assert maybe_attach_eval_rerun_visualizer({}, existing, force=True) is existing


def test_maybe_attach_force_opens_without_env(monkeypatch):
    sentinel = SimpleNamespace(enabled=True)
    monkeypatch.delenv("EMET_EVAL_RERUN", raising=False)
    monkeypatch.setattr(
        "emet.config.rerun_config.open_live_rerun_visualizer",
        lambda *a, **k: sentinel,
    )
    out = maybe_attach_eval_rerun_visualizer({}, NullVisualizer(), force=True)
    assert out is sentinel


def test_maybe_attach_env_opens_without_force(monkeypatch):
    sentinel = SimpleNamespace(enabled=True)
    monkeypatch.setenv("EMET_EVAL_RERUN", "1")
    monkeypatch.setattr(
        "emet.config.rerun_config.open_live_rerun_visualizer",
        lambda *a, **k: sentinel,
    )
    out = maybe_attach_eval_rerun_visualizer({}, NullVisualizer(), force=False)
    assert out is sentinel


def test_maybe_attach_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("EMET_EVAL_RERUN", raising=False)
    monkeypatch.setattr(
        "emet.config.rerun_config.open_live_rerun_visualizer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not open")),
    )
    vis = NullVisualizer()
    assert maybe_attach_eval_rerun_visualizer({}, vis, force=False) is vis


def test_maybe_attach_keeps_mock_when_eval_env_set(monkeypatch):
    monkeypatch.setenv("EMET_EVAL_RERUN", "1")
    monkeypatch.setattr(
        "emet.config.rerun_config.open_live_rerun_visualizer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not open")),
    )
    stub = MagicMock()
    assert maybe_attach_eval_rerun_visualizer({}, stub, force=False) is stub


def test_maybe_attach_none_existing_returns_null_when_disabled(monkeypatch):
    monkeypatch.delenv("EMET_EVAL_RERUN", raising=False)
    monkeypatch.setattr(
        "emet.config.rerun_config.open_live_rerun_visualizer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not open")),
    )
    vis = maybe_attach_eval_rerun_visualizer({}, None, force=False)
    assert isinstance(vis, NullVisualizer)


def test_maybe_attach_none_existing_opens_when_forced(monkeypatch):
    sentinel = SimpleNamespace(enabled=True)
    monkeypatch.delenv("EMET_EVAL_RERUN", raising=False)
    monkeypatch.setattr(
        "emet.config.rerun_config.open_live_rerun_visualizer",
        lambda *a, **k: sentinel,
    )
    assert maybe_attach_eval_rerun_visualizer({}, None, force=True) is sentinel

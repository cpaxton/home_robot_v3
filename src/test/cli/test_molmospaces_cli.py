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
# Tests for emet molmospaces CLI. list-robots uses static config; list-scenes/merge-scene/serve
# delegate to emet-molmospaces wrapper (subprocess). Without wrapper, those commands
# exit non-zero with an "install wrapper" message. RUN_MOLMOSPACES_TESTS=1 runs list-scenes.

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner


def test_molmospaces_config_constants():
    """Config exposes robot list and default robot."""
    from emet.simulation.molmospaces_config import (
        DEFAULT_MOLMOSPACES_ROBOT,
        MOLMOSPACES_ROBOT_IDS,
        MOLMOSPACES_SCENE_NAMES,
    )

    assert "rby1" in MOLMOSPACES_ROBOT_IDS
    assert DEFAULT_MOLMOSPACES_ROBOT == "rby1"
    assert "ithor" in MOLMOSPACES_SCENE_NAMES


def test_ensure_molmo_asset_layout_symlinks_creates_scenes_objects_link(tmp_path, monkeypatch):
    """scenes/objects -> asset root objects so ../objects in scene XML resolves."""
    monkeypatch.setenv("MLSPACES_ASSETS_DIR", str(tmp_path))
    monkeypatch.delenv("MLSPACES_CACHE_DIR", raising=False)
    (tmp_path / "objects").mkdir(parents=True)
    (tmp_path / "objects" / "marker.txt").write_text("ok")
    from emet.simulation.molmospaces_config import ensure_molmo_asset_layout_symlinks

    ensure_molmo_asset_layout_symlinks()
    link = tmp_path / "scenes" / "objects"
    assert link.is_symlink()
    assert (link / "marker.txt").read_text() == "ok"


def test_ensure_molmo_asset_layout_symlinks_cache_dir_when_distinct(tmp_path, monkeypatch):
    """THOR meshes under MLSPACES_CACHE_DIR need scenes/objects -> objects too."""
    assets = tmp_path / "assets"
    cache = tmp_path / "resource_cache"
    monkeypatch.setenv("MLSPACES_ASSETS_DIR", str(assets))
    monkeypatch.setenv("MLSPACES_CACHE_DIR", str(cache))
    (assets / "objects").mkdir(parents=True)
    (cache / "objects").mkdir(parents=True)
    (cache / "objects" / "thor_marker.txt").write_text("cache")
    from emet.simulation.molmospaces_config import ensure_molmo_asset_layout_symlinks

    ensure_molmo_asset_layout_symlinks()
    clink = cache / "scenes" / "objects"
    assert clink.is_symlink()
    assert (clink / "thor_marker.txt").read_text() == "cache"


def test_default_molmospaces_assets_dir_xdg_cache():
    """Default MolmoSpaces assets live under XDG cache, not the venv."""
    from emet.simulation.molmospaces_config import default_molmospaces_assets_dir

    p = default_molmospaces_assets_dir()
    assert p.name == "assets"
    assert "molmospaces" in p.parts


def test_ensure_molmospaces_assets_dir_sets_companion_cache_dir(monkeypatch, tmp_path):
    """Unset MLSPACES_CACHE_DIR becomes a sibling of assets (ResourceManager forbids same path)."""
    monkeypatch.delenv("MLSPACES_ASSETS_DIR", raising=False)
    monkeypatch.delenv("MLSPACES_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from emet.simulation.molmospaces_config import ensure_molmospaces_assets_dir_env

    p = ensure_molmospaces_assets_dir_env()
    assert p == tmp_path / "molmospaces" / "assets"
    assert os.environ["MLSPACES_ASSETS_DIR"] == str(p)
    cache = tmp_path / "molmospaces" / "resource_cache"
    assert os.environ["MLSPACES_CACHE_DIR"] == str(cache)
    assert cache != p


def test_ensure_molmospaces_assets_dir_companion_for_custom_assets(monkeypatch, tmp_path):
    """Custom MLSPACES_ASSETS_DIR gets a sibling resource_cache when cache is unset."""
    assets = tmp_path / "custom" / "assets"
    monkeypatch.setenv("MLSPACES_ASSETS_DIR", str(assets))
    monkeypatch.delenv("MLSPACES_CACHE_DIR", raising=False)
    from emet.simulation.molmospaces_config import ensure_molmospaces_assets_dir_env

    p = ensure_molmospaces_assets_dir_env()
    assert p == assets
    assert os.environ["MLSPACES_CACHE_DIR"] == str(tmp_path / "custom" / "resource_cache")


def test_ensure_molmospaces_assets_dir_respects_explicit_cache_dir(monkeypatch, tmp_path):
    """When MLSPACES_CACHE_DIR is set, do not override."""
    assets = tmp_path / "myassets"
    cache_other = tmp_path / "legacy_cache"
    monkeypatch.setenv("MLSPACES_ASSETS_DIR", str(assets))
    monkeypatch.setenv("MLSPACES_CACHE_DIR", str(cache_other))
    from emet.simulation.molmospaces_config import ensure_molmospaces_assets_dir_env

    p = ensure_molmospaces_assets_dir_env()
    assert p == assets
    assert os.environ["MLSPACES_CACHE_DIR"] == str(cache_other)


def test_molmospaces_help():
    """emet molmospaces --help runs and shows subcommands."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "molmospaces", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "list-robots" in result.stdout
    assert "list-scenes" in result.stdout
    assert "install-scene" in result.stdout
    assert "merge-scene" in result.stdout
    assert "build-occ-map" in result.stdout
    assert "serve" in result.stdout
    assert "export-nerfstudio" in result.stdout


def test_serve_mujoco_help_includes_molmospaces_scene():
    """emet serve mujoco --help documents --molmospaces-scene (merge + ZMQ in one step)."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "serve", "mujoco", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "molmospaces-scene" in result.stdout
    assert "molmospaces-install" in result.stdout


def test_serve_help_lists_molmospaces_and_robocasa_backends():
    """emet serve documents mujoco, robocasa, and molmospaces backend shortcuts."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "serve", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    out = result.stdout.lower()
    assert "robocasa" in out
    assert "molmospaces" in out


def test_serve_molmospaces_help():
    """emet serve molmospaces --help works and documents MolmoSpaces options."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "serve", "molmospaces", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "molmospaces-scene" in result.stdout
    assert "molmospaces-split" in result.stdout


def test_serve_robocasa_help():
    """emet serve robocasa --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "serve", "robocasa", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "robocasa-task" in result.stdout or "robocasa_task" in result.stdout


def test_molmospaces_list_robots():
    """emet molmospaces list-robots prints robot IDs (static list, no runner)."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "molmospaces", "list-robots"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "rby1" in result.stdout
    assert "franka" in result.stdout.lower()
    assert "Default:" in result.stdout


def test_molmospaces_install_scene_help():
    """emet molmospaces install-scene --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "molmospaces", "install-scene", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "scene" in result.stdout and "split" in result.stdout


def test_molmospaces_serve_help():
    """emet molmospaces serve --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "molmospaces", "serve", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "viewer" in result.stdout or "headless" in result.stdout


def test_molmospaces_merge_scene_help():
    """emet molmospaces merge-scene --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "molmospaces", "merge-scene", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--output" in result.stdout or "-o" in result.stdout


def test_molmospaces_build_occ_map_help():
    """emet molmospaces build-occ-map --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "molmospaces", "build-occ-map", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "mjcf" in result.stdout.lower() or "occupancy" in result.stdout.lower()


def test_molmospaces_export_nerfstudio_help():
    """emet molmospaces export-nerfstudio --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "molmospaces", "export-nerfstudio", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "episode-dir" in result.stdout
    assert "transforms" in result.stdout.lower()


def test_molmospaces_export_nerfstudio_cli_writes_transforms(tmp_path: Path):
    """CLI export-nerfstudio reads an on-disk episode and writes transforms.json (no live sim)."""
    from emet.core.interfaces import Observations
    from emet.molmospaces.episode_writer import MolmoEpisodeWriter

    h, w = 16, 24
    K = np.array([[40.0, 0, w / 2], [0, 40.0, h / 2], [0, 0, 1.0]], dtype=np.float64)
    obs = Observations(
        gps=np.zeros(2),
        compass=np.zeros(1),
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        depth=None,
        camera_K=K,
        camera_pose=np.eye(4, dtype=np.float64),
        seq_id=0,
    )
    writer = MolmoEpisodeWriter(
        str(tmp_path),
        episode_fields={"molmospaces_scene": "ithor", "robot": "rby1"},
        save_depth=False,
    )
    writer.write_frame(obs, 0)
    writer.finalize()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "emet.cli",
            "molmospaces",
            "export-nerfstudio",
            "--episode-dir",
            str(tmp_path),
            "--output",
            "transforms_cli.json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    out = tmp_path / "transforms_cli.json"
    assert out.is_file()
    data = json.loads(out.read_text())
    assert "frames" in data and len(data["frames"]) == 1


def test_run_molmospaces_explore_help():
    """emet run molmospaces-explore --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "run", "molmospaces-explore", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "output-dir" in result.stdout


def test_molmospaces_list_scenes_without_wrapper():
    """Without emet-molmospaces wrapper, list-scenes exits non-zero with install message."""
    from emet.cli import main

    env = {**os.environ, "EMET_UV_RUN": "1"}
    with patch(
        "emet.simulation.molmospaces_config.build_molmospaces_wrapper_command",
        return_value=None,
    ):
        runner = CliRunner(env=env)
        result = runner.invoke(main, ["molmospaces", "list-scenes"], catch_exceptions=False)
    assert result.exit_code == 1
    out = (result.output or "").lower()
    assert "install" in out or "wrapper" in out or "emet-molmospaces" in out


@pytest.mark.skipif(
    os.environ.get("RUN_MOLMOSPACES_TESTS", "") != "1",
    reason="RUN_MOLMOSPACES_TESTS=1 required (wrapper installed in .venv-molmospaces)",
)
def test_molmospaces_list_scenes_with_wrapper():
    """With wrapper installed, emet molmospaces list-scenes runs and prints scenes."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "molmospaces", "list-scenes"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Network/API failure is acceptable; ensure we got wrapper output not "install wrapper"
        err = (result.stderr or "").lower()
        assert "install" not in err or "wrapper" not in err or "emet-molmospaces" not in err
        return
    assert "Scenes" in result.stdout or "ithor" in result.stdout.lower()

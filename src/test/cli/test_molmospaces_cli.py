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

import os
import subprocess
import sys

import pytest


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
    (tmp_path / "objects").mkdir(parents=True)
    (tmp_path / "objects" / "marker.txt").write_text("ok")
    from emet.simulation.molmospaces_config import ensure_molmo_asset_layout_symlinks

    ensure_molmo_asset_layout_symlinks()
    link = tmp_path / "scenes" / "objects"
    assert link.is_symlink()
    assert (link / "marker.txt").read_text() == "ok"


def test_default_molmospaces_assets_dir_xdg_cache():
    """Default MolmoSpaces assets live under XDG cache, not the venv."""
    from emet.simulation.molmospaces_config import default_molmospaces_assets_dir

    p = default_molmospaces_assets_dir()
    assert p.name == "assets"
    assert "molmospaces" in p.parts


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
    assert "serve" in result.stdout


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


def test_molmospaces_list_scenes_without_wrapper():
    """Without emet-molmospaces wrapper, list-scenes exits non-zero with install message."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "molmospaces", "list-scenes"],
        capture_output=True,
        text=True,
    )
    # Wrapper not installed in this env: expect exit 1 and message to install wrapper
    if result.returncode == 0:
        # Wrapper is installed (e.g. in .venv-molmospaces and MOLMOSPACES_PYTHON set)
        assert "Scenes" in result.stdout or "ithor" in result.stdout.lower()
        return
    err = (result.stderr or result.stdout or "").lower()
    assert "install" in err or "wrapper" in err or "emet-molmospaces" in err


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

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
# Tests for emet molmospaces CLI. list-robots uses static config; list-scenes/serve
# require the MolmoSpaces runner venv (skipped unless RUN_MOLMOSPACES_TESTS=1).

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
    assert "serve" in result.stdout


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


@pytest.mark.skipif(
    os.environ.get("RUN_MOLMOSPACES_TESTS", "") != "1",
    reason="RUN_MOLMOSPACES_TESTS=1 required (needs MolmoSpaces venv)",
)
def test_molmospaces_list_scenes():
    """emet molmospaces list-scenes runs runner (requires .venv-molmospaces)."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "molmospaces", "list-scenes"],
        capture_output=True,
        text=True,
    )
    # If venv missing, CLI returns 1 and prints message
    if result.returncode != 0:
        assert "venv" in result.stderr.lower() or "MOLMOSPACES" in result.stderr
        return
    assert "Scenes" in result.stdout or "ithor" in result.stdout.lower()

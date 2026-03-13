# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# Tests for emet_molmospaces CLI; molmo_spaces is mocked.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def test_wrapper_cli_help():
    """Wrapper responds to list-scenes --help (command required)."""
    from emet_molmospaces.runner import main_runner

    # argparse --help raises SystemExit(0); avoid exiting pytest
    with pytest.raises(SystemExit) as exc_info:
        main_runner(["list-scenes", "--help"])
    assert exc_info.value.code == 0


def test_wrapper_list_scenes_mocked():
    """list-scenes with mocked molmo_spaces returns 0 and prints scene table."""
    from emet_molmospaces.runner import run_list_scenes

    def mock_get_scenes(name: str, split: str):
        return {split: []}  # empty list per split

    with patch("emet_molmospaces.runner._get_molmo_api") as m:
        m.return_value = (mock_get_scenes, lambda path: None)
        code = run_list_scenes()
    assert code == 0


def test_wrapper_list_scenes_via_main_runner_mocked():
    """main_runner(['list-scenes']) with mocked API returns 0."""
    from emet_molmospaces.runner import main_runner

    def mock_get_scenes(name: str, split: str):
        return {split: []}

    with patch("emet_molmospaces.runner._get_molmo_api") as m:
        m.return_value = (mock_get_scenes, lambda path: None)
        code = main_runner(["list-scenes"])
    assert code == 0


def test_wrapper_install_scene_help():
    """install-scene --help exits 0."""
    from emet_molmospaces.runner import main_runner

    with pytest.raises(SystemExit) as exc_info:
        main_runner(["install-scene", "--help"])
    assert exc_info.value.code == 0


def test_wrapper_serve_help():
    """serve --help exits 0."""
    from emet_molmospaces.runner import main_runner

    with pytest.raises(SystemExit) as exc_info:
        main_runner(["serve", "--help"])
    assert exc_info.value.code == 0


def test_wrapper_console_script_help():
    """Console script emet-molmospaces list-scenes --help works when package is installed."""
    exe = Path(sys.executable).parent / "emet-molmospaces"
    if not exe.exists():
        pytest.skip("emet-molmospaces script not in this env (run from wrapper venv)")
    result = subprocess.run(
        [str(exe), "list-scenes", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "list-scenes" in result.stdout or "scene" in result.stdout.lower()

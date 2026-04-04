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
from unittest.mock import MagicMock, patch

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


def test_wrapper_merge_scene_help():
    """merge-scene --help exits 0."""
    from emet_molmospaces.runner import main_runner

    with pytest.raises(SystemExit) as exc_info:
        main_runner(["merge-scene", "--help"])
    assert exc_info.value.code == 0


def test_xml_path_for_scene_index_legacy_list():
    from emet_molmospaces.runner import _xml_path_for_scene_index

    m = ["a.xml", "b.xml"]
    assert _xml_path_for_scene_index("procthor-10k", m, "train", 0) == "a.xml"
    assert _xml_path_for_scene_index("procthor-10k", m, "train", 1) == "b.xml"
    assert _xml_path_for_scene_index("procthor-10k", m, "train", 2) is None


def test_xml_path_for_scene_index_ithor_style():
    from emet_molmospaces.runner import _xml_path_for_scene_index

    # iTHOR map keys follow FloorPlan{N}; emet --index 0 maps to key 1 (first house).
    m = {"train": {1: "/p/FloorPlan1_physics.xml", 2: None}, "val": {}}
    assert _xml_path_for_scene_index("ithor", m, "train", 0) == "/p/FloorPlan1_physics.xml"
    assert _xml_path_for_scene_index("ithor", m, "train", 1) is None
    assert _xml_path_for_scene_index("ithor", m, "val", 0) is None


def test_xml_path_for_scene_index_procthor_variants():
    from emet_molmospaces.runner import _xml_path_for_scene_index

    m = {
        "train": {
            0: {"base": "train_0.xml", "ceiling": None, "map": None},
            1: {"ceiling": "c.xml"},
        }
    }
    assert _xml_path_for_scene_index("procthor-10k", m, "train", 0) == "train_0.xml"
    assert _xml_path_for_scene_index("procthor-10k", m, "train", 1) == "c.xml"


def test_split_scene_count():
    from emet_molmospaces.runner import _split_scene_count

    assert _split_scene_count({"train": {0: "a", 1: "b"}}, "train") == 2
    assert _split_scene_count({"train": []}, "train") == 0
    assert _split_scene_count(["x", "y"], "train") == 2


def test_install_scene_with_deps_ithor_includes_thor_objects():
    """iTHOR must call upstream with exclude_thor=False so kitchen object meshes are installed."""
    from emet_molmospaces.runner import _install_scene_with_deps

    install_fn = MagicMock()
    with patch("emet_molmospaces.runner._get_molmo_api", return_value=(None, install_fn)):
        _install_scene_with_deps("/tmp/scene.xml", "ithor")
    assert install_fn.call_args.kwargs.get("exclude_thor") is False

    install_fn.reset_mock()
    with patch("emet_molmospaces.runner._get_molmo_api", return_value=(None, install_fn)):
        _install_scene_with_deps("/tmp/scene.xml", "procthor-10k")
    assert install_fn.call_args.kwargs.get("exclude_thor") is True


def test_merge_scene_without_output_fails():
    """merge-scene without --output returns 1."""
    from emet_molmospaces.runner import main_runner

    code = main_runner(["merge-scene", "--scene", "ithor"])
    assert code == 1


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

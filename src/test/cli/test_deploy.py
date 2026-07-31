# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Unit tests for robot deploy helpers and ``emet deploy llm`` CLI."""

from click.testing import CliRunner

from emet.deploy import build_remote_bridge_import_verify_cmd
from emet.deploy_llm import CALIBAN_ORIN_VRAM_GIB, LLM_PROFILES, deploy_llm


def test_remote_bridge_import_verify_cmd_includes_emet_core_and_bridge():
    cmd = build_remote_bridge_import_verify_cmd(
        remote_emet="~/emet",
        remote_ws="~/innate-os/ros2_ws",
    )
    assert "~/emet/emet_core" in cmd
    assert "innate_mars_bridge.ros.camera" in cmd
    assert "emet.utils.image" in cmd
    assert "emet.core.server" in cmd
    assert "colcon" not in cmd
    assert '-c "import innate_mars_bridge' in cmd
    assert "-c 'import" not in cmd


def test_caliban_orin_vram_constant() -> None:
    assert CALIBAN_ORIN_VRAM_GIB >= 60
    assert "unified-7b" in LLM_PROFILES


def test_deploy_llm_help_lists_profiles_and_vram() -> None:
    from emet.cli import main

    r = CliRunner().invoke(main, ["deploy", "llm", "--help"])
    assert r.exit_code == 0, r.output
    assert "unified-7b" in r.output
    assert "dual-2b" in r.output
    assert "64" in r.output or "60" in r.output


def test_deploy_group_help_lists_llm() -> None:
    from emet.cli import main

    r = CliRunner().invoke(main, ["deploy", "--help"])
    assert r.exit_code == 0, r.output
    assert "llm" in r.output


def test_deploy_llm_missing_script_returns_error(tmp_path) -> None:
    rc = deploy_llm(host="caliban", profile="unified-7b", root=tmp_path)
    assert rc == 1

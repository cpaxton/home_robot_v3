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

"""Defaults and CLI wiring for emet.app.run_agent (no heavy LLM load)."""

from click.testing import CliRunner


def test_help_lists_discord_toggle():
    """Discord is on by default; --no-discord opts out."""
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "--no-discord" in r.output


def test_help_lists_offline_and_default_robot_ip():
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "--offline" in r.output
    assert "127.0.0.1" in r.output


def test_help_lists_command_long_and_short():
    """Scripted runs use -c / --command (repeatable)."""
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "--command" in r.output
    assert "-c" in r.output


def test_help_lists_dynamem_eqa_flag():
    """DynaMem EQA is opt-in via --eqa (heavy models)."""
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "--eqa" in r.output


def test_vl_camera_default_logic():
    """Mirror run_agent: VL model names enable camera unless --no-vl-camera."""

    def vl_include_effective(no_vl_camera: bool, vl_include_camera: bool, llm: str) -> bool:
        llm_l = llm.lower()
        is_vl_name = "-vl-" in llm_l or "vl-" in llm_l
        return (not no_vl_camera) and (vl_include_camera or is_vl_name)

    assert vl_include_effective(False, False, "qwen35-vl-9B")
    assert vl_include_effective(False, False, "qwen25-VL-7B")
    assert not vl_include_effective(True, True, "qwen35-vl-9B")
    assert not vl_include_effective(False, False, "qwen35-9B")

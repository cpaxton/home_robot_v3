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

import pytest
from click.testing import CliRunner


def test_help_lists_unified_config():
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "--config" in r.output or "-C" in r.output
    assert "--set" in r.output or "-O" in r.output


def test_help_lists_discord_toggle():
    """Discord is on by default; --no-discord opts out."""
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "--no-discord" in r.output


def test_help_robot_option_mentions_molmospaces_rby1():
    """--robot help documents MolmoSpaces → rby1 server remap."""
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "MolmoSpaces" in r.output
    assert "rby1" in r.output


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
    assert "Discord" in r.output


def test_command_mode_disables_discord_warns_without_no_discord(monkeypatch: pytest.MonkeyPatch) -> None:
    from emet.app import run_agent as ra

    captured: list[bool] = []

    def stub(**kw: object) -> None:
        captured.append(bool(kw["discord"]))

    monkeypatch.setattr(ra, "run_agent_with_robot", stub)
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--robot", "stretch", "--no-llm", "-c", "E"])
    assert r.exit_code == 0, r.output
    assert captured == [False]
    assert "Discord is disabled" in r.output
    assert "Discord" in r.output


def test_command_mode_disables_discord_no_warning_with_no_discord(monkeypatch: pytest.MonkeyPatch) -> None:
    from emet.app import run_agent as ra

    captured: list[bool] = []

    def stub(**kw: object) -> None:
        captured.append(bool(kw["discord"]))

    monkeypatch.setattr(ra, "run_agent_with_robot", stub)
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--robot", "stretch", "--no-llm", "-c", "E", "--no-discord"])
    assert r.exit_code == 0, r.output
    assert captured == [False]
    assert "Warning" not in r.output


def test_help_lists_dynamem_eqa_flag():
    """DynaMem EQA is opt-in via --eqa (heavy models)."""
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "--eqa" in r.output


def test_help_lists_share_memory_vllm_toggle():
    """Agent + --eqa can share one VL model with DynaMem (--no-share-memory-vllm opts out)."""
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "--share-memory-vllm" in r.output
    assert "--no-share-memory-vllm" in r.output
    assert "--sim-show-subprocess-output" in r.output


def test_help_lists_rerun_agent_flags():
    """Rerun is opt-in on the agent (--rerun); headless/bind flags documented."""
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "--rerun" in r.output
    assert "--headless" in r.output
    assert "--rerun-bind" in r.output


def test_default_llm_is_qwen3_vl_eqa():
    from emet.agent.loop import DEFAULT_AGENT_LLM
    from emet.app.run_agent import main

    assert DEFAULT_AGENT_LLM == "qwen3-vl-eqa"
    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "qwen3-vl-eqa" in r.output


def test_vl_camera_default_logic():
    """Mirror run_agent: VL model names enable camera unless --no-vl-camera."""

    def vl_include_effective(no_vl_camera: bool, vl_include_camera: bool, llm: str) -> bool:
        llm_l = llm.lower()
        is_vl_name = "-vl-" in llm_l or "vl-" in llm_l
        return (not no_vl_camera) and (vl_include_camera or is_vl_name)

    assert vl_include_effective(False, False, "qwen3-vl-eqa")
    assert vl_include_effective(False, False, "qwen35-vl-9B")
    assert vl_include_effective(False, False, "qwen25-VL-7B")
    assert not vl_include_effective(True, True, "qwen35-vl-9B")
    assert not vl_include_effective(False, False, "qwen35-9B")

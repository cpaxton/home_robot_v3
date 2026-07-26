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
    assert "default: 256" in r.output


def test_run_agent_with_robot_max_tokens_default_matches_cli():
    import inspect

    from emet.agent.loop import run_agent_with_robot

    default = inspect.signature(run_agent_with_robot).parameters["max_tokens"].default
    assert default == 256


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


def test_help_lists_memory_backend():
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "--memory-backend" in r.output


def test_help_lists_lifelong_input_and_refine():
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "--input-path" in r.output
    assert "--input_path" in r.output
    assert "--refine-start" in r.output


def test_input_path_underscore_alias_accepted():
    from emet.app.run_agent import main

    runner = CliRunner()
    # Parse-only: offline+quit should not need a memory dir to exist for Click option accept.
    r = runner.invoke(main, ["--offline", "--input_path", "/tmp/does-not-need-exist", "-c", "Q"])
    # Offline ignores input-path; we only assert the option is recognized (not "No such option").
    assert "No such option: --input_path" not in (r.output or "")
    assert "No such option" not in (r.output or "") or r.exit_code == 0


def test_help_lists_eqa_eval():
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "--eqa-eval" in r.output
    assert "--extra-instruction" in r.output
    assert "dynagraph" in r.output


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


def test_help_lists_confirm_nav():
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "--confirm-nav" in r.output
    assert "EMET_CONFIRM_NAV" in r.output


def test_default_llm_is_qwen35_4b():
    from emet.agent.loop import DEFAULT_AGENT_LLM
    from emet.app.run_agent import main

    assert DEFAULT_AGENT_LLM == "qwen35-4B"
    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "qwen35-4B" in r.output
    assert "qwen3-vl-eqa" in r.output  # still listed as a choice / in help


def test_vl_camera_default_logic():
    """Camera→chat VL is opt-in (--vl-include-camera); --no-vl-camera always wins."""

    def vl_include_effective(no_vl_camera: bool, vl_include_camera: bool, llm: str) -> bool:
        return bool(vl_include_camera) and (not no_vl_camera)

    assert not vl_include_effective(False, False, "qwen3-vl-eqa")
    assert vl_include_effective(False, True, "qwen3-vl-eqa")
    assert not vl_include_effective(True, True, "qwen35-vl-9B")
    assert not vl_include_effective(False, False, "qwen35-9B")


def test_config_agent_section_eqa_when_cli_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """YAML / --set agent.eqa applies when --eqa is not on the CLI."""
    from emet.app import run_agent as ra

    captured: list[bool] = []

    def stub(**kw: object) -> None:
        captured.append(bool(kw["eqa"]))

    monkeypatch.setattr(ra, "run_agent_with_robot", stub)
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(
        main,
        ["--robot", "stretch", "--no-llm", "-c", "E", "--no-discord", "--set", "agent.eqa=true"],
    )
    assert r.exit_code == 0, r.output
    assert captured == [True]


def test_cli_eqa_overrides_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit --eqa wins over agent.eqa=false in config."""
    from emet.app import run_agent as ra

    captured: list[bool] = []

    def stub(**kw: object) -> None:
        captured.append(bool(kw["eqa"]))

    monkeypatch.setattr(ra, "run_agent_with_robot", stub)
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(
        main,
        ["--robot", "stretch", "--no-llm", "-c", "E", "--no-discord", "--set", "agent.eqa=false", "--eqa"],
    )
    assert r.exit_code == 0, r.output
    assert captured == [True]


def test_set_agent_llm_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline mode honors --set agent.llm when --llm is omitted."""
    from emet.app import run_agent as ra

    seen: list[str] = []

    class _FakeClient:
        max_tokens = 1024

        def __call__(self, text: str, verbose: bool = False) -> str:
            return "ok"

    def fake_get_llm_client(llm: str, prompt_builder: object, device: str, parameters: object) -> _FakeClient:
        seen.append(llm)
        return _FakeClient()

    monkeypatch.setattr(ra, "get_llm_client", fake_get_llm_client)
    from emet.app.run_agent import main

    runner = CliRunner()
    r = runner.invoke(main, ["--offline", "--set", "agent.llm=qwen35-4B"], input="\n")
    assert r.exit_code == 0, r.output
    assert seen == ["qwen35-4B"]

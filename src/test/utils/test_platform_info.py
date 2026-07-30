# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for Jetson / Tegra platform helpers and install profile wiring."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from emet.cli import main
from emet.utils.platform_info import is_aarch64, is_tegra, jetson_install_hints


def test_is_aarch64_matches_platform_machine(monkeypatch) -> None:
    monkeypatch.setattr("emet.utils.platform_info.platform.machine", lambda: "aarch64")
    is_aarch64.cache_clear()
    assert is_aarch64() is True
    monkeypatch.setattr("emet.utils.platform_info.platform.machine", lambda: "x86_64")
    is_aarch64.cache_clear()
    assert is_aarch64() is False


def test_is_tegra_force_env(monkeypatch) -> None:
    monkeypatch.setenv("EMET_FORCE_TEGRA", "1")
    is_tegra.cache_clear()
    assert is_tegra() is True


def test_is_tegra_on_this_host() -> None:
    """On Jetson CI/dev boxes nv_tegra_release exists; elsewhere False unless forced."""
    is_tegra.cache_clear()
    assert is_tegra() is Path("/etc/nv_tegra_release").is_file()


def test_jetson_install_hints_when_forced(monkeypatch) -> None:
    monkeypatch.setenv("EMET_FORCE_TEGRA", "1")
    is_tegra.cache_clear()
    hints = jetson_install_hints()
    assert hints
    assert any("install_jetson" in h for h in hints)


def test_install_full_help_lists_jetson_profile() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["install", "full", "--help"])
    assert result.exit_code == 0
    assert "jetson" in result.stdout.lower()


def test_install_jetson_script_exists() -> None:
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts" / "install_jetson.sh"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "--profile=jetson" in text

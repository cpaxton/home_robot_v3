# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Tests for the emet CLI."""

import subprocess
import sys

import pytest


def test_cli_help():
    """CLI --help runs without error."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "emet" in result.stdout.lower()
    assert "serve" in result.stdout
    assert "run" in result.stdout
    assert "sync" in result.stdout
    assert "install" in result.stdout
    assert "show" in result.stdout
    assert "test" in result.stdout


def test_cli_version():
    """CLI --version prints version."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "0.3" in result.stdout


def test_serve_help():
    """emet serve --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "serve", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "mujoco" in result.stdout
    assert "headless" in result.stdout


def test_run_help():
    """emet run --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "dynamem" in result.stdout
    assert "mapping" in result.stdout


def test_sync_help():
    """emet sync --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "sync", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--all" in result.stdout
    assert "--sim" in result.stdout
    assert "--dynamem" in result.stdout


def test_show_help():
    """emet show --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "show", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "rrd" in result.stdout.lower()


def test_test_help():
    """emet test --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "test", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "pytest" in result.stdout.lower()


def test_install_submodules_help():
    """emet install submodules --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install", "submodules", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "submodule" in result.stdout.lower()


def test_install_sim_help():
    """emet install sim --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install", "sim", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "robocasa" in result.stdout.lower() or "sim" in result.stdout.lower()


def test_install_full_help():
    """emet install full --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install", "full", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "install" in result.stdout.lower()


def test_install_pre_commit_help():
    """emet install pre-commit --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install", "pre-commit", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "pre-commit" in result.stdout.lower()


def test_install_completion_help():
    """emet install-completion --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install-completion", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "bash" in result.stdout or "zsh" in result.stdout


def test_install_completion_bash():
    """install-completion --shell bash produces valid script."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install-completion", "--shell", "bash"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "_emet_completion" in result.stdout or "emet" in result.stdout

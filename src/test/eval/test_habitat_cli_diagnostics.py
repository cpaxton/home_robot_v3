# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI help tests for emet-habitat diagnostics flags."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HAB_CLI = REPO / ".venv-habitat" / "bin" / "emet-habitat"


@pytest.mark.skipif(not HAB_CLI.is_file(), reason="habitat venv not installed")
def test_habitat_explore_frontiers_help():
    r = subprocess.run(
        [str(HAB_CLI), "explore-frontiers", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "--max-steps" in r.stdout


@pytest.mark.skipif(not HAB_CLI.is_file(), reason="habitat venv not installed")
def test_habitat_run_batch_help_lists_habitat_nav_flag():
    r = subprocess.run(
        [str(HAB_CLI), "run-batch", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "--no-habitat-perfect-nav" in r.stdout


@pytest.mark.skipif(not HAB_CLI.is_file(), reason="habitat venv not installed")
def test_habitat_run_batch_help_lists_diagnostics_flags():
    r = subprocess.run(
        [str(HAB_CLI), "run-batch", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "--export-map" in out
    assert "--export-video" in out
    assert "--map-stride" in out


@pytest.mark.skipif(not HAB_CLI.is_file(), reason="habitat venv not installed")
def test_habitat_run_ovmm_find_batch_help_lists_run_tag():
    r = subprocess.run(
        [str(HAB_CLI), "run-ovmm-find-batch", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "--run-tag" in r.stdout


@pytest.mark.skipif(not HAB_CLI.is_file(), reason="habitat venv not installed")
def test_habitat_run_ovmm_find_batch_help_lists_agentic_flags():
    r = subprocess.run(
        [str(HAB_CLI), "run-ovmm-find-batch", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    for flag in ("--agentic-find", "--no-agentic-find", "--agentic-max-rounds", "--agentic-max-nav-steps"):
        assert flag in r.stdout, f"missing {flag}"

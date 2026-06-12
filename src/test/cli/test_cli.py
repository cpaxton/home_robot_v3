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
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Tests for the emet CLI."""

import subprocess
import sys


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
    assert "debug-da3-depth" in result.stdout


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
    assert "dynagraph" in result.stdout
    assert "graph" in result.stdout and "eqa" in result.stdout
    assert "mapping" in result.stdout
    assert "debug-da3-depth" in result.stdout


def test_run_graph_eqa_help():
    """emet run graph-eqa --help works (GraphEQA app is wired)."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "run", "graph-eqa", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "robot" in result.stdout.lower() or "robot_ip" in result.stdout


def test_run_dynagraph_help():
    """emet run --help lists dynagraph; app module --help lists merge/staleness."""
    r1 = subprocess.run(
        [sys.executable, "-m", "emet.cli", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert r1.returncode == 0
    assert "dynagraph" in r1.stdout.lower()
    r2 = subprocess.run(
        [sys.executable, "-m", "emet.app.run_dynagraph", "--help"],
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 0
    out2 = (r2.stdout + r2.stderr).lower()
    assert "staleness" in out2 or "merge" in out2


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


def test_install_robocasa_help():
    """emet install robocasa --help works (same as install sim)."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install", "robocasa", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "robocasa" in result.stdout.lower()


def test_install_full_help():
    """emet install full --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install", "full", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "install" in result.stdout.lower()
    assert "--profile" in result.stdout


def test_install_menu_help():
    """emet install menu --help lists --text-only."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install", "menu", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--text-only" in result.stdout


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


def test_sqa3d_help():
    """emet sqa3d --help lists embodied subcommands."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "sqa3d", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    out = result.stdout.lower()
    assert "run-episode" in out
    assert "run-batch" in out
    assert "run-real-sweep" in out
    assert "plot-results" in out
    assert "info" in out


def test_sqa3d_run_real_sweep_help():
    """run-real-sweep documents replay-mode and isolate-episodes."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "sqa3d", "run-real-sweep", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "--replay-mode" in out
    assert "--isolate-episodes" in out
    assert "--with-sens" in out
    assert "--all" in out
    assert "--resume" in out
    assert "--export-root" in out


def test_eval_sqa3d_help():
    """emet eval-sqa3d --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "eval-sqa3d", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--predictions" in result.stdout or "-p" in result.stdout
    assert "split" in result.stdout.lower()


def test_debug_da3_depth_help():
    """emet debug-da3-depth --help lists DA3 options (top-level subcommand)."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "debug-da3-depth", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--model-id" in result.stdout
    assert "--depth-source" in result.stdout
    assert "--meshes" in result.stdout or "--no-meshes" in result.stdout


def test_install_completion_bash():
    """install-completion --shell bash produces valid script."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install-completion", "--shell", "bash"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "_emet_completion" in result.stdout or "emet" in result.stdout

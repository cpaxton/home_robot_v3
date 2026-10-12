# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Exercise the standalone installer using local Git fixtures; no real installs."""

import json
import os
import subprocess
from pathlib import Path

import pytest

BOOTSTRAP = Path(__file__).resolve().parents[3] / "scripts/bootstrap_install.sh"


@pytest.fixture
def install_repo(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    (repo / "install.sh").write_text(
        "#!/bin/bash\n"
        "python3 -c 'import json,os,sys; "
        'json.dump({"args":sys.argv[1:],"cwd":os.getcwd(),"stdin":sys.stdin.read()},'
        'open("invocation.json","w"))'
        '\' "$@"\n'
        'exit "${FIXTURE_EXIT:-0}"\n'
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "-C", str(repo), "tag", "fixture-v1"], check=True)
    return repo


def run_bootstrap(tmp_path, install_repo, *args, piped=True, **overrides):
    env = {**os.environ, "EMET_INSTALL_REPO": str(install_repo), "EMET_INSTALL_REF": "fixture-v1"}
    env.pop("EMET_INSTALL_DIR", None)
    env.update(overrides)
    command = ["bash", "-s", "--", *args] if piped else ["bash", str(BOOTSTRAP), *args]
    return subprocess.run(
        command,
        input=BOOTSTRAP.read_text() if piped else "",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        start_new_session=True,
        timeout=15,
    )


@pytest.mark.parametrize("piped", [True, False])
def test_clone_and_forward_arguments_without_script_stdin(tmp_path, install_repo, piped):
    result = run_bootstrap(
        tmp_path,
        install_repo,
        "-y",
        "--profile",
        "minimal",
        "--no-sam2",
        piped=piped,
        EMET_INSTALL_DIR="checkout with spaces",
    )
    assert result.returncode == 0, result.stderr
    checkout = tmp_path / "checkout with spaces"
    assert json.loads((checkout / "invocation.json").read_text()) == {
        "args": ["-y", "--profile", "minimal", "--no-sam2"],
        "cwd": str(checkout),
        "stdin": "",
    }
    assert (
        subprocess.check_output(["git", "-C", str(checkout), "describe", "--tags"], text=True).strip() == "fixture-v1"
    )


def test_existing_destination_is_preserved(tmp_path, install_repo):
    checkout = tmp_path / "home_robot_v3"
    checkout.mkdir()
    marker = checkout / "keep.txt"
    marker.write_text("existing work")
    result = run_bootstrap(tmp_path, install_repo, "--yes")
    assert result.returncode != 0
    assert "Destination already exists" in result.stderr
    assert marker.read_text() == "existing work"
    assert not (checkout / "invocation.json").exists()


def test_missing_ref_does_not_run_installer(tmp_path, install_repo):
    result = run_bootstrap(tmp_path, install_repo, "-y", EMET_INSTALL_REF="missing-ref")
    assert result.returncode != 0
    assert not (tmp_path / "home_robot_v3/invocation.json").exists()


def test_noninteractive_requires_yes_before_cloning(tmp_path, install_repo):
    result = run_bootstrap(tmp_path, install_repo)
    assert result.returncode != 0
    assert "No controlling terminal" in result.stderr
    assert not (tmp_path / "home_robot_v3").exists()


def test_help_does_not_clone(tmp_path, install_repo):
    result = run_bootstrap(tmp_path, install_repo, "--help")
    assert result.returncode == 0
    assert "EMET_INSTALL_DIR" in result.stdout
    assert not (tmp_path / "home_robot_v3").exists()


def test_installer_exit_code_is_preserved(tmp_path, install_repo):
    result = run_bootstrap(tmp_path, install_repo, "-y", FIXTURE_EXIT="23")
    assert result.returncode == 23


def test_default_destination(tmp_path, install_repo):
    result = run_bootstrap(tmp_path, install_repo, "--yes")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "home_robot_v3/invocation.json").is_file()


def test_dangling_destination_symlink_is_preserved(tmp_path, install_repo):
    checkout = tmp_path / "home_robot_v3"
    checkout.symlink_to(tmp_path / "missing")
    result = run_bootstrap(tmp_path, install_repo, "-y")
    assert result.returncode != 0
    assert checkout.is_symlink()
    assert "Destination already exists" in result.stderr


def test_non_project_ref_is_rejected(tmp_path, install_repo):
    subprocess.run(["git", "-C", str(install_repo), "rm", "pyproject.toml"], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(install_repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "remove project",
        ],
        check=True,
        capture_output=True,
    )
    result = run_bootstrap(tmp_path, install_repo, "-y", EMET_INSTALL_REF="main")
    assert result.returncode != 0
    assert "missing install.sh or pyproject.toml" in result.stderr
    assert not (tmp_path / "home_robot_v3/invocation.json").exists()

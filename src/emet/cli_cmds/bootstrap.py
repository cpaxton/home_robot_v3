# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click

_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

# Sub-apps that define @click.option("--robot") and receive --robot from `emet run`.
_EMET_RUN_APPS_WITH_ROBOT = frozenset(
    {
        "dynamem",
        "agent",
        "graph-eqa",
        "dynagraph",
        "lazy-graph",
        "scene-graph",
        "molmospaces-explore",
        "debug-da3-depth",
        "debug-lingbot-depth",
    }
)


def _project_root() -> Path:
    """Return project root (parent of src/emet) for the installed ``emet`` package."""
    return Path(__file__).resolve().parents[3]


def _cwd_project_root() -> Path | None:
    """Emet checkout containing the current working directory, if any."""
    try:
        cwd = Path.cwd().resolve()
    except OSError:
        return None
    for p in (cwd, *cwd.parents):
        if (p / "pyproject.toml").is_file() and (p / "src" / "emet" / "cli.py").is_file():
            return p
    return None


def _active_project_root() -> Path:
    """Repo to use for this command: cwd checkout wins over ``emet`` install location."""
    return _cwd_project_root() or _project_root()


def _has_uv() -> bool:
    """Return True if uv is available on PATH."""
    try:
        subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _ensure_uv_project() -> None:
    """If we're in the project and uv is available, re-exec under uv run so the rest uses the project env."""
    if os.environ.get("EMET_UV_RUN"):
        return
    # Never re-exec under pytest: sys.argv there holds pytest's own args, so exec
    # would replace the test process with `uv run emet <pytest args>` (exit 2, no
    # traceback, kills the whole session via CliRunner.invoke on any subcommand).
    if os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        return
    if not _has_uv():
        return
    root = _active_project_root()
    pkg_root = _project_root()
    try:
        cwd = Path.cwd().resolve()
    except OSError:
        return
    if cwd != root and not str(cwd).startswith(str(root) + os.sep):
        return
    if not (root / "pyproject.toml").exists():
        return
    env = os.environ.copy()
    env["EMET_UV_RUN"] = "1"
    if root.resolve() != pkg_root.resolve():
        env["EMET_ACTIVE_REPO"] = str(root)
        click.secho(
            f"emet on PATH is from {pkg_root} but cwd is {root}; re-running: uv run emet …",
            fg="yellow",
            err=True,
        )
    try:
        os.execvpe("uv", ["uv", "run", "emet", *sys.argv[1:]], env)
    except Exception:
        pass


def _project_venv_python() -> Path | None:
    """Return the project .venv Python path if it exists."""
    root = _active_project_root()
    for name in ("python", "python3"):
        p = root / ".venv" / "bin" / name
        if p.exists():
            return p
    return None


def _in_project_tree() -> bool:
    """True when cwd is the project root or a subdirectory."""
    root = _active_project_root()
    if not (root / "pyproject.toml").exists():
        return False
    try:
        cwd = Path.cwd().resolve()
    except OSError:
        return False
    return cwd == root or str(cwd).startswith(str(root) + os.sep)


def _require_repo_venv_when_in_repo() -> None:
    """Fail fast when ``emet`` on PATH is not this checkout's ``.venv`` (common with old symlinks)."""
    venv_py = _project_venv_python()
    if venv_py is None or not _in_project_tree():
        return
    if Path(sys.executable).resolve() == venv_py.resolve():
        return
    root = _active_project_root()
    pkg_root = _project_root()
    click.secho(
        "Error: emet is not running from this repo's .venv.\n"
        f"  Current:   {sys.executable}\n"
        f"  Expected:  {venv_py}\n"
        f"  Cwd repo:  {root}\n"
        f"  Installed: {pkg_root}\n"
        f"  From {root}, use:\n"
        "    uv run emet …\n"
        "    # or: source .venv/bin/activate && emet …\n"
        "  Fix PATH: pip install -e .  in this repo (not home_robot_v4).",
        fg="red",
        err=True,
    )
    sys.exit(2)


def _run_module(module: str, args: list[str], env: dict | None = None) -> int:
    """Run a Python module. Returns exit code."""
    from emet.utils.pythonpath import sanitize_emet_subprocess_env

    env = sanitize_emet_subprocess_env(env)
    # Prefer project .venv so the subprocess has the same deps (e.g. robocasa for mujoco server).
    venv_py = _project_venv_python()
    if venv_py is not None:
        cmd = [str(venv_py), "-m", module] + args
    else:
        cmd = [sys.executable, "-m", module] + args
    return subprocess.call(cmd, env=env)


_MAIN_EPILOG = (
    'Tab completion: eval "$(emet install-completion --shell bash)" (bash), '
    "or use --shell zsh / --shell fish. See: emet install-completion --help"
)


def _jobs_run_id_from_output(stdout: str | None) -> str | None:
    """Best-effort parse of ``emet jobs run`` stdout (last non-empty line is job id)."""
    if not stdout:
        return None
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def _timestamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _kill_processes_on_port(port: int) -> bool:
    """Thin wrapper for emet.utils.port_utils.kill_processes_on_port."""
    from emet.utils.port_utils import kill_processes_on_port

    return kill_processes_on_port(port)

#!/usr/bin/env python
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Emet CLI — start simulations, run agents, sync deps, view logs, run tests."""

import os
import subprocess
import sys
from pathlib import Path

import click

# Enable shell completion for bash/zsh
_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _project_root() -> Path:
    """Return project root (parent of src/emet)."""
    return Path(__file__).resolve().parent.parent.parent


def _run_module(module: str, args: list[str], env: dict | None = None) -> int:
    """Run a Python module. Returns exit code."""
    cmd = [sys.executable, "-m", module] + args
    env = env or os.environ.copy()
    return subprocess.call(cmd, env=env)


_MAIN_EPILOG = (
    "Tab completion: eval \"$(emet install-completion --shell bash)\" (bash), "
    "or use --shell zsh / --shell fish. See: emet install-completion --help"
)


@click.group(
    context_settings=_CONTEXT_SETTINGS,
    epilog=_MAIN_EPILOG,
)
@click.version_option(version="0.3.3", prog_name="emet")
def main() -> None:
    """Emet — Embodied Multi-robot Environment Toolkit.

    Start simulations, run robot agents, sync dependencies, view logs, and run tests.
    """
    pass


@main.command(short_help="Start MuJoCo simulation server")
@click.argument("backend", type=click.Choice(["mujoco"]), default="mujoco")
@click.option("--use-robocasa", is_flag=True, help="Use Robocasa for scene generation")
@click.option("--headless", is_flag=True, help="Run without native viewer")
@click.option("--scene-path", type=click.Path(exists=True), help="Path to MuJoCo scene XML")
@click.option("--seed", default=0, type=int, help="Random seed")
@click.option(
    "--port-offset",
    default=0,
    type=int,
    help="Add to default ports when 4401 etc. are in use (e.g. 100 → 4501–4504)",
)
@click.argument("extra", nargs=-1, type=click.UNPROCESSED)
def serve(
    backend: str,
    use_robocasa: bool,
    headless: bool,
    scene_path: str | None,
    seed: int,
    port_offset: int,
    extra: tuple[str, ...],
) -> None:
    """Start a simulation server.

    Examples:
      emet serve
      emet serve mujoco --headless
      emet serve mujoco --use-robocasa
      emet serve mujoco --port-offset 100   # use ports 4501–4504 if default in use
    """
    if backend == "mujoco":
        args = list(extra)
        if use_robocasa:
            args.append("--use-robocasa")
        if headless:
            args.append("--headless")
        if scene_path:
            args.extend(["--scene_path", scene_path])
        args.extend(["--seed", str(seed)])
        if port_offset:
            args.extend(["--port-offset", str(port_offset)])
        sys.exit(_run_module("emet.simulation.mujoco_server", args))
    else:
        click.echo(f"Unknown backend: {backend}", err=True)
        sys.exit(1)


def _kill_processes_on_port(port: int) -> bool:
    """Kill processes using the given port. Returns True if any were killed."""
    try:
        out = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if out.returncode != 0 or not out.stdout.strip():
        return False
    pids = [s for s in out.stdout.strip().split() if s.isdigit()]
    if not pids:
        return False
    for pid in pids:
        try:
            subprocess.run(["kill", pid], check=False, capture_output=True)
        except Exception:
            pass
    return True


@main.command("kill-mujoco-server", short_help="Stop MuJoCo server (free ports)")
@click.option(
    "--port",
    default=4401,
    type=int,
    help="Kill process using this port (default: 4401, mujoco server).",
)
@click.option(
    "--all",
    "kill_all",
    is_flag=True,
    help="Kill mujoco_server by name (all instances), then free default ports 4401–4404.",
)
def kill_mujoco_server(port: int, kill_all: bool) -> None:
    """Stop MuJoCo simulation server(s) so ports are free.

    Examples:
      emet kill-mujoco-server              # kill process on port 4401
      emet kill-mujoco-server --port 4501  # kill process on port 4501
      emet kill-mujoco-server --all        # pkill mujoco_server, then free 4401–4404
    """
    killed_any = False
    if kill_all:
        r = subprocess.run(
            ["pkill", "-f", "mujoco_server"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            click.echo("Stopped mujoco_server process(es).")
            killed_any = True
        else:
            click.echo("No mujoco_server process found.", err=True)
        for p in (4401, 4402, 4403, 4404):
            if _kill_processes_on_port(p):
                click.echo(f"Freed port {p}.")
                killed_any = True
    else:
        if _kill_processes_on_port(port):
            click.echo(f"Killed process on port {port}.")
            killed_any = True
        else:
            click.echo(f"No process found on port {port}.", err=True)
    sys.exit(0 if killed_any else 1)


@main.command(
    "run",
    short_help="Run an app (dynamem, mapping, grasp, chat, ai_pickup, timing)",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
@click.argument("app", type=click.Choice(["dynamem", "mapping", "grasp", "chat", "ai_pickup", "timing"]))
@click.option("--robot-ip", "--robot_ip", default="127.0.0.1", help="Robot or simulator IP")
@click.option("--server-ip", "--server_ip", default="127.0.0.1", help="Server IP (e.g. for AnyGrasp)")
@click.option("-S", "--skip", "skip_confirmations", is_flag=True, help="Skip confirmations")
@click.option("--headless", is_flag=True, help="Run without display")
@click.option("--visual-servo", "-V", "--visual_servo", is_flag=True, help="Use visual servoing (dynamem)")
@click.option("--target-object", "--target_object", help="Target object to grasp")
@click.option("--parameter-file", "--parameter_file", help="Planner config (e.g. sim_planner.yaml)")
@click.pass_context
def run(
    ctx: click.Context,
    app: str,
    robot_ip: str,
    server_ip: str,
    skip_confirmations: bool,
    headless: bool,
    visual_servo: bool,
    target_object: str | None,
    parameter_file: str | None,
) -> None:
    """Run a robot agent or app.

    Unknown options (e.g. --match-method, --rerun-debug) are passed through to the app.

    Examples:
      emet run dynamem --robot-ip 127.0.0.1 -S
      emet run dynamem -S --visual-servo --match-method class --rerun-debug
      emet run mapping --robot-ip 127.0.0.1
      emet run grasp --target-object "red cylinder" --parameter-file sim_planner.yaml
    """
    args = list(ctx.args)
    args.extend(["--robot_ip", robot_ip])
    if app == "dynamem":
        args.extend(["--server_ip", server_ip])
        if skip_confirmations:
            args.append("-S")
        if headless:
            args.append("--headless")
        if visual_servo:
            args.append("--visual-servo")
        sys.exit(_run_module("emet.app.run_dynamem", args))
    elif app == "mapping":
        sys.exit(_run_module("emet.app.mapping", args))
    elif app == "grasp":
        if target_object:
            args.extend(["--target_object", target_object])
        if parameter_file:
            args.extend(["--parameter_file", parameter_file])
        args.append("--show_gui")
        sys.exit(_run_module("emet.app.grasp_object", args))
    elif app == "chat":
        sys.exit(_run_module("emet.app.chat", args))
    elif app == "ai_pickup":
        if skip_confirmations:
            args.append("-S")
        sys.exit(_run_module("emet.app.ai_pickup", args))
    elif app == "timing":
        args.extend(["--robot_ip", robot_ip])
        if headless:
            args.append("--headless")
        sys.exit(_run_module("emet.app.timing", args))
    else:
        click.echo(f"Unknown app: {app}", err=True)
        sys.exit(1)


_SYNC_ALL_EXTRAS = ("sim", "dynamem", "dev")  # MuJoCo, SAM-2, pytest, etc.


@main.command(short_help="Sync dependencies (uv or pip)")
@click.option("--extra", "-e", "extra_list", multiple=True, help="Extra to install (sim, dynamem, dev, etc.)")
@click.option("--all", "sync_all", is_flag=True, help="Install all common extras (sim, dynamem, dev)")
@click.option("--sim", is_flag=True, help="Include sim (MuJoCo, robocasa)")
@click.option("--dynamem", "dynamem_flag", is_flag=True, help="Include dynamem (SAM-2)")
@click.option("--dev", "dev_flag", is_flag=True, help="Include dev (pytest, black, mypy)")
@click.option("--hand-tracker", is_flag=True, help="Include hand_tracker (mediapipe)")
@click.option("--discord", is_flag=True, help="Include discord")
@click.option("--no-install", is_flag=True, help="Only sync lockfile, do not install emet")
def sync(
    extra_list: tuple[str, ...],
    sync_all: bool,
    sim: bool,
    dynamem_flag: bool,
    dev_flag: bool,
    hand_tracker: bool,
    discord: bool,
    no_install: bool,
) -> None:
    """Sync dependencies (uv sync or pip install -e .).

    Use --all for sim + dynamem + dev, or pick extras with -e or individual flags.
    Sync does not install Robocasa/robosuite; run emet install sim (or install robocasa) first.

    Examples:

      emet sync
      emet sync --all
      emet sync -e sim -e dynamem
      emet sync --sim --dynamem
      emet sync --all --hand-tracker
    """
    extras: list[str] = list(extra_list)
    if sync_all:
        extras.extend(_SYNC_ALL_EXTRAS)
    if sim:
        extras.append("sim")
    if dynamem_flag:
        extras.append("dynamem")
    if dev_flag:
        extras.append("dev")
    if hand_tracker:
        extras.append("hand_tracker")
    if discord:
        extras.append("discord")
    # Deduplicate, preserve order
    seen: set[str] = set()
    extras = [e for e in extras if e not in seen and not seen.add(e)]

    root = _project_root()
    os.chdir(root)

    if _has_uv():
        cmd = ["uv", "sync"]
        for e in extras:
            cmd.extend(["--extra", e])
        if no_install:
            cmd.append("--no-install-project")
        result = subprocess.call(cmd)
        sys.exit(result)
    else:
        # Fallback to pip
        extras_str = "[" + ",".join(extras) + "]" if extras else ""
        spec = f".{extras_str}"
        result = subprocess.call([sys.executable, "-m", "pip", "install", "-e", spec])
        sys.exit(result)


def _has_uv() -> bool:
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


@main.command(short_help="View Rerun logs (.rrd)")
@click.argument("path", type=click.Path(exists=True))
@click.option("--web", is_flag=True, help="Open in web viewer (default: native)")
def show(path: str, web: bool) -> None:
    """View Rerun logs (.rrd) or other visualization data.

    Examples:

      emet show data_0.rrd
      emet show logs/run_001.rrd --web
    """
    path_obj = Path(path).resolve()
    if not path_obj.exists():
        click.echo(f"File not found: {path_obj}", err=True)
        sys.exit(1)

    suffix = path_obj.suffix.lower()
    if suffix == ".rrd":
        env = os.environ.copy()
        if web:
            env["RERUN_VIEWER"] = "web"
        result = subprocess.call(
            [sys.executable, "-m", "rerun", str(path_obj)],
            env=env,
        )
        sys.exit(result)
    else:
        click.echo(f"Unknown format: {suffix}. Supported: .rrd (Rerun)", err=True)
        sys.exit(1)


@main.command(short_help="Run pytest")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--no-cov", "no_cov", is_flag=True, help="Disable coverage")
@click.argument("pytest_args", nargs=-1, type=click.UNPROCESSED)
def test(verbose: bool, no_cov: bool, pytest_args: tuple[str, ...]) -> None:
    """Run tests with pytest.

    Examples:

      emet test
      emet test -v
      emet test tests/test_cli.py
      emet test -k test_serve
    """
    root = _project_root()
    os.chdir(root)
    env = os.environ.copy()
    src = root / "src"
    if src.exists():
        env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "pytest"]
    if verbose:
        cmd.append("-v")
    if not no_cov and Path("pyproject.toml").exists():
        # Only add coverage if pytest-cov is likely available
        try:
            import pytest_cov  # noqa: F401
            cmd.extend(["--cov=emet", "--cov-report=term-missing"])
        except ImportError:
            pass
    cmd.extend(list(pytest_args))
    if not pytest_args:
        # Prefer src/test (project convention), then tests/
        for test_dir in (root / "src" / "test", root / "tests"):
            if test_dir.exists():
                cmd.append(str(test_dir))
                break
        else:
            cmd.append("src/emet")
    sys.exit(subprocess.call(cmd, env=env))


@main.group(short_help="Install submodules, sim, full setup, pre-commit")
def install() -> None:
    """Install submodules, simulation extras, or full setup."""
    pass


@install.command("submodules", short_help="Init and update git submodules")
@click.option("--recursive/--no-recursive", default=True, help="Recursively init submodules")
def install_submodules(recursive: bool) -> None:
    """Init and update git submodules (segment-anything-2, ok-robot, etc.).

    Examples:
      emet install submodules
      emet install submodules --no-recursive
    """
    root = _project_root()
    os.chdir(root)
    cmd = ["git", "submodule", "update", "--init"]
    if recursive:
        cmd.append("--recursive")
    result = subprocess.call(cmd)
    if result == 0:
        click.echo("Submodules initialized and updated.")
    sys.exit(result)


def _run_install_simulation(
    root: Path,
    download_assets: bool = False,
    setup_macros: bool = False,
) -> int:
    """Run scripts/install_simulation.sh (robosuite + robocasa). Returns exit code."""
    script = root / "scripts" / "install_simulation.sh"
    if not script.exists():
        click.echo(f"Script not found: {script}", err=True)
        return 1
    args = []
    if download_assets:
        args.append("-d")
    if setup_macros:
        args.append("-a")
    return subprocess.call(["bash", str(script)] + args)


@install.command("sim", short_help="Install Robocasa, robosuite")
@click.option("-d", "--download-assets", is_flag=True, help="Download Robocasa kitchen assets")
@click.option("-a", "--setup-macros", is_flag=True, help="Run Robocasa setup_macros.py")
def install_sim(download_assets: bool, setup_macros: bool) -> None:
    """Install simulation third-party deps (Robocasa + robosuite).

    Clones robosuite and robocasa into third_party and installs them.
    Not covered by sync: run this first, then emet sync -e sim.

    Examples:
      emet install sim
      emet install sim -d -a
    """
    root = _project_root()
    result = _run_install_simulation(root, download_assets, setup_macros)
    if result == 0:
        click.echo("Simulation install complete. Run: emet sync -e sim")
    sys.exit(result)


@install.command("robocasa", short_help="Install Robocasa (same as install sim)")
@click.option("-d", "--download-assets", is_flag=True, help="Download Robocasa kitchen assets")
@click.option("-a", "--setup-macros", is_flag=True, help="Run Robocasa setup_macros.py")
def install_robocasa(download_assets: bool, setup_macros: bool) -> None:
    """Install Robocasa and robosuite (same as emet install sim).

    Clones robosuite and robocasa into third_party and installs them.
    Then run: emet sync -e sim

    Examples:
      emet install robocasa
      emet install robocasa -d -a
    """
    root = _project_root()
    result = _run_install_simulation(root, download_assets, setup_macros)
    if result == 0:
        click.echo("Robocasa install complete. Run: emet sync -e sim")
    sys.exit(result)


@install.command("full", short_help="Run full install (install.sh)")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompts")
@click.option("--sim", is_flag=True, help="Include simulation extras")
@click.option("--cpu", is_flag=True, help="CPU-only (skip SAM2)")
@click.option("--no-sam2", is_flag=True, help="Skip Segment Anything 2")
def install_full(yes: bool, sim: bool, cpu: bool, no_sam2: bool) -> None:
    """Run full install (./install.sh).

    Installs uv, system deps, git-lfs, and syncs dependencies.

    Examples:
      emet install full
      emet install full -y --sim
      emet install full --cpu
    """
    root = _project_root()
    script = root / "install.sh"
    if not script.exists():
        click.echo(f"install.sh not found: {script}", err=True)
        sys.exit(1)
    args = []
    if yes:
        args.append("-y")
    if sim:
        args.append("--sim")
    if cpu:
        args.append("--cpu")
    if no_sam2:
        args.append("--no-sam2")
    result = subprocess.call(["bash", str(script)] + args)
    sys.exit(result)


@install.command("pre-commit", short_help="Install pre-commit hooks")
@click.option("--install-hooks", is_flag=True, default=True, help="Install git hook scripts")
@click.option("--run/--no-run", "run_hooks", default=False, help="Run hooks on all files after install")
def install_pre_commit(install_hooks: bool, run_hooks: bool) -> None:
    """Install pre-commit hooks (lint, format, type-check).

    Requires: emet sync --dev (installs pre-commit).

    Examples:
      emet install pre-commit
      emet install pre-commit --run
    """
    root = _project_root()
    os.chdir(root)
    try:
        import pre_commit  # noqa: F401
    except ImportError:
        click.echo("pre-commit not installed. Run: emet sync --dev", err=True)
        sys.exit(1)
    if install_hooks:
        result = subprocess.call([sys.executable, "-m", "pre_commit", "install"])
        if result != 0:
            sys.exit(result)
        click.echo("Pre-commit hooks installed.")
    if run_hooks:
        result = subprocess.call(
            [sys.executable, "-m", "pre_commit", "run", "--all-files"]
        )
        sys.exit(result)
    sys.exit(0)


@main.command("install-completion", short_help="Print shell completion script")
@click.option(
    "--shell", "-s",
    type=click.Choice(["bash", "zsh", "fish"], case_sensitive=False),
    default=None,
    help="Shell (default: auto-detect from SHELL).",
)
def install_completion(shell: str | None) -> None:
    """Print shell completion script for bash, zsh, or fish.

    Add to your shell config so that 'emet' and subcommands tab-complete:

      # Bash: add to ~/.bashrc
      eval \"$(emet install-completion --shell bash)\"

      # Zsh: add to ~/.zshrc
      eval \"$(emet install-completion --shell zsh)\"

      # Fish: add to ~/.config/fish/config.fish
      emet install-completion --shell fish | source
    """
    from click.shell_completion import get_completion_class

    if shell is None:
        shell_env = os.environ.get("SHELL", "")
        if "fish" in shell_env:
            shell = "fish"
        elif "zsh" in shell_env:
            shell = "zsh"
        else:
            shell = "bash"
    comp_cls = get_completion_class(shell)
    if comp_cls is None:
        click.echo(f"Completion not supported for {shell}", err=True)
        sys.exit(1)
    comp = comp_cls(main, {}, "emet", "_EMET_COMPLETE")
    click.echo(comp.source())


if __name__ == "__main__":
    main()

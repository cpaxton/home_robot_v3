#!/usr/bin/env python
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

"""Emet CLI — start simulations, run agents, sync deps, view logs, run tests."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import click
from click.core import ParameterSource

# Enable shell completion for bash/zsh
_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

# Sub-apps that define @click.option("--robot") and receive --robot from `emet run`.
_EMET_RUN_APPS_WITH_ROBOT = frozenset(
    {
        "dynamem",
        "agent",
        "graph-eqa",
        "dynagraph",
        "scene-graph",
        "molmospaces-explore",
        "debug-da3-depth",
        "debug-lingbot-depth",
    }
)


def _project_root() -> Path:
    """Return project root (parent of src/emet) for the installed ``emet`` package."""
    return Path(__file__).resolve().parent.parent.parent


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


@click.group(
    context_settings=_CONTEXT_SETTINGS,
    epilog=_MAIN_EPILOG,
)
@click.version_option(version="0.3.3", prog_name="emet")
def main() -> None:
    """Emet — Embodied Multi-robot Environment Toolkit.

    Start simulations, run robot agents, sync dependencies, view logs, and run tests.
    When run from inside the project directory and uv is installed, emet automatically
    uses the project environment (as if you had run uv run emet ...).
    """
    _ensure_uv_project()
    _require_repo_venv_when_in_repo()
    active = os.environ.get("EMET_ACTIVE_REPO")
    if active:
        click.secho(
            f"emet: using checkout {active} ({_project_root()})",
            fg="green",
            err=True,
        )


@main.command(short_help="Start simulation server (mujoco, robocasa, molmospaces, or habitat)")
@click.argument(
    "backend",
    type=click.Choice(["mujoco", "robocasa", "molmospaces", "habitat"]),
    default="mujoco",
)
@click.option(
    "--headless",
    is_flag=True,
    help=(
        "Run without the MuJoCo viewer and use off-screen GL when no DISPLAY is set. "
        "If you have an X server (e.g. Xvfb), set DISPLAY=:1 (or any display) and omit this flag."
    ),
)
@click.option(
    "--show-viewer-ui",
    is_flag=True,
    help="Show MuJoCo viewer side panels (rby1 / MolmoSpaces path: only when not --headless).",
)
@click.option(
    "--no-cameras",
    is_flag=True,
    help="Disable camera rendering (use on WSL when EGL camera init hangs)",
)
@click.option(
    "--use-glx",
    is_flag=True,
    help="Use GLX instead of EGL (use with Xvfb on WSL to get camera images)",
)
@click.option(
    "--scene",
    default=None,
    metavar="NAME|PATH",
    help=(
        "Scene selector: omit for default table; robocasa; MolmoSpaces catalog name (ithor, procthor-10k, …); "
        "or path to a merged MJCF. Use --split / --index with MolmoSpaces scenes."
    ),
)
@click.option(
    "--split",
    default="train",
    type=click.Choice(["train", "val", "test"]),
    help="Data split when --scene is a MolmoSpaces catalog name.",
)
@click.option(
    "--index",
    default=0,
    type=int,
    help="Scene index when --scene is a MolmoSpaces catalog name.",
)
@click.option(
    "--install-scene-if-missing",
    is_flag=True,
    help=(
        "When --scene is MolmoSpaces: download scene archive if missing "
        "(non-interactive; same as emet-molmospaces merge-scene --install-if-missing)."
    ),
)
@click.option("--seed", default=0, type=int, help="Random seed")
@click.option(
    "--steps",
    default=None,
    type=int,
    metavar="N",
    help="Stop the MuJoCo server after N physics steps (debug; rby1 / merged MJCF path).",
)
@click.option(
    "--debug-molmospaces-spawn",
    is_flag=True,
    help="Verbose MolmoSpaces base placement and post-spawn contact diagnostics (merged MJCF / Stretch subprocess).",
)
@click.option(
    "--port-offset",
    default=0,
    type=int,
    help="Add to default ports when 4401 etc. are in use (e.g. 100 → 4501–4504)",
)
@click.option(
    "--habitat-question-id",
    type=int,
    default=None,
    help="Habitat only: HM-EQA question id (loads scene + init pose from CSV)",
)
@click.option(
    "--habitat-scene-id",
    default=None,
    help="Habitat only: HM3D scene id for free play (e.g. Y8Y6ukxGMvn)",
)
@click.option(
    "--habitat-floor",
    default=0,
    type=int,
    help="Habitat only: floor index when resolving init pose from CSV",
)
@click.option(
    "--habitat-hm3d-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Habitat only: override HM3D scene root",
)
@click.option(
    "--habitat-use-semantics/--habitat-no-semantics",
    default=None,
    help="Habitat only: load HM3D semantic meshes when available",
)
@click.option(
    "--list-robocasa-tasks",
    "list_robocasa_tasks",
    is_flag=True,
    help="List all Robocasa environment names and exit. Use: emet serve robocasa --list-robocasa-tasks or emet robocasa list.",
)
@click.option(
    "--robocasa-task",
    "--robocasa_task",
    "robocasa_task",
    default="",
    help="Robocasa task name (e.g. PickPlaceCounterToCabinet). Use --list-robocasa-tasks to see all.",
)
@click.option(
    "--robot",
    default=None,
    help=(
        "Robot to simulate. Default: stretch (table, Robocasa, and MolmoSpaces when omitted). "
        "Registry robots (innate_mars, rby1, galaxea_r1) use RobosuiteZmqServer on merged MJCF paths."
    ),
)
@click.argument("extra", nargs=-1, type=click.UNPROCESSED)
def serve(
    backend: str,
    headless: bool,
    show_viewer_ui: bool,
    no_cameras: bool,
    use_glx: bool,
    scene: str | None,
    split: str,
    index: int,
    install_scene_if_missing: bool,
    seed: int,
    steps: int | None,
    debug_molmospaces_spawn: bool,
    port_offset: int,
    list_robocasa_tasks: bool,
    robocasa_task: str,
    robot: str,
    habitat_question_id: int | None,
    habitat_scene_id: str | None,
    habitat_floor: int,
    habitat_hm3d_root: Path | None,
    habitat_use_semantics: bool | None,
    extra: tuple[str, ...],
) -> None:
    """Start a simulation server.

    Backends:
      mujoco       MuJoCo server (default). Use --scene robocasa or --scene ithor for other scenes.
      robocasa     Shortcut for ``--scene robocasa``.
      molmospaces  Shortcut for ``--scene ithor`` (or pass scene name positional:
                   ``emet serve molmospaces procthor-10k``).

    List Robocasa environments (requires sim extra: ``uv sync --extra sim`` or ``emet sync -e sim`` after ``emet install sim``):
      emet robocasa list
      emet serve robocasa --list-robocasa-tasks

    Examples:
      emet serve
      DISPLAY=:1 emet serve mujoco   # Xvfb or local display; viewer works without --headless
      emet serve mujoco --headless   # True headless / no DISPLAY (EGL off-screen)
      emet serve --robot innate_mars --headless   # Innate Mars + default table (ports 4401–4404)
      emet serve robocasa
      emet serve robocasa --robot PandaOmron
      emet serve robocasa --robot galaxea_r1
      emet serve robocasa --robocasa-task PickPlaceCounterToCabinet
      emet serve robocasa --list-robocasa-tasks
      emet serve mujoco --scene robocasa --port-offset 100
      emet serve molmospaces --headless
      emet serve molmospaces ithor --index 3
      emet serve habitat --habitat-scene-id Y8Y6ukxGMvn
      emet serve habitat --habitat-question-id 17
      DISPLAY=:1 emet serve mujoco --scene ithor   # default robot: stretch
      emet serve mujoco --scene ithor --robot rby1 --headless
      emet serve mujoco --scene ithor --robot xlerobot --headless
      emet robots info xlerobot
      emet robots preview-cameras xlerobot --source local
    """
    extra_args = list(extra)
    scene_value = scene
    if backend == "habitat":
        if list_robocasa_tasks:
            click.echo("--list-robocasa-tasks is only for robocasa / --scene robocasa.", err=True)
            sys.exit(1)
        hab_args = ["serve", "--port-offset", str(int(port_offset))]
        if habitat_question_id is not None:
            hab_args.extend(["--question-id", str(int(habitat_question_id))])
        if habitat_scene_id:
            hab_args.extend(["--scene-id", str(habitat_scene_id)])
        if habitat_floor:
            hab_args.extend(["--floor", str(int(habitat_floor))])
        if habitat_hm3d_root is not None:
            hab_args.extend(["--hm3d-root", str(habitat_hm3d_root)])
        if habitat_use_semantics is True:
            hab_args.append("--use-hm3d-semantics")
        elif habitat_use_semantics is False:
            hab_args.append("--no-hm3d-semantics")
        if scene_value and str(scene_value).strip() and not habitat_scene_id:
            hab_args.extend(["--scene-id", str(scene_value).strip()])
        hab_args.extend(extra_args)
        sys.exit(_run_habitat_wrapper(hab_args))
    if backend == "robocasa":
        if scene_value and str(scene_value).strip().lower() not in ("", "robocasa"):
            click.echo("Cannot combine serve robocasa with --scene other than robocasa.", err=True)
            sys.exit(1)
        scene_value = scene_value or "robocasa"
    elif backend == "molmospaces":
        if scene_value and str(scene_value).strip().lower() == "robocasa":
            click.echo("Cannot combine serve molmospaces with --scene robocasa.", err=True)
            sys.exit(1)
        if list_robocasa_tasks:
            click.echo("--list-robocasa-tasks is only for robocasa / --scene robocasa.", err=True)
            sys.exit(1)
        if scene_value is None or not str(scene_value).strip():
            if extra_args and not str(extra_args[0]).startswith("-"):
                scene_value = str(extra_args.pop(0))
            else:
                scene_value = "ithor"
    if backend in ("mujoco", "robocasa", "molmospaces"):
        from emet.config.sim_launch_config import build_sim_launch_config_from_serve_cli
        from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv

        if list_robocasa_tasks:
            args = extra_args + ["--use-robocasa", "--list-robocasa-tasks"]
            sys.exit(_run_module("emet.simulation.mujoco_server", args))

        try:
            cfg = build_sim_launch_config_from_serve_cli(
                scene=scene_value,
                split=split,
                index=index,
                install_scene_if_missing=install_scene_if_missing,
                robot=robot,
                headless=headless,
                show_viewer_ui=show_viewer_ui,
                no_cameras=no_cameras,
                use_glx=use_glx,
                seed=seed,
                steps=steps,
                debug_molmospaces_spawn=debug_molmospaces_spawn,
                port_offset=port_offset,
                robocasa_task=robocasa_task,
            )
        except ValueError as e:
            click.echo(str(e), err=True)
            sys.exit(1)
        args = extra_args + prepare_mujoco_server_argv(cfg)
        sys.exit(_run_module("emet.simulation.mujoco_server", args))
    else:
        click.echo(f"Unknown backend: {backend}", err=True)
        sys.exit(1)


@main.group("robocasa", short_help="Robocasa simulation helpers (requires sim extra)")
def robocasa_cmd() -> None:
    """List Robocasa environments or run the server. Requires: emet install sim, then uv sync."""


@robocasa_cmd.command("list", short_help="List all Robocasa environment names")
def robocasa_list() -> None:
    """Print registered Robocasa task names. Use with: emet serve robocasa --robocasa-task <name>."""
    sys.exit(_run_module("emet.simulation.mujoco_server", ["--use-robocasa", "--list-robocasa-tasks"]))


def _run_molmospaces_wrapper(args: list[str]) -> int:
    """Run the emet-molmospaces wrapper (list-scenes, install-scene, merge-scene, serve). Returns exit code."""
    from emet.simulation.molmospaces_config import (
        build_molmospaces_wrapper_command,
        ensure_molmospaces_assets_dir_env,
    )

    cmd = build_molmospaces_wrapper_command(args)
    if cmd is None:
        click.echo(
            "MolmoSpaces wrapper not found. The package `emet-molmospaces` is part of this repo and is not "
            "published on PyPI, so `pip install emet-molmospaces` will not work.\n\n"
            "From the project root, run (sim install also creates `.venv-molmospaces` by default):\n"
            "  ./install.sh -y\n"
            "Or wrapper only:\n"
            "  ./install.sh --molmospaces -y\n\n"
            "Or create the venv and install the local packages:\n"
            "  uv venv .venv-molmospaces\n"
            "  uv pip install --python .venv-molmospaces/bin/python --no-deps -e .\n"
            "  uv pip install --python .venv-molmospaces/bin/python -e packages/emet_molmospaces\n\n"
            "Optional: set MLSPACES_ASSETS_DIR (defaults to ~/.cache/molmospaces/assets) and "
            "MLSPACES_CACHE_DIR (defaults to ~/.cache/molmospaces/resource_cache; must differ from assets; "
            "see docs/molmospaces.md).",
            err=True,
        )
        return 1
    env = os.environ.copy()
    ensure_molmospaces_assets_dir_env(env)
    return subprocess.call(cmd, cwd=_project_root(), env=env)


@main.group("dataset", short_help="Dataset inspect, export, and replay")
def dataset_cmd() -> None:
    """Learning datasets (MolmoBot-Data H5, etc.)."""


from emet.datasets.molmobot.cli import molmobot_dataset_group  # noqa: E402

dataset_cmd.add_command(molmobot_dataset_group)


@main.group("molmobot", short_help="MolmoBot policy bridge (optional venv)")
def molmobot_cmd() -> None:
    """Delegate to packages/emet_molmobot when installed."""


@molmobot_cmd.command("serve-policy")
@click.option("--hf-repo", required=True)
@click.option("--action-type", default="joint_pos", show_default=True)
@click.argument("extra_args", nargs=-1)
def molmobot_serve_policy(hf_repo: str, action_type: str, extra_args: tuple[str, ...]) -> None:
    """Run upstream MolmoBot serve_molmo.py (requires MOLMOBOT_ROOT)."""
    try:
        from emet_molmobot.runner import serve_policy
    except ImportError:
        click.echo(
            "Install packages/emet_molmobot editable or set PYTHONPATH. See docs/datasets/molmobot.md.",
            err=True,
        )
        raise SystemExit(1) from None
    argv = ["--hf-repo", hf_repo, "--action-type", action_type, *extra_args]
    raise SystemExit(serve_policy(argv))


from emet.app.robots_cli import robots_cmd

main.add_command(robots_cmd)


@main.group("molmospaces", short_help="MolmoSpaces scenes and robots (requires emet-molmospaces wrapper)")
def molmospaces_cmd() -> None:
    """Set up MolmoSpaces scenes, list robots (e.g. rby1 / Galaxea R1), and run simulation.

    list-robots works without the wrapper. list-scenes, install-scene, merge-scene, and serve
    require the local emet-molmospaces package (see docs/molmospaces.md). ``write-spawn-metadata``
    and ``build-occ-map`` are core-only offline tools (see docs/molmospaces_spawn_metadata.md).
    ``export-nerfstudio`` is core-only (reads an explore episode directory). Install wrapper with:
      ./install.sh -y   (default sim path)   or   ./install.sh --molmospaces -y   or   editable install of packages/emet_molmospaces
    """


def _run_habitat_wrapper(args: list[str]) -> int:
    """Run the emet-habitat wrapper. Returns exit code."""
    from emet.habitat.wrapper_config import build_habitat_wrapper_command, ensure_habitat_eqa_data_dir_env

    cmd = build_habitat_wrapper_command(args)
    if cmd is None:
        click.echo(
            "Habitat wrapper not found. From the project root run:\n"
            "  ./scripts/install_habitat.sh\n\n"
            "See docs/habitat/README.md.",
            err=True,
        )
        return 1
    env = os.environ.copy()
    ensure_habitat_eqa_data_dir_env(env)
    return subprocess.call(cmd, cwd=_project_root(), env=env)


@main.group("habitat", short_help="Habitat-Sim EQA harness (requires emet-habitat / .venv-habitat)")
def habitat_cmd() -> None:
    """HM-EQA / OpenEQA evaluation in Habitat driving emet GraphEQA / Dynagraph.

    Requires ``./scripts/install_habitat.sh`` (``.venv-habitat``). See docs/habitat/README.md.
    """


@habitat_cmd.command("info", short_help="Print data paths and asset status")
def habitat_info() -> None:
    sys.exit(_run_habitat_wrapper(["info"]))


@habitat_cmd.command("list-questions", short_help="List HM-EQA questions from CSV")
@click.option("--limit", default=10, type=int)
def habitat_list_questions(limit: int) -> None:
    sys.exit(_run_habitat_wrapper(["list-questions", "--limit", str(limit)]))


@habitat_cmd.command("serve", short_help="Start Habitat-Sim ZMQ server (interactive)")
@click.option("--question-id", type=int, default=None)
@click.option("--scene-id", default=None)
@click.option("--floor", default=0, type=int)
@click.option("--port-offset", default=0, type=int)
def habitat_serve(
    question_id: int | None,
    scene_id: str | None,
    floor: int,
    port_offset: int,
) -> None:
    """Same as ``emet serve habitat`` — Stretch-shaped ZMQ for dynagraph / agent."""
    args = ["serve", "--port-offset", str(port_offset)]
    if question_id is not None:
        args.extend(["--question-id", str(question_id)])
    if scene_id:
        args.extend(["--scene-id", scene_id])
    if floor:
        args.extend(["--floor", str(floor)])
    sys.exit(_run_habitat_wrapper(args))


@habitat_cmd.command("run-episode", short_help="Run one HM-EQA episode")
@click.option("--question-id", default=0, type=int)
@click.option("--method", type=click.Choice(["graph_eqa", "dynagraph"]), default="dynagraph")
@click.option("--mock-llm", is_flag=True, default=False)
@click.option("--max-planning-steps", default=5, type=int)
def habitat_run_episode(
    question_id: int,
    method: str,
    mock_llm: bool,
    max_planning_steps: int,
) -> None:
    args = [
        "run-episode",
        "--question-id",
        str(question_id),
        "--method",
        method,
        "--max-planning-steps",
        str(max_planning_steps),
    ]
    if mock_llm:
        args.append("--mock-llm")
    sys.exit(_run_habitat_wrapper(args))


@habitat_cmd.command("compare-batch", short_help="GraphEQA vs Dynagraph on same questions")
@click.option("--question-start", default=0, type=int)
@click.option("--question-end", default=5, type=int)
@click.option("--mock-llm", is_flag=True, default=False)
@click.option("--max-planning-steps", default=20, type=int)
def habitat_compare_batch(
    question_start: int,
    question_end: int,
    mock_llm: bool,
    max_planning_steps: int,
) -> None:
    args = [
        "compare-batch",
        "--question-start",
        str(question_start),
        "--question-end",
        str(question_end),
        "--max-planning-steps",
        str(max_planning_steps),
        "--output",
        f"{os.path.expanduser('~')}/.cache/habitat_eqa/results/compare_q{question_start}-{question_end}.json",
    ]
    if mock_llm:
        args.append("--mock-llm")
    sys.exit(_run_habitat_wrapper(args))


@molmospaces_cmd.command("list-robots", short_help="List supported robot IDs")
def molmospaces_list_robots() -> None:
    """Print MolmoSpaces robot IDs and emet vendored robots with MJCF (xlerobot, franka_fr3, …)."""
    from emet.app.robots_cli import CANONICAL_ROBOT_KEYS
    from emet.simulation.molmospaces_config import DEFAULT_MOLMOSPACES_ROBOT, MOLMOSPACES_ROBOT_IDS
    from emet.utils.assets import get_robot_mjcf_path

    click.echo("MolmoSpaces wrapper IDs: " + ", ".join(MOLMOSPACES_ROBOT_IDS))
    click.echo(f"MolmoSpaces default: {DEFAULT_MOLMOSPACES_ROBOT}")
    click.echo("")
    click.echo("Emet registry robots (use with emet serve mujoco --robot <key>):")
    for key in CANONICAL_ROBOT_KEYS:
        mjcf = get_robot_mjcf_path(key)
        tag = "mjcf" if mjcf and mjcf.is_file() else "no mjcf"
        click.echo(f"  {key} ({tag})")


@molmospaces_cmd.command("list-scenes", short_help="List scene names and split sizes")
def molmospaces_list_scenes() -> None:
    """Print available MolmoSpaces scene names and split counts. Requires emet-molmospaces wrapper."""
    sys.exit(_run_molmospaces_wrapper(["list-scenes"]))


@molmospaces_cmd.command("install-scene", short_help="Install a scene and optionally write XML path")
@click.option("--scene", default="ithor", help="Scene name (e.g. ithor, procthor-10k)")
@click.option("--split", default="train", type=click.Choice(["train", "val", "test"]))
@click.option("--index", default=0, type=int, help="Scene index within split")
@click.option(
    "--scene-path", type=click.Path(path_type=Path), default=None, help="Write installed scene XML to this path"
)
@click.option(
    "--install-if-missing",
    is_flag=True,
    help="Download/link the scene archive without prompting if it is not on disk yet.",
)
def molmospaces_install_scene(
    scene: str, split: str, index: int, scene_path: Path | None, install_if_missing: bool
) -> None:
    """Download and install a MolmoSpaces scene; optionally copy the scene XML to a path."""
    args = ["install-scene", "--scene", scene, "--split", split, "--index", str(index)]
    if scene_path is not None:
        args.extend(["--scene-path", str(scene_path)])
    if install_if_missing:
        args.append("--install-if-missing")
    sys.exit(_run_molmospaces_wrapper(args))


@molmospaces_cmd.command(
    "write-spawn-metadata",
    short_help="Measure spawn hints from a merged MJCF and update molmospaces_spawn.json",
)
@click.option(
    "--robot",
    required=True,
    help="Robot id merged into the MJCF (e.g. stretch, rby1, galaxea_r1, innate_mars).",
)
@click.option(
    "--mjcf",
    "mjcf_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Merged scene+robot MJCF (from merge-scene or a temp path from emet serve molmospaces).",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override output JSON (default: <robot_mjcf_dir>/molmospaces_spawn.json).",
)
@click.option(
    "--no-merge",
    is_flag=True,
    help="Replace the file instead of merging measured keys into existing JSON.",
)
@click.option("--base-body", default="base_link", show_default=True, help="Free-joint base body name.")
@click.option("--seed-x", default=0.0, show_default=True, type=float, help="World X (m) for measurement pose.")
@click.option("--seed-y", default=0.0, show_default=True, type=float, help="World Y (m) for measurement pose.")
def molmospaces_write_spawn_metadata(
    robot: str,
    mjcf_path: Path,
    output_path: Path | None,
    no_merge: bool,
    base_body: str,
    seed_x: float,
    seed_y: float,
) -> None:
    """Offline spawn tuning: kinematic floor settle → foot clearance / base height in JSON.

    Full workflow and JSON field reference: docs/molmospaces_spawn_metadata.md
    """
    from emet.app.write_molmospaces_spawn_metadata import run_write

    out = run_write(
        robot,
        mjcf_path,
        output_path=output_path,
        merge_existing=not no_merge,
        base_body_name=base_body,
        seed_x=seed_x,
        seed_y=seed_y,
    )
    click.echo(f"Wrote {out}")


@molmospaces_cmd.command("build-occ-map", short_help="Build vendored iTHOR orthographic occupancy map from merged MJCF")
@click.argument("mjcf", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: same directory as MJCF).",
)
@click.option("--agent-radius", default=0.32, type=float, show_default=True)
@click.option("--px-per-m", default=120, type=int, show_default=True)
def molmospaces_build_occ_map(mjcf: Path, output_dir: Path | None, agent_radius: float, px_per_m: int) -> None:
    """Orthographic segmentation occupancy (Molmo-style) for spawn QA; writes occupancy.png + occupancy_meta.json."""
    from emet.app.build_molmo_occupancy_map import run_build

    png, meta = run_build(mjcf, output_dir, agent_radius=agent_radius, px_per_m=px_per_m)
    click.echo(f"Wrote {png}")
    click.echo(f"Wrote {meta}")


@molmospaces_cmd.command("merge-scene", short_help="Write merged scene+robot MJCF for emet serve mujoco")
@click.option("--scene", default="ithor", help="Scene name")
@click.option("--split", default="train", type=click.Choice(["train", "val", "test"]))
@click.option("--index", default=0, type=int, help="Scene index within split")
@click.option("--robot", default="stretch", help="Robot ID (default: stretch)")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Where to write the merged MJCF (for emet serve mujoco --scene_path)",
)
@click.option(
    "--install-if-missing",
    is_flag=True,
    help="Download/link the scene archive without prompting if it is not on disk yet.",
)
def molmospaces_merge_scene(
    scene: str, split: str, index: int, robot: str, output: Path, install_if_missing: bool
) -> None:
    """Install scene if needed, merge rby1 MJCF, write XML for ZMQ server + emet run agent."""
    args = [
        "merge-scene",
        "--scene",
        scene,
        "--split",
        split,
        "--index",
        str(index),
        "--robot",
        robot,
        "--output",
        str(output),
    ]
    if install_if_missing:
        args.append("--install-if-missing")
    sys.exit(_run_molmospaces_wrapper(args))


@molmospaces_cmd.command("serve", short_help="Run MolmoSpaces simulation (scene + robot)")
@click.option("--scene", default="ithor", help="Scene name")
@click.option("--split", default="train", type=click.Choice(["train", "val", "test"]))
@click.option("--index", default=0, type=int, help="Scene index")
@click.option("--robot", default="stretch", help="Robot ID (default: stretch)")
@click.option("--headless", is_flag=True, help="Run without viewer")
@click.option("--viewer", is_flag=True, help="Open MuJoCo viewer")
@click.option("--rerun", type=str, default="", metavar="PORT_OR_PATH", help="Log to rerun (port or RRD path)")
@click.option(
    "--scene-path", type=click.Path(path_type=Path), default=None, help="Write installed scene XML path to this file"
)
@click.option(
    "--install-if-missing",
    is_flag=True,
    help="Download/link the scene archive without prompting if it is not on disk yet.",
)
def molmospaces_serve(
    scene: str,
    split: str,
    index: int,
    robot: str,
    headless: bool,
    viewer: bool,
    rerun: str,
    scene_path: Path | None,
    install_if_missing: bool,
) -> None:
    """Run MuJoCo simulation with a MolmoSpaces scene and robot. Use --viewer to see the sim."""
    args = ["serve", "--scene", scene, "--split", split, "--index", str(index), "--robot", robot]
    if headless:
        args.append("--headless")
    if viewer:
        args.append("--viewer")
    if rerun:
        args.extend(["--rerun", rerun])
    if scene_path is not None:
        args.extend(["--scene-path", str(scene_path)])
    if install_if_missing:
        args.append("--install-if-missing")
    sys.exit(_run_molmospaces_wrapper(args))


@molmospaces_cmd.command("export-nerfstudio", short_help="Build transforms.json from explore episode metadata")
@click.option(
    "--episode-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
    required=True,
    help="Directory containing metadata.jsonl and images/ from molmospaces-explore.",
)
@click.option(
    "--output",
    "-o",
    type=str,
    default="transforms.json",
    show_default=True,
    help="Output filename inside episode-dir (default transforms.json).",
)
def molmospaces_export_nerfstudio(episode_dir: str, output: str) -> None:
    """Convert ``metadata.jsonl`` + images into NERFStudio-style ``transforms.json``."""
    from pathlib import Path

    from emet.molmospaces.episode_writer import export_nerfstudio_transforms

    p = export_nerfstudio_transforms(Path(episode_dir), output_name=output)
    click.echo(str(p))


def _kill_processes_on_port(port: int) -> bool:
    """Thin wrapper for emet.utils.port_utils.kill_processes_on_port."""
    from emet.utils.port_utils import kill_processes_on_port

    return kill_processes_on_port(port)


@main.group(
    "jobs",
    invoke_without_command=True,
    short_help="List and manage queued/running eval experiments",
)
@click.pass_context
def jobs_group(ctx: click.Context) -> None:
    """Track paper evals / overnight smokes (registry + process scan).

    Queue scripts register under ``~/runs/emet/jobs/`` (``EMET_JOBS_DIR``).

    \b
    Examples:
      emet jobs
      emet jobs list --all
      emet jobs report
      emet jobs status JOB_ID
      emet jobs cancel JOB_ID
      emet jobs logs JOB_ID --tail 50
      emet jobs run --name eqa-smoke -- ./scripts/run_….sh OUT
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(jobs_list, show_all=False, as_json=False, scan=True)


@jobs_group.command("list", short_help="List registered jobs (default: non-terminal)")
@click.option("--all", "show_all", is_flag=True, help="Include done/failed/cancelled.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON array.")
@click.option(
    "--scan/--no-scan",
    default=True,
    show_default=True,
    help="Also show unmanaged eval processes from pgrep.",
)
def jobs_list(show_all: bool, as_json: bool, scan: bool) -> None:
    """List jobs from the registry; optionally scan for unmanaged eval PIDs."""
    from emet.utils.job_registry import (
        format_job_header,
        format_job_row,
        format_scanned_header,
        format_scanned_row,
        list_jobs,
        scan_eval_processes,
    )

    jobs = list_jobs(include_terminal=show_all)
    if as_json:
        payload: dict[str, Any] = {"jobs": [j.to_dict() for j in jobs]}
        if scan:
            payload["unmanaged"] = [
                {"pid": s.pid, "cmd": s.cmd, "pattern": s.matched_pattern}
                for s in scan_eval_processes()
                if s.pid not in {j.pid for j in jobs if j.pid is not None}
            ]
        click.echo(json.dumps(payload, indent=2))
        return

    if not jobs:
        click.echo("(no registered jobs)" if show_all else "(no active registered jobs)")
    else:
        click.echo(format_job_header())
        for job in jobs:
            click.echo(format_job_row(job))

    if scan:
        registered_pids = {j.pid for j in jobs if j.pid is not None}
        unmanaged = [s for s in scan_eval_processes() if s.pid not in registered_pids]
        if unmanaged:
            click.echo("")
            click.echo(f"Unmanaged eval processes ({len(unmanaged)}, not in registry):")
            click.echo(format_scanned_header())
            for s in unmanaged[:40]:
                click.echo(format_scanned_row(s))
            if len(unmanaged) > 40:
                click.echo(f"  … {len(unmanaged) - 40} more")


@jobs_group.command("status", short_help="Show one job record")
@click.argument("job_id")
@click.option("--json", "as_json", is_flag=True, help="Emit full JSON record.")
def jobs_status(job_id: str, as_json: bool) -> None:
    from emet.utils.job_registry import (
        compute_job_progress,
        format_job_detail,
        format_progress_brief,
        load_job,
        refresh_job_liveness,
    )

    job = load_job(job_id)
    if job is None:
        click.echo(f"unknown job: {job_id}", err=True)
        sys.exit(1)
    job = refresh_job_liveness(job)
    if as_json:
        payload = job.to_dict()
        prog = compute_job_progress(job)
        payload["progress"] = {
            "units_done": prog.units_done,
            "units_total": prog.units_total,
            "phase": prog.phase,
            "current_id": prog.current_id,
            "elapsed_s": prog.elapsed_s,
            "rate_s_per_unit": prog.rate_s_per_unit,
            "eta_s": prog.eta_s,
            "source": prog.source,
            "brief": format_progress_brief(prog),
        }
        click.echo(json.dumps(payload, indent=2))
    else:
        click.echo(format_job_detail(job))


@jobs_group.command(
    "report",
    short_help="Progress + per-episode scores (defaults to running job)",
)
@click.argument("job_id", required=False, default=None)
@click.option(
    "--question",
    "-q",
    "question_id",
    type=int,
    default=None,
    help="Deep-dive one question id: episode row + agentic trace (phrases, verify scores, stale re-verifies).",
)
@click.option("--arm", default=None, help="Restrict --question to one arm (classic/agentic).")
@click.option("--json", "as_json", is_flag=True, help="Emit structured JSON.")
def jobs_report(job_id: str | None, question_id: int | None, arm: str | None, as_json: bool) -> None:
    """Scorecard for an H2H / eval OUT dir.

    With no JOB_ID, picks the active running/waiting job (most recent if several),
    else the newest finished job with an ``out_dir``. Pass ``--question ID`` for a
    per-episode trace analysis.
    """
    from emet.utils.job_registry import (
        format_job_report,
        format_question_report,
        job_report_dict,
        question_report_dict,
        resolve_report_job,
    )

    job = resolve_report_job(job_id)
    if job is None:
        if job_id:
            click.echo(f"unknown job: {job_id}", err=True)
        else:
            click.echo("no job to report (registry empty)", err=True)
        sys.exit(1)
    if question_id is not None:
        if as_json:
            click.echo(json.dumps(question_report_dict(job, question_id, arm=arm), indent=2))
        else:
            click.echo(format_question_report(job, question_id, arm=arm))
        return
    if as_json:
        click.echo(json.dumps(job_report_dict(job), indent=2))
    else:
        click.echo(format_job_report(job))


@jobs_group.command("cancel", short_help="Cancel a registered job (kill process tree)")
@click.argument("job_id")
@click.option("--grace-sec", type=float, default=10.0, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Emit full JSON record.")
def jobs_cancel(job_id: str, grace_sec: float, as_json: bool) -> None:
    from emet.utils.job_registry import cancel_job, format_job_detail

    try:
        job = cancel_job(job_id, grace_s=grace_sec)
    except KeyError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    if as_json:
        click.echo(json.dumps(job.to_dict(), indent=2))
    else:
        click.echo(f"cancelled {job.id}")
        click.echo(format_job_detail(job))

@jobs_group.command("logs", short_help="Tail a job log (log_path or out_dir/*.log)")
@click.argument("job_id")
@click.option("--tail", "n_tail", type=int, default=40, show_default=True)
def jobs_logs(job_id: str, n_tail: int) -> None:
    from emet.utils.job_registry import load_job

    job = load_job(job_id)
    if job is None:
        click.echo(f"unknown job: {job_id}", err=True)
        sys.exit(1)
    candidates: list[Path] = []
    if job.log_path:
        candidates.append(Path(job.log_path))
    if job.out_dir:
        od = Path(job.out_dir)
        for name in (
            "queue.log",
            "orchestrator.log",
            "nohup.log",
            "phase1_smoke.log",
            "eqa_smoke.log",
        ):
            candidates.append(od / name)
        candidates.extend(sorted(od.glob("*.log")))
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        click.echo("no log file found for job", err=True)
        sys.exit(1)
    click.echo(f"# {path}", err=True)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-max(1, n_tail) :]:
        click.echo(line)


@jobs_group.command("register", short_help="Register a job (for scripts)")
@click.option("--name", required=True, help="Short job name.")
@click.option("--cmd", default="", help="Command summary.")
@click.option("--out-dir", type=click.Path(), default=None)
@click.option("--log-path", type=click.Path(), default=None)
@click.option("--repo", type=click.Path(), default=None)
@click.option("--wait-pid", multiple=True, type=int, help="PIDs to wait on (repeatable).")
@click.option("--pid", type=int, default=None, help="Controller PID if already running.")
@click.option(
    "--status",
    type=click.Choice(["queued", "waiting", "running", "done", "failed", "cancelled"]),
    default="queued",
)
def jobs_register(
    name: str,
    cmd: str,
    out_dir: str | None,
    log_path: str | None,
    repo: str | None,
    wait_pid: tuple[int, ...],
    pid: int | None,
    status: str,
) -> None:
    """Print the new job id on stdout (scripts should capture it)."""
    from emet.utils.job_registry import register_job

    job = register_job(
        name=name,
        cmd=cmd,
        out_dir=out_dir,
        log_path=log_path,
        repo=repo or str(_project_root()),
        wait_pids=list(wait_pid),
        pid=pid,
        status=status,  # type: ignore[arg-type]
    )
    click.echo(job.id)


@jobs_group.command("update", short_help="Update job status / pid / progress (for scripts)")
@click.argument("job_id")
@click.option(
    "--status",
    type=click.Choice(["queued", "waiting", "running", "done", "failed", "cancelled"]),
    default=None,
)
@click.option("--pid", type=int, default=None)
@click.option("--cmd", default=None)
@click.option("--out-dir", type=click.Path(), default=None)
@click.option("--log-path", type=click.Path(), default=None)
@click.option("--error", default=None)
@click.option("--units-done", type=int, default=None, help="Completed work units (for ETA).")
@click.option("--units-total", type=int, default=None, help="Total work units (for ETA).")
@click.option("--phase", default=None, help="Current phase label (e.g. classic, agentic).")
@click.option("--current-id", default=None, help="Current unit id (e.g. question id).")
@click.option(
    "--meta",
    multiple=True,
    help="Extra meta KEY=VALUE (repeatable). Progress keys also accepted here.",
)
def jobs_update(
    job_id: str,
    status: str | None,
    pid: int | None,
    cmd: str | None,
    out_dir: str | None,
    log_path: str | None,
    error: str | None,
    units_done: int | None,
    units_total: int | None,
    phase: str | None,
    current_id: str | None,
    meta: tuple[str, ...],
) -> None:
    from emet.utils.job_registry import update_job

    meta_update: dict = {}
    for item in meta:
        if "=" not in item:
            click.echo(f"ignore --meta {item!r} (want KEY=VALUE)", err=True)
            continue
        k, v = item.split("=", 1)
        meta_update[k.strip()] = v.strip()

    try:
        job = update_job(
            job_id,
            status=status,  # type: ignore[arg-type]
            pid=pid,
            cmd=cmd,
            out_dir=out_dir,
            log_path=log_path,
            error=error,
            meta_update=meta_update or None,
            units_done=units_done,
            units_total=units_total,
            phase=phase,
            current_id=current_id,
        )
    except KeyError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    click.echo(json.dumps(job.to_dict(), indent=2))


@jobs_group.command(
    "run",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    short_help="Register + nohup a command as a managed job",
)
@click.option("--name", required=True, help="Short job name.")
@click.option("--out-dir", type=click.Path(), default=None, help="Artifact directory.")
@click.option(
    "--wait-pid",
    multiple=True,
    type=int,
    help="Wait for these PIDs before starting (repeatable).",
)
@click.option(
    "--need-mib",
    type=int,
    default=None,
    help="If set, run emet eval wait before the command.",
)
@click.option(
    "--cpu-safe/--no-cpu-safe",
    default=None,
    help="Pin job away from turbo P-cores (default: on when --need-mib is set).",
)
@click.option(
    "--gpu-exclusive/--no-gpu-exclusive",
    default=None,
    help="Wait for other active Habitat/VLM/MuJoCo jobs (default: on when --need-mib is set).",
)
@click.option("--foreground", is_flag=True, help="Run in foreground (no nohup).")
@click.pass_context
def jobs_run(
    ctx: click.Context,
    name: str,
    out_dir: str | None,
    wait_pid: tuple[int, ...],
    need_mib: int | None,
    cpu_safe: bool | None,
    gpu_exclusive: bool | None,
    foreground: bool,
) -> None:
    """Register + nohup a command as a managed job.

    \b
    Example:
      emet jobs run --name improve-eqa --need-mib 14000 -- \\
        ./scripts/run_dynagraph_dynamic_improve_smokes.sh OUT
    """
    import shlex

    from emet.utils.job_registry import active_gpu_job_pids, register_job, update_job

    cmd_args = list(ctx.args)
    if cmd_args and cmd_args[0] == "--":
        cmd_args = cmd_args[1:]
    if not cmd_args:
        click.echo("usage: emet jobs run --name NAME -- CMD [ARGS…]", err=True)
        sys.exit(2)

    root = _project_root()
    out = Path(out_dir).expanduser() if out_dir else Path.home() / "runs" / "emet" / "jobs_runs" / name
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "job.log"
    cmd_str = " ".join(shlex.quote(a) for a in cmd_args)

    use_cpu_safe = bool(cpu_safe) if cpu_safe is not None else (need_mib is not None)
    use_gpu_excl = bool(gpu_exclusive) if gpu_exclusive is not None else (need_mib is not None)

    wait_pids = list(wait_pid)
    if use_gpu_excl:
        for extra in active_gpu_job_pids():
            if extra not in wait_pids:
                wait_pids.append(extra)
                click.echo(f"gpu-exclusive: will wait for pid {extra}", err=True)

    job = register_job(
        name=name,
        cmd=cmd_str,
        out_dir=out,
        log_path=log_path,
        repo=str(root),
        wait_pids=wait_pids,
        status="queued",
    )
    click.echo(f"registered  {job.id}", err=True)
    click.echo(f"name        {name}", err=True)
    click.echo(f"out_dir     {out}", err=True)
    click.echo(f"log         {log_path}", err=True)

    wrapper = out / "job_wrapper.sh"
    wait_lines = ""
    for wpid in wait_pids:
        wait_lines += f'while kill -0 {int(wpid)} 2>/dev/null; do sleep 15; done\n'
    need_block = ""
    if need_mib is not None:
        need_block = (
            f'NEED_MIB={int(need_mib)} "$EMET_BIN" eval wait --need-mib {int(need_mib)}\n'
            f'"$EMET_BIN" eval status || true\n'
        )
    cpu_block = ""
    if use_cpu_safe:
        cpu_block = (
            '"$EMET_BIN" eval affinity --apply --pid $$ || {\n'
            '  echo "ERROR: cpu-safe affinity failed (fail-closed)" >&2\n'
            '  exit 2\n'
            "}\n"
        )
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'cd "{root}"\n'
        f'export EMET_JOB_ID="{job.id}"\n'
        f'JOB_ID="{job.id}"\n'
        f'EMET_BIN="{root}/.venv/bin/emet"\n'
        'if [ ! -x "$EMET_BIN" ]; then EMET_BIN="emet"; fi\n'
        f'"$EMET_BIN" jobs update "$JOB_ID" --status waiting --pid $$\n'
        f"{wait_lines}"
        f"{need_block}"
        f"{cpu_block}"
        f'"$EMET_BIN" jobs update "$JOB_ID" --status running --pid $$\n'
        "set +e\n"
        f"{cmd_str}\n"
        "rc=$?\n"
        "set -e\n"
        'if [ "$rc" -eq 0 ]; then\n'
        f'  "$EMET_BIN" jobs update "$JOB_ID" --status done\n'
        "else\n"
        f'  "$EMET_BIN" jobs update "$JOB_ID" --status failed --error "exit $rc"\n'
        "fi\n"
        "exit $rc\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    if foreground:
        update_job(job.id, status="running", pid=os.getpid())
        rc = subprocess.call(["bash", str(wrapper)])
        sys.exit(rc)

    # Detach: start_new_session so cancel can killpg
    with log_path.open("a", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            ["bash", str(wrapper)],
            cwd=str(root),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    update_job(job.id, status="queued", pid=proc.pid)
    click.echo(f"pid         {proc.pid}", err=True)
    click.echo(job.id)


@main.group("eval", short_help="GPU preflight and eval process cleanup")
def eval_group() -> None:
    """GPU preflight and stale-process cleanup for paper evals / overnight smokes.

    Prefer these over sourcing ``scripts/gpu_preflight.sh`` from an interactive shell.
    Overnight bash scripts may still source that file; it delegates here when possible.

    Examples:
      emet eval status
      emet eval diagnose
      emet eval check --need-mib 12000
      emet eval wait --need-mib 12000
      emet eval kill-stale
    """


@eval_group.command("status", short_help="Show free VRAM and GPU compute apps")
def eval_status() -> None:
    """Print GPU free/total MiB and nvidia-smi compute apps (read-only)."""
    from emet.utils.gpu_preflight import format_status_lines

    for line in format_status_lines():
        click.echo(line)


@eval_group.command(
    "diagnose",
    short_help="Explain GPU/EGL readiness (empty nvidia-smi ≠ Habitat OK)",
)
def eval_diagnose() -> None:
    """Read-only Habitat/HM-EQA readiness notes for agents.

    Empty compute apps does **not** prove Magnum EGL can create a WindowlessContext.
    Also flags empty ``CUDA_VISIBLE_DEVICES``, missing ``.venv-habitat``, and recent
    ``emet`` segfault hints from dmesg when readable.
    """
    from pathlib import Path

    from emet.utils.gpu_preflight import diagnose_eval_environment

    ok, lines = diagnose_eval_environment(repo_root=str(Path.cwd()))
    for line in lines:
        click.echo(line)
    if not ok:
        sys.exit(1)


@eval_group.command("check", short_help="Exit 1 if free VRAM below threshold")
@click.option(
    "--need-mib",
    type=int,
    default=None,
    help="Minimum free VRAM in MiB (default: NEED_MIB or 12000).",
)
def eval_check(need_mib: int | None) -> None:
    """One-shot GPU memory gate (same role as ``gpu_preflight.sh --check``)."""
    from emet.utils.gpu_preflight import check_gpu_memory, list_compute_apps

    ok, msg = check_gpu_memory(need_mib)
    click.echo(msg)
    if not ok:
        for app in list_compute_apps():
            click.echo(
                f"  pid={app.pid} {app.process_name} {app.used_memory}".rstrip(),
                err=True,
            )
        sys.exit(1)


@eval_group.command("wait", short_help="Block until free VRAM is stably above threshold")
@click.option(
    "--need-mib",
    type=int,
    default=None,
    help="Minimum free VRAM in MiB (default: NEED_MIB or 12000).",
)
def eval_wait(need_mib: int | None) -> None:
    """Wait for consecutive stable free-VRAM reads (``gpu_preflight.sh --wait``)."""
    from emet.utils.gpu_preflight import wait_gpu_stable

    ok = wait_gpu_stable(need_mib, log=lambda m: click.echo(m, err=True))
    if not ok:
        click.echo("WARNING: GPU wait timed out; free VRAM still below threshold", err=True)
        sys.exit(1)


@eval_group.command("kill-stale", short_help="Stop orphaned eval/sim/uv GPU workers")
@click.option(
    "--no-gpu",
    "no_gpu",
    is_flag=True,
    help="Only match process patterns; do not kill nvidia-smi compute apps.",
)
@click.option(
    "--settle-sec",
    type=float,
    default=None,
    help="Sleep after pattern kills (default: GPU_SETTLE_SEC or 15).",
)
def eval_kill_stale(no_gpu: bool, settle_sec: float | None) -> None:
    """SIGTERM→SIGKILL stale mujoco/habitat/dynagraph/uv trees; skip caller ancestry.

    Protects this process and its parents, plus ``EMET_GPU_PROTECT_PIDS``.
    Patterns also match ``uv run emet {run,test,serve}`` in *other* terminals —
    do not run this while intentional GPU work is still in progress elsewhere;
    set ``EMET_GPU_PROTECT_PIDS`` or use ``emet eval wait`` instead.
    For port-only MuJoCo cleanup see ``emet kill-mujoco-server``.
    """
    from emet.utils.gpu_preflight import kill_stale_eval_processes

    click.echo(
        "kill-stale: matching sim/eval/uv emet trees "
        "(other terminals' pytest/serve may match; EMET_GPU_PROTECT_PIDS to keep)",
        err=True,
    )
    n = kill_stale_eval_processes(
        kill_gpu_apps=not no_gpu,
        settle_s=settle_sec,
        log=lambda m: click.echo(m, err=True),
    )
    click.echo(f"kill-stale done (signaled≈{n})")


@eval_group.command("affinity", short_help="Show or apply turbo-CPU exclusion mask")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON summary.")
@click.option("--apply", "do_apply", is_flag=True, help="Pin current process (or --pid).")
@click.option("--pid", type=int, default=None, help="Target PID (default: this process).")
@click.option(
    "--fail-open",
    is_flag=True,
    help="Do not exit non-zero if turbo CPUs remain after apply.",
)
def eval_affinity(as_json: bool, do_apply: bool, pid: int | None, fail_open: bool) -> None:
    """Exclude logical CPUs whose max freq is ≥ ``EMET_EXCLUDE_CPU_MIN_MHZ`` (default 6000)."""
    from emet.eval.harness import affinity_summary_dict, apply_eval_affinity

    if do_apply:
        try:
            summary = apply_eval_affinity(pid=pid, fail_closed=not fail_open)
        except RuntimeError as exc:
            click.echo(f"ERROR: {exc}", err=True)
            sys.exit(2)
        if as_json:
            click.echo(json.dumps(summary, indent=2))
        else:
            click.echo(
                f"affinity pid={summary.get('pid')} mask={summary.get('applied')} "
                f"turbo_excluded={summary.get('turbo_cpus')}"
            )
        return

    summary = affinity_summary_dict()
    if as_json:
        click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(
            f"taskset {summary['taskset']}  "
            f"(exclude>={summary['exclude_min_mhz']} MHz turbo={summary['turbo_cpus']})"
        )


@eval_group.command("recover", short_help="status + diagnose + wait (post-crash preflight)")
@click.option(
    "--need-mib",
    type=int,
    default=None,
    help="Minimum free VRAM in MiB (default: NEED_MIB or 12000).",
)
@click.option(
    "--skip-wait",
    is_flag=True,
    help="Only status+diagnose; do not block on free VRAM.",
)
def eval_recover(need_mib: int | None, skip_wait: bool) -> None:
    """One-shot recovery gate after agent death / host reboot / failed HM-EQA job."""
    from emet.eval.harness import affinity_summary_dict
    from emet.utils.gpu_preflight import (
        diagnose_eval_environment,
        format_status_lines,
        wait_gpu_stable,
    )

    for line in format_status_lines():
        click.echo(line)
    ok, lines = diagnose_eval_environment(repo_root=str(_project_root()))
    for line in lines:
        click.echo(line)
    aff = affinity_summary_dict()
    click.echo(
        f"affinity: prefer taskset {aff['taskset']} "
        f"(turbo {aff['turbo_cpus']} excluded by emet jobs --cpu-safe / emet hmeqa)"
    )
    if not ok:
        sys.exit(1)
    if skip_wait:
        return
    if not wait_gpu_stable(need_mib, log=lambda m: click.echo(m, err=True)):
        click.echo("WARNING: GPU wait timed out; free VRAM still below threshold", err=True)
        sys.exit(1)
    click.echo("recover: GPU ready — next: emet hmeqa resume  (or emet jobs)")


@main.group("hmeqa", short_help="HM-EQA classic vs agentic H2H helpers")
def hmeqa_group() -> None:
    """Dogfood entrypoints for Habitat HM-EQA head-to-head runs.

    Prefer these over hand-built ``env … taskset … ./scripts/run_hmeqa_*.sh`` lines.

    \b
    Examples:
      emet eval recover --need-mib 12000
      emet hmeqa resume
      emet hmeqa status
      emet hmeqa h2h --out OUT --resume --ids 15,68,105,17
    """


@hmeqa_group.command("status", short_help="Show OUT progress, crashes, scored counts")
@click.argument("out_dir", required=False)
def hmeqa_status(out_dir: str | None) -> None:
    from emet.eval.harness import count_crash_markers, resolve_hmeqa_out

    out = resolve_hmeqa_out(out_dir)
    click.echo(f"OUT={out}")
    progress = out / "progress.json"
    if progress.is_file():
        click.echo(progress.read_text(encoding="utf-8").rstrip())
    else:
        click.echo("(no progress.json)")
    scored_c = len([p for p in out.glob("classic_q*.jsonl") if p.stat().st_size > 0])
    scored_a = len([p for p in out.glob("agentic_q*.jsonl") if p.stat().st_size > 0])
    crashes = count_crash_markers(out)
    click.echo(f"scored classic={scored_c} agentic={scored_a} crash_markers={crashes}")
    for cap in sorted(out.glob("native_crash_*.log")) + sorted(out.glob("host_freeze_*.log")):
        click.echo(f"capsule {cap.name}")


@hmeqa_group.command("summarize", short_help="Run summarize_hmeqa_agentic_h2h.py on OUT")
@click.argument("out_dir", required=False)
def hmeqa_summarize(out_dir: str | None) -> None:
    from emet.eval.harness import resolve_hmeqa_out

    out = resolve_hmeqa_out(out_dir)
    script = _project_root() / "scripts" / "summarize_hmeqa_agentic_h2h.py"
    rc = subprocess.call([sys.executable, str(script), str(out)], cwd=str(_project_root()))
    sys.exit(rc)


def _hmeqa_launch(
    *,
    out: Path,
    resume: bool,
    arms: str,
    ids: str,
    coverage_qids: str,
    cooldown: int,
    crash_policy: str,
    streak_abort: int,
    agentic_verifier: str,
    require_verified: bool,
    agentic_router: bool,
    job_name: str,
    need_mib: int,
    foreground: bool,
) -> None:
    """Register H2H via ``emet jobs run`` (cpu-safe + gpu-exclusive defaults)."""
    import shlex

    root = _project_root()
    script = root / "scripts" / "run_hmeqa_agentic_h2h.sh"
    env_parts = [
        "EMET_ALLOW_SDPA_ATTN=1",
        "EMET_EQA_TRACE=1",
        f"ARMS={arms}",
        f"HOLDOUT_IDS={ids}",
        f"COVERAGE_QIDS={coverage_qids}",
        f"EPISODE_COOLDOWN_SEC={int(cooldown)}",
        f"NATIVE_CRASH_POLICY={crash_policy}",
        f"NATIVE_CRASH_STREAK_ABORT={int(streak_abort)}",
        f"EMET_EQA_AGENTIC_VERIFIER={agentic_verifier}",
        f"EMET_EQA_AGENTIC_REQUIRE_VERIFIED={int(require_verified)}",
        f"EMET_EQA_AGENTIC_ROUTER={int(agentic_router)}",
    ]
    if resume:
        env_parts.append("RESUME=1")
    inner = (
        "env "
        + " ".join(env_parts)
        + " "
        + shlex.quote(str(script))
        + " "
        + shlex.quote(str(out))
    )
    # Re-enter CLI so jobs run applies mutex/affinity wrapper.
    cmd = [
        sys.executable,
        "-m",
        "emet.cli",
        "jobs",
        "run",
        "--name",
        job_name,
        "--need-mib",
        str(int(need_mib)),
        "--out-dir",
        str(out),
    ]
    if foreground:
        cmd.append("--foreground")
    cmd.extend(["--", "bash", "-lc", inner])
    click.echo(f"launching via emet jobs: OUT={out} resume={int(resume)} arms={arms}", err=True)
    rc = subprocess.call(cmd, cwd=str(root))
    sys.exit(rc)


@hmeqa_group.command("h2h", short_help="Launch classic vs agentic H2H via emet jobs")
@click.argument("out_dir", required=False)
@click.option("--resume", is_flag=True, help="Skip non-empty per-qid jsonl.")
@click.option("--arms", default="classic,agentic", show_default=True)
@click.option(
    "--ids",
    "holdout_ids",
    default=None,
    help="Comma-separated question ids (default: bal-32 list).",
)
@click.option("--coverage-qids", default="15,28,47", show_default=True)
@click.option("--cooldown", type=int, default=20, show_default=True, help="EPISODE_COOLDOWN_SEC")
@click.option(
    "--crash-policy",
    type=click.Choice(["skip", "abort"]),
    default="skip",
    show_default=True,
    help="skip=continue after settle; abort=stop batch on first native crash.",
)
@click.option(
    "--streak-abort",
    type=int,
    default=2,
    show_default=True,
    help="Under skip: abort after N consecutive native crashes (0=never).",
)
@click.option(
    "--agentic-verifier",
    type=click.Choice(["none", "owlv2", "yoloe"]),
    default="none",
    show_default=True,
    help="Hybrid presence backend for the agentic arm.",
)
@click.option(
    "--require-verified/--allow-unverified",
    default=True,
    show_default=True,
    help="Abstain instead of force-submitting without fused evidence.",
)
@click.option(
    "--agentic-router/--no-agentic-router",
    default=False,
    show_default=True,
    help="Use VLM tool routing (fallback policy is deterministic).",
)
@click.option("--job-name", default="hmeqa-h2h", show_default=True)
@click.option("--need-mib", type=int, default=12000, show_default=True)
@click.option("--foreground", is_flag=True)
def hmeqa_h2h(
    out_dir: str | None,
    resume: bool,
    arms: str,
    holdout_ids: str | None,
    coverage_qids: str,
    cooldown: int,
    crash_policy: str,
    streak_abort: int,
    agentic_verifier: str,
    require_verified: bool,
    agentic_router: bool,
    job_name: str,
    need_mib: int,
    foreground: bool,
) -> None:
    from emet.eval.harness import DEFAULT_BAL32_IDS, resolve_hmeqa_out

    if out_dir:
        out = Path(out_dir).expanduser().resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = Path.home() / "runs" / "emet" / f"hmeqa_agentic_h2h_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    ids = holdout_ids or DEFAULT_BAL32_IDS
    _hmeqa_launch(
        out=out,
        resume=resume,
        arms=arms,
        ids=ids,
        coverage_qids=coverage_qids,
        cooldown=cooldown,
        crash_policy=crash_policy,
        streak_abort=streak_abort,
        agentic_verifier=agentic_verifier,
        require_verified=require_verified,
        agentic_router=agentic_router,
        job_name=job_name,
        need_mib=need_mib,
        foreground=foreground,
    )


@hmeqa_group.command("resume", short_help="Resume latest (or given) H2H OUT under safe defaults")
@click.argument("out_dir", required=False)
@click.option("--arms", default="classic,agentic", show_default=True)
@click.option("--ids", "holdout_ids", default=None, help="Override ids (default: from STATUS / bal-32).")
@click.option("--coverage-qids", default="15,28,47", show_default=True)
@click.option("--cooldown", type=int, default=30, show_default=True)
@click.option("--crash-policy", type=click.Choice(["skip", "abort"]), default="skip", show_default=True)
@click.option("--streak-abort", type=int, default=2, show_default=True)
@click.option(
    "--agentic-verifier",
    type=click.Choice(["none", "owlv2", "yoloe"]),
    default="none",
    show_default=True,
)
@click.option("--require-verified/--allow-unverified", default=True, show_default=True)
@click.option("--agentic-router/--no-agentic-router", default=False, show_default=True)
@click.option("--job-name", default="hmeqa-h2h-resume", show_default=True)
@click.option("--need-mib", type=int, default=12000, show_default=True)
@click.option("--foreground", is_flag=True)
def hmeqa_resume(
    out_dir: str | None,
    arms: str,
    holdout_ids: str | None,
    coverage_qids: str,
    cooldown: int,
    crash_policy: str,
    streak_abort: int,
    agentic_verifier: str,
    require_verified: bool,
    agentic_router: bool,
    job_name: str,
    need_mib: int,
    foreground: bool,
) -> None:
    from emet.eval.harness import (
        DEFAULT_BAL32_IDS,
        detect_host_freeze,
        resolve_hmeqa_out,
        write_host_freeze_capsule,
    )

    out = resolve_hmeqa_out(out_dir)
    freeze = detect_host_freeze(out)
    if freeze:
        cap = write_host_freeze_capsule(out, freeze)
        click.echo(f"host-freeze capsule → {cap}", err=True)
    ids = holdout_ids or DEFAULT_BAL32_IDS
    # Prefer HOLDOUT from orchestrator.log if present
    orch = out / "orchestrator.log"
    if holdout_ids is None and orch.is_file():
        import re as _re

        text = orch.read_text(encoding="utf-8", errors="replace")
        m = _re.search(r"ids=([0-9,]+)", text)
        if m:
            ids = m.group(1)
    _hmeqa_launch(
        out=out,
        resume=True,
        arms=arms,
        ids=ids,
        coverage_qids=coverage_qids,
        cooldown=cooldown,
        crash_policy=crash_policy,
        streak_abort=streak_abort,
        agentic_verifier=agentic_verifier,
        require_verified=require_verified,
        agentic_router=agentic_router,
        job_name=job_name,
        need_mib=need_mib,
        foreground=foreground,
    )


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

    For broader orphan cleanup (dynagraph / Habitat / ``uv run emet`` trees), use
    ``emet eval kill-stale``.

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


@main.command("view-mujoco", short_help="Open MuJoCo viewer for a robot MJCF (no ZMQ server)")
@click.option(
    "--robot",
    default="innate_mars",
    help="Robot key for get_robot_mjcf_path (e.g. innate_mars, rby1, galaxea_r1).",
)
@click.option(
    "--merge-scene",
    is_flag=True,
    help="Merge scene_environment.xml (table, objects) the same way as emet serve mujoco.",
)
@click.option(
    "--no-extras",
    is_flag=True,
    help="With innate_mars only: load robot MJCF alone (no grid floor / extra lights). Ignored with --merge-scene.",
)
def view_mujoco(robot: str, merge_scene: bool, show_viewer_ui: bool, no_extras: bool) -> None:
    """Open the native MuJoCo viewer to inspect a robot model (requires ``uv sync --extra sim``).

    Uses ``launch_passive``: close the window or Ctrl+C to exit. Needs a desktop ``DISPLAY`` (or X forwarding).

    Examples:
      emet view-mujoco --robot innate_mars
      emet view-mujoco --robot innate_mars --merge-scene
      emet view-mujoco --robot innate_mars --no-extras
    """
    import time

    import mujoco
    import mujoco.viewer

    from emet.utils.assets import get_robot_mjcf_path

    robot_key = robot.lower().replace("-", "_")
    if merge_scene:
        from emet.simulation.mujoco_server import _load_default_scene_with_robot

        model = _load_default_scene_with_robot(robot_key)
        if model is None:
            click.echo(
                "Could not build merged model (scene_environment.xml or robot MJCF missing).",
                err=True,
            )
            sys.exit(1)
    else:
        p = get_robot_mjcf_path(robot_key)
        if p is None or not p.is_file():
            click.echo(
                f"No MuJoCo XML for {robot!r} (see get_robot_mjcf_path in emet.utils.assets).",
                err=True,
            )
            sys.exit(1)
        use_extras = robot_key == "innate_mars" and not no_extras
        extras_p = p.parent / "innate_mars_visual_extras.xml" if use_extras else None
        if use_extras and extras_p is not None and extras_p.is_file():
            import os
            import tempfile

            robot_abs = str(p.resolve())
            extras_abs = str(extras_p.resolve())
            wrapper = (
                '<?xml version="1.0"?>\n'
                '<mujoco model="innate_mars_view">\n'
                f'  <include file="{robot_abs}"/>\n'
                f'  <include file="{extras_abs}"/>\n'
                "</mujoco>\n"
            )
            fd, tmp = tempfile.mkstemp(suffix=".xml", prefix="view_", dir=str(p.parent))
            os.close(fd)
            tmp_path = Path(tmp)
            try:
                tmp_path.write_text(wrapper)
                model = mujoco.MjModel.from_xml_path(str(tmp_path))
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        else:
            model = mujoco.MjModel.from_xml_path(str(p))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=show_viewer_ui,
            show_right_ui=show_viewer_ui,
        ) as viewer:
            click.echo("MuJoCo viewer open — close the window or Ctrl+C to exit.")
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.01)
    except Exception as e:
        click.echo(
            f"Viewer failed ({e!r}). On headless hosts use X11 forwarding or run with a local DISPLAY.",
            err=True,
        )
        sys.exit(1)


@main.command("show-memory", short_help="Open a saved memory in Rerun")
@click.argument(
    "path",
    type=click.Path(path_type=Path),
    default="saved_memory",
    required=False,
)
@click.option("--open-browser", is_flag=True, help="Open browser to Rerun web viewer")
def show_memory(path: str, open_browser: bool) -> None:
    """Load a memory directory and display it in Rerun.

    PATH defaults to saved_memory. Must be a directory with manifest.json
    (common memory format from create-and-print-memory or backend save).

    Examples:
      emet show-memory
      emet show-memory ./my_memory --open-browser
    """
    args = [str(path)]
    if open_browser:
        args.append("--open-browser")
    sys.exit(_run_module("emet.app.show_memory", args))


@main.command("graph-memory-show", short_help="Print formatted scene graph from saved memory directory")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--max-nodes", type=int, default=None, help="Truncate long node lists")
def graph_memory_show(path: Path, max_nodes: int | None) -> None:
    """Load common memory format (graph_eqa) and print nodes/edges to stdout."""

    args = [str(path)]
    if max_nodes is not None:
        args.extend(["--max-nodes", str(max_nodes)])
    sys.exit(_run_module("emet.app.graph_memory_show", args))


@main.command(
    "reprocess-graph-eqa-cache",
    short_help="Rebuild GraphEQA graph from saved memory frames (offline)",
    context_settings={"ignore_unknown_options": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def reprocess_graph_eqa_cache(args: tuple[str, ...]) -> None:
    """Re-run instance→graph logic on ``frames/`` under a saved memory directory.

    Example: ``emet reprocess-graph-eqa-cache ./saved_memory -o ./saved_memory_v2``
    """
    if not args:
        raise click.UsageError("Missing INPUT_DIR. See emet reprocess-graph-eqa-cache --help.")
    sys.exit(_run_module("emet.app.reprocess_graph_eqa_cache", list(args)))


@main.command("print", short_help="Print summary of a saved memory directory")
@click.argument(
    "path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
)
def print_memory(path: Path) -> None:
    """Load and print a summary of a saved memory directory.

    Use this with memory saved by emet run dynamem, emet run create-and-print-memory,
    or other runs that write the common memory format (manifest.json, point_cloud.npz, etc.).
    """
    from emet.memory.utils import print_memory_from_path

    print_memory_from_path(str(path))


@main.group("connect", short_help="Save or show robot connection (host, user) for deploy/view")
def connect_cmd() -> None:
    """Save and reuse connection details so deploy and view-bridge default to the right robot."""
    pass


@connect_cmd.command("save", short_help="Save a connection as active (or named)")
@click.argument("host")
@click.option("--user", "-u", default="root", help="SSH user")
@click.option("--password", "-p", default=None, help="Password (or set EMET_ROBOT_PASSWORD); omit to use SSH key")
@click.option("--name", "-n", default=None, help="Profile name (default: host)")
@click.option("--robot", default=None, help="Emet robot id (e.g. innate_mars) stored in profile")
@click.option(
    "--workspace",
    default=None,
    help="Remote ROS2 workspace on robot (e.g. ~/innate-os/ros2_ws for innate-os Mars)",
)
@click.option("--emet-dir", default=None, help="Remote emet_core install dir (default ~/emet)")
@click.option("--no-active", is_flag=True, help="Do not set as active connection")
def connect_save(
    host: str,
    user: str,
    password: str | None,
    name: str | None,
    robot: str | None,
    workspace: str | None,
    emet_dir: str | None,
    no_active: bool,
) -> None:
    """Save host and user; optional password.

    When saved as active (default), also updates ``~/.stretch/robot_ip.txt`` for legacy tools.
    Use ``--no-active`` to add/update a named profile without changing the active host.
    """
    pwd = password or os.environ.get("EMET_ROBOT_PASSWORD")
    from emet.utils.connection import save_connection

    conn_name = save_connection(
        host=host,
        user=user,
        password=pwd,
        name=name,
        set_active=not no_active,
        workspace=workspace,
        emet_dir=emet_dir,
        robot=robot,
    )
    click.echo(f"Saved connection '{conn_name}' (host={host}, user={user}).")
    if not no_active:
        click.echo("Set as active. Use: emet deploy, emet view-bridge (omit --robot-ip to use this).")


@connect_cmd.command("list", short_help="List saved connections")
def connect_list() -> None:
    """List all saved connections and which is active."""
    from emet.utils.connection import list_connections

    items = list_connections()
    if not items:
        click.echo("No connections saved. Use: emet connect save <host> [--user USER]")
        return
    for name, is_active in items:
        mark = " (active)" if is_active else ""
        click.echo(f"  {name}{mark}")


@connect_cmd.command("show", short_help="Show active connection")
def connect_show() -> None:
    """Show the active connection used by deploy and view-bridge when --robot-ip is omitted."""
    from emet.utils.connection import get_active_connection

    conn = get_active_connection()
    if conn is None:
        click.echo("No active connection. Use: emet connect save <host> [--user USER]")
        sys.exit(1)
    click.echo(f"host: {conn.get('host', '')}")
    click.echo(f"user: {conn.get('user', '')}")
    if conn.get("robot"):
        click.echo(f"robot: {conn.get('robot')}")
    if conn.get("workspace"):
        click.echo(f"workspace: {conn.get('workspace')}")
    if conn.get("emet_dir"):
        click.echo(f"emet_dir: {conn.get('emet_dir')}")
    if "password" in conn:
        click.echo("password: (set)")


@main.group("mars", short_help="Innate Mars hardware bridge (innate-os + ZMQ)")
def mars_cmd() -> None:
    """Deploy and start the innate Mars ZMQ bridge on a Jetson running innate-os."""
    pass


@mars_cmd.command("start", short_help="Deploy (optional) and start ZMQ bridge on robot")
@click.option("--ip", "--host", "-H", "host", default=None, help="Robot hostname or IP")
@click.option("--username", "--user", "-u", "user", default=None, help="SSH user (e.g. jetson1)")
@click.option("--password", "-p", default=None, help="SSH password (or EMET_ROBOT_PASSWORD)")
@click.option("--connection", "-c", "connection_name", default=None, help="Saved connection profile")
@click.option("--deploy", is_flag=True, help="Rsync emet_core + bridge and colcon build before start")
@click.option(
    "--onboard-da3",
    is_flag=True,
    help="Run Depth Anything 3 on the Jetson; publish depth over ZMQ (implies --deploy when set)",
)
@click.option("--preview", is_flag=True, help="Run preview-cameras after bridge startup")
@click.option("--wait-s", default=20.0, show_default=True, help="Seconds to wait before status check")
@click.option("--no-save", is_flag=True, help="Do not update saved connection profile")
def mars_start_cmd(
    host: str | None,
    user: str | None,
    password: str | None,
    connection_name: str | None,
    deploy: bool,
    onboard_da3: bool,
    preview: bool,
    wait_s: float,
    no_save: bool,
) -> None:
    """Start innate_mars_bridge on the robot (inside innate-os tmux + Zenoh).

    Requires innate-os running on the robot (``innate service start``).

    Examples:
      emet mars start --ip herman --username jetson1
      emet mars start --ip herman --username jetson1 --deploy --preview
      emet mars start --connection herman --onboard-da3 --deploy
    """
    from emet.mars import mars_start

    if onboard_da3 and not deploy:
        deploy = True

    mars_start(
        host=host,
        user=user,
        password=password,
        connection_name=connection_name,
        save_profile=not no_save,
        deploy=deploy,
        preview=preview,
        onboard_da3=onboard_da3,
        wait_s=wait_s,
    )


@mars_cmd.command("stop", short_help="Stop ZMQ bridge on robot")
@click.option("--ip", "--host", "-H", "host", default=None, help="Robot hostname or IP")
@click.option("--username", "--user", "-u", "user", default=None, help="SSH user")
@click.option("--password", "-p", default=None, help="SSH password (or EMET_ROBOT_PASSWORD)")
@click.option("--connection", "-c", "connection_name", default=None, help="Saved connection profile")
def mars_stop_cmd(
    host: str | None,
    user: str | None,
    password: str | None,
    connection_name: str | None,
) -> None:
    """Stop innate_mars_bridge on the robot."""
    from emet.mars import resolve_mars_target, stop_bridge_on_robot

    host, user, password, _, _ = resolve_mars_target(
        host=host,
        user=user,
        password=password,
        connection_name=connection_name,
    )
    stop_bridge_on_robot(host, user, password)


@mars_cmd.command("status", short_help="Show bridge process and ZMQ ports on robot")
@click.option("--ip", "--host", "-H", "host", default=None, help="Robot hostname or IP")
@click.option("--username", "--user", "-u", "user", default=None, help="SSH user")
@click.option("--password", "-p", default=None, help="SSH password (or EMET_ROBOT_PASSWORD)")
@click.option("--connection", "-c", "connection_name", default=None, help="Saved connection profile")
def mars_status_cmd(
    host: str | None,
    user: str | None,
    password: str | None,
    connection_name: str | None,
) -> None:
    """Print bridge process, ZMQ ports, and recent tmux log on the robot."""
    from emet.mars import bridge_status_on_robot, resolve_mars_target

    host, user, password, _, _ = resolve_mars_target(
        host=host,
        user=user,
        password=password,
        connection_name=connection_name,
    )
    bridge_status_on_robot(host, user, password, profile=connection_name or host)


@main.command("view-bridge", short_help="View images and state from robot bridge")
@click.option("--robot-ip", "--robot_ip", default="", help="Robot IP (default: active connection)")
def view_bridge(robot_ip: str) -> None:
    """Connect to the robot's ZMQ bridge and display head/EE camera images and state.
    Use after starting the bridge on the robot (e.g. ros2 launch innate_mars_bridge server.launch.py).
    """
    sys.exit(_run_module("emet.app.view_bridge", ["--robot-ip", robot_ip] if robot_ip else []))


@main.command(
    "preview-cameras",
    short_help="Montage robot cameras (local MJCF or ZMQ) for diagnostics",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
@click.pass_context
def preview_cameras(ctx: click.Context) -> None:
    """Save a PNG strip of stereo + arm cameras, optionally post to Discord.

    Runs ``emet.app.preview_robot_cameras``: default local merged scene MJCF preview; use ``--source zmq``
    for one frame from observation port (4401).

    Examples:
      emet preview-cameras
      emet preview-cameras --source zmq --robot innate_mars
      emet preview-cameras --discord --caption "check head aim"
      emet preview-cameras --nod --nod-out-dir ./nod_caps --nod-motion bounce
      emet preview-cameras --nod --nod-arm --nod-out-dir ./nod_caps --nod-arm-joint joint5
    """
    sys.exit(_run_module("emet.app.preview_robot_cameras", list(ctx.args)))


@main.command(short_help="Deploy emet_core and innate_mars_bridge to robot")
@click.option("--host", "-H", default=None, help="Robot host (default: active connection)")
@click.option("--user", "-u", default=None, help="SSH user (default: from connection or root)")
@click.option("--password", "-p", default=None, help="SSH password (or EMET_ROBOT_PASSWORD)")
@click.option("--connection", "-c", "connection_name", default=None, help="Use saved connection by name")
@click.option("--workspace", "-w", default="~/ament_ws", help="Remote ROS2 workspace path")
@click.option("--emet-dir", default="~/emet", help="Remote dir for emet_core (e.g. ~/emet)")
@click.option("--start-bridge", is_flag=True, help="Start bridge on robot after deploy (nohup in background)")
def deploy(
    host: str | None,
    user: str | None,
    password: str | None,
    connection_name: str | None,
    workspace: str,
    emet_dir: str,
    start_bridge: bool,
) -> None:
    """Deploy emet_core and innate_mars_bridge to the robot via rsync and SSH.

    Syncs src/emet_core and src/innate_mars_bridge to the robot, runs pip install
    for emet_core, and colcon build for the bridge. Use emet connect save <host> first
    to set default host/user, or pass --host and --user.

    Examples:
      emet connect save 192.168.1.43 --user jetson1
      emet deploy
      emet deploy --host 192.168.1.43 --user jetson1 --start-bridge
    """
    from emet.deploy import deploy as deploy_impl

    deploy_impl(
        host=host,
        user=user,
        password=password,
        connection_name=connection_name,
        workspace=workspace,
        emet_dir=emet_dir,
        start_bridge=start_bridge,
        root=_project_root(),
    )


@main.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@click.argument(
    "app",
    type=click.Choice(
        [
            "dynamem",
            "scene-graph",
            "graph-eqa",
            "dynagraph",
            "mapping",
            "grasp",
            "chat",
            "agent",
            "web-chat",
            "ai_pickup",
            "timing",
            "discord",
            "create-and-print-memory",
            "molmospaces-explore",
            "debug-da3-depth",
            "debug-lingbot-depth",
            "graph-eqa-habitat",
        ]
    ),
)
@click.option("--robot-ip", "--robot_ip", default="127.0.0.1", help="Robot or simulator IP")
@click.option(
    "--robot",
    default="stretch",
    help=(
        "Robot backend (stretch, innate_mars, rby1, galaxea_r1). Must match emet serve mujoco --robot. "
        "If you omit this flag, ``emet run`` does not forward ``--robot`` to sub-apps—so e.g. "
        "``emet run agent --agent-config configs/agent_innate_mars.yaml`` uses top-level ``robot:`` from that YAML."
    ),
)
@click.option("--server-ip", "--server_ip", default="127.0.0.1", help="Server IP (e.g. for AnyGrasp)")
@click.option("-S", "--skip", "skip_confirmations", is_flag=True, help="Skip confirmations")
@click.option("--headless", is_flag=True, help="Run without display")
@click.option("--visual-servo", "-V", "--visual_servo", is_flag=True, help="Use visual servoing (dynamem)")
@click.option("--target-object", "--target_object", help="Target object to grasp (grasp) or pick (dynamem)")
@click.option("--target-receptacle", "--target_receptacle", help="Target receptacle to place on (dynamem)")
@click.option("--parameter-file", "--parameter_file", help="Planner config (e.g. sim_planner.yaml)")
@click.option(
    "--port-offset",
    default=0,
    type=int,
    help="Add to default ZMQ ports (e.g. 100 → 4501-4504). Must match the server.",
)
@click.pass_context
def run(
    ctx: click.Context,
    app: str,
    robot_ip: str,
    robot: str,
    server_ip: str,
    skip_confirmations: bool,
    headless: bool,
    visual_servo: bool,
    target_object: str | None,
    target_receptacle: str | None,
    parameter_file: str | None,
    port_offset: int,
) -> None:
    """Run a robot agent or app.

    Unknown options (e.g. --match-method, --rerun-debug) are passed through to the app.

    Examples:
      emet run dynamem --robot-ip 127.0.0.1 -S
      emet run dynamem -S --port-offset 100
      emet run dynamem -S --visual-servo --match-method class --target-object apple --target-receptacle plate
      emet run molmospaces-explore --output-dir ./ep0 --steps 40
      emet run molmospaces-explore --start-sim --robot xlerobot --scene ithor --headless \\
        --output-dir ./ep0 --steps 40
      # --robot optional: omit to read emet_robot_id from the running ZMQ server
      emet run mapping --robot-ip 127.0.0.1
      emet run grasp --target-object "red cylinder" --parameter-file sim_planner.yaml
      emet run discord --robot-ip 192.168.1.15 --task pickup   # requires DISCORD_TOKEN in env
      emet run debug-da3-depth --robot innate_mars   # DA3 depth + point cloud in Rerun (or: emet debug-da3-depth)
    """
    _require_repo_venv_when_in_repo()
    args = list(ctx.args)
    if app != "graph-eqa-habitat":
        args.extend(["--robot_ip", robot_ip])
    if app in _EMET_RUN_APPS_WITH_ROBOT:
        # Do not inject ``--robot stretch`` when the user omitted ``--robot`` on ``emet run``: the wrapper's
        # default would override ``robot:`` from ``--agent-config`` (run_agent) or MolmoSpaces discovery
        # (robot_backend=None). Forward ``--robot`` only when explicitly set (CLI or env).
        if ctx.get_parameter_source("robot") != ParameterSource.DEFAULT:
            args.extend(["--robot", robot])
    if port_offset:
        args.extend(["--port-offset", str(port_offset)])
    if app == "dynamem":
        args.extend(["--server_ip", server_ip])
        if skip_confirmations:
            args.append("-S")
        if headless:
            args.append("--headless")
        if visual_servo:
            args.append("--visual-servo")
        if target_object:
            args.extend(["--target_object", target_object])
        if target_receptacle:
            args.extend(["--target_receptacle", target_receptacle])
        sys.exit(_run_module("emet.app.run_dynamem", args))
    elif app == "scene-graph":
        args.extend(["--server_ip", server_ip])
        if skip_confirmations:
            args.append("-S")
        if headless:
            args.append("--headless")
        if visual_servo:
            args.append("--visual-servo")
        if target_object:
            args.extend(["--target_object", target_object])
        if target_receptacle:
            args.extend(["--target_receptacle", target_receptacle])
        sys.exit(_run_module("emet.app.run_scene_graph", args))
    elif app == "graph-eqa":
        sys.exit(_run_module("emet.app.run_graph_eqa", args))
    elif app == "dynagraph":
        sys.exit(_run_module("emet.app.run_dynagraph", args))
    elif app == "graph-eqa-habitat":
        sys.exit(_run_module("emet.app.run_graph_eqa_habitat", args))
    elif app == "molmospaces-explore":
        if headless:
            args.append("--headless")
        sys.exit(_run_module("emet.app.run_molmospaces_explore", args))
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
    elif app == "agent":
        sys.exit(_run_module("emet.app.run_agent", args))
    elif app == "web-chat":
        sys.exit(_run_module("emet.app.web_chat", args))
    elif app == "ai_pickup":
        if skip_confirmations:
            args.append("-S")
        sys.exit(_run_module("emet.app.ai_pickup", args))
    elif app == "timing":
        args.extend(["--robot_ip", robot_ip])
        if headless:
            args.append("--headless")
        sys.exit(_run_module("emet.app.timing", args))
    elif app == "discord":
        args.extend(["--robot_ip", robot_ip])
        if server_ip:
            args.extend(["--server_ip", server_ip])
        if parameter_file:
            args.extend(["--parameter_file", parameter_file])
        sys.exit(_run_module("emet.app.run_discord", args))
    elif app == "create-and-print-memory":
        args.extend(["--robot-ip", robot_ip])
        sys.exit(_run_module("emet.app.create_and_print_memory", args))
    elif app == "debug-da3-depth":
        sys.exit(_run_module("emet.app.debug_da3_depth", args))
    elif app == "debug-lingbot-depth":
        sys.exit(_run_module("emet.app.debug_lingbot_depth", args))
    else:
        click.echo(f"Unknown app: {app}", err=True)
        sys.exit(1)


_SYNC_ALL_EXTRAS = ("dev", "sim", "hand_tracker", "dynamem", "da3")


@main.command(short_help="Sync dependencies (uv or pip)")
@click.option("--extra", "-e", "extra_list", multiple=True, help="Extra to install (sim, dynamem, dev, etc.)")
@click.option(
    "--all",
    "sync_all",
    is_flag=True,
    help="Install all common extras (same as defaults: dev, sim, hand_tracker, dynamem, da3)",
)
@click.option("--sim", is_flag=True, help="Include sim (MuJoCo, robocasa)")
@click.option("--dynamem", "dynamem_flag", is_flag=True, help="Include dynamem (SAM-2)")
@click.option("--dev", "dev_flag", is_flag=True, help="Include dev (pytest, black, mypy)")
@click.option("--hand-tracker", is_flag=True, help="Include hand_tracker (mediapipe)")
@click.option("--no-install", is_flag=True, help="Only sync lockfile, do not install emet")
def sync(
    extra_list: tuple[str, ...],
    sync_all: bool,
    sim: bool,
    dynamem_flag: bool,
    dev_flag: bool,
    hand_tracker: bool,
    no_install: bool,
) -> None:
    """Sync dependencies (uv sync or pip install -e .).

    With **uv**, a plain ``emet sync`` runs ``uv sync``, which installs
    **default dependency groups** from ``pyproject.toml`` (dev, sim, hand_tracker,
    dynamem, da3). Use ``uv sync --no-default-groups`` for base dependencies only.

    robosuite/robocasa are installed editable by: ``emet install sim`` (not from the lockfile).
    A plain ``emet sync`` reinstalls them when present under ``third_party/`` (default groups
    include sim). ``emet sync -e dynamem`` alone does not.

    Examples:

      emet sync
      emet sync --all
      emet sync -e sim -e dynamem
      emet sync --sim --dynamem
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
    # Deduplicate, preserve order
    extras = list(dict.fromkeys(extras))

    root = _project_root()
    os.chdir(root)

    # Sim extra: pip-installable deps (mujoco, stretch-urdf, etc.) are in pyproject.toml.
    # robosuite/robocasa live in third_party/ and are *not* lockfile deps — ``uv sync`` removes them unless
    # we reinstall with ``uv pip install -e`` after sync whenever sim is part of this sync.
    want_sim_third_party = (not extras) or ("sim" in extras)
    if want_sim_third_party:
        missing = [name for name in ("robosuite", "robocasa") if not (root / "third_party" / name).is_dir()]
        if missing:
            click.echo(
                f"Note: third_party missing ({', '.join(missing)}). "
                "Sim pip deps (mujoco, etc.) will install, but robosuite/robocasa need: emet install sim",
                err=True,
            )

    if extras:
        click.echo("Syncing extras: " + ", ".join(extras))

    if _has_uv():
        cmd = ["uv", "sync"]
        for e in extras:
            cmd.extend(["--extra", e])
        if no_install:
            cmd.append("--no-install-project")
        result = subprocess.call(cmd)
        if result != 0:
            sys.exit(result)
        if want_sim_third_party:
            result = _uv_pip_install_sim_third_party_editables(root)
            if result != 0:
                sys.exit(result)
        sys.exit(result)
    else:
        # Fallback to pip
        extras_str = "[" + ",".join(extras) + "]" if extras else ""
        spec = f".{extras_str}"
        result = subprocess.call([sys.executable, "-m", "pip", "install", "-e", spec])
        sys.exit(result)


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


@main.command(
    short_help="Run pytest (use uv: uv run emet test)",
    context_settings={**_CONTEXT_SETTINGS, "ignore_unknown_options": True},
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--no-cov", "no_cov", is_flag=True, help="Disable coverage")
@click.option(
    "--no-sim",
    "no_sim_tests",
    is_flag=True,
    help="Disable sim tests (RUN_SIM_TESTS=0); sim tests run by default",
)
@click.argument("pytest_args", nargs=-1, type=click.UNPROCESSED)
def test(
    verbose: bool,
    no_cov: bool,
    no_sim_tests: bool,
    pytest_args: tuple[str, ...],
) -> None:
    """Run tests with pytest.

    Uses the project environment: run from repo root with uv so dev deps (pytest,
    pytest-timeout) are available. Sim tests (e.g. red cylinder in MuJoCo) run by default;
    use --no-sim to skip them for a faster run.

      uv sync
      uv run emet test
      uv run emet test -v
      uv run emet test --no-sim           # skip sim tests (faster)
      uv run emet test -v src/test/memory/test_memory_backends_smoke.py
      uv run emet test src/test/mapping/test_red_cylinder_in_sim.py -k innate_mars
      uv run emet test -k test_red_cylinder
      Heavy VLLM tests (@pytest.mark.vllm_load) are excluded by default; see docs/plans/TESTING_VLLM_LOAD.md
    """
    root = _project_root()
    os.chdir(root)
    env = os.environ.copy()
    if no_sim_tests:
        env["RUN_SIM_TESTS"] = "0"
    else:
        env["RUN_SIM_TESTS"] = "1"
    # Prefer project .venv so pytest and deps match the project (e.g. pytest-timeout)
    venv_py = _project_venv_python()
    python = str(venv_py) if venv_py is not None else sys.executable
    src = root / "src"
    if src.exists():
        # Prepend project src; drop ROS site-packages so ament-* pytest plugins
        # (auto-loaded via PYTHONPATH) do not break src/test/habitat collection.
        prev_parts = [
            p
            for p in env.get("PYTHONPATH", "").split(os.pathsep)
            if p and "/opt/ros/" not in p.replace("\\", "/")
        ]
        env["PYTHONPATH"] = os.pathsep.join([str(src), *prev_parts])

    cmd = [python, "-m", "pytest"]
    if verbose:
        cmd.append("-v")
    if not no_cov and (root / "pyproject.toml").exists():
        try:
            import pytest_cov  # noqa: F401

            cmd.extend(["--cov=emet", "--cov-report=term-missing"])
        except ImportError:
            pass
    cmd.extend(list(pytest_args))
    if not pytest_args:
        # pytest uses testpaths from pyproject.toml ([tool.pytest.ini_options] testpaths = ["src/test"])
        pass
    sys.exit(subprocess.call(cmd, env=env, cwd=root))


_SIM_THIRD_PARTY_DIRS = ("robosuite", "robosuite_models", "robocasa")


def _third_party_sim_editable_paths(root: Path) -> list[Path]:
    """Paths under ``third_party/`` to install with ``uv pip install -e`` (not declared in uv.lock)."""
    return [root / "third_party" / name for name in _SIM_THIRD_PARTY_DIRS if (root / "third_party" / name).is_dir()]


def _uv_pip_install_sim_third_party_editables(root: Path) -> int:
    """Re-register robosuite / robosuite_models / robocasa after ``uv sync`` (sync prunes undeclared packages)."""
    paths = _third_party_sim_editable_paths(root)
    if not paths or not _has_uv():
        return 0
    click.echo("Installing robosuite/robocasa from third_party (not in uv.lock)...")
    args: list[str] = []
    for p in paths:
        args.extend(["-e", str(p)])
    # Do not use ``--no-deps``: robocasa/robosuite need install_requires (e.g. h5py, imageio).
    return subprocess.call(["uv", "pip", "install"] + args, cwd=root)


@main.command("clean", short_help="Remove third-party sim clones (robosuite, robocasa, etc.)")
@click.option(
    "-y",
    "--yes",
    "skip_confirm",
    is_flag=True,
    help="Skip confirmation; remove without prompting",
)
def clean(skip_confirm: bool) -> None:
    """Remove third-party directories created by emet install sim / install robocasa.

    Deletes third_party/robosuite, third_party/robosuite_models, and third_party/robocasa.
    Re-run emet install sim (or install robocasa) to clone and install them again.

    Examples:
      emet clean
      emet clean -y
    """
    root = _project_root()
    third_party = root / "third_party"
    if not third_party.exists():
        click.echo("Nothing to clean: third_party/ does not exist.")
        return
    to_remove = [third_party / name for name in _SIM_THIRD_PARTY_DIRS if (third_party / name).exists()]
    if not to_remove:
        click.echo("Nothing to clean: none of third_party/robosuite, robosuite_models, robocasa exist.")
        return
    if not skip_confirm:
        click.echo("The following directories will be removed:")
        for p in to_remove:
            click.echo(f"  {p}")
        click.confirm("Continue?", default=False, abort=True)
    for p in to_remove:
        click.echo(f"Removing {p}...")
        shutil.rmtree(p)
    click.echo("Done. Re-run emet install sim to reinstall.")


@main.group(
    short_help="Install submodules, sim, full setup, pre-commit",
    invoke_without_command=True,
)
@click.pass_context
def install(ctx: click.Context) -> None:
    """Install submodules, simulation extras, or full setup.

    With no subcommand, opens the interactive install menu (manage sub-assets and sync).
    Use a subcommand for direct install: emet install sim, emet install full, etc.
    """
    if ctx.invoked_subcommand is not None:
        return
    from emet.install_ui import run_install_menu

    sys.exit(run_install_menu())


@install.command("gh", short_help="Install GitHub CLI (apt)")
@click.option("-y", "--yes", "non_interactive", is_flag=True, help="Run apt-get without prompting")
def install_gh(non_interactive: bool) -> None:
    """Install the GitHub CLI (``gh``) for pull requests and issues.

    Package name is declared in ``pyproject.toml`` under ``[tool.emet.system-packages]``.
    After install, authenticate once: ``gh auth login``.

    Examples:
      emet install gh -y
    """
    from emet.dev_system_packages import ensure_apt_package

    sys.exit(ensure_apt_package("gh", non_interactive=non_interactive))


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
    skip_download_assets: bool = False,
    setup_macros: bool = False,
) -> int:
    """Run scripts/install_simulation.sh (robosuite + robocasa). Returns exit code.
    Asset download is part of installation by default; pass skip_download_assets to omit.
    """
    script = root / "scripts" / "install_simulation.sh"
    if not script.exists():
        click.echo(f"Script not found: {script}", err=True)
        return 1
    args = []
    if skip_download_assets:
        args.append("-n")
    if setup_macros:
        args.append("-a")
    env = os.environ.copy()
    env["EMET_PYTHON"] = sys.executable
    if _has_uv():
        env["EMET_USE_UV"] = "1"
    return subprocess.call(["bash", str(script)] + args, env=env)


@install.command("sim", short_help="Install Robocasa, robosuite")
@click.option(
    "-n",
    "--no-download-assets",
    "skip_download_assets",
    is_flag=True,
    help="Skip downloading Robocasa kitchen assets (~10GB)",
)
@click.option(
    "-a",
    "--setup-macros",
    is_flag=True,
    help="Force run macro setup (overwrite existing macros_private.py); by default macros are set up only when missing",
)
@click.option("--no-sync", is_flag=True, help="Skip uv sync and third_party reinstall after clone/install")
def install_sim(skip_download_assets: bool, setup_macros: bool, no_sync: bool) -> None:
    """Install simulation third-party deps (Robocasa + robosuite).

    Clones robosuite and robocasa, runs macro setup when missing (silences
    "No private macro file" warnings), downloads kitchen assets (~10GB), then
    runs ``uv sync`` and reinstalls editable ``third_party/`` robosuite /
    robosuite_models / robocasa (not in uv.lock; sync would otherwise remove them).
    Use -n to skip the asset download (e.g. CI).

    Examples:
      emet install sim
      emet install sim -a
      emet install sim --no-download-assets --no-sync
    """
    root = _project_root()
    result = _run_install_simulation(root, skip_download_assets, setup_macros)
    if result != 0:
        sys.exit(result)
    if no_sync:
        click.echo("Simulation install complete. Run: uv sync")
        return
    click.echo("Syncing project env (uv sync); then reinstalling third_party sim packages...")
    os.chdir(root)
    if _has_uv():
        result = subprocess.call(["uv", "sync"], cwd=root)
        if result == 0:
            result = _uv_pip_install_sim_third_party_editables(root)
    else:
        result = subprocess.call([sys.executable, "-m", "pip", "install", "-e", ".[sim]"])
    if result == 0:
        click.echo("Simulation install complete.")
        click.echo("Verify: uv run python -c \"import robocasa; print('robocasa OK')\"")
    sys.exit(result)


@install.command("robocasa", short_help="Install Robocasa (same as install sim)")
@click.option(
    "-n",
    "--no-download-assets",
    "skip_download_assets",
    is_flag=True,
    help="Skip downloading Robocasa kitchen assets (~10GB)",
)
@click.option(
    "-a",
    "--setup-macros",
    is_flag=True,
    help="Force run macro setup (overwrite existing macros_private.py); by default macros are set up only when missing",
)
@click.option("--no-sync", is_flag=True, help="Skip uv sync and third_party reinstall after clone/install")
def install_robocasa(skip_download_assets: bool, setup_macros: bool, no_sync: bool) -> None:
    """Install Robocasa and robosuite (same as emet install sim).

    Clones robosuite and robocasa, runs macro setup when missing, downloads
    kitchen assets (~10GB), then runs ``uv sync`` and reinstalls editable
    third_party sim packages. Use -n to skip the asset download.

    Examples:
      emet install robocasa
      emet install robocasa -a
      emet install robocasa --no-download-assets
    """
    root = _project_root()
    result = _run_install_simulation(root, skip_download_assets, setup_macros)
    if result != 0:
        sys.exit(result)
    if no_sync:
        click.echo("Robocasa install complete. Run: uv sync")
        return
    click.echo("Syncing project env (uv sync); then reinstalling third_party sim packages...")
    os.chdir(root)
    if _has_uv():
        result = subprocess.call(["uv", "sync"], cwd=root)
        if result == 0:
            result = _uv_pip_install_sim_third_party_editables(root)
    else:
        result = subprocess.call([sys.executable, "-m", "pip", "install", "-e", ".[sim]"])
    if result == 0:
        click.echo("Robocasa install complete.")
        click.echo("Verify: uv run python -c \"import robocasa; print('robocasa OK')\"")
    sys.exit(result)


@install.command("menu", short_help="Interactive menu to manage sub-assets")
@click.option(
    "--text-only",
    "text_only",
    is_flag=True,
    help="ASCII menu only (skip Rich plan wizard when Rich is installed).",
)
def install_menu(text_only: bool) -> None:
    """Open a menu to install or update sub-assets.

    With Rich (dev extra), starts a colored plan: choose defaults, then review
    before running commands. Use --text-only for the legacy ASCII asset menu only.

    Examples:
      emet install menu
      emet install menu --text-only
    """
    from emet.install_ui import run_install_menu

    sys.exit(run_install_menu(text_only=text_only))


@install.command("full", short_help="Run full install (install.sh)")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompts (non-interactive apt, link emet)")
@click.option(
    "--profile",
    type=click.Choice(["minimal", "standard", "full"], case_sensitive=False),
    default=None,
    help=(
        "Install profile forwarded to install.sh / EMET_INSTALL_PROFILE: "
        "standard (default)=no sim unless --sim; full=legacy sim-on-by-default; minimal=same as standard today."
    ),
)
@click.option("--sim", is_flag=True, help="Include simulation extras")
@click.option("--cpu", is_flag=True, help="CPU-only (skip SAM2)")
@click.option("--no-sam2", is_flag=True, help="Skip Segment Anything 2")
@click.option(
    "--molmospaces",
    is_flag=True,
    help="Create .venv-molmospaces (MolmoSpaces; Python 3.11+). Forwards to install.sh --molmospaces",
)
@click.option(
    "--no-molmospaces",
    "no_molmospaces",
    is_flag=True,
    help="Skip MolmoSpaces wrapper even when sim installs (forwards to install.sh --no-molmospaces).",
)
@click.option(
    "--all",
    "install_all",
    is_flag=True,
    help="Same as install.sh --all (includes MolmoSpaces among other bundles)",
)
def install_full(
    yes: bool,
    profile: str | None,
    sim: bool,
    cpu: bool,
    no_sam2: bool,
    molmospaces: bool,
    no_molmospaces: bool,
    install_all: bool,
) -> None:
    """Run full install (./install.sh).

    Installs uv, system deps, git-lfs, and syncs dependencies.

    By default the shell script does **not** install Robocasa/simulation; pass ``--sim`` or ``--all``,
    or set ``EMET_INSTALL_PROFILE=full`` for the old behavior. With sim enabled, ``install.sh`` also
    creates ``.venv-molmospaces`` when ``packages/emet_molmospaces`` is present unless you pass
    ``--no-molmospaces``. ``-y`` does not enable MolmoSpaces by itself — pass ``--molmospaces`` or ``--all``.

    Examples:
      emet install full
      emet install full -y --sim
      emet install full -y --profile full
      emet install full -y --no-molmospaces
      emet install full -y --molmospaces
      emet install full -y --all
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
    if profile:
        args.append(f"--profile={profile}")
    if sim:
        args.append("--sim")
    if cpu:
        args.append("--cpu")
    if no_sam2:
        args.append("--no-sam2")
    if install_all:
        args.append("--all")
    if molmospaces:
        args.append("--molmospaces")
    if no_molmospaces:
        args.append("--no-molmospaces")
    env = os.environ.copy()
    if profile:
        env["EMET_INSTALL_PROFILE"] = profile.lower()
    result = subprocess.call(["bash", str(script)] + args, env=env)
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
        result = subprocess.call([sys.executable, "-m", "pre_commit", "run", "--all-files"])
        sys.exit(result)
    sys.exit(0)


@main.command("install-completion", short_help="Print shell completion script")
@click.option(
    "--shell",
    "-s",
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


from emet.app.capture import main as _capture_app  # noqa: E402

_capture_app.short_help = "One ZMQ frame + metadata (shortcut: zmq_obs capture profile)"
main.add_command(_capture_app)

from emet.app.stream import main as _stream_app  # noqa: E402

_stream_app.short_help = "Live ZMQ → Rerun (shortcut: zmq_obs stream profile)"
main.add_command(_stream_app)

# Full Click options (not a thin wrapper) so `emet debug-da3-depth --help` lists all flags.
from emet.app.debug_da3_depth import main as _debug_da3_depth_app  # noqa: E402

_debug_da3_depth_app.short_help = "Live DA3 depth + point cloud from ZMQ (Rerun)"
main.add_command(_debug_da3_depth_app)

from emet.app.debug_lingbot_depth import main as _debug_lingbot_depth_app  # noqa: E402

_debug_lingbot_depth_app.short_help = "Live LingBot-Map depth + pose from ZMQ (Rerun)"
main.add_command(_debug_lingbot_depth_app)

from emet.app.export_sim_gt import main as _export_sim_gt_app  # noqa: E402

_export_sim_gt_app.short_help = "Export Robocasa sim GT objects (3D bounds + head 2D boxes)"
main.add_command(_export_sim_gt_app)

from emet.app.tune_graph_fusion import main as _tune_graph_fusion_app  # noqa: E402

_tune_graph_fusion_app.short_help = "Grid-search GraphObjectFusion vs GT + calibration frames"
main.add_command(_tune_graph_fusion_app)

from emet.app.eval_calibration import main as _eval_calibration_app  # noqa: E402

_eval_calibration_app.short_help = "Score calibration frames vs sim GT (spatial recall)"
main.add_command(_eval_calibration_app)

from emet.app.eval_dynagraph import main as _eval_dynagraph_app  # noqa: E402

_eval_dynagraph_app.short_help = "Unified Dynagraph episode eval (explore, graph, fusion, EQA)"
main.add_command(_eval_dynagraph_app)

from emet.app.eval_sqa3d import eval_sqa3d_main as _eval_sqa3d_app  # noqa: E402
from emet.app.eval_sqa3d import sqa3d_group as _sqa3d_group  # noqa: E402

_eval_sqa3d_app.short_help = "Score SQA3D QA predictions (EM@1)"
main.add_command(_eval_sqa3d_app)
main.add_command(_sqa3d_group)

from emet.app.eval_robovista import robovista_group as _robovista_group  # noqa: E402

_robovista_group.short_help = "RoboVista offline robot-centric MCQ-VQA"
main.add_command(_robovista_group)

if __name__ == "__main__":
    main()

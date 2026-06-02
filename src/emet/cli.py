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

import os
import shutil
import subprocess
import sys
from pathlib import Path

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
    }
)


def _project_root() -> Path:
    """Return project root (parent of src/emet)."""
    return Path(__file__).resolve().parent.parent.parent


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
    root = _project_root()
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
    try:
        os.execvpe("uv", ["uv", "run", "emet"] + sys.argv[1:], env)
    except Exception:
        pass


def _project_venv_python() -> Path | None:
    """Return the project .venv Python path if it exists."""
    root = _project_root()
    for name in ("python", "python3"):
        p = root / ".venv" / "bin" / name
        if p.exists():
            return p
    return None


def _run_module(module: str, args: list[str], env: dict | None = None) -> int:
    """Run a Python module. Returns exit code."""
    env = env or os.environ.copy()
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


@main.command(short_help="Start simulation server (mujoco or robocasa)")
@click.argument(
    "backend",
    type=click.Choice(["mujoco", "robocasa"]),
    default="mujoco",
)
@click.option(
    "--use-robocasa",
    is_flag=True,
    help="Use Robocasa for scene generation (ignored if backend is robocasa).",
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
@click.option("--scene-path", type=click.Path(exists=True), help="Path to MuJoCo scene XML")
@click.option(
    "--molmospaces-scene",
    default=None,
    metavar="NAME",
    help=(
        "MolmoSpaces scene (e.g. ithor). Runs merge via emet-molmospaces into a temp MJCF, "
        "then starts the ZMQ server (same as merge-scene + emet serve mujoco). Requires wrapper: "
        "install.sh --molmospaces. First-time use may need scene packages under MLSPACES_ASSETS_DIR "
        "(you will be prompted to download unless --molmospaces-install). "
        "Incompatible with --scene-path and --use-robocasa."
    ),
)
@click.option(
    "--molmospaces-split",
    default="train",
    type=click.Choice(["train", "val", "test"]),
    help="Split when using --molmospaces-scene.",
)
@click.option(
    "--molmospaces-index",
    default=0,
    type=int,
    help="Scene index when using --molmospaces-scene.",
)
@click.option(
    "--molmospaces-install",
    is_flag=True,
    help=(
        "If the MolmoSpaces scene archive is not on disk yet, download/link it without prompting "
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
    help="Verbose MolmoSpaces base placement and post-spawn contact diagnostics (non-stretch server).",
)
@click.option(
    "--port-offset",
    default=0,
    type=int,
    help="Add to default ports when 4401 etc. are in use (e.g. 100 → 4501–4504)",
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
    default="stretch",
    help=(
        "Robot to simulate. 'stretch' (default) uses the Stretch-MuJoCo server. "
        "Registry robots (e.g. innate_mars, rby1, galaxea_r1) load the default table "
        "scene merged with that robot's MJCF and use the generic ZMQ sim (RobosuiteZmqServer). "
        "Robosuite-native names (PandaOmron, Tiago, GR1) use the stock robosuite robot in Robocasa."
    ),
)
@click.argument("extra", nargs=-1, type=click.UNPROCESSED)
def serve(
    backend: str,
    use_robocasa: bool,
    headless: bool,
    show_viewer_ui: bool,
    no_cameras: bool,
    use_glx: bool,
    scene_path: str | None,
    molmospaces_scene: str | None,
    molmospaces_split: str,
    molmospaces_index: int,
    molmospaces_install: bool,
    seed: int,
    steps: int | None,
    debug_molmospaces_spawn: bool,
    port_offset: int,
    list_robocasa_tasks: bool,
    robocasa_task: str,
    robot: str,
    extra: tuple[str, ...],
) -> None:
    """Start a simulation server.

    Backends:
      mujoco   MuJoCo server (default). Add --use-robocasa for Robocasa scenes.
      robocasa Shortcut for mujoco with Robocasa (same as emet serve mujoco --use-robocasa).

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
      emet serve mujoco --use-robocasa --port-offset 100
      DISPLAY=:1 emet serve mujoco --molmospaces-scene ithor --robot rby1   # MolmoSpaces + rby1 (needs wrapper)
      emet serve mujoco --molmospaces-scene ithor --headless   # same, headless / no window
    """
    use_robocasa_flag = use_robocasa or (backend == "robocasa")
    if backend == "mujoco" or backend == "robocasa":
        from emet.config.sim_launch_config import build_sim_launch_config_from_serve_cli
        from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv

        if list_robocasa_tasks:
            args = list(extra) + ["--use-robocasa", "--list-robocasa-tasks"]
            sys.exit(_run_module("emet.simulation.mujoco_server", args))

        try:
            cfg = build_sim_launch_config_from_serve_cli(
                molmospaces_scene=molmospaces_scene,
                molmospaces_split=molmospaces_split,
                molmospaces_index=molmospaces_index,
                molmospaces_install=molmospaces_install,
                use_robocasa=use_robocasa_flag,
                scene_path=scene_path,
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
        args = list(extra) + prepare_mujoco_server_argv(cfg)
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


@main.group("molmospaces", short_help="MolmoSpaces scenes and robots (requires emet-molmospaces wrapper)")
def molmospaces_cmd() -> None:
    """Set up MolmoSpaces scenes, list robots (e.g. rby1 / Galaxea R1), and run simulation.

    list-robots works without the wrapper. list-scenes, install-scene, merge-scene, and serve
    require the local emet-molmospaces package (see docs/molmospaces.md). ``export-nerfstudio`` is
    core-only (reads an explore episode directory). Install wrapper with:
      ./install.sh -y   (default sim path)   or   ./install.sh --molmospaces -y   or   editable install of packages/emet_molmospaces
    """


@molmospaces_cmd.command("list-robots", short_help="List supported robot IDs")
def molmospaces_list_robots() -> None:
    """Print MolmoSpaces robot IDs (rby1, rby1m, franka_*, etc.). Default robot is rby1 (Galaxea R1 family)."""
    from emet.simulation.molmospaces_config import DEFAULT_MOLMOSPACES_ROBOT, MOLMOSPACES_ROBOT_IDS

    click.echo("Robots: " + ", ".join(MOLMOSPACES_ROBOT_IDS))
    click.echo(f"Default: {DEFAULT_MOLMOSPACES_ROBOT}")


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
@click.option("--robot", default="rby1", help="Robot ID (default: rby1)")
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
@click.option("--robot", default="rby1", help="Robot ID (default: rby1, Galaxea R1 family)")
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
@click.option("--no-active", is_flag=True, help="Do not set as active connection")
def connect_save(host: str, user: str, password: str | None, name: str | None, no_active: bool) -> None:
    """Save host and user; optional password. Updates ~/.stretch/robot_ip.txt for legacy tools."""
    pwd = password or os.environ.get("EMET_ROBOT_PASSWORD")
    from emet.utils.connection import save_connection

    conn_name = save_connection(
        host=host,
        user=user,
        password=pwd,
        name=name,
        set_active=not no_active,
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
    if "password" in conn:
        click.echo("password: (set)")


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

    Syncs src/emet_core and src/innate_mars_bridge to the robot, runs pip install -e
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
            "debug-circle-rerun",
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
      # --robot optional: omit to read emet_robot_id from the running ZMQ server
      emet run mapping --robot-ip 127.0.0.1
      emet run grasp --target-object "red cylinder" --parameter-file sim_planner.yaml
      emet run discord --robot-ip 192.168.1.15 --task pickup   # requires DISCORD_TOKEN in env
      emet run debug-da3-depth --robot innate_mars   # DA3 depth + point cloud in Rerun (or: emet debug-da3-depth)
      emet run debug-circle-rerun   # pole-ring calibration → Rerun (or: emet debug-circle-rerun)
    """
    args = list(ctx.args)
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
    elif app == "molmospaces-explore":
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
    else:
        click.echo(f"Unknown app: {app}", err=True)
        sys.exit(1)


_SYNC_ALL_EXTRAS = ("dev", "sim", "hand_tracker", "dynamem", "da3")


@main.command(short_help="Sync dependencies (uv or pip)")
@click.option("--extra", "-e", "extra_list", multiple=True, help="Extra to install (sim, dynamem, dev, etc.)")
@click.option("--all", "sync_all", is_flag=True, help="Install all common extras (same as defaults: dev, sim, hand_tracker, dynamem, da3)")
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
    if src.exists() and "PYTHONPATH" not in env:
        env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")

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


# Full Click options (not a thin wrapper) so `emet debug-da3-depth --help` lists all flags.
from emet.app.debug_da3_depth import main as _debug_da3_depth_app  # noqa: E402

_debug_da3_depth_app.short_help = "Live DA3 depth + point cloud from ZMQ (Rerun)"
main.add_command(_debug_da3_depth_app)


if __name__ == "__main__":
    main()

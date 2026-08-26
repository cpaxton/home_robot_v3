# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from emet.cli_cmds.bootstrap import (
    _kill_processes_on_port,
    _run_module,
)
from emet.cli_cmds.habitat import _run_habitat_wrapper


@click.command(short_help="Start simulation server (mujoco, robocasa, molmospaces, habitat) or LLM HTTP API")
@click.argument(
    "backend",
    type=click.Choice(["mujoco", "robocasa", "molmospaces", "habitat", "llm"]),
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
@click.option(
    "--llm",
    "llm_key",
    default="qwen25-7B",
    show_default=True,
    help="For ``serve llm``: emet llm key (e.g. qwen25-14B, qwen35-9B).",
)
@click.option(
    "--host",
    "llm_host",
    default="0.0.0.0",
    show_default=True,
    help="For ``serve llm``: bind address (0.0.0.0 for LAN).",
)
@click.option(
    "--port",
    "llm_port",
    default=8000,
    show_default=True,
    type=int,
    help="For ``serve llm``: HTTP port (OpenAI-compatible /v1).",
)
@click.option(
    "--device",
    "llm_device",
    default="auto",
    show_default=True,
    help="For ``serve llm``: auto | cuda | cpu | mps.",
)
@click.option(
    "--max-tokens",
    "llm_max_tokens",
    default=512,
    show_default=True,
    type=int,
    help="For ``serve llm``: default max_new_tokens.",
)
@click.option(
    "--api-key",
    "llm_api_key",
    default=None,
    help="For ``serve llm``: optional Bearer token (or EMET_LLM_SERVE_API_KEY).",
)
@click.option(
    "--vl/--no-vl",
    "llm_vl",
    default=False,
    show_default=True,
    help="For ``serve llm``: load multimodal VLM (image_url). Default port becomes 8001.",
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
    llm_key: str,
    llm_host: str,
    llm_port: int,
    llm_device: str,
    llm_max_tokens: int,
    llm_api_key: str | None,
    llm_vl: bool,
    extra: tuple[str, ...],
) -> None:
    """Start a simulation server or OpenAI-compatible LLM HTTP API.

    Backends:
      mujoco       MuJoCo server (default). Use --scene robocasa or --scene ithor for other scenes.
      robocasa     Shortcut for ``--scene robocasa``.
      molmospaces  Shortcut for ``--scene ithor`` (or pass scene name positional:
                   ``emet serve molmospaces procthor-10k``).
      llm          OpenAI-compatible text LLM on ``/v1/chat/completions`` (see docs/llm_serve.md).

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
      emet serve llm --llm qwen25-14B --host 0.0.0.0 --port 8000
      emet serve llm --vl --host 0.0.0.0 --port 8001
      emet robots info xlerobot
      emet robots preview-cameras xlerobot --source local
    """
    if backend == "llm":
        from emet.llms.openai_server import (
            DEFAULT_LLM_SERVE_MODEL,
            DEFAULT_VL_SERVE_MODEL,
            DEFAULT_VL_SERVE_PORT,
            resolve_serve_device,
            serve_openai_llm,
        )

        use_vl = bool(llm_vl)
        # Click defaults --llm/--port for all serve backends; nudge VL defaults when --vl.
        resolved_llm = llm_key
        if use_vl and resolved_llm == DEFAULT_LLM_SERVE_MODEL:
            resolved_llm = DEFAULT_VL_SERVE_MODEL
        resolved_port = int(llm_port)
        if use_vl and resolved_port == 8000:
            resolved_port = DEFAULT_VL_SERVE_PORT
        resolved = resolve_serve_device(llm_device)
        click.echo(f"emet serve llm: llm={resolved_llm} device={resolved} bind={llm_host}:{resolved_port} vl={use_vl}")
        serve_openai_llm(
            llm=resolved_llm,
            host=llm_host,
            port=resolved_port,
            device=resolved,
            max_tokens=int(llm_max_tokens),
            api_key=llm_api_key,
            multimodal=use_vl,
        )
        return

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


@click.command("grasp-oracle", short_help="Fake MolmoSpaces grasp predictor (ZMQ REP)")
@click.option(
    "--bind",
    default="tcp://127.0.0.1:5558",
    show_default=True,
    help="ZMQ REP bind address for grasp predict requests.",
)
@click.option(
    "--grasps-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="MolmoSpaces grasps root (default: $MLSPACES_ASSETS_DIR/grasps).",
)
@click.option(
    "--tcp-frame",
    default="droid",
    show_default=True,
    type=click.Choice(["droid", "rum"]),
    help="Gripper TCP frame correction applied to object-local grasps.",
)
def grasp_oracle_cmd(bind: str, grasps_dir: Path | None, tcp_frame: str) -> None:
    """Serve MolmoSpaces NPZ grasps over ZMQ (robot-agnostic fake grasp predictor).

    Example::

      emet grasp-oracle --bind tcp://127.0.0.1:5558
    """
    from emet.perception.grasps.zmq_server import serve_grasp_oracle

    serve_grasp_oracle(bind=bind, grasps_dir=grasps_dir, tcp_frame=tcp_frame)


@click.group("robocasa", short_help="Robocasa simulation helpers (requires sim extra)")
def robocasa_cmd() -> None:
    """List Robocasa environments or run the server. Requires: emet install sim, then uv sync."""


@robocasa_cmd.command("list", short_help="List all Robocasa environment names")
def robocasa_list() -> None:
    """Print registered Robocasa task names. Use with: emet serve robocasa --robocasa-task <name>."""
    sys.exit(_run_module("emet.simulation.mujoco_server", ["--use-robocasa", "--list-robocasa-tasks"]))


@click.command("kill-mujoco-server", short_help="Stop MuJoCo server (free ports)")
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


@click.command("view-mujoco", short_help="Open MuJoCo viewer for a robot MJCF (no ZMQ server)")
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


def register(main: click.Group) -> None:
    main.add_command(serve)
    main.add_command(grasp_oracle_cmd)
    main.add_command(robocasa_cmd)
    main.add_command(kill_mujoco_server)
    main.add_command(view_mujoco)

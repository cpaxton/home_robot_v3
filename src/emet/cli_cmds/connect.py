# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import json
import os
import sys
from typing import Any

import click

from emet.cli_cmds.bootstrap import (
    _project_root,
    _run_module,
)


@click.group("connect", short_help="Save or show robot connection (host, user) for deploy/view")
def connect_cmd() -> None:
    """Save and reuse connection details so deploy and view-bridge default to the right robot."""
    pass


@connect_cmd.command("save", short_help="Save a connection as active (or named)")
@click.argument("host")
@click.option("--user", "-u", default="root", help="SSH user")
@click.option("--password", "-p", default=None, help="Password (or set EMET_ROBOT_PASSWORD); omit to use SSH key")
@click.option("--name", "-n", default=None, help="Profile name (default: host)")
@click.option(
    "--robot",
    default=None,
    help="Emet robot id (stretch | innate_mars) — used by emet deploy when --robot is omitted",
)
@click.option(
    "--config",
    "profile_config",
    default=None,
    help="Default unified YAML for emet run agent / stream when --config is omitted (e.g. configs/agent_innate_mars.yaml)",
)
@click.option(
    "--workspace",
    default=None,
    help="Remote ROS2 workspace (Stretch: ~/ament_ws; Mars: ~/innate-os/ros2_ws)",
)
@click.option("--emet-dir", default=None, help="Remote emet_core install dir (default ~/emet)")
@click.option("--no-active", is_flag=True, help="Do not set as active connection")
def connect_save(
    host: str,
    user: str,
    password: str | None,
    name: str | None,
    robot: str | None,
    profile_config: str | None,
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
        config=profile_config,
    )
    bits = [f"host={host}", f"user={user}"]
    if robot:
        bits.append(f"robot={robot}")
    if profile_config:
        bits.append(f"config={profile_config}")
    click.echo(f"Saved connection '{conn_name}' ({', '.join(bits)}).")
    if not no_active:
        click.echo("Set as active. Use: emet deploy, emet view-bridge (omit --robot-ip to use this).")


@connect_cmd.command("list", short_help="List saved connections")
def connect_list() -> None:
    """List all saved connections and which is active."""
    from emet.utils.connection import get_connection, list_connections

    items = list_connections()
    if not items:
        click.echo("No connections saved. Use: emet connect save <host> [--user USER]")
        return
    for name, is_active in items:
        mark = " (active)" if is_active else ""
        conn = get_connection(name) or {}
        cfg = conn.get("config")
        extra = f"  config={cfg}" if cfg else ""
        click.echo(f"  {name}{mark}{extra}")


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
    if conn.get("config"):
        click.echo(f"config: {conn.get('config')}")
    if conn.get("workspace"):
        click.echo(f"workspace: {conn.get('workspace')}")
    if conn.get("emet_dir"):
        click.echo(f"emet_dir: {conn.get('emet_dir')}")
    if "password" in conn:
        click.echo("password: (set)")


@connect_cmd.command("use", short_help="Set active connection by name")
@click.argument("name")
def connect_use(name: str) -> None:
    """Mark a saved profile active so deploy / capture / mars omit --host."""
    from emet.utils.connection import get_connection, set_active

    if not set_active(name):
        click.echo(f"Unknown connection {name!r}. Use: emet connect list", err=True)
        sys.exit(1)
    conn = get_connection(name) or {}
    robot = conn.get("robot") or "?"
    click.echo(f"Active connection: {name} ({conn.get('user')}@{conn.get('host')}, robot={robot})")


@click.group("llm", short_help="Remote OpenAI text/VL health + smoke (LAN Jetson / workstation)")
def llm_cmd() -> None:
    """Probe and smoke OpenAI-compatible text/VL servers.

    See docs/llm_serve.md. Pass ``--host`` (or ``EMET_LLM_HOST``). unified-7b serves
    text+VL on ``:8000``; dual-2b keeps VL on ``:8001`` (``--vl-port 8001``).
    """


def _llm_targets_from_host(
    *,
    host: str | None,
    port: int,
    vl_port: int | None,
    text_url: str | None,
    vl_url: str | None,
    check_text: bool,
    check_vl: bool,
) -> tuple[str | None, str | None]:
    from emet.llms.remote_ops import (
        DEFAULT_VL_PORT,
        openai_base_for_host,
        resolve_llm_host,
    )

    resolved = resolve_llm_host(host)
    text_target: str | None = None
    vl_target: str | None = None
    if check_text:
        if text_url is not None and text_url.strip() != "":
            text_target = text_url
        elif resolved:
            text_target = openai_base_for_host(resolved, port)
        else:
            env = (os.environ.get("EMET_OPENAI_BASE_URL") or "").strip()
            text_target = env or None
    if check_vl:
        if vl_url is not None and vl_url.strip() != "":
            vl_target = vl_url
        elif resolved:
            vl_target = openai_base_for_host(resolved, vl_port if vl_port is not None else DEFAULT_VL_PORT)
        else:
            env = (os.environ.get("EMET_VL_ENDPOINT") or os.environ.get("EMET_OPENAI_BASE_URL") or "").strip()
            vl_target = env or None
    return text_target, vl_target


@llm_cmd.command("health", short_help="GET /health for text and/or VL endpoints")
@click.option("--host", default=None, help="LAN host (or EMET_LLM_HOST). Builds http://HOST:PORT/v1.")
@click.option("--port", default=8000, show_default=True, type=int, help="Text/OpenAI port with --host.")
@click.option("--vl-port", default=None, type=int, help="VL port with --host (default same as --port / 8000).")
@click.option(
    "--text",
    "text_url",
    default=None,
    help="Text base URL override. Empty string skips. Else --host or EMET_OPENAI_BASE_URL.",
)
@click.option(
    "--vl",
    "vl_url",
    default=None,
    help="VL base URL override. Empty string skips. Else --host or EMET_VL_ENDPOINT.",
)
@click.option("--text-only", is_flag=True, help="Only check text endpoint.")
@click.option("--vl-only", is_flag=True, help="Only check VL endpoint.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
def llm_health_cmd(
    host: str | None,
    port: int,
    vl_port: int | None,
    text_url: str | None,
    vl_url: str | None,
    text_only: bool,
    vl_only: bool,
    as_json: bool,
) -> None:
    """Check ``/health`` readiness for LAN LLM/VLM servers."""
    from emet.llms.remote_ops import fetch_health

    check_text = not vl_only
    check_vl = not text_only
    if text_url is not None and text_url.strip() == "":
        check_text = False
    if vl_url is not None and vl_url.strip() == "":
        check_vl = False
    text_target, vl_target = _llm_targets_from_host(
        host=host,
        port=port,
        vl_port=vl_port,
        text_url=text_url,
        vl_url=vl_url,
        check_text=check_text,
        check_vl=check_vl,
    )
    if check_text and text_target is None:
        raise click.UsageError("pass --host / EMET_LLM_HOST, --text URL, or EMET_OPENAI_BASE_URL")
    if check_vl and vl_target is None:
        raise click.UsageError("pass --host / EMET_LLM_HOST, --vl URL, or EMET_VL_ENDPOINT")

    results: dict[str, Any] = {}
    ok_all = True
    if text_target is not None:
        r = fetch_health(text_target)
        results["text"] = {"ok": r.ok, "url": r.url, "payload": r.payload, "error": r.error}
        ok_all = ok_all and r.ok
        if not as_json:
            status = "ready" if r.ok else "DOWN"
            click.echo(f"text {status}  {r.url}" + (f"  err={r.error}" if r.error else f"  {r.payload}"))
    if vl_target is not None:
        r = fetch_health(vl_target)
        results["vl"] = {"ok": r.ok, "url": r.url, "payload": r.payload, "error": r.error}
        ok_all = ok_all and r.ok
        if not as_json:
            status = "ready" if r.ok else "DOWN"
            click.echo(f"vl   {status}  {r.url}" + (f"  err={r.error}" if r.error else f"  {r.payload}"))
    if as_json:
        click.echo(json.dumps(results, indent=2, default=str))
    sys.exit(0 if ok_all else 1)


@llm_cmd.command("smoke", short_help="Chat-completions smoke for text and/or VL")
@click.option("--host", default=None, help="LAN host (or EMET_LLM_HOST). Builds http://HOST:PORT/v1.")
@click.option("--port", default=8000, show_default=True, type=int, help="Text/OpenAI port with --host.")
@click.option("--vl-port", default=None, type=int, help="VL port with --host (default 8000; dual-2b: 8001).")
@click.option("--text", "text_url", default=None, help="Text base URL override.")
@click.option("--vl", "vl_url", default=None, help="VL base URL override.")
@click.option("--text-only", is_flag=True, help="Only smoke text.")
@click.option("--vl-only", is_flag=True, help="Only smoke VL.")
@click.option(
    "--image",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Optional image for VL smoke (else a tiny synthetic RGB).",
)
def llm_smoke_cmd(
    host: str | None,
    port: int,
    vl_port: int | None,
    text_url: str | None,
    vl_url: str | None,
    text_only: bool,
    vl_only: bool,
    image: str | None,
) -> None:
    """POST a short completion to text and/or VL OpenAI servers."""
    from emet.llms.remote_ops import smoke_chat_completions, smoke_vl_completions

    check_text = not vl_only
    check_vl = not text_only
    if text_url is not None and text_url.strip() == "":
        check_text = False
    if vl_url is not None and vl_url.strip() == "":
        check_vl = False
    text_target, vl_target = _llm_targets_from_host(
        host=host,
        port=port,
        vl_port=vl_port,
        text_url=text_url,
        vl_url=vl_url,
        check_text=check_text,
        check_vl=check_vl,
    )
    if check_text and text_target is None:
        raise click.UsageError("pass --host / EMET_LLM_HOST, --text URL, or EMET_OPENAI_BASE_URL")
    if check_vl and vl_target is None:
        raise click.UsageError("pass --host / EMET_LLM_HOST, --vl URL, or EMET_VL_ENDPOINT")
    failed = False
    if check_text and text_target is not None:
        click.echo(f"[llm smoke] text {text_target}")
        try:
            out = smoke_chat_completions(text_target)
            click.echo(f"  -> {out!r}")
        except Exception as exc:
            click.echo(f"  FAIL: {type(exc).__name__}: {exc}", err=True)
            failed = True
    if check_vl and vl_target is not None:
        click.echo(f"[llm smoke] vl {vl_target}" + (f" image={image}" if image else " (synthetic)"))
        try:
            out = smoke_vl_completions(vl_target, image_path=image)
            click.echo(f"  -> {out!r}")
        except Exception as exc:
            click.echo(f"  FAIL: {type(exc).__name__}: {exc}", err=True)
            failed = True
    sys.exit(1 if failed else 0)


@click.group("mars", short_help="Innate Mars hardware bridge (innate-os + ZMQ)")
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
@click.option(
    "--onboard-dinov3",
    is_flag=True,
    help="Run DINOv3 vits16 on the Jetson; publish dinov3_head over ZMQ (implies --deploy when set)",
)
@click.option("--preview", is_flag=True, help="Run preview-cameras after bridge startup")
@click.option(
    "--video-rtsp",
    is_flag=True,
    help="Experimental: RTSP side channel (EMET_MARS_VIDEO_RTSP=1; requires site launcher on robot)",
)
@click.option(
    "--metadata-only-obs",
    is_flag=True,
    help="Skinny 4401 (lidar/poses only); scaled JPEG on 4404 (EMET_ZMQ_OBS_INCLUDE_IMAGES=0)",
)
@click.option("--wait-s", default=20.0, show_default=True, help="Seconds to wait before status check")
@click.option("--no-save", is_flag=True, help="Do not update saved connection profile")
def mars_start_cmd(
    host: str | None,
    user: str | None,
    password: str | None,
    connection_name: str | None,
    deploy: bool,
    onboard_da3: bool,
    onboard_dinov3: bool,
    preview: bool,
    video_rtsp: bool,
    metadata_only_obs: bool,
    wait_s: float,
    no_save: bool,
) -> None:
    """Start innate_mars_bridge on the robot (inside innate-os tmux + Zenoh).

    Requires innate-os running on the robot (``innate service start``).

    Examples:
      emet mars start --ip MARS_IP --username jetson1
      emet mars start --ip MARS_IP --username jetson1 --deploy --preview
      emet mars start --connection mars --onboard-da3 --deploy
    """
    from emet.mars import mars_start

    if onboard_da3 and not deploy:
        deploy = True
    if onboard_dinov3 and not deploy:
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
        onboard_dinov3=onboard_dinov3,
        video_rtsp=video_rtsp,
        metadata_only_obs=metadata_only_obs,
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

    host, user, password, workspace, _ = resolve_mars_target(
        host=host,
        user=user,
        password=password,
        connection_name=connection_name,
    )
    bridge_status_on_robot(
        host,
        user,
        password,
        profile=connection_name or host,
        workspace=workspace,
    )


@click.command("view-bridge", short_help="View images and state from robot bridge")
@click.option("--robot-ip", "--robot_ip", default="", help="Robot IP (default: active connection)")
def view_bridge(robot_ip: str) -> None:
    """Connect to the robot's ZMQ bridge and display head/EE camera images and state.
    Use after starting the bridge on the robot
    (``ros2 launch stretch_ros2_bridge server.launch.py`` or
    ``ros2 launch innate_mars_bridge server.launch.py`` / ``emet mars start``).
    """
    sys.exit(_run_module("emet.app.view_bridge", ["--robot-ip", robot_ip] if robot_ip else []))


@click.command(
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


@click.group(
    "deploy",
    invoke_without_command=True,
    short_help="Deploy Stretch/Mars bridge to a robot, or LLM/VLM to a Jetson",
)
@click.option("--host", "-H", default=None, help="Robot host (default: active connection)")
@click.option(
    "--user",
    "-u",
    default=None,
    help="SSH user (default: from connection, else hello-robot / jetson1 by robot)",
)
@click.option("--password", "-p", default=None, help="SSH password (or EMET_ROBOT_PASSWORD)")
@click.option("--connection", "-c", "connection_name", default=None, help="Use saved connection by name")
@click.option(
    "--robot",
    default=None,
    help="Bridge target: stretch | innate_mars (default: connection profile robot, else stretch)",
)
@click.option(
    "--workspace",
    "-w",
    default="~/ament_ws",
    help="Remote ROS2 workspace (Stretch default ~/ament_ws; Mars uses profile or ~/innate-os/ros2_ws)",
)
@click.option("--emet-dir", default="~/emet", help="Remote dir for emet_core (e.g. ~/emet)")
@click.option(
    "--start-bridge",
    is_flag=True,
    help="Start bridge after deploy (Stretch: nohup; Mars: innate-os tmux)",
)
@click.pass_context
def deploy(
    ctx: click.Context,
    host: str | None,
    user: str | None,
    password: str | None,
    connection_name: str | None,
    robot: str | None,
    workspace: str,
    emet_dir: str,
    start_bridge: bool,
) -> None:
    """Deploy robot bridge code or a Jetson LAN LLM/VLM host.

    Bare ``emet deploy`` syncs ``emet_core`` + the robot bridge package:

    - ``--robot stretch`` → ``stretch_ros2_bridge`` into ``~/ament_ws``
    - ``--robot innate_mars`` → ``innate_mars_bridge`` into ``~/innate-os/ros2_ws``

    Robot defaults from the active ``emet connect`` profile when ``--robot`` is omitted.
    Use ``emet deploy llm --host HOST`` for OpenAI Jetson serve (AGX Orin ~60–64 GiB).

    Examples:
      emet connect save STRETCH_IP --user hello-robot --robot stretch --name stretch
      emet deploy --robot stretch --start-bridge
      emet connect save MARS_IP --user jetson1 --robot innate_mars --name mars
      emet deploy --connection mars
      emet mars start --connection mars --deploy
      emet deploy llm --host ORIN_HOST --profile unified-7b
    """
    if ctx.invoked_subcommand is not None:
        return
    from emet.deploy import deploy as deploy_impl

    deploy_impl(
        host=host,
        user=user,
        password=password,
        connection_name=connection_name,
        workspace=workspace,
        emet_dir=emet_dir,
        start_bridge=start_bridge,
        robot=robot,
        root=_project_root(),
    )


@deploy.command("llm", short_help="Deploy Jetson OpenAI LLM/VLM (Orin ~64 GiB)")
@click.option(
    "--profile",
    type=click.Choice(["dual-2b", "unified-7b", "2b", "7b", "big"]),
    default="unified-7b",
    show_default=True,
    help=(
        "dual-2b: CausalLM text :8000 + Qwen2-VL-2B :8001. "
        "unified-7b: one Qwen2-VL-7B on :8000 for text+captions "
        "(fits ~60–64 GiB Orin VRAM; frees eMMC vs dual 7B weights)."
    ),
)
@click.option(
    "--host",
    "-H",
    default=None,
    help="LLM host (required unless EMET_LLM_HOST / EMET_CALIBAN_HOST). Example: --host ORIN_HOST",
)
@click.option("--model", default=None, help="Override HF model id for the VL container.")
@click.option("--port", default=None, type=int, help="Override serve port (unified-7b→8000, dual-2b→8001).")
@click.option("--name", "container_name", default=None, help="Docker container name override.")
def deploy_llm_cmd(
    profile: str,
    host: str | None,
    model: str | None,
    port: int | None,
    container_name: str | None,
) -> None:
    """Rsync VL weights and start the Tegra-CUDA OpenAI container on a Jetson host.

    AGX Orin has ~64 GiB unified memory — enough for Qwen2-VL-7B fp16 (unified-7b).
    eMMC cannot hold both a 7B CausalLM and a 7B VL; use dual-2b for a small VL
    beside text, or unified-7b for the larger single model.

    Quantization (bitsandbytes / AWQ / Quanto) is **not** available on the JP5
    Tegra-CUDA image yet — pip installs replace NVIDIA torch. Stay on fp16 or
    use a JP6/vLLM container; see docs/llm_serve.md § Quantization on Jetson.

    Examples:
      emet deploy llm --host ORIN_HOST --profile unified-7b
      emet deploy llm --host ORIN_HOST --profile dual-2b
      emet llm health --host ORIN_HOST
      emet llm smoke --host ORIN_HOST --vl-only
    """
    from emet.deploy_llm import deploy_llm

    sys.exit(
        deploy_llm(
            host=host,
            profile=profile,
            model=model,
            port=port,
            name=container_name,
            root=_project_root(),
        )
    )


def register(main: click.Group) -> None:
    main.add_command(connect_cmd)
    main.add_command(llm_cmd)
    main.add_command(mars_cmd)
    main.add_command(view_bridge)
    main.add_command(preview_cameras)
    main.add_command(deploy)

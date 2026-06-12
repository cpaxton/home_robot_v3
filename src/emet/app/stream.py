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

"""Live ZMQ → Rerun stream (cameras, base pose, MJCF mesh). Optional mapping backend updates."""

from __future__ import annotations

import os
import time

import click
from click.core import ParameterSource

from emet.app.capture import _resolve_host, _resolve_robot
from emet.app.robot_cli import create_robot_client_from_cli
from emet.app.stream_agent_factory import (
    STREAM_BACKENDS,
    create_stream_agent,
    format_stream_stats,
    resolve_stream_backend,
    stream_agent_update,
    stream_stats,
)
from emet.robots import DEFAULT_DYNAV_CONFIG_YAML


def _stream_zmq_only(
    *,
    robot_key: str,
    host: str,
    port_offset: int,
    headless: bool,
    rerun_native: bool,
    rerun_show_panels: bool,
    rerun_debug: bool,
    allow_missing_depth: bool,
) -> None:
    robot_client = create_robot_client_from_cli(
        robot_key,
        host,
        port_offset=port_offset,
        enable_rerun_server=True,
        rerun_headless=headless,
        rerun_native_viewer=rerun_native,
        rerun_show_panels=rerun_show_panels,
        rerun_debug=rerun_debug,
        start_immediately=False,
        allow_missing_depth=allow_missing_depth,
    )

    click.echo(f"Connecting to {robot_key} at tcp://{host}:4401+ …")
    if not robot_client.start():
        raise SystemExit(
            f"Failed to connect to ZMQ on {host} (start sim or bridge first; match --robot to the server)."
        )

    click.echo("Streaming to Rerun. Ctrl+C to stop.")
    click.echo("Viewer: http://localhost:9090?url=ws://localhost:9877")
    try:
        while robot_client.is_running():
            time.sleep(0.25)
    except KeyboardInterrupt:
        click.echo("Stopping…")
    finally:
        robot_client.stop()


def _stream_with_backend(
    *,
    backend: str,
    robot_key: str,
    host: str,
    port_offset: int,
    dynav_config: str,
    headless: bool,
    rerun_native: bool,
    rerun_show_panels: bool,
    rerun_debug: bool,
    allow_missing_depth: bool,
    cpu_only: bool,
    hz: float,
    max_steps: int,
    no_sensor_perception: bool,
    no_instance_graph: bool,
    compare_to_gt: bool,
    map_only: bool,
    dynav_from_default: bool,
) -> None:
    click.echo(f"Starting {backend} stream for {robot_key} at tcp://{host}:4401+ …")
    if map_only:
        click.echo("Map-only: skipping SigLIP/YoloE/VLM/scene-graph models (voxel + depth only).")
    bundle = create_stream_agent(
        backend,  # type: ignore[arg-type]
        robot=robot_key,
        host=host,
        port_offset=port_offset,
        dynav_config=dynav_config,
        enable_rerun=True,
        headless=headless,
        rerun_native=rerun_native,
        rerun_show_panels=rerun_show_panels,
        rerun_debug=rerun_debug,
        allow_missing_depth=allow_missing_depth,
        cpu_only=cpu_only,
        use_sensor_perception=not no_sensor_perception,
        use_instance_graph=not no_instance_graph,
        compare_to_gt=compare_to_gt,
        map_only=map_only,
        dynav_from_default=dynav_from_default,
    )
    agent = bundle.agent
    backend = bundle.backend
    if backend == "ground_truth":
        from emet.app.run_dynagraph import _ensure_ground_truth_ready

        _ensure_ground_truth_ready(agent, context="stream")

    click.echo(f"Config: {bundle.dynav_resolved}")
    click.echo("Streaming cameras + memory to Rerun. Ctrl+C to stop.")
    click.echo("Viewer: http://localhost:9090?url=ws://localhost:9877")

    period_s = 1.0 / max(0.05, float(hz))
    step = 0
    last_status_t = 0.0
    try:
        while agent.robot.is_running():
            t0 = time.monotonic()
            stream_agent_update(agent, backend)
            step += 1
            now = time.monotonic()
            if now - last_status_t >= 5.0:
                stats = stream_stats(agent, backend, dynav_resolved=bundle.dynav_resolved)
                click.echo(f"  step {step}: {format_stream_stats(stats)}")
                last_status_t = now
            if max_steps > 0 and step >= max_steps:
                click.echo(f"Reached --max-steps {max_steps}; stopping.")
                break
            elapsed = time.monotonic() - t0
            sleep_s = period_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        click.echo("Stopping…")
    finally:
        agent.robot.stop()


@click.command("stream")
@click.option(
    "--ip",
    "--robot-ip",
    "robot_ip",
    default="127.0.0.1",
    show_default=True,
    help="ZMQ host (sim on localhost, or robot hostname/IP)",
)
@click.option("--connection", "-c", "connection_name", default=None, help="Saved connection profile (host/robot)")
@click.option(
    "--robot",
    default="stretch",
    show_default=True,
    help="Robot backend (stretch, innate_mars, rby1, galaxea_r1, …)",
)
@click.option("--port-offset", default=0, type=int, show_default=True, help="Add to default ZMQ ports (4401+)")
@click.option(
    "--backend",
    type=click.Choice(STREAM_BACKENDS),
    default=None,
    help=(
        "Mapping memory backend for continuous updates: dynamem, graph_eqa, dynagraph, "
        "ground_truth, svm, scene_graph. Omit for cameras-only stream."
    ),
)
@click.option("--map", "run_map", is_flag=True, help="Alias for --backend dynamem")
@click.option("--graph", "run_graph", is_flag=True, help="Alias for --backend dynagraph")
@click.option(
    "--dynav-config",
    "--dynav_config",
    default=DEFAULT_DYNAV_CONFIG_YAML,
    show_default=True,
    help="Planner/dynav YAML for mapping backends (hardware Mars: dynav_innate_mars.yaml)",
)
@click.option("--hz", default=1.0, show_default=True, help="Update rate when a mapping backend is active")
@click.option("--max-steps", default=0, type=int, show_default=True, help="Stop after N updates (0 = until Ctrl+C)")
@click.option("--cpu-only", is_flag=True, help="CPU-only models for mapping backends")
@click.option(
    "--map-only",
    is_flag=True,
    help="Voxel/obstacle map only: skip SigLIP, YoloE, VLM, and scene-graph models (DA3 still runs when depth is missing)",
)
@click.option(
    "--no-sensor-perception",
    is_flag=True,
    help="graph_eqa/dynagraph: voxel labels only (no VLM scene labels)",
)
@click.option(
    "--no-instance-graph",
    is_flag=True,
    help="graph_eqa/dynagraph: disable YoloE instance masks for graph nodes",
)
@click.option(
    "--ground-truth",
    is_flag=True,
    help="Alias for --backend ground_truth (sim GT graph from emet_session)",
)
@click.option(
    "--compare-to-gt",
    is_flag=True,
    help="dynagraph: overlay sim GT reference layer alongside the sensor graph",
)
@click.option("--headless", is_flag=True, help="Rerun web server only (no auto-open browser)")
@click.option("--rerun-native", is_flag=True, help="Native Rerun desktop viewer instead of browser")
@click.option("--rerun-show-panels", is_flag=True, help="Show Rerun blueprint/selection panel")
@click.option("--rerun-debug", is_flag=True, help="Log periodic ZMQ/Rerun stream status")
@click.option(
    "--rerun-bind",
    is_flag=True,
    help="Bind Rerun to 0.0.0.0 for remote viewing (Tailscale, SSH tunnel, etc.)",
)
@click.option(
    "--allow-missing-depth",
    is_flag=True,
    help="Accept RGB-only ZMQ frames (default for innate_mars hardware bridge)",
)
@click.pass_context
def main(
    ctx: click.Context,
    robot_ip: str,
    connection_name: str | None,
    robot: str,
    port_offset: int,
    backend: str | None,
    run_map: bool,
    run_graph: bool,
    dynav_config: str,
    hz: float,
    max_steps: int,
    cpu_only: bool,
    map_only: bool,
    no_sensor_perception: bool,
    no_instance_graph: bool,
    ground_truth: bool,
    compare_to_gt: bool,
    headless: bool,
    rerun_native: bool,
    rerun_show_panels: bool,
    rerun_debug: bool,
    rerun_bind: bool,
    allow_missing_depth: bool,
) -> None:
    """Stream live ZMQ observations to Rerun until Ctrl+C.

    By default logs cameras, base pose, and MJCF mesh only. Pass ``--backend`` (or legacy
    ``--map`` / ``--graph``) to run continuous mapping updates in Rerun.

    Examples:
      emet stream
      emet stream --backend dynamem --robot innate_mars --ip herman
      emet stream --backend dynagraph --dynav-config dynav_innate_mars.yaml --ip herman
      emet stream --backend graph_eqa --robot stretch
      emet stream --backend ground_truth --robot stretch
      emet stream --backend scene_graph
    """
    resolved_backend = resolve_stream_backend(
        backend=backend,
        run_map=run_map,
        run_graph=run_graph,
        ground_truth=ground_truth,
    )
    if map_only and resolved_backend is None:
        resolved_backend = "dynamem"
    if map_only and cpu_only:
        click.echo("Note: --map-only already skips heavy perception models; --cpu-only is optional.")
    if (no_sensor_perception or no_instance_graph) and resolved_backend not in (
        "graph_eqa",
        "dynagraph",
        "ground_truth",
        None,
    ):
        raise click.UsageError("--no-sensor-perception and --no-instance-graph apply to graph_eqa/dynagraph.")
    if compare_to_gt and resolved_backend != "dynagraph":
        raise click.UsageError("--compare-to-gt requires --backend dynagraph (or --graph).")

    if rerun_bind:
        os.environ["RERUN_BIND_ALL"] = "1"

    ip_from_default = ctx.get_parameter_source("robot_ip") == ParameterSource.DEFAULT
    robot_from_default = ctx.get_parameter_source("robot") == ParameterSource.DEFAULT
    host = _resolve_host(robot_ip, connection_name, ip_from_default=ip_from_default)
    robot_key = _resolve_robot(robot, connection_name, robot_from_default=robot_from_default)
    dynav_from_default = ctx.get_parameter_source("dynav_config") == ParameterSource.DEFAULT

    if not allow_missing_depth and robot_key == "innate_mars":
        allow_missing_depth = True

    if resolved_backend:
        _stream_with_backend(
            backend=resolved_backend,
            robot_key=robot_key,
            host=host,
            port_offset=port_offset,
            dynav_config=dynav_config,
            headless=headless,
            rerun_native=rerun_native,
            rerun_show_panels=rerun_show_panels,
            rerun_debug=rerun_debug,
            allow_missing_depth=allow_missing_depth,
            cpu_only=cpu_only,
            hz=hz,
            max_steps=max_steps,
            no_sensor_perception=no_sensor_perception,
            no_instance_graph=no_instance_graph,
            compare_to_gt=compare_to_gt,
            map_only=map_only,
            dynav_from_default=dynav_from_default,
        )
    else:
        _stream_zmq_only(
            robot_key=robot_key,
            host=host,
            port_offset=port_offset,
            headless=headless,
            rerun_native=rerun_native,
            rerun_show_panels=rerun_show_panels,
            rerun_debug=rerun_debug,
            allow_missing_depth=allow_missing_depth,
        )

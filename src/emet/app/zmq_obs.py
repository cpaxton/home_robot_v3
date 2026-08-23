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

"""Unified ZMQ observation CLI: ``emet capture`` and ``emet stream``.

Both commands are profile shortcuts into :func:`run_zmq_obs`. Implementation stack:

- ``zmq_cli_resolve`` — host/robot from ``--ip`` or ``--connection``
- ``stream_agent_factory`` — ``--backend`` agent constructors
- ``zmq_mapping_session`` — shared ``agent.update()`` loop

User guide: ``docs/zmq_obs.md``. CLI flag index: ``docs/cli.md`` (capture/stream sections).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import click
import cv2
import numpy as np
from click.core import ParameterSource

import emet.utils.compression as compression
from emet.app.config_cli import emet_config_options, load_runtime_from_cli
from emet.app.preview_robot_cameras import (
    _decode_obs_message,
    _recv_zmq_obs,
    _save_montage_rgb,
    build_montage,
)
from emet.app.robot_cli import create_robot_client_from_cli
from emet.app.stream_agent_factory import (
    STREAM_BACKENDS,
    is_localhost_host,
    load_stream_parameters,
    resolve_stream_backend,
    resolve_stream_dynav_config,
)
from emet.app.zmq_mapping_session import echo_mapping_status, run_mapping_session
from emet.config.stream_config import load_stream_config_from_parameters
from emet.robots import get_robot_spec

ZmqObsProfile = Literal["capture", "stream"]


@dataclass
class ZmqObsRun:
    profile: ZmqObsProfile
    ctx: click.Context
    robot_ip: str
    connection_name: str | None
    robot: str | None
    port_offset: int
    backend: str | None
    emet_config: str
    config_sets: tuple[str, ...]
    agent_config: str | None
    dynav_config: str | None
    cpu_only: bool
    headless: bool
    ip_from_default: bool
    robot_from_default: bool
    dynav_from_default: bool
    # stream-oriented
    cameras_only: bool = False
    hz: float = 1.0
    max_steps: int = 0
    no_sensor_perception: bool = False
    no_instance_graph: bool = False
    compare_to_gt: bool = False
    rerun_native: bool = False
    rerun_show_panels: bool = False
    rerun_debug: bool = False
    rerun_bind: bool = False
    allow_missing_depth: bool = False
    verbose: bool = False
    # capture-oriented
    recv_port: int | None = None
    timeout_ms: int = 9000
    out_dir: Path | None = None
    no_rerun: bool = False
    rerun_hold_s: float = 30.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _rgb_channel_means(rgb: np.ndarray) -> dict[str, float]:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return {}
    return {
        "r_mean": float(arr[..., 0].mean()),
        "g_mean": float(arr[..., 1].mean()),
        "b_mean": float(arr[..., 2].mean()),
    }


def _obs_metadata(raw: dict[str, Any], *, robot: str, host: str, recv_port: int) -> dict[str, Any]:
    skip_keys = {"rgb", "rgb_right", "rgb_tertiary", "depth", "sensor_depth", "segmentation"}
    meta: dict[str, Any] = {
        "robot": robot,
        "host": host,
        "recv_port": recv_port,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for key, value in raw.items():
        if key in skip_keys:
            continue
        meta[key] = _json_safe(value)
    if raw.get("rgb") is not None:
        try:
            meta["rgb_channel_means"] = _rgb_channel_means(compression.from_jpg(raw["rgb"]))
        except Exception:
            pass
    return meta


def _save_capture_images(
    raw: dict[str, Any],
    spec_names: list[str],
    out_dir: Path,
) -> dict[str, Path]:
    imgs, labels = _decode_obs_message(raw, spec_names)
    if not imgs:
        raise click.ClickException("Observation had no decodable RGB.")
    saved: dict[str, Path] = {}
    image_keys = ["rgb", "rgb_right", "rgb_tertiary"]
    for idx, (img, label) in enumerate(zip(imgs, labels, strict=False)):
        key = image_keys[idx] if idx < len(image_keys) else f"cam_{idx}"
        stem = label.replace(" ", "_").replace("/", "_")
        path = out_dir / f"{key}_{stem}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved[key] = path
    montage_path = out_dir / "montage.png"
    _save_montage_rgb(montage_path, build_montage(imgs, labels))
    saved["montage"] = montage_path
    return saved


def _save_artifacts(
    *,
    robot_key: str,
    host: str,
    port_offset: int,
    recv_port: int | None,
    timeout_ms: int,
    out_dir: Path | None,
) -> tuple[Path, dict[str, Any]]:
    spec = get_robot_spec(robot_key)
    if spec is None:
        raise click.ClickException(f"Unknown robot {robot_key!r}.")

    obs_port = int(recv_port if recv_port is not None else 4401 + int(port_offset))
    spec_names = list(spec.camera_names)

    click.echo(f"Capturing from tcp://{host}:{obs_port} ({robot_key})…")
    raw = _recv_zmq_obs(host, obs_port, timeout_ms)
    if raw is None:
        raise click.ClickException(
            f"No observation on tcp://{host}:{obs_port} within {timeout_ms} ms (start sim or bridge first)."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = out_dir or Path("runs") / "capture" / f"{robot_key}_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)

    saved_paths = _save_capture_images(raw, spec_names, dest)
    meta = _obs_metadata(raw, robot=robot_key, host=host, recv_port=obs_port)
    meta_path = dest / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    click.echo(f"Wrote {dest.resolve()}/")
    for name, path in saved_paths.items():
        click.echo(f"  {name}: {path.name}")
    return meta_path, meta


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


def _run_mapping(
    run: ZmqObsRun,
    *,
    backend: str,
    robot_key: str,
    host: str,
    config_path: str,
    allow_missing_depth: bool,
    max_steps: int,
    enable_rerun: bool,
    rerun_hold_s: float,
) -> dict[str, Any]:
    if run.profile == "stream":
        click.echo(f"Starting {backend} stream for {robot_key} at tcp://{host}:4401+ …")
        click.echo("Streaming cameras + memory to Rerun. Ctrl+C to stop.")
        click.echo("Viewer: http://localhost:9090?url=ws://localhost:9877")
    else:
        click.echo(f"Running one {backend} update…")

    dynav_resolved = resolve_stream_dynav_config(
        robot_key,
        host,
        config_path,
        dynav_from_default=run.dynav_from_default,
    )
    params, _ = load_stream_parameters(
        robot_key,
        host,
        dynav_resolved,
        overrides=list(run.config_sets) if run.config_sets else None,
    )
    stream_cfg = load_stream_config_from_parameters(params)

    try:
        result = run_mapping_session(
            backend=backend,
            robot=robot_key,
            host=host,
            port_offset=run.port_offset,
            dynav_config=config_path,
            enable_rerun=enable_rerun,
            headless=run.headless,
            rerun_native=run.rerun_native,
            rerun_show_panels=run.rerun_show_panels,
            rerun_debug=run.rerun_debug,
            allow_missing_depth=allow_missing_depth,
            cpu_only=run.cpu_only,
            hz=run.hz,
            max_steps=max_steps,
            no_sensor_perception=run.no_sensor_perception,
            no_instance_graph=run.no_instance_graph,
            compare_to_gt=run.compare_to_gt,
            dynav_from_default=run.dynav_from_default,
            config_overrides=list(run.config_sets) if run.config_sets else None,
            rerun_hold_s=rerun_hold_s,
            verbose=run.verbose,
            stream_cfg=stream_cfg,
            on_ready=lambda bundle: click.echo(f"Config: {bundle.dynav_resolved}"),
            on_status=echo_mapping_status if run.profile == "stream" else None,
        )
    except KeyboardInterrupt:
        if run.profile == "stream":
            click.echo("Stopping…")
        raise

    return result.final_stats


def run_zmq_obs(run: ZmqObsRun) -> None:
    """Run the unified capture or stream profile (see ``docs/zmq_obs.md``)."""
    if run.rerun_bind:
        os.environ["RERUN_BIND_ALL"] = "1"

    runtime = load_runtime_from_cli(
        run.ctx,
        emet_config=run.emet_config,
        config_sets=run.config_sets,
        agent_config=run.agent_config,
        dynav_config=run.dynav_config,
        robot=run.robot,
        robot_ip=run.robot_ip,
        connection=run.connection_name,
        port_offset=run.port_offset,
    )
    host = runtime.host
    robot_key = runtime.robot_id
    config_path = runtime.config_path
    if runtime.robot_source == "zmq":
        click.echo(f"Using robot from ZMQ server: {robot_key!r} (pass --robot to override).")

    allow_missing_depth = run.allow_missing_depth
    if not allow_missing_depth:
        allow_missing_depth = runtime.allow_missing_depth

    meta_path: Path | None = None
    meta: dict[str, Any] | None = None
    if run.profile == "capture" or run.out_dir is not None:
        meta_path, meta = _save_artifacts(
            robot_key=robot_key,
            host=host,
            port_offset=run.port_offset,
            recv_port=run.recv_port,
            timeout_ms=run.timeout_ms,
            out_dir=run.out_dir,
        )

    if run.profile == "capture":
        resolved_backend = run.backend
        mapping_max_steps = 1 if resolved_backend else 0
        enable_rerun = not run.no_rerun
        rerun_hold_s = 0.0 if run.no_rerun else run.rerun_hold_s
    else:
        resolved_backend = resolve_stream_backend(backend=run.backend, cameras_only=run.cameras_only)
        if (run.no_sensor_perception or run.no_instance_graph) and resolved_backend not in (
            "static_graph",
            "graph_eqa",
            "dynagraph",
            "ground_truth",
            None,
        ):
            raise click.UsageError("--no-sensor-perception and --no-instance-graph apply to static_graph/dynagraph.")
        if run.compare_to_gt and resolved_backend != "dynagraph":
            raise click.UsageError("--compare-to-gt requires --backend dynagraph.")
        if resolved_backend is None and not run.cameras_only and not is_localhost_host(host):
            resolved_backend = "dynamem"
            click.echo(
                f"Remote host {host!r}: defaulting to --backend dynamem "
                "(pass --backend dynagraph|static_graph|svm|… or --cameras-only to override)."
            )
        mapping_max_steps = run.max_steps
        enable_rerun = True
        rerun_hold_s = 0.0

    if resolved_backend:
        map_stats = _run_mapping(
            run,
            backend=resolved_backend,
            robot_key=robot_key,
            host=host,
            config_path=config_path,
            allow_missing_depth=allow_missing_depth,
            max_steps=mapping_max_steps,
            enable_rerun=enable_rerun,
            rerun_hold_s=rerun_hold_s,
        )
        if meta_path is not None and meta is not None:
            meta["map"] = map_stats
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            click.echo(
                f"Map: {map_stats.get('n_voxel_observations', 0)} observations, "
                f"{map_stats.get('n_voxel_explored_cells', 0)} explored cells"
            )
    elif run.profile == "stream":
        _stream_zmq_only(
            robot_key=robot_key,
            host=host,
            port_offset=run.port_offset,
            headless=run.headless,
            rerun_native=run.rerun_native,
            rerun_show_panels=run.rerun_show_panels,
            rerun_debug=run.rerun_debug,
            allow_missing_depth=allow_missing_depth,
        )


_BACKEND_HELP = (
    "Memory backend: dynamem, voxel_only, dynagraph, static_graph (alias graph_eqa), "
    "svm, scene_graph, ground_truth. Full table in docs/zmq_obs.md."
)


def _zmq_connection_options(fn):
    fn = click.option(
        "--ip",
        "--robot-ip",
        "robot_ip",
        default="127.0.0.1",
        show_default=True,
        help="ZMQ host (sim on localhost, or robot hostname/IP)",
    )(fn)
    fn = click.option(
        "--connection", "-c", "connection_name", default=None, help="Saved connection profile (host + robot)"
    )(fn)
    fn = click.option(
        "--robot",
        default=None,
        help="Robot backend (optional: config, connection profile, or ZMQ discovery on localhost).",
    )(fn)
    fn = click.option("--port-offset", default=0, type=int, show_default=True, help="Add to default ZMQ ports (4401+)")(
        fn
    )
    return fn


def _zmq_mapping_options(fn):
    fn = click.option(
        "--backend",
        type=click.Choice(STREAM_BACKENDS),
        default=None,
        help=_BACKEND_HELP,
    )(fn)
    fn = click.option("--cpu-only", is_flag=True, help="CPU-only models for mapping backends")(fn)
    fn = click.option("--headless", is_flag=True, help="Rerun web server only (no auto-open browser)")(fn)
    return fn


def _ctx_to_run(ctx: click.Context, profile: ZmqObsProfile, **kwargs: Any) -> ZmqObsRun:
    dynav_from_default = (
        ctx.get_parameter_source("dynav_config") == ParameterSource.DEFAULT
        and ctx.get_parameter_source("emet_config") == ParameterSource.DEFAULT
    )
    return ZmqObsRun(
        profile=profile,
        ctx=ctx,
        ip_from_default=ctx.get_parameter_source("robot_ip") == ParameterSource.DEFAULT,
        robot_from_default=ctx.get_parameter_source("robot") == ParameterSource.DEFAULT,
        dynav_from_default=dynav_from_default,
        **kwargs,
    )


@_zmq_connection_options
@_zmq_mapping_options
@emet_config_options(include_connection=False)
@click.option("--recv-port", default=None, type=int, help="Observation SUB port (default 4401 + port-offset)")
@click.option("--timeout-ms", default=9000, type=int, show_default=True, help="ZMQ receive timeout for artifact save")
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output directory (default: runs/capture/<timestamp>)",
)
@click.option("--no-rerun", is_flag=True, help="Skip Rerun when --backend is set (capture profile)")
@click.option(
    "--rerun-hold-s", default=30.0, show_default=True, help="Keep Rerun open this long after a capture-profile map step"
)
@click.command("capture")
@click.pass_context
def capture_main(
    ctx: click.Context,
    robot_ip: str,
    connection_name: str | None,
    robot: str | None,
    port_offset: int,
    backend: str | None,
    cpu_only: bool,
    headless: bool,
    recv_port: int | None,
    timeout_ms: int,
    out_dir: Path | None,
    no_rerun: bool,
    rerun_hold_s: float,
    emet_config: str,
    config_sets: tuple[str, ...],
    agent_config: str | None,
    dynav_config: str | None,
) -> None:
    """One ZMQ frame + montage/metadata on disk; optional single mapping update.

    Profile shortcut into ``emet.app.zmq_obs`` (same runner as ``emet stream``).
    Always saves artifacts; with ``--backend``, runs exactly one mapping step.

    \b
    Examples:

      emet capture

      emet capture --connection herman

      emet capture --connection herman --backend voxel_only --no-rerun

    See docs/zmq_obs.md for backends and hardware notes.
    """
    run_zmq_obs(
        _ctx_to_run(
            ctx,
            "capture",
            robot_ip=robot_ip,
            connection_name=connection_name,
            robot=robot,
            port_offset=port_offset,
            backend=backend,
            emet_config=emet_config,
            config_sets=config_sets,
            agent_config=agent_config,
            dynav_config=dynav_config,
            cpu_only=cpu_only,
            headless=headless,
            recv_port=recv_port,
            timeout_ms=timeout_ms,
            out_dir=out_dir,
            no_rerun=no_rerun,
            rerun_hold_s=rerun_hold_s,
        )
    )


@_zmq_connection_options
@_zmq_mapping_options
@emet_config_options(include_connection=False)
@click.option(
    "--cameras-only",
    is_flag=True,
    help="Rerun cameras + MJCF mesh only (no mapping); overrides remote dynamem default",
)
@click.option("--hz", default=1.0, show_default=True, help="Update rate when a mapping backend is active")
@click.option(
    "--max-steps",
    default=0,
    type=int,
    show_default=True,
    help="Stop after N mapping updates (0 = run until Ctrl+C; 3 means exactly 3 updates then exit)",
)
@click.option("--no-sensor-perception", is_flag=True, help="static_graph/dynagraph: voxel labels only (no VLM)")
@click.option("--no-instance-graph", is_flag=True, help="static_graph/dynagraph: disable YoloE instance masks")
@click.option("--compare-to-gt", is_flag=True, help="dynagraph: overlay sim GT reference layer")
@click.option("--rerun-native", is_flag=True, help="Native Rerun desktop viewer instead of browser")
@click.option("--rerun-show-panels", is_flag=True, help="Show Rerun blueprint/selection panel")
@click.option("--rerun-debug", is_flag=True, help="Verbose ZMQ/Rerun client debug (not DA3 timing)")
@click.option("--verbose", is_flag=True, help="Per-step map status + DA3 INFO timing (default: quiet, status every 5s)")
@click.option("--rerun-bind", is_flag=True, help="Bind Rerun to 0.0.0.0 for remote viewing")
@click.option(
    "--allow-missing-depth",
    is_flag=True,
    help="Accept RGB-only ZMQ frames (default for innate_mars hardware bridge)",
)
@click.option("--recv-port", default=None, type=int, help="Observation SUB port when saving artifacts via --out-dir")
@click.option("--timeout-ms", default=9000, type=int, show_default=True, help="ZMQ timeout when saving artifacts")
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Save one montage + metadata before streaming (optional)",
)
@click.command("stream")
@click.pass_context
def stream_main(
    ctx: click.Context,
    robot_ip: str,
    connection_name: str | None,
    robot: str | None,
    port_offset: int,
    backend: str | None,
    cpu_only: bool,
    headless: bool,
    cameras_only: bool,
    hz: float,
    max_steps: int,
    no_sensor_perception: bool,
    no_instance_graph: bool,
    compare_to_gt: bool,
    rerun_native: bool,
    rerun_show_panels: bool,
    rerun_debug: bool,
    verbose: bool,
    rerun_bind: bool,
    allow_missing_depth: bool,
    recv_port: int | None,
    timeout_ms: int,
    out_dir: Path | None,
    emet_config: str,
    config_sets: tuple[str, ...],
    agent_config: str | None,
    dynav_config: str | None,
) -> None:
    """Live ZMQ → Rerun until Ctrl+C (or ``--max-steps``).

    Profile shortcut into ``emet.app.zmq_obs`` (same runner as ``emet capture``).
    Remote hosts default to ``--backend dynamem`` unless ``--cameras-only``.

    \b
    Examples:

      emet stream --cameras-only

      emet stream --connection herman --backend voxel_only

      emet stream --connection herman --out-dir /tmp/frame --backend dynamem --max-steps 5

    Viewer: http://localhost:9090?url=ws://localhost:9877

    See docs/zmq_obs.md for backends, artifact save, and hardware notes.
    """
    run_zmq_obs(
        _ctx_to_run(
            ctx,
            "stream",
            robot_ip=robot_ip,
            connection_name=connection_name,
            robot=robot,
            port_offset=port_offset,
            backend=backend,
            emet_config=emet_config,
            config_sets=config_sets,
            agent_config=agent_config,
            dynav_config=dynav_config,
            cpu_only=cpu_only,
            headless=headless,
            cameras_only=cameras_only,
            hz=hz,
            max_steps=max_steps,
            no_sensor_perception=no_sensor_perception,
            no_instance_graph=no_instance_graph,
            compare_to_gt=compare_to_gt,
            rerun_native=rerun_native,
            rerun_show_panels=rerun_show_panels,
            rerun_debug=rerun_debug,
            verbose=verbose,
            rerun_bind=rerun_bind,
            allow_missing_depth=allow_missing_depth,
            recv_port=recv_port,
            timeout_ms=timeout_ms,
            out_dir=out_dir,
        )
    )

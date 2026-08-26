# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Robot link diagnostics: ZMQ observation stream benchmarks (CPU only)."""

from __future__ import annotations

import sys

import click

from emet.app.zmq_cli_resolve import resolve_cli_host
from emet.app.zmq_stream_benchmark import BenchmarkOptions, run_benchmark, summarize


def _zmq_host_options(fn):
    fn = click.option(
        "--ip",
        "--robot-ip",
        "robot_ip",
        default="127.0.0.1",
        show_default=True,
        help="ZMQ host (sim on localhost, or robot hostname/IP)",
    )(fn)
    fn = click.option(
        "--connection",
        "-c",
        "connection_name",
        default=None,
        help="Saved connection profile (host + robot)",
    )(fn)
    fn = click.option(
        "--port",
        type=int,
        default=4401,
        show_default=True,
        help="Observation SUB port",
    )(fn)
    fn = click.option(
        "--port-offset",
        type=int,
        default=0,
        show_default=True,
        help="Add to --port (multi-robot sim)",
    )(fn)
    return fn


def run_comm_benchmark(
    *,
    robot_ip: str,
    connection_name: str | None,
    port: int,
    port_offset: int,
    seconds: float,
    frames: int,
    timeout_ms: int,
    decode: bool,
    no_conflate: bool,
    warmup_frames: int,
    json_out: str | None,
    verbose: bool,
) -> int:
    """Run the ZMQ full-observation stream benchmark; return process exit code."""
    host = resolve_cli_host(
        robot_ip,
        connection_name,
        ip_from_default=robot_ip.strip() == "127.0.0.1",
    )
    obs_port = port + port_offset
    opts = BenchmarkOptions(
        host=host,
        port=obs_port,
        seconds=seconds,
        frames=frames,
        timeout_ms=timeout_ms,
        conflate=not no_conflate,
        decode=decode,
        warmup_frames=warmup_frames,
        verbose=verbose,
    )
    click.echo(f"Benchmarking ZMQ obs at tcp://{host}:{obs_port} …")
    try:
        result = run_benchmark(opts)
    except TimeoutError as exc:
        click.echo(str(exc), err=True)
        return 1
    click.echo(summarize(result, json_out=json_out))
    return 0


@click.group("comm", short_help="Robot link / ZMQ comms diagnostics")
def comm_group() -> None:
    """CPU-only tools for the robot observation ZMQ stream (port 4401+).

    Measures frame rate, wire size, jitter, unpickle/decode cost, and slim-wire
    flags (duplicate JPEG aliases, float32 lidar). No perception models or GPU.

    Examples::

      emet comm benchmark --connection herman --seconds 30
      emet comm benchmark --connection herman --seconds 15 --decode
      emet comm obs --connection herman --frames 5
    """


@comm_group.command("benchmark", short_help="Sustained ZMQ obs stream benchmark")
@_zmq_host_options
@click.option("--seconds", type=float, default=10.0, show_default=True, help="Stream duration")
@click.option("--frames", type=int, default=0, show_default=True, help="Stop after N frames (overrides --seconds)")
@click.option("--timeout-ms", type=int, default=15_000, show_default=True, help="ZMQ receive timeout")
@click.option("--decode", is_flag=True, help="Measure client JPEG/JP2 decode cost (CPU only)")
@click.option(
    "--no-conflate",
    is_flag=True,
    help="Disable CONFLATE=1 (expose server drops via step gaps)",
)
@click.option("--warmup-frames", type=int, default=2, show_default=True, help="Frames to discard before timing")
@click.option("--json-out", default=None, help="Write machine-readable stats to this path")
@click.option("--verbose", is_flag=True, help="Per-frame wire/payload lines")
def comm_benchmark_cmd(
    robot_ip: str,
    connection_name: str | None,
    port: int,
    port_offset: int,
    seconds: float,
    frames: int,
    timeout_ms: int,
    decode: bool,
    no_conflate: bool,
    warmup_frames: int,
    json_out: str | None,
    verbose: bool,
) -> None:
    """Benchmark sustained full-observation streaming (same runner as ``scripts/benchmark_zmq_obs_stream.py``)."""
    sys.exit(
        run_comm_benchmark(
            robot_ip=robot_ip,
            connection_name=connection_name,
            port=port,
            port_offset=port_offset,
            seconds=seconds,
            frames=frames,
            timeout_ms=timeout_ms,
            decode=decode,
            no_conflate=no_conflate,
            warmup_frames=warmup_frames,
            json_out=json_out,
            verbose=verbose,
        )
    )


@comm_group.command("obs", short_help="Alias: quick ZMQ obs benchmark (default 10 frames)")
@_zmq_host_options
@click.option("--seconds", type=float, default=0.0, show_default=True, help="Stream duration (0 = use --frames)")
@click.option("--frames", type=int, default=10, show_default=True, help="Stop after N frames")
@click.option("--timeout-ms", type=int, default=15_000, show_default=True, help="ZMQ receive timeout")
@click.option("--decode", is_flag=True, help="Measure client JPEG/JP2 decode cost (CPU only)")
@click.option("--no-conflate", is_flag=True, help="Disable CONFLATE=1")
@click.option("--warmup-frames", type=int, default=2, show_default=True)
@click.option("--json-out", default=None, help="Write machine-readable stats to this path")
@click.option("--verbose", is_flag=True, help="Per-frame wire/payload lines")
def comm_obs_cmd(
    robot_ip: str,
    connection_name: str | None,
    port: int,
    port_offset: int,
    seconds: float,
    frames: int,
    timeout_ms: int,
    decode: bool,
    no_conflate: bool,
    warmup_frames: int,
    json_out: str | None,
    verbose: bool,
) -> None:
    """Short obs-stream probe (default 10 frames). Same metrics as ``comm benchmark``."""
    sys.exit(
        run_comm_benchmark(
            robot_ip=robot_ip,
            connection_name=connection_name,
            port=port,
            port_offset=port_offset,
            seconds=seconds,
            frames=frames,
            timeout_ms=timeout_ms,
            decode=decode,
            no_conflate=no_conflate,
            warmup_frames=warmup_frames,
            json_out=json_out,
            verbose=verbose,
        )
    )


@comm_group.command("video", short_help="Preview RTSP streams from emet_session.video_streams")
@click.option("--ip", "--robot-ip", "robot_ip", default="127.0.0.1", show_default=True)
@click.option("--connection", "-c", "connection_name", default=None)
@click.option("--port-offset", type=int, default=0, show_default=True)
@click.option("--seconds", type=float, default=0.0, show_default=True, help="Preview duration (0 = until q/Esc)")
@click.option("--ffplay", is_flag=True, help="Spawn ffplay per stream instead of OpenCV windows")
def comm_video_cmd(
    robot_ip: str,
    connection_name: str | None,
    port_offset: int,
    seconds: float,
    ffplay: bool,
) -> None:
    """Open RTSP camera URLs advertised in session capabilities (Mars ``--video-rtsp``)."""
    from emet.app.comm_video import run_comm_video

    try:
        code = run_comm_video(
            robot_ip=robot_ip,
            connection_name=connection_name,
            port_offset=port_offset,
            seconds=seconds,
            ffplay=ffplay,
        )
    except (RuntimeError, TimeoutError) as exc:
        click.echo(str(exc), err=True)
        code = 1
    sys.exit(code)

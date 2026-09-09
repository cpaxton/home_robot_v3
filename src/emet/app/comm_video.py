# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Open RTSP camera streams advertised in ``emet_session.capabilities.video_streams``."""

from __future__ import annotations

import subprocess
import sys
import time

import click
import cv2

from emet.app.zmq_cli_resolve import resolve_cli_host
from emet.controller.generic_zmq_client import GenericZmqClient


def _video_streams_from_session(client: GenericZmqClient, timeout_s: float = 15.0) -> dict[str, str]:
    if not client.start(log_startup_timeout=False):
        raise RuntimeError("ZMQ client failed to start")
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        sess = client.get_emet_session()
        if sess:
            streams = (sess.get("capabilities") or {}).get("video_streams")
            if isinstance(streams, dict) and streams:
                return {str(k): str(v) for k, v in streams.items()}
        time.sleep(0.1)
    raise TimeoutError(f"No video_streams in emet_session after {timeout_s:.0f}s")


def _opencv_preview(streams: dict[str, str], seconds: float) -> None:
    caps: dict[str, cv2.VideoCapture] = {}
    for name, url in streams.items():
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            click.echo(f"Failed to open {name}: {url}", err=True)
            continue
        caps[name] = cap
        click.echo(f"Opened {name}: {url}")
    if not caps:
        raise RuntimeError("No RTSP streams opened")
    t_end = time.monotonic() + seconds if seconds > 0 else None
    while True:
        for name, cap in caps.items():
            ok, frame = cap.read()
            if ok and frame is not None:
                cv2.imshow(f"comm_video:{name}", frame)
        key = cv2.waitKey(1)
        if key == ord("q") or key == 27:
            break
        if t_end is not None and time.monotonic() >= t_end:
            break
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


def run_comm_video(
    *,
    robot_ip: str,
    connection_name: str | None,
    port_offset: int,
    seconds: float,
    ffplay: bool,
) -> int:
    host = resolve_cli_host(
        robot_ip,
        connection_name,
        ip_from_default=robot_ip.strip() == "127.0.0.1",
    )
    client = GenericZmqClient(robot_ip=host, port_offset=port_offset, robot="innate_mars")
    try:
        streams = _video_streams_from_session(client)
    finally:
        client.finish()

    if ffplay:
        for name, url in streams.items():
            click.echo(f"ffplay {name}: {url}")
            subprocess.Popen(["ffplay", "-window_title", f"comm_video:{name}", url])  # noqa: S603
        if seconds > 0:
            time.sleep(seconds)
        return 0

    _opencv_preview(streams, seconds)
    return 0


@click.command("video")
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
    """View RTSP URLs from ``capabilities.video_streams`` (requires ``emet mars start --video-rtsp``)."""
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

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Optional RTSP video side channel for Innate Mars (experimental; site launcher required)."""

from __future__ import annotations

import os
import subprocess
import time
from typing import Any

from emet.core.zmq_server_env import (
    zmq_video_rtsp_host,
    zmq_video_rtsp_port,
)

_RTSP_STARTUP_GRACE_S = 0.25


def mars_rtsp_stream_urls() -> dict[str, str] | None:
    """Build ``capabilities.video_streams`` entries when RTSP is enabled and running."""
    host = zmq_video_rtsp_host()
    if not host:
        return None
    port = zmq_video_rtsp_port()
    base = f"rtsp://{host}:{port}"
    return {
        "head_left": f"{base}/head_left",
        "head_right": f"{base}/head_right",
        "ee": f"{base}/ee",
    }


def mars_rtsp_launch_command() -> str | None:
    """Shell command to start RTSP server (ROS image topics → H.264)."""
    if os.environ.get("EMET_MARS_VIDEO_RTSP", "").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    port = zmq_video_rtsp_port()
    script = os.environ.get(
        "EMET_MARS_VIDEO_RTSP_SCRIPT",
        os.path.expanduser("~/innate-os/ros2_ws/src/innate_mars_bridge/scripts/mars_video_rtsp.sh"),
    )
    if not os.path.isfile(script):
        return None
    return f"bash {script} {port}"


def rtsp_subprocess_alive(proc: subprocess.Popen[Any] | None) -> bool:
    if proc is None:
        return False
    return proc.poll() is None


def start_mars_rtsp_subprocess() -> subprocess.Popen[Any] | None:
    cmd = mars_rtsp_launch_command()
    if cmd is None:
        return None
    proc = subprocess.Popen(cmd, shell=True)
    time.sleep(_RTSP_STARTUP_GRACE_S)
    if proc.poll() is not None:
        return None
    return proc


def mars_rtsp_capabilities(proc: subprocess.Popen[Any] | None) -> dict[str, str] | None:
    """Return advertised stream URLs only when the RTSP subprocess is alive."""
    if not rtsp_subprocess_alive(proc):
        return None
    return mars_rtsp_stream_urls()

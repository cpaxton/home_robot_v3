# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Optional RTSP video side channel for Innate Mars (GStreamer on Jetson)."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from emet.core.zmq_server_env import (
    zmq_video_rtsp_host,
    zmq_video_rtsp_port,
)


def mars_rtsp_stream_urls() -> dict[str, str] | None:
    """Build ``capabilities.video_streams`` entries when RTSP is enabled."""
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
    return f"bash {script} {port}"


def start_mars_rtsp_subprocess() -> subprocess.Popen[Any] | None:
    cmd = mars_rtsp_launch_command()
    if cmd is None:
        return None
    return subprocess.Popen(cmd, shell=True)

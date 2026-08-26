#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Innate Mars head stereo (+ EE) RTSP via GStreamer NVENC when available.
# Usage: mars_video_rtsp.sh [PORT]
# Requires ROS image topics and Jetson GStreamer plugins (nvv4l2h264enc).

set -euo pipefail
PORT="${1:-8554}"

if ! command -v gst-launch-1.0 >/dev/null 2>&1; then
  echo "gst-launch-1.0 not found; install GStreamer on the robot." >&2
  exit 1
fi

# Minimal RTSP server: one mount head_left (extend with multi-pipeline or MediaMTX for prod).
exec gst-launch-1.0 -e \
  rtspserver name=rtsp port="$PORT" \
  rosimagesrc topic=/mars/main_camera/left/image_raw ! \
  videoconvert ! video/x-raw,format=I420 ! \
  nvv4l2h264enc maxperf-enable=1 bitrate=4000000 ! rtph264pay name=pay0 pt=96 \
  rtsp.attach pay0

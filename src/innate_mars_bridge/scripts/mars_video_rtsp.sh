#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Experimental RTSP side channel for Innate Mars head/EE cameras.
# Usage: mars_video_rtsp.sh [PORT]
#
# Default Mars launch (--video-rtsp) sets EMET_MARS_VIDEO_RTSP=1 but does not ship a
# working GStreamer pipeline here — configure a site launcher on the Jetson:
#
#   export EMET_MARS_VIDEO_RTSP_LAUNCHER=/path/to/your_rtsp_server.sh
#   # or override the whole script path:
#   export EMET_MARS_VIDEO_RTSP_SCRIPT=/path/to/your_rtsp_server.sh
#
# MediaMTX + a ROS image→RTSP republisher is the expected production layout.

set -euo pipefail
PORT="${1:-8554}"

if [[ -n "${EMET_MARS_VIDEO_RTSP_LAUNCHER:-}" ]]; then
  exec bash "${EMET_MARS_VIDEO_RTSP_LAUNCHER}" "${PORT}"
fi

echo "mars_video_rtsp: no RTSP launcher configured (port ${PORT})." >&2
echo "Set EMET_MARS_VIDEO_RTSP_LAUNCHER to a site-specific script on the robot." >&2
echo "See docs/environment_variables.md (EMET_MARS_VIDEO_RTSP_*)." >&2
exit 1

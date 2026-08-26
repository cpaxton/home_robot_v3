# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import os
import subprocess
import textwrap

from innate_mars_bridge.video_rtsp import (
    mars_rtsp_capabilities,
    mars_rtsp_launch_command,
    rtsp_subprocess_alive,
    start_mars_rtsp_subprocess,
)


def test_mars_rtsp_launch_command_none_when_disabled():
    os.environ.pop("EMET_MARS_VIDEO_RTSP", None)
    assert mars_rtsp_launch_command() is None


def test_mars_rtsp_capabilities_none_when_subprocess_exits(tmp_path, monkeypatch):
    script = tmp_path / "fail_rtsp.sh"
    script.write_text("#!/usr/bin/env bash\nexit 1\n")
    script.chmod(0o755)
    monkeypatch.setenv("EMET_MARS_VIDEO_RTSP", "1")
    monkeypatch.setenv("EMET_MARS_VIDEO_RTSP_SCRIPT", str(script))
    monkeypatch.setenv("EMET_MARS_VIDEO_RTSP_HOST", "192.168.1.10")
    monkeypatch.setattr("innate_mars_bridge.video_rtsp._RTSP_STARTUP_GRACE_S", 0.05)

    proc = start_mars_rtsp_subprocess()
    assert proc is None or not rtsp_subprocess_alive(proc)
    assert mars_rtsp_capabilities(proc) is None


def test_mars_rtsp_capabilities_present_when_subprocess_alive(tmp_path, monkeypatch):
    script = tmp_path / "sleep_rtsp.sh"
    script.write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        trap 'exit 0' TERM
        while true; do sleep 1; done
    """))
    script.chmod(0o755)
    monkeypatch.setenv("EMET_MARS_VIDEO_RTSP", "1")
    monkeypatch.setenv("EMET_MARS_VIDEO_RTSP_SCRIPT", str(script))
    monkeypatch.setenv("EMET_MARS_VIDEO_RTSP_HOST", "mars.local")
    monkeypatch.setattr("innate_mars_bridge.video_rtsp._RTSP_STARTUP_GRACE_S", 0.05)

    proc = start_mars_rtsp_subprocess()
    assert proc is not None
    try:
        assert rtsp_subprocess_alive(proc)
        urls = mars_rtsp_capabilities(proc)
        assert urls is not None
        assert urls["head_left"].startswith("rtsp://mars.local:")
    finally:
        proc.terminate()
        proc.wait(timeout=2)

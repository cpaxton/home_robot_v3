# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared gate for SQA3D tests that render through Open3D offscreen.

``o3d.visualization.rendering.OffscreenRenderer`` needs a working GL/EGL context. When
one is missing (headless CI, contended GPU, wedged driver after a Habitat teardown) it
does not raise — it kills the interpreter with SIGSEGV, which takes down the whole
pytest run and, under the Cursor agent, the tool host with it. Probe once in a
subprocess so those tests skip cleanly instead.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_PROBE = "import open3d as o3d;r = o3d.visualization.rendering.OffscreenRenderer(32, 32);del r;print('ok')"

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")

_probe_result: bool | None = None


def open3d_offscreen_available() -> bool:
    """True when a subprocess can build an Open3D OffscreenRenderer without crashing."""
    global _probe_result
    if _probe_result is None:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _PROBE],
                capture_output=True,
                timeout=120,
            )
            _probe_result = proc.returncode == 0 and b"ok" in proc.stdout
        except (OSError, subprocess.SubprocessError):
            _probe_result = False
    return _probe_result


@pytest.fixture
def open3d_offscreen() -> None:
    """Skip a test unless Open3D offscreen rendering actually works here."""
    if not RUN_SIM_TESTS:
        pytest.skip("RUN_SIM_TESTS=0")
    if not open3d_offscreen_available():
        pytest.skip("Open3D OffscreenRenderer unavailable (no usable GL/EGL context)")

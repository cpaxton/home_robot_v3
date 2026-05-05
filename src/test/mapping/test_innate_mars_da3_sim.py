# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Optional integration: innate_mars + DynaMem + Depth Anything 3 in MuJoCo (slow; downloads weights).
#   RUN_DA3_TESTS=1 uv run emet test src/test/mapping/test_innate_mars_da3_sim.py -v

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent

_run_da3 = os.environ.get("RUN_DA3_TESTS", "").strip() == "1"
_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def _wait_for_port(host: str, port: int, timeout_sec: float = 30) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2) as _:
                return True
        except (TimeoutError, OSError):
            time.sleep(0.5)
    return False


@pytest.mark.skipif(not _run_da3, reason="Set RUN_DA3_TESTS=1 (requires da3 extra and model download)")
@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(600)
def test_innate_mars_dynamem_da3_rotate_smoke():
    pytest.importorskip("depth_anything_3")
    pytest.importorskip("mujoco")

    proc = None
    robot = None
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC_ROOT)] + env.get("PYTHONPATH", "").split(os.pathsep))
    if sys.platform == "linux":
        env["MUJOCO_GL"] = "egl"

    try:
        server_cmd = [
            sys.executable,
            "-m",
            "emet.simulation.mujoco_server",
            "--headless",
            "--robot",
            "innate_mars",
        ]
        proc = subprocess.Popen(
            server_cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not _wait_for_port("127.0.0.1", 4401, timeout_sec=45):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait(timeout=5)
            pytest.fail("MuJoCo server did not bind to 4401 within 45s. stderr:\n" + stderr)

        from emet.app.robot_cli import create_robot_client_from_cli
        from emet.controller.task.dynamem import DynamemTaskExecutor
        from emet.core.parameters import get_parameters

        robot = create_robot_client_from_cli(
            "innate_mars",
            "127.0.0.1",
            enable_rerun_server=False,
            start_immediately=True,
            allow_missing_depth=True,
        )
        parameters = get_parameters("dynav_innate_mars.yaml")
        executor = DynamemTaskExecutor(
            robot,
            parameters,
            skip_confirmations=True,
            cpu_only=True,
        )
        executor([("rotate_in_place", "")])

        voxel_map = executor.agent.get_voxel_map()
        sem = voxel_map.semantic_memory
        assert sem._points is not None, "DA3 depth updates should populate semantic memory points"
        assert int(sem._points.numel()) > 0
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

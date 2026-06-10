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
# Optional integration: innate_mars + DynaMem + LingBot-Map in MuJoCo.
#   RUN_LINGBOT_TESTS=1 uv run emet test src/test/mapping/test_innate_mars_lingbot_sim.py -v
#
# Requires: .venv-lingbot-map, LINGBOT_MAP_CHECKPOINT, CUDA

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent

_run_lingbot = os.environ.get("RUN_LINGBOT_TESTS", "").strip() == "1"
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


def _gpu_free_gib() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        free, _total = torch.cuda.mem_get_info()
        return float(free) / (1024**3)
    except Exception:
        return 0.0


@pytest.mark.skipif(not _run_lingbot, reason="Set RUN_LINGBOT_TESTS=1 (requires lingbot venv + checkpoint + CUDA)")
@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(900)
def test_innate_mars_dynamem_lingbot_rotate_smoke():
    ckpt = os.environ.get("LINGBOT_MAP_CHECKPOINT", "")
    venv_py = Path(os.environ.get("LINGBOT_MAP_VENV", _SRC_ROOT.parent / ".venv-lingbot-map")) / "bin" / "python"
    if not ckpt or not Path(ckpt).is_file():
        pytest.skip("LINGBOT_MAP_CHECKPOINT not set or missing")
    if not venv_py.is_file():
        pytest.skip(f"LingBot venv missing: {venv_py} (run ./scripts/install_lingbot_map.sh)")
    free_gib = _gpu_free_gib()
    if free_gib < 8.0:
        pytest.skip(f"Need ~8GiB free GPU for LingBot subprocess infer (have {free_gib:.1f}GiB free)")

    proc = None
    robot = None
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
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
        proc = subprocess.Popen(server_cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if not _wait_for_port("127.0.0.1", 4401, timeout_sec=45):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait(timeout=5)
            pytest.fail("MuJoCo server did not bind to 4401. stderr:\n" + stderr)

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
        parameters = get_parameters("dynav_innate_mars_lingbot.yaml")
        executor = DynamemTaskExecutor(
            robot,
            parameters,
            skip_confirmations=True,
            cpu_only=True,
        )
        executor([("rotate_in_place", "")])

        voxel_map = executor.agent.get_voxel_map()
        sem = voxel_map.semantic_memory
        assert sem._points is not None
        assert int(sem._points.numel()) > 0
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

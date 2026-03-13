# Copyright (c) Hello Robot, Inc.
#
# Integration test: start MuJoCo server with Robocasa scene (PickPlaceCounterToCabinet;
# PickPlaceCounterToSink preferred but often fails placement in this setup), connect,
# run one rotate_in_place to build memory, then assert the unified memory backend
# finds at least one object. Run with:
#   uv run emet test -v src/test/simulation/test_robocasa_memory_after_spin.py
# Sim tests run by default; use RUN_SIM_TESTS=0 or emet test --no-sim to skip.
# Requires sim extra and robocasa assets (emet install sim, download_robocasa_assets).

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent
# Run sim tests by default; skip only when explicitly disabled (e.g. RUN_SIM_TESTS=0)
_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def _wait_for_port(host: str, port: int, timeout_sec: float = 60) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2) as _:
                return True
        except (OSError, socket.timeout):
            time.sleep(0.5)
    return False


@pytest.mark.skipif(
    not RUN_SIM_TESTS,
    reason="Set RUN_SIM_TESTS=0 to skip (sim tests run by default)",
)
@pytest.mark.timeout(180)
def test_robocasa_memory_after_spin():
    """
    Start MuJoCo server with Robocasa kitchen scene (default: PickPlaceCounterToCabinet),
    run rotate_in_place once, then assert the unified memory backend finds at least
    one object (placed/counter objects get added to memory).
    """
    proc = None
    robot = None
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_SRC_ROOT)] + env.get("PYTHONPATH", "").split(os.pathsep)
    )
    if sys.platform == "linux":
        env["MUJOCO_GL"] = "egl"

    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "emet.simulation.mujoco_server",
                "--headless",
                "--use-robocasa",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not _wait_for_port("127.0.0.1", 4401, timeout_sec=90):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait(timeout=5)
            pytest.fail(
                "MuJoCo server (robocasa) did not bind to 4401 within 90s. stderr:\n"
                + stderr
            )

        from emet.controller.task.dynamem import DynamemTaskExecutor
        from emet.controller.zmq_client import StretchZmqClient
        from emet.core.parameters import get_parameters
        from emet.memory.backend import get_memory_backend

        robot = StretchZmqClient(
            robot_ip="127.0.0.1",
            enable_rerun_server=False,
            start_immediately=True,
        )
        parameters = get_parameters("dynav_config.yaml")
        executor = DynamemTaskExecutor(
            robot,
            parameters,
            skip_confirmations=True,
            cpu_only=True,
        )
        executor([("rotate_in_place", "")])

        backend = get_memory_backend("dynamem", voxel_map=executor.agent.get_voxel_map())
        # Default task PickPlaceCounterToCabinet (and e.g. CounterToSink): counter/cabinet objects
        candidates = [
            "bowl",
            "cup",
            "bottle",
            "plate",
            "can",
            "box",
            "apple",
            "pot",
            "pan",
            "red cylinder",
            "blue cube",
        ]
        found_any = False
        for name in candidates:
            check = backend.check_memory_for_object(name)
            if check.confidence > 0 and check.location_xyz is not None:
                found_any = True
                break
        assert found_any, (
            f"After one spin in Robocasa scene, at least one of {candidates} "
            "should be in memory (placed object). Check detector/encoder."
        )
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

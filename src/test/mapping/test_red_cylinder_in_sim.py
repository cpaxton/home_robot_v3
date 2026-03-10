# Copyright (c) Hello Robot, Inc.
#
# Integration test: robot moves around in the default MuJoCo scene (rotate_in_place),
# then we assert that the memory backend can find the red cylinder. Ensures the full
# stack (sim → camera → encoder → semantic memory → localize_text) works.
#
# Run from project root with full env (e.g. after `pip install -e .` or `emet sync -e sim`):
#   RUN_SIM_TESTS=1 pytest src/test/mapping/test_red_cylinder_in_sim.py -v
# With timeout (requires pytest-timeout): same command; test is marked with 120s timeout.
# On Linux, MuJoCo uses EGL (headless).

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Ensure subprocess can import emet (pytest adds src to path; subprocess does not)
_SRC_ROOT = Path(__file__).resolve().parent.parent.parent

# Skip unless explicitly requested (sim is slow and needs display/EGL)
RUN_SIM_TESTS = os.environ.get("RUN_SIM_TESTS", "").strip() in ("1", "true", "yes")


def _wait_for_port(host: str, port: int, timeout_sec: float = 30) -> bool:
    """Return True when something is accepting connections on host:port."""
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
    reason="Set RUN_SIM_TESTS=1 to run (starts MuJoCo sim and Dynamem)",
)
@pytest.mark.timeout(120)
def test_red_cylinder_detected_in_sim():
    """
    Robot moves around in the default MuJoCo scene (rotate_in_place to build map),
    then we find the red cylinder via localize_text. Fails if not found within 120s.
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
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not _wait_for_port("127.0.0.1", 4401, timeout_sec=45):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait(timeout=5)
            pytest.fail(
                "MuJoCo server did not bind to 4401 within 45s. stderr:\n" + stderr
            )

        from emet.controller.task.dynamem import DynamemTaskExecutor
        from emet.controller.zmq_client import HomeRobotZmqClient
        from emet.core.parameters import get_parameters

        robot = HomeRobotZmqClient(
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

        voxel_map = executor.agent.get_voxel_map()
        result = voxel_map.localize_text("red cylinder", return_debug=True)
        point, debug_text = result[0], result[1]

        assert point is not None, (
            "localize_text('red cylinder') should return a target in sim. "
            f"debug: {debug_text!r}"
        )
        target = point.squeeze()
        assert target.shape == (3,), "target should be 3D (x, y, z)"

        # Default scene: red cylinder (object2) at roughly (0.08, -0.55, 0.6)
        expected = np.array([0.08, -0.55, 0.6], dtype=np.float64)
        np.testing.assert_allclose(
            target.cpu().numpy() if hasattr(target, "cpu") else target,
            expected,
            atol=0.25,
            err_msg="Target point should be near red cylinder in default scene",
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

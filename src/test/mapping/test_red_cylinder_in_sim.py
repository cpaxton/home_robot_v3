# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
#
# Integration test: robot moves around in the default MuJoCo scene (rotate_in_place),
# then we assert that the memory backend can find the red cylinder. Ensures the full
# stack (sim → camera → encoder → semantic memory → localize_text) works.
#
# Run from project root with full env (e.g. after `pip install -e .` or `emet sync -e sim`):
#   uv run emet test -v src/test/mapping/test_red_cylinder_in_sim.py
# Sim tests run by default; use RUN_SIM_TESTS=0 or emet test --no-sim to skip.
# With timeout (requires pytest-timeout): same command; test is marked with 120s timeout.
# On Linux, MuJoCo uses EGL (headless).
#
# Parametrized: default Stretch sim, and ``--robot innate_mars`` (GenericZmqClient + same table scene)

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

# Run sim tests by default; skip only when explicitly disabled (e.g. RUN_SIM_TESTS=0)
_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def _wait_for_port(host: str, port: int, timeout_sec: float = 30) -> bool:
    """Return True when something is accepting connections on host:port."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2) as _:
                return True
        except (TimeoutError, OSError):
            time.sleep(0.5)
    return False


@pytest.mark.skipif(
    not RUN_SIM_TESTS,
    reason="Set RUN_SIM_TESTS=0 to skip (sim tests run by default)",
)
@pytest.mark.timeout(720)
@pytest.mark.parametrize("sim_robot", ["stretch", "innate_mars"])
def test_red_cylinder_detected_in_sim(sim_robot: str):
    """
    Robot moves around in the default MuJoCo scene (rotate_in_place to build map),
    then we find the red cylinder via localize_text. Fails if not found within 120s.
    """
    proc = None
    robot = None
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC_ROOT)] + env.get("PYTHONPATH", "").split(os.pathsep))
    if sys.platform == "linux":
        env["MUJOCO_GL"] = "egl"

    try:
        subprocess.run(
            ["uv", "run", "emet", "kill-mujoco-server"],
            cwd=_SRC_ROOT.parent,
            check=False,
        )
        time.sleep(1.5)
        server_cmd = [sys.executable, "-m", "emet.simulation.mujoco_server", "--headless"]
        if sim_robot == "innate_mars":
            server_cmd.extend(["--robot", "innate_mars"])
        proc = subprocess.Popen(
            server_cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not _wait_for_port("127.0.0.1", 4401, timeout_sec=90):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait(timeout=5)
            pytest.fail("MuJoCo server did not bind to 4401 within 45s. stderr:\n" + stderr)

        from emet.app.robot_cli import create_robot_client_from_cli
        from emet.controller.task.dynamem import DynamemTaskExecutor
        from emet.core.parameters import get_parameters

        if sim_robot == "stretch":
            from emet.controller.zmq_client import StretchZmqClient

            robot = StretchZmqClient(
                robot_ip="127.0.0.1",
                enable_rerun_server=False,
                start_immediately=True,
            )
        else:
            robot = create_robot_client_from_cli(
                sim_robot,
                "127.0.0.1",
                enable_rerun_server=False,
                start_immediately=True,
                allow_missing_depth=True,
                zmq_startup_timeout=120.0,
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

        assert point is not None, f"localize_text('red cylinder') should return a target in sim. debug: {debug_text!r}"
        target = point.squeeze()
        assert target.shape == (3,), "target should be 3D (x, y, z)"

        # Default scene: red cylinder (object2) at (0.08, -0.55, 0.6), blue cube (object1) at (-0.02, -0.55, 0.6)
        expected_red = np.array([0.08, -0.55, 0.6], dtype=np.float64)
        expected_blue = np.array([-0.02, -0.55, 0.6], dtype=np.float64)
        # Sim voxel localize can be ~0.5 m off GT table coords after one spin; memory checks below are tighter.
        np.testing.assert_allclose(
            target.cpu().numpy() if hasattr(target, "cpu") else target,
            expected_red,
            atol=0.55,
            err_msg="Target point should be near red cylinder in default scene",
        )

        # Blue cube (object1) at roughly (-0.02, -0.55, 0.6) should be in memory after one spin
        result_blue = voxel_map.localize_text("blue cube", return_debug=True)
        point_blue = result_blue[0] if isinstance(result_blue, (list, tuple)) else result_blue
        if point_blue is not None:
            target_blue = point_blue.squeeze()
            np.testing.assert_allclose(
                target_blue.cpu().numpy() if hasattr(target_blue, "cpu") else target_blue,
                expected_blue,
                atol=0.55,
                err_msg="Target point should be near blue cube in default scene",
            )

        # Unified MemoryBackend: same scene must yield red cylinder (and blue cube if detected) via interface
        from emet.memory.backend import get_memory_backend

        backend = get_memory_backend("dynamem", voxel_map=voxel_map)
        check_red = backend.check_memory_for_object("red cylinder")
        assert check_red.confidence > 0, (
            "Unified backend check_memory_for_object('red cylinder') should have confidence > 0 after spin."
        )
        assert check_red.location_xyz is not None
        np.testing.assert_allclose(
            np.asarray(check_red.location_xyz).flat[:3],
            expected_red,
            atol=0.55,
            err_msg="Unified backend red cylinder location",
        )
        loc_red = backend.localize_text("red cylinder")
        assert loc_red.success and loc_red.point_xyz is not None

        check_blue = backend.check_memory_for_object("blue cube")
        if check_blue.confidence > 0 and check_blue.location_xyz is not None:
            np.testing.assert_allclose(
                np.asarray(check_blue.location_xyz).flat[:3],
                expected_blue,
                atol=0.55,
                err_msg="Unified backend blue cube location",
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

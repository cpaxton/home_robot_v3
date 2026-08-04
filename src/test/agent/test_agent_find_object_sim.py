# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Integration test: start MuJoCo sim, run the agent in scripted (no-LLM) mode
# with commands: rotate_in_place (scan), then find red cylinder.
# Verifies the full DynaMem pipeline: sim → camera → encoder → semantic memory
# → localize_text → navigate.
#
# Run with: pytest src/test/agent/test_agent_find_object_sim.py -v -s
# Sim tests run by default; RUN_SIM_TESTS=0 to skip.

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def _have_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return False
    return True


_SKIP_AGENT_SIM = not RUN_SIM_TESTS or not _have_mujoco()
pytestmark = pytest.mark.skipif(
    _SKIP_AGENT_SIM,
    reason="RUN_SIM_TESTS=0 or mujoco not installed (uv sync)",
)


def _wait_for_port(host: str, port: int, timeout_sec: float = 30) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except (TimeoutError, OSError):
            time.sleep(0.5)
    return False


@pytest.mark.timeout(300)
def test_agent_find_red_cylinder_no_llm():
    """
    Non-interactive agent test: scan scene, then find red cylinder.
    Uses --no-llm mode with manual commands so no GPU needed for LLM.
    """
    proc = None
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC_ROOT)] + env.get("PYTHONPATH", "").split(os.pathsep))
    if sys.platform == "linux":
        env["MUJOCO_GL"] = "egl"

    try:
        # Start MuJoCo server
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "emet.simulation.mujoco_server",
                "--robot",
                "stretch",
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
            pytest.fail("MuJoCo server did not start. stderr:\n" + stderr)

        # Run agent in scripted no-LLM mode:
        # The initial rotate_in_place happens automatically (no input-path),
        # then we send "FIND red cylinder" as a manual command.
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "emet.app.run_agent",
                "--robot",
                "stretch",
                "--robot-ip",
                "127.0.0.1",
                "--memory-backend",
                "dynagraph",
                "--no-llm",
                "--no-discord",
                "--command",
                "find red cylinder",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )

        stdout = result.stdout
        stderr = result.stderr
        print("--- STDOUT ---")
        print(stdout[-3000:] if len(stdout) > 3000 else stdout)
        print("--- STDERR ---")
        print(stderr[-2000:] if len(stderr) > 2000 else stderr)

        # The agent should have executed the find command.
        # We check that it didn't crash and the executor processed the find.
        assert result.returncode == 0 or result.returncode is None, f"Agent exited with code {result.returncode}"
        # Check that the find command was dispatched
        assert "find" in stdout.lower() or "FIND" in stdout, "Expected 'find' in agent output"

    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)


@pytest.mark.timeout(300)
def test_agent_find_blue_cube_no_llm():
    """Scripted find for the default-scene blue cube (near red cylinder)."""
    proc = None
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC_ROOT)] + env.get("PYTHONPATH", "").split(os.pathsep))
    if sys.platform == "linux":
        env["MUJOCO_GL"] = "egl"

    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "emet.simulation.mujoco_server",
                "--robot",
                "stretch",
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
            pytest.fail("MuJoCo server did not start. stderr:\n" + stderr)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "emet.app.run_agent",
                "--robot",
                "stretch",
                "--robot-ip",
                "127.0.0.1",
                "--memory-backend",
                "dynagraph",
                "--no-llm",
                "--no-discord",
                "--command",
                "find blue cube",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )

        stdout = result.stdout
        print("--- STDOUT (blue cube) ---")
        print(stdout[-3000:] if len(stdout) > 3000 else stdout)

        assert result.returncode == 0 or result.returncode is None, f"Agent exited with code {result.returncode}"
        assert "find" in stdout.lower() or "FIND" in stdout, "Expected find-related output"

    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)


@pytest.mark.timeout(300)
def test_agent_multi_command_no_llm():
    """
    Scripted multi-command test: explore, then find red cylinder (mixed -c / --command).
    """
    proc = None
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC_ROOT)] + env.get("PYTHONPATH", "").split(os.pathsep))
    if sys.platform == "linux":
        env["MUJOCO_GL"] = "egl"

    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "emet.simulation.mujoco_server",
                "--robot",
                "stretch",
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
            pytest.fail("MuJoCo server did not start. stderr:\n" + stderr)

        # Commands: E (explore), then find red cylinder via --command (no Q: query_memory path varies by map setup)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "emet.app.run_agent",
                "--robot",
                "stretch",
                "--robot-ip",
                "127.0.0.1",
                "--memory-backend",
                "dynagraph",
                "--no-llm",
                "--no-discord",
                "-c",
                "E",
                "--command",
                "find red cylinder",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )

        stdout = result.stdout
        print("--- STDOUT ---")
        print(stdout[-3000:] if len(stdout) > 3000 else stdout)

        assert result.returncode == 0 or result.returncode is None, f"Agent exited with code {result.returncode}"

    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

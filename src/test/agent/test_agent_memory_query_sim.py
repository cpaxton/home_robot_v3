# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Integration test: start MuJoCo sim, scan scene, then use the agent's query_memory
# tool to ask about objects. Verifies blue cube and red cylinder are reported.
#
# Run with: pytest src/test/agent/test_agent_memory_query_sim.py -v
# Sim tests run by default; RUN_SIM_TESTS=0 to skip.

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def _wait_for_port(host: str, port: int, timeout_sec: float = 30) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except (OSError, socket.timeout):
            time.sleep(0.5)
    return False


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(180)
def test_agent_query_memory_finds_objects_in_sim():
    """
    Full pipeline: MuJoCo sim → scan → build memory → query_memory tool → answer mentions objects.
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
            [sys.executable, "-m", "emet.simulation.mujoco_server", "--headless"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not _wait_for_port("127.0.0.1", 4401, timeout_sec=45):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait(timeout=5)
            pytest.fail("MuJoCo server did not start. stderr:\n" + stderr)

        from emet.controller.task.dynamem import DynamemTaskExecutor
        from emet.controller.zmq_client import StretchZmqClient
        from emet.core.parameters import get_parameters
        from emet.memory.backend import get_memory_backend
        from emet.agent.tools import get_tools

        robot = StretchZmqClient(
            robot_ip="127.0.0.1",
            enable_rerun_server=False,
            start_immediately=True,
        )
        parameters = get_parameters("dynav_config.yaml")
        executor = DynamemTaskExecutor(
            robot, parameters, skip_confirmations=True, cpu_only=True,
        )
        executor([("rotate_in_place", "")])

        voxel_map = executor.agent.get_voxel_map()
        memory_backend = get_memory_backend("dynamem", voxel_map=voxel_map)

        context = {
            "executor": executor,
            "robot": robot,
            "memory_backend": memory_backend,
            "xyt_for_query": None,
            "planner": getattr(executor.agent, "planner", None),
        }
        tools = get_tools(context)
        query_tool = next(t for t in tools if t.name == "query_memory")

        # Ask about red cylinder
        answer_red = query_tool.func(question="Is there a red cylinder?")
        assert isinstance(answer_red, str) and len(answer_red) > 0
        # The answer from DynaMem is based on localize_text, so it should have non-zero confidence
        # Check directly via backend
        check_red = memory_backend.check_memory_for_object("red cylinder")
        assert check_red.confidence > 0, "red cylinder should be in memory after scan"

        # Ask about blue cube
        answer_blue = query_tool.func(question="Is there a blue cube?")
        assert isinstance(answer_blue, str) and len(answer_blue) > 0
        check_blue = memory_backend.check_memory_for_object("blue cube")
        assert check_blue.confidence > 0, "blue cube should be in memory after scan"

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

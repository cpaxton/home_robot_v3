# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""DynaMem rotate + explore smoke on packaged default table with XLeRobot."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[2]


def _truthy(env: str) -> bool:
    return os.environ.get(env, "").strip().lower() in ("1", "true", "yes", "on")


_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def _skip_reason() -> str | None:
    if not RUN_SIM_TESTS:
        return "RUN_SIM_TESTS=0 skips sim-heavy tests."
    if not _truthy("RUN_XLEROBOT_DYNAMEM"):
        return "Set RUN_XLEROBOT_DYNAMEM=1 to run XLeRobot DynaMem smoke."
    return None


_SKIP = _skip_reason()


def _wait_port(port: int, timeout_sec: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


@pytest.mark.timeout(300)
@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "skipped")
def test_xlerobot_dynamem_default_table_rotate_and_explore():
    """Default-table MuJoCo + GenericZmqClient: mapping grid populates after rotate/explore."""
    recv_port = 4401
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC_ROOT)] + env.get("PYTHONPATH", "").split(os.pathsep))
    if sys.platform == "linux":
        env["MUJOCO_GL"] = "egl"
    env.setdefault("EMET_ZMQ_STARTUP_TIMEOUT", "120")
    env["EMET_SIM_NAV_TELEPORT"] = "1"

    cmd = [sys.executable, "-m", "emet.simulation.mujoco_server", "--robot", "xlerobot", "--headless"]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    robot_client = None
    try:
        if not _wait_port(recv_port):
            err = proc.stderr.read().decode() if proc.stderr else ""
            pytest.fail(f"XLeRobot server did not bind {recv_port}. stderr:\n{err}")

        from emet.config.embodied_agent_config import legacy_embodied_agent_off
        from emet.controller.controller_dynamem import DynamemController
        from emet.controller.generic_zmq_client import GenericZmqClient
        from emet.core.parameters import get_parameters
        from emet.robots.xlerobot import XLeRobotBackend

        spec = XLeRobotBackend().get_spec()
        robot_client = GenericZmqClient(
            robot_spec=spec,
            robot_ip="127.0.0.1",
            enable_rerun_server=False,
            start_immediately=True,
            allow_missing_depth=False,
        )
        params = get_parameters("dynav_config.yaml")
        params["use_instance_memory"] = False
        params["depth_source"] = "sensor"
        agent = DynamemController(
            robot_client,
            params,
            save_rerun=False,
            cpu_only=True,
            realtime_updates=True,
            embodied_agent=legacy_embodied_agent_off(),
        )
        agent.start()
        for _ in range(6):
            agent.update()
        agent.rotate_in_place()
        for _ in range(4):
            agent.update()
        _, explored = agent.voxel_map.get_2d_map()
        explored_n = int(explored.float().sum().item())
        explore_ok = sum(1 for _ in range(2) if agent.run_exploration())
        assert explored_n >= 80, f"expected explored cells after rotate; got {explored_n}"
        assert explore_ok >= 1, "expected at least one frontier exploration step to succeed"
    finally:
        if robot_client is not None:
            try:
                robot_client.stop()
            except Exception:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=4)

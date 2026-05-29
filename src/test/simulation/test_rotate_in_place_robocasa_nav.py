# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Robocasa + innate_mars: rotate_in_place must keep world (x,y) at spawn (only θ changes).

Run:
  uv run emet test -v src/test/simulation/test_rotate_in_place_robocasa_nav.py
"""

from __future__ import annotations

import os
import re
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

_NAV_GOAL_RE = re.compile(
    r"frame=(\w+).*goal_world=\[([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\].*nav_world=(\w+)"
)
_SPAWN_RE = re.compile(
    r"spawn / navigation_origin \(world\): \(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)"
)


def _wait_for_port(host: str, port: int, timeout_sec: float = 120) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2) as _:
                return True
        except (TimeoutError, OSError):
            time.sleep(0.5)
    return False


def _parse_sim_nav_goals(stderr_text: str) -> list[tuple[float, float, float, bool, str]]:
    goals: list[tuple[float, float, float, bool, str]] = []
    for line in stderr_text.splitlines():
        if "[sim_nav]" not in line or "goal_world=" not in line:
            continue
        m = _NAV_GOAL_RE.search(line)
        if m:
            frame = m.group(1)
            wx, wy, wt = float(m.group(2)), float(m.group(3)), float(m.group(4))
            nav_world = m.group(5) == "True"
            goals.append((wx, wy, wt, nav_world, frame))
    return goals


def _parse_spawn_origin(stderr_text: str) -> np.ndarray | None:
    for line in stderr_text.splitlines():
        m = _SPAWN_RE.search(line)
        if m:
            return np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])
    return None


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(300)
def test_rotate_in_place_robocasa_innate_mars_world_xy_stays_at_spawn():
    """Eight rotate steps: server world goals must stay within a few m of navigation_origin."""
    proc = None
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC_ROOT)] + env.get("PYTHONPATH", "").split(os.pathsep))
    env["EMET_SIM_NAV_DEBUG"] = "1"
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
                "--robot",
                "innate_mars",
                "--seed",
                "0",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if not _wait_for_port("127.0.0.1", 4401, timeout_sec=120):
            proc.terminate()
            out, err = proc.communicate(timeout=10)
            pytest.fail(f"Server did not bind 4401 within 120s.\nstderr:\n{err[-8000:]}\nstdout:\n{out[-2000:]}")

        time.sleep(2.0)

        from emet.app.robot_cli import create_robot_client_from_cli
        from emet.controller.controller_dynamem import DynamemController
        from emet.core.parameters import get_parameters

        robot = create_robot_client_from_cli(
            "innate_mars",
            "127.0.0.1",
            enable_rerun_server=False,
            start_immediately=True,
            allow_missing_depth=True,
            zmq_startup_timeout=60.0,
        )
        params = get_parameters("dynav_innate_mars.yaml")
        agent = DynamemController(
            robot,
            params,
            cpu_only=True,
            realtime_updates=False,
            save_rerun=False,
        )
        # Shorter per-step timeout: rotate only needs θ to settle, not long XY drives.
        orig_wait = getattr(robot, "_wait_at_goal", None)
        if orig_wait is not None:

            def _wait_at_goal_fast(timeout: float = 30.0, target_xyt=None):
                return orig_wait(timeout=min(timeout, 12.0), target_xyt=target_xyt)

            robot._wait_at_goal = _wait_at_goal_fast  # type: ignore[method-assign]
        agent.rotate_in_place()

        proc.terminate()
        try:
            _, stderr_text = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr_text = proc.communicate(timeout=10)

        origin = _parse_spawn_origin(stderr_text)
        goals = _parse_sim_nav_goals(stderr_text)

        assert origin is not None, (
            "Expected [sim_nav] startup banner with spawn origin in server stderr. "
            f"stderr tail:\n{stderr_text[-4000:]}"
        )
        assert len(goals) >= 1, (
            "Expected at least one [sim_nav] navigation goal during rotate_in_place. "
            f"stderr tail:\n{stderr_text[-6000:]}"
        )

        max_xy_jump = 0.0
        for wx, wy, _wt, nav_world, frame in goals:
            assert not nav_world, f"rotate_in_place must not use nav_world (got frame={frame!r})"
            assert frame in ("spawn_compose", "spawn_compose_corrected_world"), (
                f"rotate expects episode_compose on server, got frame={frame!r}"
            )
            jump = float(np.hypot(wx - origin[0], wy - origin[1]))
            max_xy_jump = max(max_xy_jump, jump)
            assert jump < 3.0, (
                f"rotate goal world ({wx:.3f}, {wy:.3f}) is {jump:.2f}m from spawn {origin[:2]} — "
                "rotate_in_place should only change θ"
            )

        assert max_xy_jump < 1.5, (
            f"rotate world xy should stay at spawn (max jump {max_xy_jump:.3f}m, origin={origin.tolist()})"
        )

    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

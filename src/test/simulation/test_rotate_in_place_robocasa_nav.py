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
    r"frame=(\w+).*goal_world=\[([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]"
    r".*base_before=\[([-\d.]+),\s*([-\d.]+)\].*Δxy=([-\d.]+)m.*nav_world=(\w+)"
)
# rotate_in_place uses nav_relative: goal XY must match base_before (θ-only); base may differ from spawn banner
_MAX_GOAL_XY_OFFSET_FROM_BASE_M = 0.05
_MAX_BASE_XY_DRIFT_DURING_ROTATE_M = 0.35
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


def _parse_sim_nav_goals(stderr_text: str) -> list[tuple[float, float, float, float, float, float, bool, str]]:
    """(goal wx, wy, wt, base bx, by, Δxy m, nav_world, frame)."""
    goals: list[tuple[float, float, float, float, float, float, bool, str]] = []
    for line in stderr_text.splitlines():
        if "[sim_nav]" not in line or "goal_world=" not in line:
            continue
        m = _NAV_GOAL_RE.search(line)
        if m:
            frame = m.group(1)
            wx, wy, wt = float(m.group(2)), float(m.group(3)), float(m.group(4))
            bx, by = float(m.group(5)), float(m.group(6))
            dxy = float(m.group(7))
            nav_world = m.group(8) == "True"
            goals.append((wx, wy, wt, bx, by, dxy, nav_world, frame))
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
            zmq_startup_timeout=120.0,
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

        max_goal_offset = 0.0
        max_base_drift = 0.0
        prev_base: tuple[float, float] | None = None
        for wx, wy, _wt, bx, by, dxy, nav_world, frame in goals:
            assert not nav_world, f"rotate_in_place must not use nav_world (got frame={frame!r})"
            assert frame == "relative_delta_world", (
                f"rotate expects nav_relative world delta, got frame={frame!r}"
            )
            max_goal_offset = max(max_goal_offset, dxy, float(np.hypot(wx - bx, wy - by)))
            if prev_base is not None:
                max_base_drift = max(
                    max_base_drift,
                    float(np.hypot(bx - prev_base[0], by - prev_base[1])),
                )
            prev_base = (bx, by)

        assert max_goal_offset < _MAX_GOAL_XY_OFFSET_FROM_BASE_M, (
            f"rotate goals must not translate in XY (max goal-vs-base offset {max_goal_offset:.3f}m)"
        )
        assert max_base_drift < _MAX_BASE_XY_DRIFT_DURING_ROTATE_M, (
            f"rotate should not translate base (max base drift {max_base_drift:.3f}m, "
            f"origin={origin.tolist()})"
        )

    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

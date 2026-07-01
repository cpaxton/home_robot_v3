# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Dynagraph ground-truth mode: default MuJoCo table scene via ZMQ sim_object_placements."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from emet.memory.graph_eqa.sim_ground_truth_graph import (
    build_ground_truth_graph_from_session,
    ground_truth_alignment_report,
    read_sim_object_placements,
)

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def _wait_for_port(host: str, port: int, timeout_sec: float = 90) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except (TimeoutError, OSError):
            time.sleep(0.5)
    return False


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(360)
def test_dynagraph_ground_truth_default_mujoco_scene():
    """
    MuJoCo default table scene: emet_session exposes sim_object_placements;
    GT graph builder produces red cylinder + blue cube nodes without VLM.
    """
    proc = None
    robot = None
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC_ROOT)] + env.get("PYTHONPATH", "").split(os.pathsep))
    if sys.platform == "linux":
        env["MUJOCO_GL"] = "egl"
    env["PYTHONUNBUFFERED"] = "1"

    port_offset = int(os.environ.get("EMET_TEST_PORT_OFFSET", str(os.getpid() % 400 + 50)))
    recv_port = 4401 + port_offset

    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "emet.simulation.mujoco_server",
                "--headless",
                "--port-offset",
                str(port_offset),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not _wait_for_port("127.0.0.1", recv_port, timeout_sec=90):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait(timeout=5)
            pytest.fail("MuJoCo server did not start. stderr:\n" + stderr)
        time.sleep(15)

        from emet.controller.zmq_client import StretchZmqClient
        from emet.memory.graph_eqa import GraphEQAMemory

        robot = StretchZmqClient(
            robot_ip="127.0.0.1",
            enable_rerun_server=False,
            start_immediately=True,
            port_offset=port_offset,
        )
        session = robot.get_emet_session()
        assert session is not None, "expected emet_session on observation"
        placements = read_sim_object_placements(session)
        assert placements is not None and len(placements) >= 3

        from emet.simulation.sim_object_placements import assert_default_table_gt

        assert_default_table_gt(placements)

        obs = robot.get_observation()
        rgb = np.asarray(obs.rgb, dtype=np.uint8)
        mem = GraphEQAMemory(defer_llm_clients=True)
        n_added, gt = build_ground_truth_graph_from_session(mem, rgb, session)
        assert n_added >= 2
        assert gt is not None

        g = mem.to_string().lower()
        assert "red" in g
        assert "blue" in g
        assert "table" in g

        report = ground_truth_alignment_report(mem, gt, max_dist_xy=0.05)
        assert "NO GT match" not in report, report

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

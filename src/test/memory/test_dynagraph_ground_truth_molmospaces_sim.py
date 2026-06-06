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

"""Dynagraph ground-truth mode on MolmoSpaces iTHOR via unified ``--scene ithor``."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from emet.config.sim_launch_config import SimLaunchMolmospaces
from emet.memory.graph_eqa.sim_ground_truth_graph import (
    build_ground_truth_graph_from_session,
    read_sim_object_placements,
)
from emet.simulation.molmospaces_config import build_molmospaces_wrapper_command
from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent

_run_molmo = os.environ.get("RUN_MOLMOSPACES_TESTS", "0").strip().lower()
RUN_MOLMOSPACES_TESTS = _run_molmo in ("1", "true", "yes", "on")


def _molmospaces_wrapper_available() -> bool:
    return build_molmospaces_wrapper_command(["--help"]) is not None


def _wait_for_port(host: str, port: int, timeout_sec: float = 180) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except (TimeoutError, OSError):
            time.sleep(0.5)
    return False


@pytest.mark.skipif(not RUN_MOLMOSPACES_TESTS, reason="RUN_MOLMOSPACES_TESTS=0")
@pytest.mark.skipif(not _molmospaces_wrapper_available(), reason="MolmoSpaces wrapper not installed")
@pytest.mark.timeout(600)
def test_dynagraph_ground_truth_molmospaces_ithor():
    """
    MolmoSpaces iTHOR: emet_session exposes molmospaces GT placements;
    GT graph builder produces nodes without VLM.
    """
    proc = None
    robot = None
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC_ROOT)] + env.get("PYTHONPATH", "").split(os.pathsep))
    if sys.platform == "linux":
        env["MUJOCO_GL"] = "egl"
    env["PYTHONUNBUFFERED"] = "1"

    port_offset = int(os.environ.get("EMET_TEST_PORT_OFFSET", str(os.getpid() % 400 + 60)))
    recv_port = 4401 + port_offset

    sim = SimLaunchMolmospaces(
        scene="ithor",
        split="train",
        index=0,
        robot="stretch",
        headless=True,
        port_offset=port_offset,
    )
    server_argv = prepare_mujoco_server_argv(sim)

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "emet.simulation.mujoco_server", *server_argv],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not _wait_for_port("127.0.0.1", recv_port, timeout_sec=180):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait(timeout=5)
            pytest.fail("MolmoSpaces MuJoCo server did not start. stderr:\n" + stderr)
        time.sleep(25)

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
        env_desc = session.get("environment") or {}
        assert env_desc.get("kind") == "molmospaces", env_desc

        placements = read_sim_object_placements(session)
        assert placements is not None and len(placements) >= 5, (
            f"got {0 if placements is None else len(placements)} placements"
        )

        obs = robot.get_observation()
        rgb = np.asarray(obs.rgb, dtype=np.uint8)
        mem = GraphEQAMemory(defer_llm_clients=True)
        n_added, gt = build_ground_truth_graph_from_session(mem, rgb, session)
        assert n_added >= 5, f"expected >=5 GT nodes, got {n_added}"
        assert gt is not None

    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)

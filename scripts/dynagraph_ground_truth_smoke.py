#!/usr/bin/env python3
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

"""Smoke: ground-truth graph build + export sidecars for default / robocasa / MolmoSpaces sim."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PORT_OFFSET = int(os.environ.get("EMET_GT_PORT_OFFSET", str(os.getpid() % 400 + 100)))


def _wait_port(port: int, timeout: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _build_server_argv(scene: str) -> list[str]:
    from emet.config.sim_launch_config import build_sim_launch_config_from_serve_cli
    from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv

    scene_arg: str | None
    if scene in ("default", "table"):
        scene_arg = None
    elif scene == "robocasa":
        scene_arg = "robocasa"
    else:
        scene_arg = scene

    cfg = build_sim_launch_config_from_serve_cli(
        scene=scene_arg,
        split="train",
        index=0,
        install_scene_if_missing=False,
        robot="stretch",
        headless=True,
        show_viewer_ui=False,
        no_cameras=False,
        use_glx=False,
        seed=0,
        steps=None,
        debug_molmospaces_spawn=False,
        port_offset=PORT_OFFSET,
        robocasa_task="PickPlaceCounterToCabinet",
    )
    return prepare_mujoco_server_argv(cfg)


def _run_gt_smoke(scene: str) -> int:
    from emet.controller.zmq_client import StretchZmqClient
    from emet.memory.graph_eqa import GraphEQAMemory
    from emet.memory.graph_eqa.sim_ground_truth_graph import (
        build_ground_truth_graph_from_session,
        ground_truth_alignment_report,
        read_sim_object_placements,
    )
    from emet.memory.headless_export import export_dynagraph_episode
    from emet.simulation.sim_object_placements import assert_default_table_gt

    recv_port = 4401 + PORT_OFFSET
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["MUJOCO_GL"] = "egl"
    env["PYTHONUNBUFFERED"] = "1"

    server_argv = _build_server_argv(scene)
    server_cmd = [sys.executable, "-m", "emet.simulation.mujoco_server", *server_argv]
    print(f"Starting server scene={scene!r} (ZMQ recv ~{recv_port})…", file=sys.stderr)
    server = subprocess.Popen(server_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    robot = None
    try:
        bind_timeout = 180.0 if scene == "ithor" else 120.0
        if not _wait_port(recv_port, timeout=bind_timeout):
            out = server.stdout.read() if server.stdout else ""
            print(f"FAIL: server did not bind port {recv_port}\n{out}", file=sys.stderr)
            return 1
        settle = 25.0 if scene == "ithor" else 18.0
        time.sleep(settle)

        robot = StretchZmqClient(
            robot_ip="127.0.0.1",
            enable_rerun_server=False,
            start_immediately=True,
            port_offset=PORT_OFFSET,
        )
        session = robot.get_emet_session()
        if session is None:
            print("FAIL: missing emet_session", file=sys.stderr)
            return 1

        placements = read_sim_object_placements(session)
        if scene in ("default", "table"):
            assert_default_table_gt(placements)
            min_nodes = 2
        elif scene == "robocasa":
            if not placements or len(placements) < 5:
                print(
                    f"FAIL: robocasa placements too small: {0 if not placements else len(placements)}", file=sys.stderr
                )
                return 1
            min_nodes = 5
        else:
            env_desc = session.get("environment") or {}
            if env_desc.get("kind") != "molmospaces":
                print(f"FAIL: expected molmospaces environment, got {env_desc!r}", file=sys.stderr)
                return 1
            if not placements or len(placements) < 5:
                print(f"FAIL: molmo placements too small: {0 if not placements else len(placements)}", file=sys.stderr)
                return 1
            min_nodes = 5

        obs = robot.get_observation()
        rgb = np.asarray(obs.rgb, dtype=np.uint8)
        mem = GraphEQAMemory(defer_llm_clients=True)
        n_added, gt = build_ground_truth_graph_from_session(mem, rgb, session)
        if n_added < min_nodes or gt is None:
            print(f"FAIL: GT graph nodes={n_added}", file=sys.stderr)
            return 1

        report = ground_truth_alignment_report(mem, gt, max_dist_xy=0.05)
        if "NO GT match" in report:
            print(f"FAIL: alignment report\n{report}", file=sys.stderr)
            return 1

        with tempfile.TemporaryDirectory(prefix="dynagraph_gt_") as tmp:
            export_dynagraph_episode(
                mem,
                None,
                tmp,
                title="Scene graph (Dynagraph GT smoke)",
                ground_truth_mode=True,
                sim_object_placements=placements,
                gt_alignment_report_text=report,
            )
            if not Path(tmp, "sim_object_placements.json").is_file():
                print("FAIL: sim_object_placements.json missing", file=sys.stderr)
                return 1
            if not Path(tmp, "scene_graph_report.txt").is_file():
                print("FAIL: scene_graph_report.txt missing", file=sys.stderr)
                return 1

        print(f"PASS: dynagraph ground-truth smoke (scene={scene}, nodes={n_added})", file=sys.stderr)
        return 0
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynagraph ground-truth smoke test")
    parser.add_argument(
        "--scene",
        default="default",
        choices=("default", "table", "robocasa", "ithor"),
        help="Sim scene: default table (CI), robocasa, or MolmoSpaces ithor (needs wrapper)",
    )
    args = parser.parse_args()
    os.chdir(REPO)
    return _run_gt_smoke(args.scene)


if __name__ == "__main__":
    raise SystemExit(main())

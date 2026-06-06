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

"""Smoke: full ``emet run dynagraph --ground-truth --export`` against default MuJoCo scene."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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


def main() -> int:
    os.chdir(REPO)
    recv_port = 4401 + PORT_OFFSET
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["MUJOCO_GL"] = "egl"
    env["PYTHONUNBUFFERED"] = "1"

    server_cmd = [
        sys.executable,
        "-m",
        "emet.simulation.mujoco_server",
        "--headless",
        "--port-offset",
        str(PORT_OFFSET),
    ]
    print(f"Starting server (ZMQ recv ~{recv_port})…", file=sys.stderr)
    server = subprocess.Popen(server_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        if not _wait_port(recv_port, timeout=120.0):
            out = server.stdout.read() if server.stdout else ""
            print(f"FAIL: server did not bind port {recv_port}\n{out}", file=sys.stderr)
            return 1
        time.sleep(20)

        with tempfile.TemporaryDirectory(prefix="dynagraph_gt_") as tmp:
            client_cmd = [
                sys.executable,
                "-m",
                "emet.app.run_dynagraph",
                "--no-rerun",
                "--cpu-only",
                "--ground-truth",
                "--not_rotate_in_place",
                "--port-offset",
                str(PORT_OFFSET),
                "--export",
                tmp,
            ]
            print("Running dynagraph ground-truth export…", file=sys.stderr)
            res = subprocess.run(client_cmd, env=env, capture_output=True, text=True, timeout=600)
            combined = (res.stdout or "") + (res.stderr or "")
            print(combined, file=sys.stderr)
            if res.returncode != 0:
                print(f"FAIL: dynagraph exit {res.returncode}", file=sys.stderr)
                return res.returncode
            low = combined.lower()
            if "red" not in low or "blue" not in low or "table" not in low:
                print("FAIL: export output missing red/blue/table GT labels", file=sys.stderr)
                return 1
            report_path = Path(tmp) / "scene_graph_report.txt"
            if not report_path.is_file():
                print("FAIL: scene_graph_report.txt missing", file=sys.stderr)
                return 1
            placements_path = Path(tmp) / "sim_object_placements.json"
            if not placements_path.is_file():
                print("FAIL: sim_object_placements.json missing", file=sys.stderr)
                return 1

        print("PASS: dynagraph ground-truth export", file=sys.stderr)
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=3)


if __name__ == "__main__":
    raise SystemExit(main())

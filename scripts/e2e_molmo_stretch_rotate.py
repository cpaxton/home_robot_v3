#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""E2E: emet serve mujoco (MolmoSpaces iTHOR + Stretch) + rotate_in_place like dynamem."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def wait_port(port: int, timeout: float) -> bool:
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
    recv_port = 4402
    log_path = REPO / "molmo_e2e_server.log"

    serve_cmd = [
        "uv",
        "run",
        "emet",
        "serve",
        "mujoco",
        "--molmospaces-scene",
        "ithor",
        "--molmospaces-split",
        "train",
        "--molmospaces-index",
        "0",
        "--robot",
        "stretch",
        "--headless",
        "--no-cameras",
    ]
    print("SERVE:", " ".join(serve_cmd), flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            serve_cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env={**os.environ, "MUJOCO_GL": "egl"},
        )

    try:
        if not wait_port(recv_port, timeout=150.0):
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
            print("FAIL: recv port not open\n", tail, file=sys.stderr)
            return 1

        print("ZMQ recv port open; waiting for MuJoCo subprocess + observations…", flush=True)
        from emet.controller.zmq_client import StretchZmqClient
        from emet.utils.geometry import angle_difference

        client = StretchZmqClient(
            robot_ip="127.0.0.1",
            enable_rerun_server=False,
            start_immediately=True,
            allow_missing_depth=True,
        )

        obs = None
        for i in range(120):
            obs = client.get_observation(timeout=2.0)
            if obs is not None:
                print(f"Got observation after {i + 1}s", flush=True)
                break
            time.sleep(1)
        if obs is None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
            print("FAIL: no observations\n", tail, file=sys.stderr)
            return 1

        sess = client.get_emet_session() or {}
        teleport = (sess.get("capabilities") or {}).get("teleport_base")
        print(f"session teleport_base={teleport} env={sess.get('environment')}", flush=True)
        if not teleport:
            print("FAIL: teleport_base not advertised", file=sys.stderr)
            return 1

        # Same sequence as controller_dynamem.rotate_in_place (simplified)
        client.move_to_nav_posture()
        client.look_front(blocking=True)
        time.sleep(0.5)

        t0 = client.get_base_pose().copy()
        print(f"start spawn xyt={t0}", flush=True)
        thetas = [float(t0[2])]
        for step in range(8):
            goal = np.array([t0[0], t0[1], thetas[-1] + np.pi / 4], dtype=np.float64)
            client.move_base_to(goal, blocking=True, timeout=30.0)
            cur = client.get_base_pose()
            thetas.append(float(cur[2]))
            d = abs(angle_difference(thetas[-1], thetas[-2]))
            print(f"  step {step + 1}: theta={cur[2]:.4f} delta={d:.4f} rad", flush=True)

        total = abs(angle_difference(thetas[-1], thetas[0]))
        print(f"net yaw change: {total:.4f} rad (full scan returns near 0)", flush=True)

        log_tail = log_path.read_text(encoding="utf-8", errors="replace")
        teleport_logs = [ln for ln in log_tail.splitlines() if "MolmoSpaces teleport nav" in ln]
        print(f"server teleport log lines: {len(teleport_logs)}", flush=True)
        if teleport_logs:
            print("  last:", teleport_logs[-1][:120], flush=True)

        client.stop()

        step_deltas = [abs(angle_difference(thetas[i + 1], thetas[i])) for i in range(len(thetas) - 1)]
        sum_delta = float(sum(step_deltas))
        print(f"sum_step_deltas={sum_delta:.4f}", flush=True)
        if sum_delta < 1.0 or min(step_deltas) < 0.3:
            print("FAIL: insufficient rotation", file=sys.stderr)
            if "teleport_base failed" in log_tail or "teleport did not reach" in log_tail:
                print("Server teleport errors present — see molmo_e2e_server.log", file=sys.stderr)
            return 1
        if len(teleport_logs) < 4:
            print("WARN: expected multiple MolmoSpaces teleport log lines", flush=True)
        print("PASS: MolmoSpaces Stretch rotation E2E", flush=True)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())

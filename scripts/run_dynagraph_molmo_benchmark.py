#!/usr/bin/env python3
"""MolmoSpaces Dynagraph benchmark: explore + export + eval-dynagraph."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("DYNAGRAPH_MOLMO_BENCH", "/tmp/dynagraph_molmo_bench"))
PORT = 4401


def _wait_port(timeout: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def main() -> int:
    out = BASE / "ithor0_stretch"
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["EMET_SIM_NAV_TELEPORT"] = "1"
    env["EMET_ZMQ_STARTUP_TIMEOUT"] = "120"

    server = subprocess.Popen(
        [
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
        ],
        cwd=REPO,
        stdout=open(out / "server.log", "w"),
        stderr=subprocess.STDOUT,
    )
    try:
        if not _wait_port():
            print("FAIL: server timeout", file=sys.stderr)
            return 1
        time.sleep(12)
        dyn = subprocess.run(
            [
                "uv",
                "run",
                "emet",
                "run",
                "dynagraph",
                "--robot",
                "stretch",
                "--robot-ip",
                "127.0.0.1",
                "--no-rerun",
                "--cpu-only",
                "--no-sensor-perception",
                "--explore-loop",
                "--explore-max-iters",
                "8",
                "--export",
                str(out / "episode"),
            ],
            cwd=REPO,
            env=env,
            timeout=600,
        )
        if dyn.returncode not in (0, 250):
            print(f"FAIL: dynagraph exit {dyn.returncode}", file=sys.stderr)
            return 1
        ev = subprocess.run(
            [
                "uv",
                "run",
                "emet",
                "eval-dynagraph",
                "--episode",
                str(out / "episode"),
                "-o",
                str(out / "dynagraph_eval.json"),
            ],
            cwd=REPO,
            timeout=120,
        )
        if ev.returncode != 0:
            return 1
        metrics = json.loads((out / "dynagraph_eval.json").read_text())
        nodes = metrics.get("graph", {}).get("node_count", 0)
        explored = metrics.get("explore", {}).get("explored_area_m2", 0)
        print(f"PASS: nodes={nodes} explored_m2={explored}")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())

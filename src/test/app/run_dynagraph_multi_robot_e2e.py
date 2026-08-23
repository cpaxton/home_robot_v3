#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Run Dynagraph Robocasa E2E for innate_mars / galaxea_r1 and compare floor metrics.

Stretch Robocasa (RobosuiteZmqServer + GenericZmqClient) lives on branch
``feature/stretch-robocasa-robosuite``; this harness uses stretch_mujoco on main path only.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from emet.memory.floor_metrics import (
    compare_explored_floor_metrics,
    explored_vs_spawn_summary,
    load_floor_metrics,
)

ROBOTS = ("innate_mars", "galaxea_r1")
BASE = Path("/tmp/dynagraph_e2e_compare")
SEED = 0
LAYOUT = 1
STYLE = 1
SEND_PORT = 4401
EXPLORE_ITERS = 15
EXPLORE_FAILURES = 5
SERVER_WAIT_S = 180
DYNAGRAPH_TIMEOUT_S = 900


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _wait_port(port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.5)
    return False


def _kill_servers() -> None:
    subprocess.run(["uv", "run", "emet", "kill-mujoco-server"], cwd=Path(__file__).resolve().parents[3], check=False)
    subprocess.run(["pkill", "-f", "emet serve mujoco"], check=False)
    time.sleep(1.5)


def _run_robot(robot: str) -> dict:
    repo = Path(__file__).resolve().parents[3]
    out_dir = BASE / robot
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server.log"

    _kill_servers()
    server_cmd = [
        "uv",
        "run",
        "emet",
        "serve",
        "mujoco",
        "--use-robocasa",
        "--robot",
        robot,
        "--headless",
        "--seed",
        str(SEED),
    ]
    print(f"\n=== {robot}: starting server ===", flush=True)
    with open(log_path, "w", encoding="utf-8") as log_f:
        server = subprocess.Popen(
            server_cmd,
            cwd=repo,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    try:
        if not _wait_port(SEND_PORT, SERVER_WAIT_S):
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"{robot}: server did not bind port {SEND_PORT} within {SERVER_WAIT_S}s\n{log_tail}")
        time.sleep(12.0)

        graph_dir = out_dir / "graph"
        if graph_dir.exists():
            shutil.rmtree(graph_dir)

        dyn_cmd = [
            "uv",
            "run",
            "emet",
            "run",
            "dynagraph",
            "--robot",
            robot,
            "--robot-ip",
            "127.0.0.1",
            "--dynav-config",
            "dynav_config.yaml",
            "--no-rerun",
            "--cpu-only",
            "--explore-loop",
            "--explore-max-iters",
            str(EXPLORE_ITERS),
            "--explore-max-failures",
            str(EXPLORE_FAILURES),
            "--export",
            str(out_dir / "graph"),
        ]
        env = os.environ.copy()
        env["EMET_ZMQ_STARTUP_TIMEOUT"] = "120"
        env["EMET_SIM_NAV_TELEPORT"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        print(f"=== {robot}: running dynagraph ===", flush=True)
        t_dyn_start = time.time()
        proc = subprocess.run(
            dyn_cmd,
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=DYNAGRAPH_TIMEOUT_S,
            env=env,
        )
        (out_dir / "dynagraph.stdout").write_text(proc.stdout, encoding="utf-8")
        (out_dir / "dynagraph.stderr").write_text(proc.stderr, encoding="utf-8")
        metrics_path = out_dir / "graph" / "floor_metrics.json"
        if proc.returncode != 0:
            export_fresh = metrics_path.is_file() and metrics_path.stat().st_mtime >= t_dyn_start
            if not export_fresh:
                raise RuntimeError(
                    f"{robot}: dynagraph exit {proc.returncode}\nstdout tail:\n{proc.stdout[-3000:]}\n"
                    f"stderr tail:\n{proc.stderr[-3000:]}"
                )
            print(
                f"{robot}: dynagraph exit {proc.returncode} but fresh export present; treating as success",
                flush=True,
            )

        if not metrics_path.is_file():
            raise FileNotFoundError(f"{robot}: missing {metrics_path}")
        metrics = load_floor_metrics(out_dir / "graph")
        vs = explored_vs_spawn_summary(metrics)
        print(
            f"{robot}: explored={metrics.get('explored_area_m2')} m² "
            f"scene_walkable={vs.get('scene_walkable_area_m2')} m² "
            f"spawn_eroded={vs.get('spawn_walkable_area_m2')} m²",
            flush=True,
        )
        return metrics
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        _kill_servers()


def main() -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    failures: list[str] = []

    report_only = os.environ.get("DYNAGRAPH_E2E_REPORT_ONLY", "").strip().lower() in ("1", "true", "yes")
    if report_only:
        for robot in ROBOTS:
            try:
                results[robot] = load_floor_metrics(BASE / robot / "graph")
            except Exception as e:
                failures.append(f"{robot}: {e}")
    else:
        for robot in ROBOTS:
            try:
                results[robot] = _run_robot(robot)
            except Exception as e:
                failures.append(f"{robot}: {e}")
                print(f"FAIL {robot}: {e}", file=sys.stderr, flush=True)

    report = {"robots": results, "pairwise_explored": {}, "spawn_vs_explored": {}, "scene_walkable_m2": {}}
    ok_robots = list(results.keys())
    for robot in ok_robots:
        report["spawn_vs_explored"][robot] = explored_vs_spawn_summary(results[robot])
        spawn = results[robot].get("spawn_floor_map") or {}
        report["scene_walkable_m2"][robot] = spawn.get("scene_walkable_area_m2")
    for i, a in enumerate(ok_robots):
        for b in ok_robots[i + 1 :]:
            key = f"{a}_vs_{b}"
            report["pairwise_explored"][key] = compare_explored_floor_metrics(results[a], results[b], rtol_area=0.35)

    report_path = BASE / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote {report_path}", flush=True)

    if len(ok_robots) >= 2:
        ref = ok_robots[0]
        for other in ok_robots[1:]:
            cmp = report["pairwise_explored"][f"{ref}_vs_{other}"]
            print(
                f"compare {ref} vs {other}: area_match={cmp['area_match']} "
                f"cells_delta={cmp['cell_delta']} "
                f"({cmp['left']['area_m2']:.2f} vs {cmp['right']['area_m2']:.2f} m²)",
                flush=True,
            )
        scene_vals = [v for v in report["scene_walkable_m2"].values() if v is not None]
        if scene_vals:
            smin, smax = min(scene_vals), max(scene_vals)
            print(f"scene_walkable_area_m2 range: {smin:.3f} .. {smax:.3f}", flush=True)

    if failures:
        print("\nFailures:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    if len(ok_robots) < len(ROBOTS):
        return 1
    if len(ok_robots) >= 2 and not all(v.get("area_match") for v in report["pairwise_explored"].values()):
        print("WARN: explored floor areas differ between robots", file=sys.stderr)
        return 1
    scene_vals = [v for v in report.get("scene_walkable_m2", {}).values() if v is not None]
    if len(scene_vals) >= 2:
        smin, smax = min(scene_vals), max(scene_vals)
        if smax > 1e-9 and (smax - smin) / smax > 0.05:
            print("WARN: spawner scene walkable area differs >5% across robots", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

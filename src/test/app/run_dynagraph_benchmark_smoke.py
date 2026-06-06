#!/usr/bin/env python3
"""Dynagraph benchmark smoke: default table, Robocasa, MolmoSpaces (CI-friendly tiers)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
BASE = Path(os.environ.get("DYNAGRAPH_BENCH_BASE", "/tmp/dynagraph_bench_smoke"))
SEND_PORT = 4401
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
    subprocess.run(["uv", "run", "emet", "kill-mujoco-server"], cwd=REPO, check=False)
    subprocess.run(["pkill", "-f", "emet serve mujoco"], check=False)
    time.sleep(1.5)


def _run_dynagraph(cmd: list[str], *, timeout: int = DYNAGRAPH_TIMEOUT_S) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["EMET_ZMQ_STARTUP_TIMEOUT"] = "120"
    env["EMET_SIM_NAV_TELEPORT"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout, env=env)


def _node_count_from_export(export_dir: Path) -> float:
    from emet.memory.graph_eqa.dynagraph_eval import compute_dynagraph_eval

    try:
        m = compute_dynagraph_eval(export_dir)
        return float(m.get("graph", {}).get("node_count", 0))
    except Exception:
        report = export_dir / "scene_graph_report.txt"
        if report.is_file():
            import re

            text = report.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"Nodes\s*\((\d+)\)", text, re.I)
            return float(m.group(1)) if m else 0.0
        return 0.0


def _serve_and_run(
    server_cmd: list[str],
    dyn_cmd: list[str],
    export_dir: Path,
    *,
    log_name: str = "server.log",
) -> dict[str, Any]:
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    log_path = export_dir.parent / log_name
    _kill_servers()
    with open(log_path, "w", encoding="utf-8") as log_f:
        server = subprocess.Popen(
            server_cmd, cwd=REPO, stdout=log_f, stderr=subprocess.STDOUT, text=True
        )
    try:
        if not _wait_port(SEND_PORT, SERVER_WAIT_S):
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError(f"server did not bind {SEND_PORT}\n{tail}")
        time.sleep(12.0)
        proc = _run_dynagraph(dyn_cmd)
        combined = (proc.stdout or "") + (proc.stderr or "")
        nodes = _node_count_from_export(export_dir)
        return {
            "exit_code": proc.returncode,
            "node_count": nodes,
            "exported": "Exported graph memory to" in combined,
            "combined_tail": combined[-4000:],
        }
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        _kill_servers()


def tier_default_table() -> dict[str, Any]:
    out_dir = BASE / "default_table"
    res = _serve_and_run(
        ["uv", "run", "emet", "serve", "mujoco", "--robot", "stretch", "--headless"],
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
            "--ground-truth",
            "--not_rotate_in_place",
            "--export",
            str(out_dir),
        ],
        out_dir,
    )
    res["tier"] = "default_table_gt"
    res["pass"] = res["exported"] and res["node_count"] >= 2.0
    return res


def tier_default_perception() -> dict[str, Any]:
    out_dir = BASE / "default_table_perception"
    res = _serve_and_run(
        ["uv", "run", "emet", "serve", "mujoco", "--robot", "stretch", "--headless"],
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
            "--explore-loop",
            "--explore-max-iters",
            "5",
            "--export",
            str(out_dir),
        ],
        out_dir,
    )
    res["tier"] = "default_table_perception"
    res["pass"] = res["exported"] and res["node_count"] >= 1.0
    return res


def tier_robocasa() -> dict[str, Any]:
    out_dir = BASE / "robocasa_innate_mars"
    res = _serve_and_run(
        [
            "uv",
            "run",
            "emet",
            "serve",
            "mujoco",
            "--use-robocasa",
            "--robot",
            "innate_mars",
            "--headless",
            "--seed",
            "0",
        ],
        [
            "uv",
            "run",
            "emet",
            "run",
            "dynagraph",
            "--robot",
            "innate_mars",
            "--robot-ip",
            "127.0.0.1",
            "--dynav-config",
            "dynav_config.yaml",
            "--no-rerun",
            "--cpu-only",
            "--explore-loop",
            "--explore-max-iters",
            "5",
            "--export",
            str(out_dir),
        ],
        out_dir,
    )
    metrics_path = out_dir / "floor_metrics.json"
    res["tier"] = "robocasa_innate_mars"
    res["pass"] = (
        res["exported"]
        and metrics_path.is_file()
        and res["node_count"] >= 0.0  # perception graph may be 0; floor export must succeed
    )
    return res


def tier_molmospaces() -> dict[str, Any]:
    out_dir = BASE / "molmo_stretch"
    res = _serve_and_run(
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
            "5",
            "--export",
            str(out_dir),
        ],
        out_dir,
        log_name="molmo_server.log",
    )
    res["tier"] = "molmospaces_ithor0"
    res["pass"] = res["exported"] and (out_dir / "floor_metrics.json").is_file()
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynagraph benchmark smoke tiers")
    parser.add_argument("--default", action="store_true", help="Default table GT + perception")
    parser.add_argument("--robocasa", action="store_true")
    parser.add_argument("--molmo", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    run_all = args.all or not (args.default or args.robocasa or args.molmo)

    results: list[dict[str, Any]] = []
    if run_all or args.default:
        results.append(tier_default_table())
        results.append(tier_default_perception())
    if run_all or args.robocasa:
        results.append(tier_robocasa())
    if run_all or args.molmo:
        results.append(tier_molmospaces())

    report = {"results": results, "all_pass": all(r.get("pass") for r in results)}
    BASE.mkdir(parents=True, exist_ok=True)
    rep_path = BASE / "benchmark_report.json"
    rep_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {rep_path}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

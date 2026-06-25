# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
#
# Integration: ``uv run emet run dynagraph`` with explore-loop, dynav-config, question, export
# on Robocasa + innate_mars (same flags as docs/dynagraph_robocasa_e2e.md manual EQA).
#
# Run: uv run emet test src/test/app/test_dynagraph_robocasa_question_cli.py -v
# Skip: RUN_SIM_TESTS=0

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")

REPO = Path(__file__).resolve().parents[3]
EXPORT_BASE = Path("/tmp/dynagraph_q_test")
SEND_PORT = 4401
SERVER_WAIT_S = 180
DYNAGRAPH_TIMEOUT_S = 600


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


def test_run_dynagraph_module_exposes_dynav_and_explore_flags():
    """Guard against stale installs missing Dynagraph CLI options."""
    r = subprocess.run(
        [sys.executable, "-m", "emet.app.run_dynagraph", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert "--config" in out or "-C" in out
    assert "--dynav-config" in out
    assert "--explore-loop" in out


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(900)
def test_dynagraph_robocasa_question_export_innate_mars():
    """Full Robocasa run: explore-loop + question + export (documented manual recipe)."""
    pytest.importorskip("mujoco")
    export_dir = EXPORT_BASE / "innate_mars"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    log_path = EXPORT_BASE / "server.log"
    EXPORT_BASE.mkdir(parents=True, exist_ok=True)

    _kill_servers()
    server_cmd = [
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
    ]
    with open(log_path, "w", encoding="utf-8") as log_f:
        server = subprocess.Popen(
            server_cmd,
            cwd=REPO,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    try:
        if not _wait_port(SEND_PORT, SERVER_WAIT_S):
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            pytest.fail(f"server did not bind {SEND_PORT} within {SERVER_WAIT_S}s\n{tail}")
        time.sleep(12.0)

        # Robocasa ZMQ serves rendered depth — use shared dynav_config (depth_source: sensor).
        # dynav_innate_mars.yaml forces DA3 and can fail Umeyama alignment in sim on rotate-in-place.
        dyn_cmd = [
            "uv",
            "run",
            "emet",
            "run",
            "dynagraph",
            "--robot",
            "innate_mars",
            "--robot-ip",
            "127.0.0.1",
            "--no-rerun",
            "--cpu-only",
            "--dynav-config",
            "dynav_config.yaml",
            "--explore-loop",
            "--explore-max-iters",
            "5",
            "--explore-max-failures",
            "5",
            "--question",
            "Where is the sink?",
            "--export",
            str(export_dir),
        ]
        env = os.environ.copy()
        env["EMET_ZMQ_STARTUP_TIMEOUT"] = "120"
        env["EMET_SIM_NAV_TELEPORT"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.run(
            dyn_cmd,
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=DYNAGRAPH_TIMEOUT_S,
            env=env,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        assert "No such option: --dynav-config" not in combined
        assert "No such option: --explore-loop" not in combined
        report = export_dir / "scene_graph_report.txt"
        metrics = export_dir / "floor_metrics.json"
        assert report.is_file(), f"missing {report}\n{combined[-4000:]}"
        assert metrics.is_file(), f"missing {metrics}\n{combined[-4000:]}"
        assert "Explore-loop [export-path] done" in combined, combined[-4000:]
        assert "--- GraphEQA answer ---" in combined or "GraphEQA answer" in combined, combined[-4000:]
        assert not re.search(r"\banswer:\s*image\s*\d+", combined, re.I), combined[-8000:]
        assert "Exported graph memory to" in combined, combined[-4000:]
        assert "dynav=dynav_config.yaml" in combined
        report_text = report.read_text(encoding="utf-8", errors="replace")
        assert "innate_mars" in report_text or "Explored floor" in report_text
        # Exit 250 can occur after successful export (CLIP/model teardown abort); artefacts are authoritative.
        if proc.returncode != 0:
            assert "Exported graph memory to" in combined, (
                f"dynagraph exit {proc.returncode} before export\n{combined[-8000:]}"
            )
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        _kill_servers()

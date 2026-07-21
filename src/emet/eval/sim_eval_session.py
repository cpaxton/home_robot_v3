# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Shared MuJoCo subprocess boot/teardown for benchmark eval harnesses."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from emet.simulation.sim_subprocess import wait_for_sim_tcp_port


def zmq_recv_port(port_offset: int) -> int:
    return 4401 + int(port_offset)


def benchmark_subprocess_env(repo: Path, *, cpu_only: bool) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MUJOCO_GL", "egl")
    env["EMET_SIM_NAV_TELEPORT"] = "1"
    env["EMET_ZMQ_STARTUP_TIMEOUT"] = "120"
    env["PYTHONUNBUFFERED"] = "1"
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def bind_timeout_s(sim_kind: str) -> float:
    return 180.0 if sim_kind in ("molmospaces", "robocasa") else 120.0


def sim_settle_s(sim_kind: str, *, cpu_only: bool) -> float:
    settle = 25.0 if sim_kind in ("molmospaces", "robocasa") else 15.0
    if cpu_only:
        settle += 15.0
    return settle


def terminate_server(proc: subprocess.Popen[Any]) -> None:
    from emet.utils.process_tree import terminate_process_tree

    terminate_process_tree(proc, grace_s=15.0)


@dataclass
class BenchmarkSimServer:
    server: subprocess.Popen[Any]
    env: dict[str, str]
    recv_port: int
    sim_kind: str
    port_offset: int


def launch_benchmark_sim_server(
    sim_cfg: Any,
    *,
    repo: Path,
    cpu_only: bool,
    cwd: Path | None = None,
    server_stderr: Any | None = None,
) -> BenchmarkSimServer:
    """Start headless ``mujoco_server`` and block until the ZMQ recv port is ready."""
    from dataclasses import replace

    from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv

    port_offset = int(getattr(sim_cfg, "port_offset", 0))
    sim_kind = str(getattr(sim_cfg, "kind", ""))
    recv_port = zmq_recv_port(port_offset)
    env = benchmark_subprocess_env(repo, cpu_only=cpu_only)
    sim_cfg = replace(sim_cfg, port_offset=port_offset, headless=True)
    server_cmd = [
        sys.executable,
        "-m",
        "emet.simulation.mujoco_server",
        *prepare_mujoco_server_argv(sim_cfg),
    ]
    from emet.utils.process_tree import popen_session

    stderr_target = subprocess.DEVNULL if server_stderr is None else server_stderr
    proc = popen_session(
        server_cmd,
        cwd=str(cwd or repo),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=stderr_target,
    )
    try:
        wait_for_sim_tcp_port(
            "127.0.0.1",
            recv_port,
            proc=proc,
            timeout_sec=bind_timeout_s(sim_kind),
        )
    except (TimeoutError, RuntimeError) as exc:
        terminate_server(proc)
        raise RuntimeError(f"sim server did not bind port {recv_port}") from exc
    time.sleep(sim_settle_s(sim_kind, cpu_only=cpu_only))
    return BenchmarkSimServer(
        server=proc,
        env=env,
        recv_port=recv_port,
        sim_kind=sim_kind,
        port_offset=port_offset,
    )


def terminate_benchmark_sim_server(sim: BenchmarkSimServer) -> None:
    terminate_server(sim.server)


@contextmanager
def benchmark_sim_server(
    sim_cfg: Any,
    *,
    repo: Path,
    cpu_only: bool,
    cwd: Path | None = None,
    server_stderr: Any | None = None,
) -> Iterator[BenchmarkSimServer]:
    sim = launch_benchmark_sim_server(
        sim_cfg,
        repo=repo,
        cpu_only=cpu_only,
        cwd=cwd,
        server_stderr=server_stderr,
    )
    try:
        yield sim
    finally:
        terminate_benchmark_sim_server(sim)


def connect_benchmark_robot(sim_cfg: Any, port_offset: int) -> Any:
    """ZMQ client for an already-running benchmark sim server."""
    from emet.app.robot_cli import create_robot_client_from_cli

    robot = create_robot_client_from_cli(
        str(getattr(sim_cfg, "robot", "stretch")),
        "127.0.0.1",
        port_offset=int(port_offset),
        enable_rerun_server=False,
        start_immediately=True,
        allow_missing_depth=True,
    )
    robot.move_to_nav_posture()
    robot.set_velocity(v=30.0, w=15.0)
    return robot

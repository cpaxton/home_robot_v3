# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Spawn ``emet.simulation.mujoco_server`` as a subprocess for ``emet run agent --start-sim``."""

from __future__ import annotations

import atexit
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emet.config.sim_launch_config import SimLaunchConfig

_SIM_PROC: subprocess.Popen[bytes] | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _shutdown_sim_process() -> None:
    global _SIM_PROC
    p = _SIM_PROC
    if p is None:
        return
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            p.kill()
    _SIM_PROC = None


def wait_for_sim_tcp_port(
    host: str,
    port: int,
    *,
    proc: subprocess.Popen[bytes],
    timeout_sec: float = 120.0,
    poll_sec: float = 0.35,
) -> None:
    """Block until *port* accepts a TCP connection or *proc* exits."""
    deadline = time.time() + timeout_sec
    last_err: OSError | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"MuJoCo server process exited before binding (code={proc.returncode}). "
                "Check stderr above or run `emet serve mujoco` with the same flags."
            )
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return
        except OSError as e:
            last_err = e
            time.sleep(poll_sec)
    msg = f"Timed out after {timeout_sec:.0f}s waiting for sim on {host}:{port}"
    if last_err is not None:
        msg += f" ({last_err})"
    raise TimeoutError(msg)


def spawn_mujoco_server_subprocess(cfg: SimLaunchConfig) -> subprocess.Popen[bytes]:
    """Start ``python -m emet.simulation.mujoco_server``; register atexit shutdown; wait for ZMQ send port."""
    from emet.config.sim_launch_config import SimLaunchMolmospaces
    from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv
    from emet.utils.port_utils import get_ports

    global _SIM_PROC
    if _SIM_PROC is not None and _SIM_PROC.poll() is None:
        raise RuntimeError("A sim subprocess is already running in this process.")

    argv = prepare_mujoco_server_argv(cfg)
    cmd = [sys.executable, "-m", "emet.simulation.mujoco_server", *argv]
    proc = subprocess.Popen(
        cmd,
        cwd=str(_repo_root()),
    )
    _SIM_PROC = proc
    atexit.register(_shutdown_sim_process)

    ports = get_ports(int(cfg.port_offset))
    wait_for_sim_tcp_port("127.0.0.1", int(ports.send), proc=proc)
    # Brief pause so ZMQ bind is fully ready before client connects
    time.sleep(0.4)
    if isinstance(cfg, SimLaunchMolmospaces):
        # First connection can race right after TCP accept; small extra delay for merge I/O
        time.sleep(0.2)
    return proc

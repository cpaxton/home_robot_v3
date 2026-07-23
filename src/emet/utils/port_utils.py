# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Helpers for network ports: defaults, offset computation, freeing ports."""

import os
import subprocess
from typing import NamedTuple


class ZmqPorts(NamedTuple):
    """The four ZMQ ports used by the simulation server and robot clients."""

    send: int
    recv: int
    state: int
    servo: int


DEFAULT_SEND_PORT = 4401
DEFAULT_RECV_PORT = 4402
DEFAULT_STATE_PORT = 4403
DEFAULT_SERVO_PORT = 4404


def get_ports(port_offset: int = 0) -> ZmqPorts:
    """Compute the four ZMQ ports given an offset from the defaults.

    >>> get_ports(0)
    ZmqPorts(send=4401, recv=4402, state=4403, servo=4404)
    >>> get_ports(100)
    ZmqPorts(send=4501, recv=4502, state=4503, servo=4504)
    """
    return ZmqPorts(
        send=DEFAULT_SEND_PORT + port_offset,
        recv=DEFAULT_RECV_PORT + port_offset,
        state=DEFAULT_STATE_PORT + port_offset,
        servo=DEFAULT_SERVO_PORT + port_offset,
    )


def kill_processes_on_port(port: int, *, listeners_only: bool = False) -> bool:
    """Kill processes using the given port. Returns True if any were killed.

    When *listeners_only* is True, only ``TCP LISTEN`` PIDs are signaled. Plain
    ``lsof -i:PORT`` also matches connected *clients* (e.g. dynagraph talking to
    MuJoCo) — killing those SIGTERMs the eval process itself (exit 241 / -15).
    """
    cmd = ["lsof", "-t", f"-i:{int(port)}"]
    if listeners_only:
        cmd.extend(["-sTCP:LISTEN"])
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if out.returncode != 0 or not out.stdout.strip():
        return False
    pids = [s for s in out.stdout.strip().split() if s.isdigit()]
    if not pids:
        return False
    # Never signal our own process / process group leader by accident.
    self_pid = str(os.getpid())
    self_pgid = str(os.getpgid(0))
    killed = False
    for pid in pids:
        if pid in (self_pid, self_pgid):
            continue
        try:
            subprocess.run(["kill", pid], check=False, capture_output=True)
            killed = True
        except Exception:
            pass
    return killed


def release_zmq_ports(port_offset: int = 0) -> list[int]:
    """Kill **listeners** on the four default ZMQ ports for *port_offset*.

    Used before answer-only EQA so MuJoCo EGL is not sharing the GPU with Qwen
    (vision ``generate`` was observed stuck at ~5% GPU util for 25+ minutes).
    Returns the ports that had a process killed.

    Important: only LISTEN sockets are targeted. Connected clients on the same
    ports must not be killed (that was SIGTERM-ing dynagraph mid-EQA setup).
    """
    ports = get_ports(int(port_offset or 0))
    freed: list[int] = []
    for p in (ports.send, ports.recv, ports.state, ports.servo):
        if kill_processes_on_port(p, listeners_only=True):
            freed.append(p)
    return freed

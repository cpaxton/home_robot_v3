# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Helpers for network ports: defaults, offset computation, freeing ports."""

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


def kill_processes_on_port(port: int) -> bool:
    """Kill processes using the given port. Returns True if any were killed."""
    try:
        out = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
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
    for pid in pids:
        try:
            subprocess.run(["kill", pid], check=False, capture_output=True)
        except Exception:
            pass
    return True

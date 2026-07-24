# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Regression: release_zmq_ports must not SIGTERM connected ZMQ clients."""

from __future__ import annotations

import socket
import subprocess
from unittest.mock import patch

from emet.utils.port_utils import get_ports, kill_processes_on_port, release_zmq_ports


def test_get_ports_offset() -> None:
    assert get_ports(0) == (4401, 4402, 4403, 4404)
    assert get_ports(100).send == 4501


def test_kill_processes_on_port_listeners_only_skips_connected_client() -> None:
    """Plain lsof -i:PORT matches clients; listeners_only must not."""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    client = socket.socket()
    client.connect(("127.0.0.1", port))
    try:
        # Client connection exists; listeners_only targets the listener PID only.
        # Same-process listener is skipped by the self-pid guard → no kill.
        assert kill_processes_on_port(port, listeners_only=True) is False
        # Socket still usable (we were not signaled).
        client.send(b"x")
    finally:
        client.close()
        listener.close()


def test_release_zmq_ports_uses_listeners_only() -> None:
    """Guard against regressing to plain lsof that kills dynagraph clients."""
    with patch("emet.utils.port_utils.kill_processes_on_port", return_value=True) as mock_kill:
        freed = release_zmq_ports(42)
    assert freed == [4443, 4444, 4445, 4446]
    assert mock_kill.call_count == 4
    for call in mock_kill.call_args_list:
        assert call.kwargs.get("listeners_only") is True


def test_kill_processes_on_port_skips_self_pid() -> None:
    with patch("emet.utils.port_utils.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=f"{__import__('os').getpid()}\n",
            stderr="",
        )
        assert kill_processes_on_port(59999, listeners_only=False) is False
        # lsof once; no kill invocation for self
        assert mock_run.call_count == 1

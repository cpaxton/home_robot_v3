# Copyright (c) Chris Paxton 2026

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from emet.utils.process_tree import kill_process_tree, popen_session, terminate_process_tree


def test_terminate_process_tree_kills_grandchild(tmp_path: Path) -> None:
    marker = tmp_path / "alive"
    script = (
        "import subprocess, sys, time, pathlib\n"
        f"m = pathlib.Path({str(marker)!r})\n"
        "child = subprocess.Popen([\n"
        "    sys.executable, '-c',\n"
        "    'import pathlib,time; m=pathlib.Path(' + repr(str(m)) + ');\\n'\n"
        "    'while True:\\n'\n"
        "    ' m.write_text(str(time.time())); time.sleep(0.15)'\n"
        "])\n"
        "time.sleep(120)\n"
    )
    proc = popen_session([sys.executable, "-c", script])
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not marker.is_file():
        time.sleep(0.05)
    assert marker.is_file(), "grandchild never started"
    terminate_process_tree(proc, grace_s=2.0)
    assert proc.poll() is not None
    time.sleep(0.5)
    t0 = float(marker.read_text())
    time.sleep(0.5)
    t1 = float(marker.read_text())
    assert t1 == t0, "grandchild survived terminate_process_tree"


def test_kill_process_tree_immediate() -> None:
    proc = popen_session([sys.executable, "-c", "import time; time.sleep(60)"])
    kill_process_tree(proc)
    assert proc.poll() is not None


def test_terminate_noop_when_already_exited() -> None:
    proc = popen_session([sys.executable, "-c", "pass"])
    proc.wait(timeout=5)
    terminate_process_tree(proc, grace_s=1.0)
    kill_process_tree(proc)


def test_process_tree_refuses_unsafe_mock_pid() -> None:
    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = 1

    with patch("emet.utils.process_tree.os.killpg") as mock_killpg:
        with patch("emet.utils.process_tree.os.kill") as mock_kill:
            with pytest.raises(ValueError, match="unsafe process PID/PGID 1"):
                terminate_process_tree(proc)
            with pytest.raises(ValueError, match="unsafe process PID/PGID 1"):
                kill_process_tree(proc)

    mock_killpg.assert_not_called()
    mock_kill.assert_not_called()

# Copyright (c) Chris Paxton 2026

"""Process-group helpers so eval/sim children cannot outlive their parents.

``uv run …`` and nested Python workers often leave GPU orphans when only the
direct child is ``kill()``-ed. Prefer :func:`popen_session` + :func:`terminate_process_tree`.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any


def popen_session(
    cmd: list[str],
    **kwargs: Any,
) -> subprocess.Popen[Any]:
    """``subprocess.Popen`` in a new session so the whole tree can be signaled."""
    kwargs.setdefault("start_new_session", True)
    return subprocess.Popen(cmd, **kwargs)


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def terminate_process_tree(
    proc: subprocess.Popen[Any] | None,
    *,
    grace_s: float = 15.0,
) -> None:
    """SIGTERM the process group, then SIGKILL if still alive after ``grace_s``."""
    if proc is None or proc.poll() is not None:
        return
    pid = int(proc.pid)
    _signal_group(pid, signal.SIGTERM)
    try:
        proc.wait(timeout=float(grace_s))
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_group(pid, signal.SIGKILL)
    try:
        proc.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        pass


def kill_process_tree(proc: subprocess.Popen[Any] | None) -> None:
    """Immediate SIGKILL of the process group (timeouts / hard abort)."""
    if proc is None or proc.poll() is not None:
        return
    _signal_group(int(proc.pid), signal.SIGKILL)
    try:
        proc.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        pass


def wait_briefly(seconds: float = 0.25) -> None:
    """Tiny sleep for OS reaping after kills (tests / scripts)."""
    time.sleep(float(seconds))

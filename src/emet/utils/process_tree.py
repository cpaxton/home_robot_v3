# Copyright (c) Chris Paxton 2026

"""Process-group helpers so eval/sim children cannot outlive their parents.

``uv run …`` and nested Python workers often leave GPU orphans when only the
direct child is ``kill()``-ed. Prefer :func:`popen_session` + :func:`terminate_process_tree`.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import time
from typing import Any

_ACTIVE_PROCESSES: list[subprocess.Popen[Any]] = []
_CLEANUP_HOOKS_INSTALLED = False


def _cleanup_active_processes() -> None:
    for proc in reversed(list(_ACTIVE_PROCESSES)):
        terminate_process_tree(proc, grace_s=5.0)


def _install_cleanup_hooks() -> None:
    global _CLEANUP_HOOKS_INSTALLED
    if _CLEANUP_HOOKS_INSTALLED:
        return
    _CLEANUP_HOOKS_INSTALLED = True
    atexit.register(_cleanup_active_processes)

    def _handler(signum: int, frame: Any) -> None:
        _cleanup_active_processes()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def popen_session(
    cmd: list[str],
    **kwargs: Any,
) -> subprocess.Popen[Any]:
    """``subprocess.Popen`` in a new session so the whole tree can be signaled."""
    kwargs.setdefault("start_new_session", True)
    proc = subprocess.Popen(cmd, **kwargs)
    _ACTIVE_PROCESSES.append(proc)
    _install_cleanup_hooks()
    return proc


def _signal_group(pid: int, sig: int) -> None:
    if pid <= 1:
        raise ValueError(f"refusing to signal unsafe process PID/PGID {pid}")
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
        if proc in _ACTIVE_PROCESSES:
            _ACTIVE_PROCESSES.remove(proc)
        return
    try:
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
    finally:
        if proc in _ACTIVE_PROCESSES:
            _ACTIVE_PROCESSES.remove(proc)


def kill_process_tree(proc: subprocess.Popen[Any] | None) -> None:
    """Immediate SIGKILL of the process group (timeouts / hard abort)."""
    if proc is None or proc.poll() is not None:
        if proc in _ACTIVE_PROCESSES:
            _ACTIVE_PROCESSES.remove(proc)
        return
    try:
        _signal_group(int(proc.pid), signal.SIGKILL)
        try:
            proc.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            pass
    finally:
        if proc in _ACTIVE_PROCESSES:
            _ACTIVE_PROCESSES.remove(proc)


def wait_briefly(seconds: float = 0.25) -> None:
    """Tiny sleep for OS reaping after kills (tests / scripts)."""
    time.sleep(float(seconds))

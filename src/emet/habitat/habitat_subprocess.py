# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Spawn ``emet-habitat serve`` as a subprocess for ``emet run dynagraph --start-habitat``."""

from __future__ import annotations

import atexit
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from emet.simulation.sim_subprocess import wait_for_sim_tcp_port

_HABITAT_PROC: subprocess.Popen[bytes] | None = None
_PREV_SIGINT: Callable[..., Any] | int | None = None
_PREV_SIGTERM: Callable[..., Any] | int | None = None
_SIGNALS_INSTALLED = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def shutdown_habitat_server_subprocess() -> None:
    """Terminate the Habitat serve subprocess (idempotent)."""
    from emet.utils.process_tree import terminate_process_tree

    global _HABITAT_PROC
    proc = _HABITAT_PROC
    if proc is None:
        return
    terminate_process_tree(proc, grace_s=12.0)
    _HABITAT_PROC = None


def _shutdown_habitat_process() -> None:
    shutdown_habitat_server_subprocess()


def _on_signal(signum: int, frame: object | None) -> None:
    shutdown_habitat_server_subprocess()
    if signum == signal.SIGINT:
        prev = _PREV_SIGINT
        if prev is signal.SIG_DFL:
            raise KeyboardInterrupt
        if prev is not None and prev is not signal.SIG_IGN and callable(prev):
            prev(signum, frame)
            return
        raise KeyboardInterrupt
    if signum == signal.SIGTERM:
        prev = _PREV_SIGTERM
        if prev is signal.SIG_DFL:
            raise SystemExit(143)
        if prev is not None and prev is not signal.SIG_IGN and callable(prev):
            prev(signum, frame)
            return
        raise SystemExit(143)


def _ensure_habitat_signal_handlers() -> None:
    global _PREV_SIGINT, _PREV_SIGTERM, _SIGNALS_INSTALLED
    if _SIGNALS_INSTALLED:
        return
    _PREV_SIGINT = signal.signal(signal.SIGINT, _on_signal)
    _PREV_SIGTERM = signal.signal(signal.SIGTERM, _on_signal)
    _SIGNALS_INSTALLED = True


def build_habitat_serve_argv(
    *,
    question_id: int | None = None,
    scene_id: str | None = None,
    floor: int = 0,
    port_offset: int = 0,
    use_hm3d_semantics: bool | None = None,
    hm3d_root: str | None = None,
) -> list[str]:
    argv = ["serve", "--port-offset", str(int(port_offset))]
    if question_id is not None:
        argv.extend(["--question-id", str(int(question_id))])
    if scene_id is not None and str(scene_id).strip():
        argv.extend(["--scene-id", str(scene_id).strip()])
    if floor:
        argv.extend(["--floor", str(int(floor))])
    if use_hm3d_semantics is True:
        argv.append("--use-hm3d-semantics")
    elif use_hm3d_semantics is False:
        argv.append("--no-hm3d-semantics")
    if hm3d_root:
        argv.extend(["--hm3d-root", str(hm3d_root)])
    return argv


def spawn_habitat_server_subprocess(
    *,
    question_id: int | None = None,
    scene_id: str | None = None,
    floor: int = 0,
    port_offset: int = 0,
    use_hm3d_semantics: bool | None = None,
    hm3d_root: str | None = None,
    silence_sim_output: bool = True,
) -> subprocess.Popen[bytes]:
    """Start ``emet-habitat serve`` and wait for the ZMQ send port."""
    from emet.habitat.wrapper_config import build_habitat_wrapper_command, ensure_habitat_eqa_data_dir_env
    from emet.utils.port_utils import get_ports

    global _HABITAT_PROC
    if _HABITAT_PROC is not None and _HABITAT_PROC.poll() is None:
        raise RuntimeError("A Habitat serve subprocess is already running in this process.")

    argv = build_habitat_serve_argv(
        question_id=question_id,
        scene_id=scene_id,
        floor=floor,
        port_offset=port_offset,
        use_hm3d_semantics=use_hm3d_semantics,
        hm3d_root=hm3d_root,
    )
    cmd = build_habitat_wrapper_command(argv)
    if cmd is None:
        raise RuntimeError("Habitat wrapper not found. From the project root run: ./scripts/install_habitat.sh")
    env = dict(__import__("os").environ)
    ensure_habitat_eqa_data_dir_env(env)
    from emet.utils.process_tree import popen_session

    out_err: int | None = subprocess.DEVNULL if silence_sim_output else None
    proc = popen_session(
        cmd,
        cwd=str(_repo_root()),
        stdin=subprocess.DEVNULL,
        stdout=out_err,
        stderr=out_err,
        env=env,
    )
    _HABITAT_PROC = proc
    atexit.register(_shutdown_habitat_process)
    _ensure_habitat_signal_handlers()

    ports = get_ports(int(port_offset))
    try:
        wait_for_sim_tcp_port("127.0.0.1", int(ports.send), proc=proc, timeout_sec=180.0)
    except BaseException:
        shutdown_habitat_server_subprocess()
        raise
    time.sleep(0.5)
    return proc

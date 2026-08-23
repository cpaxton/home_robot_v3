# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Managed out-of-process VL worker for OVMM batches."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from emet.utils.process_tree import popen_session, terminate_process_tree


def allocate_local_vl_port() -> int:
    """Reserve an ephemeral localhost port for the managed VL server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def local_vl_endpoint(port: int) -> str:
    """Return the OpenAI endpoint spec consumed by ``create_dynamem_vllm``."""
    return f"openai@http://127.0.0.1:{int(port)}/v1"


def _health_url(port: int) -> str:
    return f"http://127.0.0.1:{int(port)}/health"


def wait_for_vl_worker(
    port: int,
    proc: subprocess.Popen[Any],
    *,
    timeout_s: float = 300.0,
    poll_s: float = 0.5,
) -> dict[str, Any]:
    """Wait until the local server reports its model ready."""
    deadline = time.monotonic() + float(timeout_s)
    last_error = "worker did not respond"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"managed VL worker exited with code {proc.returncode}: {last_error}")
        try:
            with urllib.request.urlopen(_health_url(port), timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict) and payload.get("ready") is True:
                return payload
            last_error = str(payload)
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(float(poll_s))
    raise TimeoutError(f"managed VL worker was not ready after {timeout_s:g}s: {last_error}")


@dataclass
class ManagedVLWorker:
    """Start and stop one local OpenAI-compatible VL server."""

    port: int | None = None
    timeout_s: float = 300.0
    log_path: str | None = None

    _proc: subprocess.Popen[Any] | None = None
    _endpoint: str | None = None
    health: dict[str, Any] | None = None

    @property
    def endpoint(self) -> str:
        if self._endpoint is None:
            raise RuntimeError("managed VL worker has not been started")
        return self._endpoint

    @property
    def process(self) -> subprocess.Popen[Any] | None:
        return self._proc

    def start(self) -> str:
        if self._proc is not None and self._proc.poll() is None:
            return self.endpoint
        port = int(self.port or allocate_local_vl_port())
        self.port = port
        cmd = [
            sys.executable,
            "-m",
            "emet.cli",
            "serve",
            "llm",
            "--vl",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        log_handle = None
        if self.log_path:
            path = os.path.abspath(os.path.expanduser(self.log_path))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            log_handle = open(path, "a", encoding="utf-8")
        try:
            worker_env = {
                **os.environ,
                "EMET_ALLOW_SDPA_ATTN": "1",
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "4"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "4"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "4"),
                "TOKENIZERS_PARALLELISM": "false",
            }
            self._proc = popen_session(
                cmd,
                stdout=log_handle or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                text=True,
                env=worker_env,
            )
            self._endpoint = local_vl_endpoint(port)
            self.health = wait_for_vl_worker(port, self._proc, timeout_s=self.timeout_s)
            return self.endpoint
        except BaseException:
            terminate_process_tree(self._proc)
            self._proc = None
            if log_handle is not None:
                log_handle.close()
            raise
        finally:
            if log_handle is not None:
                log_handle.close()

    def stop(self) -> None:
        terminate_process_tree(self._proc)
        self._proc = None

    def __enter__(self) -> ManagedVLWorker:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

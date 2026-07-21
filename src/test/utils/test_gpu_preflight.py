# Copyright (c) Chris Paxton 2026

from __future__ import annotations

import os

from emet.utils import gpu_preflight as gp


def test_check_gpu_memory_ok(monkeypatch):
    monkeypatch.setattr(
        gp,
        "gpu_memory_info",
        lambda: gp.GpuMemoryInfo(free_mib=16000, total_mib=24564),
    )
    ok, msg = gp.check_gpu_memory(12000)
    assert ok is True
    assert "16000" in msg


def test_check_gpu_memory_low(monkeypatch):
    monkeypatch.setattr(
        gp,
        "gpu_memory_info",
        lambda: gp.GpuMemoryInfo(free_mib=1000, total_mib=24564),
    )
    ok, msg = gp.check_gpu_memory(12000)
    assert ok is False
    assert "insufficient" in msg


def test_wait_gpu_stable_succeeds(monkeypatch):
    monkeypatch.setattr(gp, "gpu_free_mib", lambda: 15000)
    sleeps: list[float] = []
    logs: list[str] = []
    assert (
        gp.wait_gpu_stable(
            12000,
            stable_checks=2,
            interval_s=0.01,
            log=logs.append,
            sleep_fn=sleeps.append,
        )
        is True
    )
    assert len(logs) >= 2


def test_protected_pids_includes_self():
    assert os.getpid() in gp.protected_pids()


def test_kill_matching_pids_skips_protected(monkeypatch):
    signaled: list[tuple[int, int]] = []

    monkeypatch.setattr(gp, "_process_args", lambda pid: f"fake-{pid}")
    monkeypatch.setattr(
        gp.os,
        "kill",
        lambda pid, sig: signaled.append((pid, int(sig))),
    )
    monkeypatch.setattr(gp, "_pid_alive", lambda _pid: False)

    n = gp.kill_matching_pids(
        [111, 222],
        protected={111},
        log=lambda _m: None,
        escalate_s=0.0,
    )
    assert n == 1
    assert all(pid == 222 for pid, _ in signaled)


def test_kill_stale_uses_patterns(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(gp, "_pgrep_f", lambda pattern: calls.append(pattern) or [999001])
    monkeypatch.setattr(
        gp,
        "kill_matching_pids",
        lambda pids, **kw: len(pids),
    )
    monkeypatch.setattr(gp, "list_compute_apps", lambda: [])
    n = gp.kill_stale_eval_processes(
        kill_gpu_apps=False,
        settle_s=0.0,
        escalate_s=0.0,
        sleep_fn=lambda _s: None,
        log=lambda _m: None,
        protected={1},
    )
    assert n >= 1
    assert any("mujoco_server" in p for p in calls)


def test_format_status_lines(monkeypatch):
    monkeypatch.setattr(
        gp,
        "gpu_memory_info",
        lambda: gp.GpuMemoryInfo(free_mib=8000, total_mib=24000),
    )
    monkeypatch.setattr(gp, "list_compute_apps", lambda: [])
    lines = gp.format_status_lines()
    assert lines[0].startswith("GPU:")
    assert "8000" in lines[0]

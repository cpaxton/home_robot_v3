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


def test_habitat_egl_error_in_text():
    assert gp.habitat_egl_error_in_text(
        "Platform::WindowlessEglApplication::tryCreateContext(): "
        "unable to find CUDA device 0 among 2 EGL devices in total"
    )
    assert gp.habitat_egl_error_in_text(
        "WindowlessContext: Unable to create windowless context"
    )
    assert not gp.habitat_egl_error_in_text("episode finished correctly")


def test_diagnose_flags_empty_cuda_visible(monkeypatch, tmp_path):
    hab = tmp_path / ".venv-habitat" / "bin" / "emet-habitat"
    hab.parent.mkdir(parents=True)
    hab.write_text("#!/bin/sh\n")
    hab.chmod(0o755)
    monkeypatch.setattr(
        gp,
        "gpu_memory_info",
        lambda: gp.GpuMemoryInfo(free_mib=20000, total_mib=24564),
    )
    monkeypatch.setattr(gp, "list_compute_apps", lambda: [])
    monkeypatch.setattr(gp, "recent_emet_segfault_hint", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    ok, lines = gp.diagnose_eval_environment(repo_root=str(tmp_path))
    assert ok is False
    assert any("CUDA_VISIBLE_DEVICES is empty" in ln for ln in lines)
    assert any("empty nvidia-smi compute apps" in ln for ln in lines)

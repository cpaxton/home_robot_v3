# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tail-able status log helper (scripts/status_log.sh).

Recovery after an agent/session death depends on the *last* records of
``~/runs/emet/STATUS.log`` being self-contained and ending in an actionable
``next:`` line, so these tests read the tail the way a human would.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER = REPO_ROOT / "scripts" / "status_log.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def run_snippet(snippet: str, tmp_path: Path) -> tuple[str, str]:
    """Run bash with the helper sourced; return (global log, per-run log)."""
    status_log = tmp_path / "runs" / "STATUS.log"
    out_dir = tmp_path / "run_out"
    script = f"set -euo pipefail\nsource {HELPER}\n{snippet}\n"
    subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "EMET_STATUS_LOG": str(status_log),
            "OUT": str(out_dir),
        },
    )
    per_run = out_dir / "STATUS.log"
    return (
        status_log.read_text() if status_log.exists() else "",
        per_run.read_text() if per_run.exists() else "",
    )


def test_note_record_is_self_contained(tmp_path: Path) -> None:
    global_log, per_run_log = run_snippet(
        'status_open "$OUT" hmeqa-test\n'
        'STATUS_PROGRESS="6/64 classic q14"\n'
        'status_note RUNNING "episode classic q14 started" "wait — job alive"\n'
        'status_close DONE "finished" "review $OUT"\n',
        tmp_path,
    )
    assert global_log == per_run_log
    running = [line for line in global_log.splitlines() if "RUNNING" in line]
    assert running and "hmeqa-test" in running[0] and "6/64 classic q14" in running[0]
    assert "    what: episode classic q14 started" in global_log
    assert "    next: wait — job alive" in global_log


def test_tail_ends_with_next_action(tmp_path: Path) -> None:
    global_log, _ = run_snippet(
        'status_open "$OUT" hmeqa-test\n'
        'status_note RUNNING "episode started" "wait"\n'
        'status_close CRASH "SIGSEGV in classic q14" "read the crash capsule"\n',
        tmp_path,
    )
    tail = global_log.splitlines()[-5:]
    assert tail[-1].startswith("    next: read the crash capsule")
    assert any("CRASH" in line for line in tail)
    assert any(line.startswith("    out:") for line in tail)


def test_unexpected_exit_records_resume_command(tmp_path: Path) -> None:
    global_log, _ = run_snippet(
        'status_open "$OUT" hmeqa-test\n'
        'STATUS_RESUME_CMD="RESUME=1 ./scripts/run_hmeqa_agentic_h2h.sh $OUT"\n'
        'status_note RUNNING "episode started" "wait"\n'
        "exit 139\n",
        tmp_path,
    )
    assert "EXIT" in global_log
    assert "rc=139" in global_log
    assert "RESUME=1 ./scripts/run_hmeqa_agentic_h2h.sh" in global_log


def test_sigterm_records_exit(tmp_path: Path) -> None:
    """`emet jobs cancel` sends SIGTERM; the run must still leave a recovery record."""
    status_log = tmp_path / "runs" / "STATUS.log"
    out_dir = tmp_path / "run_out"
    script = (
        f"set -euo pipefail\nsource {HELPER}\n"
        f'status_open "{out_dir}" hmeqa-test\n'
        'STATUS_RESUME_CMD="RESUME=1 ./scripts/my_eval.sh"\n'
        "sleep 30\n"
    )
    proc = subprocess.Popen(
        ["bash", "-c", script],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "EMET_STATUS_LOG": str(status_log)},
        start_new_session=True,
    )
    deadline = time.time() + 10.0
    while not status_log.parent.exists() and time.time() < deadline:
        time.sleep(0.05)
    # Signal the whole tree, as terminate_process_tree / `emet jobs cancel` do:
    # bash defers traps while a foreground child (the episode) is running.
    os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=10)

    text = status_log.read_text()
    assert "EXIT" in text
    assert "killed by SIGTERM" in text
    assert "RESUME=1 ./scripts/my_eval.sh" in text


def test_close_disarms_exit_trap(tmp_path: Path) -> None:
    global_log, _ = run_snippet(
        'status_open "$OUT" hmeqa-test\nstatus_close DONE "all units finished" "review results"\n',
        tmp_path,
    )
    assert "EXIT" not in global_log
    assert global_log.count("===") == 1


def test_latest_symlink_points_at_run_dir(tmp_path: Path) -> None:
    run_snippet('status_open "$OUT" hmeqa-test\nstatus_close DONE "done" "none"\n', tmp_path)
    latest = tmp_path / "runs" / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == (tmp_path / "run_out").resolve()


def test_job_id_line_present_when_set(tmp_path: Path) -> None:
    global_log, _ = run_snippet(
        'export EMET_JOB_ID=20260725_101522_3b3b11\nstatus_open "$OUT" hmeqa-test\nstatus_close DONE "done" "none"\n',
        tmp_path,
    )
    assert "uv run emet jobs status 20260725_101522_3b3b11" in global_log

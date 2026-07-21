# Copyright (c) Chris Paxton 2026

from __future__ import annotations

import os

from emet.utils import job_registry as jr


def test_register_list_update_cancel(tmp_path, monkeypatch):
    monkeypatch.setenv("EMET_JOBS_DIR", str(tmp_path / "jobs"))
    job = jr.register_job(
        name="unit-test-job",
        cmd="echo hi",
        out_dir=tmp_path / "out",
        status="queued",
        wait_pids=[1],
    )
    assert job.id
    assert (tmp_path / "jobs" / f"{job.id}.json").is_file()

    active = jr.list_jobs(include_terminal=False)
    assert any(j.id == job.id for j in active)

    updated = jr.update_job(job.id, status="running", pid=os.getpid())
    assert updated.status == "running"
    assert updated.pid == os.getpid()

    # cancel must not kill the pytest process (pid == self); still marks cancelled
    cancelled = jr.cancel_job(job.id, grace_s=0.1)
    assert cancelled.status == "cancelled"
    assert os.getpid() > 0  # still alive

    active2 = jr.list_jobs(include_terminal=False)
    assert not any(j.id == job.id for j in active2)
    all_jobs = jr.list_jobs(include_terminal=True)
    assert any(j.id == job.id and j.status == "cancelled" for j in all_jobs)


def test_refresh_marks_dead_pid_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("EMET_JOBS_DIR", str(tmp_path / "jobs"))
    job = jr.register_job(name="dead", status="running", pid=999999991)
    refreshed = jr.refresh_job_liveness(jr.load_job(job.id))  # type: ignore[arg-type]
    assert refreshed is not None
    assert refreshed.status == "failed"


def test_format_job_row_columns():
    job = jr.JobRecord(
        id="20260721_140000_abc123",
        name="demo",
        status="queued",
        pid=None,
        out_dir="/tmp/out",
    )
    header = jr.format_job_header()
    row = jr.format_job_row(job)
    assert "STATUS" in header and "PID" in header
    assert "queued" in row
    assert "demo" in row
    # PID column should be a bare dash, not "pid=-"
    assert "pid=" not in row
    assert "     -" in row or row.split()[2] == "-"


def test_summarize_eval_cmd_extracts_script_and_out():
    cmd = (
        "python scripts/eval_dynamic_exploration.py --out-dir "
        "/home/cpaxton/runs/emet/dynagraph_fix_verify_20260721_114903/eqa_smoke "
        "--port-offset-base 220"
    )
    script, out = jr.summarize_eval_cmd(cmd)
    assert script == "eval_dynamic_exploration.py"
    assert "eqa_smoke" in out
    assert "port-offset" not in out

    detail = jr.format_job_detail(jr.JobRecord(id="x", name="n", status="running", pid=1))
    assert "id:        x" in detail
    assert "status:    running" in detail

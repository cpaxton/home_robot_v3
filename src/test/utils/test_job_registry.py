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
    assert "STATUS" in header and "PID" in header and "PROGRESS" in header
    assert "queued" in row
    assert "demo" in row
    # PID column should be a bare dash, not "pid=-"
    assert "pid=" not in row
    assert "     -" in row or row.split()[2] == "-"
    # No progress yet → placeholder dash in PROGRESS column
    assert " - " in row or row.count("-") >= 2


def test_compute_job_progress_eta_from_meta():
    now = 1_000_000.0
    job = jr.JobRecord(
        id="j1",
        name="h2h",
        status="running",
        created_at=now - 800.0,  # 800s elapsed, 8 units → 100s/unit
        meta={"units_done": 8, "units_total": 64, "phase": "classic", "current_id": "17"},
    )
    prog = jr.compute_job_progress(job, now=now)
    assert prog.units_done == 8
    assert prog.units_total == 64
    assert prog.phase == "classic"
    assert prog.current_id == "17"
    assert prog.rate_s_per_unit == 100.0
    assert prog.eta_s == 5600.0  # 56 remaining * 100s
    assert prog.source == "meta"
    brief = jr.format_progress_brief(prog)
    assert "8/64" in brief
    assert "classic" in brief
    assert "q17" in brief
    assert "ETA" in brief


def test_progress_file_overlays_meta(tmp_path, monkeypatch):
    monkeypatch.setenv("EMET_JOBS_DIR", str(tmp_path / "jobs"))
    out = tmp_path / "run"
    job = jr.JobRecord(
        id="j2",
        name="h2h",
        status="running",
        out_dir=str(out),
        created_at=1_000_000.0 - 100.0,
        meta={"units_done": 1, "units_total": 10, "phase": "classic"},
    )
    jr.write_progress_file(out, units_done=4, units_total=10, phase="agentic", current_id="21")
    prog = jr.compute_job_progress(job, now=1_000_000.0)
    assert prog.units_done == 4
    assert prog.phase == "agentic"
    assert prog.current_id == "21"
    assert prog.source == "meta+file"

    registered = jr.register_job(name="x", out_dir=out, status="running")
    updated = jr.update_job(
        registered.id,
        units_done=5,
        units_total=10,
        phase="agentic",
        current_id="22",
    )
    assert updated.meta["units_done"] == 5
    disk = jr.read_progress_file(out)
    assert disk["units_done"] == 5
    assert disk["current_id"] == "22"

    detail = jr.format_job_detail(
        jr.JobRecord(
            id="x",
            name="n",
            status="running",
            pid=1,
            out_dir=str(out),
            meta={"units_done": 5, "units_total": 10, "phase": "agentic"},
            created_at=1_000_000.0 - 50.0,
        )
    )
    assert "progress:" in detail
    assert "5/10" in detail


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

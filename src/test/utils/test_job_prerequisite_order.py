# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Queued dependencies must finish before their GPU resource is acquired."""

from click.testing import CliRunner

from emet.cli_cmds import jobs


def test_wait_pid_precedes_gpu_lock_and_gpu_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_project_root", lambda: tmp_path)
    # Inspect the generated supervisor without launching a process or acquiring
    # the host lock. The production wrapper still performs all resource checks.
    monkeypatch.setattr(jobs.subprocess, "call", lambda *args, **kwargs: 0)
    result = CliRunner().invoke(
        jobs.jobs_group,
        [
            "run",
            "--name",
            "dependency-test",
            "--out-dir",
            str(tmp_path / "run"),
            "--wait-pid",
            "12345",
            "--need-mib",
            "12000",
            "--cpu-safe",
            "--gpu-exclusive",
            "--foreground",
            "--",
            "true",
        ],
    )
    assert result.exit_code == 0, result.output
    wrapper = (tmp_path / "run/job_wrapper.sh").read_text()
    assert wrapper.index("while pid_is_running") < wrapper.index("flock -w")
    assert wrapper.index("flock -w") < wrapper.index("eval wait")
    assert wrapper.index("eval wait") < wrapper.index("--status running")

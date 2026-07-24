# Copyright (c) Chris Paxton 2026

"""Lightweight local job registry for paper evals / overnight smokes.

Jobs are JSON files under ``~/runs/emet/jobs/`` (override with ``EMET_JOBS_DIR``).
Queue scripts call :func:`register_job` / :func:`update_job`; ``emet jobs`` lists
and cancels them. Unmanaged eval processes can still be discovered via
:func:`scan_eval_processes`.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

JobStatus = Literal["queued", "waiting", "running", "done", "failed", "cancelled"]

TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "failed", "cancelled"})

EVAL_SCAN_PATTERNS: tuple[str, ...] = (
    r"eval_dynamic_exploration\.py",
    r"queue_dynagraph_eqa_evals\.sh",
    r"run_dynagraph_dynamic_",
    r"run_overnight_",
    r"run_simulation_smoke_battery",
    r"emet\.app\.run_dynagraph",
    r"emet run (dynagraph|graph-eqa|dynamem)",
    r"emet\.simulation\.mujoco_server",
    r"emet-habitat",
)


def jobs_dir() -> Path:
    raw = os.environ.get("EMET_JOBS_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / "runs" / "emet" / "jobs").resolve()


def _now() -> float:
    return time.time()


def _new_id() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:6]}"


@dataclass
class JobRecord:
    id: str
    name: str
    status: JobStatus = "queued"
    pid: int | None = None
    pgid: int | None = None
    cmd: str = ""
    out_dir: str | None = None
    log_path: str | None = None
    repo: str | None = None
    wait_pids: list[int] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobRecord:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs.setdefault("wait_pids", [])
        kwargs.setdefault("meta", {})
        return cls(**kwargs)


def _job_path(job_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id)
    return jobs_dir() / f"{safe}.json"


def ensure_jobs_dir() -> Path:
    d = jobs_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_job(job: JobRecord) -> Path:
    ensure_jobs_dir()
    job.updated_at = _now()
    path = _job_path(job.id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(job.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def load_job(job_id: str) -> JobRecord | None:
    path = _job_path(job_id)
    if not path.is_file():
        return None
    try:
        return JobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def list_jobs(*, include_terminal: bool = False) -> list[JobRecord]:
    ensure_jobs_dir()
    out: list[JobRecord] = []
    for path in sorted(jobs_dir().glob("*.json")):
        try:
            job = JobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not include_terminal and job.status in TERMINAL_STATUSES:
            # Keep recently finished for a short window in list? Prefer --all for terminal.
            continue
        out.append(refresh_job_liveness(job))
    out.sort(key=lambda j: j.created_at)
    return out


def register_job(
    *,
    name: str,
    cmd: str = "",
    out_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    repo: str | Path | None = None,
    wait_pids: list[int] | None = None,
    pid: int | None = None,
    status: JobStatus = "queued",
    meta: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> JobRecord:
    job = JobRecord(
        id=job_id or _new_id(),
        name=str(name),
        status=status,
        pid=int(pid) if pid is not None else None,
        pgid=_pgid_of(int(pid)) if pid is not None else None,
        cmd=str(cmd or ""),
        out_dir=str(out_dir) if out_dir is not None else None,
        log_path=str(log_path) if log_path is not None else None,
        repo=str(repo) if repo is not None else None,
        wait_pids=[int(p) for p in (wait_pids or [])],
        meta=dict(meta or {}),
    )
    save_job(job)
    return job


def update_job(
    job_id: str,
    *,
    status: JobStatus | None = None,
    pid: int | None = None,
    cmd: str | None = None,
    out_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    error: str | None = None,
    meta_update: dict[str, Any] | None = None,
) -> JobRecord:
    job = load_job(job_id)
    if job is None:
        raise KeyError(f"unknown job id: {job_id}")
    if status is not None:
        job.status = status
    if pid is not None:
        job.pid = int(pid)
        job.pgid = _pgid_of(int(pid))
    if cmd is not None:
        job.cmd = str(cmd)
    if out_dir is not None:
        job.out_dir = str(out_dir)
    if log_path is not None:
        job.log_path = str(log_path)
    if error is not None:
        job.error = str(error)
    if meta_update:
        job.meta.update(meta_update)
    save_job(job)
    return job


def _pgid_of(pid: int) -> int | None:
    try:
        return int(os.getpgid(pid))
    except (ProcessLookupError, PermissionError, OSError):
        return None


def pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def refresh_job_liveness(job: JobRecord) -> JobRecord:
    """Mark running/waiting jobs failed if their PID disappeared without DONE."""
    if job.status in TERMINAL_STATUSES:
        return job
    if job.pid is not None and not pid_alive(job.pid):
        if job.status in ("running", "waiting", "queued"):
            # If out_dir has DONE, treat as done.
            if job.out_dir and (Path(job.out_dir) / "DONE").is_file():
                job.status = "done"
            else:
                job.status = "failed"
                job.error = job.error or f"pid {job.pid} exited without DONE"
            save_job(job)
    return job


def cancel_job(job_id: str, *, grace_s: float = 10.0) -> JobRecord:
    from emet.utils.process_tree import terminate_process_tree

    job = load_job(job_id)
    if job is None:
        raise KeyError(f"unknown job id: {job_id}")
    if job.status in TERMINAL_STATUSES:
        return job
    pid = job.pid
    self_pid = os.getpid()
    if pid is not None and pid_alive(pid) and pid != self_pid:
        # Never signal our own process (tests / mistaken register).
        class _Proc:
            def __init__(self, p: int) -> None:
                self.pid = p

            def poll(self) -> int | None:
                return None if pid_alive(self.pid) else 0

            def wait(self, timeout: float | None = None) -> int:
                deadline = time.monotonic() + (timeout if timeout is not None else 30.0)
                while time.monotonic() < deadline:
                    if not pid_alive(self.pid):
                        return 0
                    time.sleep(0.2)
                raise subprocess.TimeoutExpired(["job", str(self.pid)], timeout or 30.0)

        terminate_process_tree(_Proc(pid), grace_s=grace_s)
        pgid = job.pgid or _pgid_of(pid)
        if pgid is not None and pgid not in (pid, self_pid):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
    job.status = "cancelled"
    job.error = job.error or "cancelled via emet jobs cancel"
    save_job(job)
    return job


@dataclass(frozen=True)
class ScannedProcess:
    pid: int
    cmd: str
    matched_pattern: str


def scan_eval_processes(
    patterns: tuple[str, ...] = EVAL_SCAN_PATTERNS,
) -> list[ScannedProcess]:
    """Discover eval-related processes (may include registered job PIDs)."""
    found: list[ScannedProcess] = []
    seen: set[int] = set()
    for pattern in patterns:
        try:
            proc = subprocess.run(
                ["pgrep", "-af", pattern],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if not parts:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            if pid in seen or pid == os.getpid():
                continue
            cmd = parts[1] if len(parts) > 1 else ""
            if "emet jobs" in cmd:
                continue
            seen.add(pid)
            found.append(ScannedProcess(pid=pid, cmd=cmd, matched_pattern=pattern))
    found.sort(key=lambda s: s.pid)
    return found


def _ellipsize(text: str, width: int) -> str:
    if width <= 1 or len(text) <= width:
        return text
    if width <= 2:
        return text[:width]
    # Prefer keeping the useful tail (paths / flags) visible.
    return "…" + text[-(width - 1) :]


def _format_age(created_at: float, *, now: float | None = None) -> str:
    age_m = max(0.0, ((now if now is not None else _now()) - created_at) / 60.0)
    if age_m < 60:
        return f"{age_m:.0f}m"
    return f"{age_m / 60.0:.1f}h"


def format_job_header() -> str:
    return f"{'ID':<26}  {'STATUS':<10}  {'PID':>8}  {'AGE':>6}  {'NAME':<22}  OUT"


def format_job_row(job: JobRecord) -> str:
    pid_s = "-" if job.pid is None else str(job.pid)
    name = _ellipsize(job.name, 22)
    out = _ellipsize(job.out_dir or "-", 52)
    return (
        f"{job.id:<26}  {job.status:<10}  {pid_s:>8}  {_format_age(job.created_at):>6}  "
        f"{name:<22}  {out}"
    )


def format_job_detail(job: JobRecord) -> str:
    """Human-readable multi-line status (not JSON)."""
    lines = [
        f"id:        {job.id}",
        f"name:      {job.name}",
        f"status:    {job.status}",
        f"pid:       {job.pid if job.pid is not None else '-'}",
        f"age:       {_format_age(job.created_at)}",
        f"out_dir:   {job.out_dir or '-'}",
        f"log_path:  {job.log_path or '-'}",
    ]
    if job.wait_pids:
        lines.append(f"wait_pids: {', '.join(str(p) for p in job.wait_pids)}")
    if job.error:
        lines.append(f"error:     {job.error}")
    if job.cmd:
        lines.append(f"cmd:       {_ellipsize(job.cmd, 120)}")
    if job.repo:
        lines.append(f"repo:      {job.repo}")
    return "\n".join(lines)


_SCRIPT_RE = re.compile(
    r"(?:^|[\s/])("
    r"eval_dynamic_exploration\.py|"
    r"queue_dynagraph_eqa_evals\.sh|"
    r"run_dynagraph_dynamic_[^\s]+|"
    r"run_overnight_[^\s]+|"
    r"run_simulation_smoke_battery[^\s]*|"
    r"emet\.app\.run_dynagraph|"
    r"emet\.simulation\.mujoco_server|"
    r"emet-habitat|"
    r"job_wrapper\.sh"
    r")\b"
)
_OUT_FLAG_RE = re.compile(
    r"(?:--(?:out[-_]?dir|result[-_]?dir|output[-_]?dir)|-O)\s+(\S+)",
    re.IGNORECASE,
)
_RUNS_PATH_RE = re.compile(r"(/?(?:home/[^/\s]+/)?runs/emet/\S+)")


def summarize_eval_cmd(cmd: str) -> tuple[str, str]:
    """Return ``(script, out_hint)`` for compact unmanaged-process rows."""
    script = "-"
    m = _SCRIPT_RE.search(cmd)
    if m:
        script = Path(m.group(1)).name
    else:
        parts = cmd.split()
        for i, tok in enumerate(parts):
            base = Path(tok).name
            if base == "emet" and i + 1 < len(parts) and parts[i + 1] == "run":
                rest = parts[i + 2 : i + 4]
                script = ("emet run " + " ".join(rest)).strip() or "emet run"
                break
    out = "-"
    om = _OUT_FLAG_RE.search(cmd)
    if om:
        out = om.group(1).rstrip("\"'")
    else:
        pm = _RUNS_PATH_RE.search(cmd)
        if pm:
            out = pm.group(1).rstrip("\"',")
    # Shorten home prefix for display
    home = str(Path.home())
    if out.startswith(home + "/"):
        out = "~" + out[len(home) :]
    return script, out


def format_scanned_header() -> str:
    return f"  {'PID':>8}  {'SCRIPT':<36}  OUT"


def format_scanned_row(proc: ScannedProcess, *, out_width: int = 72) -> str:
    script, out = summarize_eval_cmd(proc.cmd)
    return f"  {proc.pid:>8}  {_ellipsize(script, 36):<36}  {_ellipsize(out, out_width)}"

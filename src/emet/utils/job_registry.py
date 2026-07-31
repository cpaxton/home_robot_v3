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
from collections import Counter
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
    # Human “what / why” for queue visibility (``emet jobs`` list + status).
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobRecord:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs.setdefault("wait_pids", [])
        kwargs.setdefault("meta", {})
        kwargs.setdefault("description", "")
        # Legacy: some scripts stashed a note under meta.
        if not str(kwargs.get("description") or "").strip():
            meta = kwargs.get("meta") or {}
            if isinstance(meta, dict):
                for key in ("description", "note", "why"):
                    val = meta.get(key)
                    if isinstance(val, str) and val.strip():
                        kwargs["description"] = val.strip()
                        break
        return cls(**kwargs)


def job_description(job: JobRecord) -> str:
    """Resolved description for display (field, else legacy meta note)."""
    text = str(job.description or "").strip()
    if text:
        return text
    meta = job.meta or {}
    for key in ("description", "note", "why"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


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


_GPU_JOB_HINT = re.compile(
    r"emet-habitat|run_hmeqa|hmeqa|habitat|qwen|vlm|mujoco|robocasa|dynagraph|"
    r"dynamem|sqa3d|ovmm|need.mib|--need-mib|EMET_ALLOW_SDPA",
    re.IGNORECASE,
)


def looks_like_gpu_job(job: JobRecord) -> bool:
    """Heuristic: Habitat/VLM/MuJoCo/HM-EQA-style commands share one GPU."""
    blob = f"{job.name} {job.cmd}"
    return bool(_GPU_JOB_HINT.search(blob))


def active_gpu_job_pids(*, exclude_job_id: str | None = None) -> list[int]:
    """PIDs of non-terminal GPU-like jobs (for ``--gpu-exclusive`` waits)."""
    pids: list[int] = []
    for job in list_jobs(include_terminal=False):
        if exclude_job_id and job.id == exclude_job_id:
            continue
        if job.status not in ("queued", "waiting", "running"):
            continue
        if not looks_like_gpu_job(job):
            continue
        if job.pid is not None and pid_alive(job.pid):
            pids.append(int(job.pid))
    return pids


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
    description: str | None = None,
) -> JobRecord:
    desc = str(description or "").strip()
    meta_dict = dict(meta or {})
    if desc and "description" not in meta_dict:
        meta_dict.setdefault("note", desc)
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
        meta=meta_dict,
        description=desc,
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
    units_done: int | None = None,
    units_total: int | None = None,
    phase: str | None = None,
    current_id: str | None = None,
    write_progress_json: bool = True,
    description: str | None = None,
) -> JobRecord:
    job = load_job(job_id)
    if job is None:
        raise KeyError(f"unknown job id: {job_id}")
    if description is not None:
        job.description = str(description).strip()
        if job.description:
            job.meta["note"] = job.description
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
    progress_meta: dict[str, Any] = {}
    if units_done is not None:
        progress_meta["units_done"] = int(units_done)
    if units_total is not None:
        progress_meta["units_total"] = int(units_total)
    if phase is not None:
        progress_meta["phase"] = str(phase)
    if current_id is not None:
        progress_meta["current_id"] = str(current_id)
    if meta_update:
        job.meta.update(meta_update)
    if progress_meta:
        job.meta.update(progress_meta)
    save_job(job)
    if write_progress_json and job.out_dir and progress_meta:
        write_progress_file(job.out_dir, **progress_meta)
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


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60.0:.0f}m"
    return f"{seconds / 3600.0:.1f}h"


@dataclass(frozen=True)
class JobProgress:
    """Derived progress / ETA for a job (from meta and/or out_dir/progress.json)."""

    units_done: int | None = None
    units_total: int | None = None
    phase: str | None = None
    current_id: str | None = None
    elapsed_s: float = 0.0
    rate_s_per_unit: float | None = None
    eta_s: float | None = None
    source: str = "none"

    @property
    def fraction(self) -> float | None:
        if self.units_done is None or self.units_total is None or self.units_total <= 0:
            return None
        return min(1.0, max(0.0, float(self.units_done) / float(self.units_total)))


def progress_json_path(out_dir: str | Path | None) -> Path | None:
    if not out_dir:
        return None
    return Path(out_dir).expanduser() / "progress.json"


def read_progress_file(out_dir: str | Path | None) -> dict[str, Any]:
    path = progress_json_path(out_dir)
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_progress_file(out_dir: str | Path, **fields: Any) -> Path:
    """Atomically write ``out_dir/progress.json`` (merge with existing keys)."""
    root = Path(out_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "progress.json"
    cur = read_progress_file(root)
    cur.update({k: v for k, v in fields.items() if v is not None})
    cur["updated_at"] = _now()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cur, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compute_job_progress(job: JobRecord, *, now: float | None = None) -> JobProgress:
    """Merge ``job.meta`` progress with optional ``out_dir/progress.json`` (file wins on conflict)."""
    now_t = now if now is not None else _now()
    file_prog = read_progress_file(job.out_dir)
    meta = dict(job.meta or {})
    # File overlays meta so heartbeats on disk work even without jobs update.
    merged: dict[str, Any] = {**meta, **file_prog}
    units_done = _coerce_int(merged.get("units_done"))
    units_total = _coerce_int(merged.get("units_total"))
    phase = merged.get("phase")
    phase_s = str(phase).strip() if phase is not None and str(phase).strip() else None
    current = merged.get("current_id")
    current_s = str(current).strip() if current is not None and str(current).strip() else None
    elapsed = max(0.0, now_t - float(job.created_at))
    rate: float | None = None
    eta: float | None = None
    if units_done is not None and units_done > 0 and elapsed > 0:
        rate = elapsed / float(units_done)
        if units_total is not None and units_total >= units_done:
            eta = rate * float(units_total - units_done)
    source = "none"
    if file_prog and meta:
        source = "meta+file"
    elif file_prog:
        source = "file"
    elif any(k in meta for k in ("units_done", "units_total", "phase", "current_id")):
        source = "meta"
    return JobProgress(
        units_done=units_done,
        units_total=units_total,
        phase=phase_s,
        current_id=current_s,
        elapsed_s=elapsed,
        rate_s_per_unit=rate,
        eta_s=eta,
        source=source,
    )


def format_progress_brief(prog: JobProgress) -> str:
    """Compact progress for list rows, e.g. ``8/64 classic q17 ~2.4h``."""
    if prog.units_done is None and prog.units_total is None and not prog.phase:
        return "-"
    parts: list[str] = []
    if prog.units_done is not None and prog.units_total is not None:
        parts.append(f"{prog.units_done}/{prog.units_total}")
    elif prog.units_done is not None:
        parts.append(f"{prog.units_done}/?")
    if prog.phase:
        parts.append(str(prog.phase))
    if prog.current_id:
        parts.append(f"q{prog.current_id}" if str(prog.current_id).isdigit() else str(prog.current_id))
    if prog.eta_s is not None and prog.units_total is not None:
        parts.append(f"ETA {_format_duration(prog.eta_s)}")
    elif prog.rate_s_per_unit is not None:
        parts.append(f"{_format_duration(prog.rate_s_per_unit)}/u")
    return " ".join(parts) if parts else "-"


def format_job_header() -> str:
    return f"{'ID':<26}  {'STATUS':<10}  {'PID':>8}  {'AGE':>6}  {'NAME':<18}  {'PROGRESS':<28}  OUT"


def format_job_row(job: JobRecord) -> str:
    pid_s = "-" if job.pid is None else str(job.pid)
    name = _ellipsize(job.name, 18)
    out = _ellipsize(job.out_dir or "-", 40)
    prog = format_progress_brief(compute_job_progress(job))
    prog_s = _ellipsize(prog, 28)
    line = (
        f"{job.id:<26}  {job.status:<10}  {pid_s:>8}  {_format_age(job.created_at):>6}  {name:<18}  {prog_s:<28}  {out}"
    )
    desc = job_description(job)
    if desc:
        line = f"{line}\n  why: {_ellipsize(desc, 96)}"
    return line


@dataclass
class VizArtifactDir:
    """One image-bearing directory under a job OUT (for status/report)."""

    rel: str
    n_png: int
    kind: str  # frontier_picks | maps | figures | topdown | other


def discover_out_viz_artifacts(out_dir: str | Path | None) -> list[VizArtifactDir]:
    """Find PNG dirs under an H2H / eval OUT (bundles, figures, frontier_picks)."""
    if not out_dir:
        return []
    root = Path(out_dir).expanduser()
    if not root.is_dir():
        return []
    found: list[VizArtifactDir] = []

    def _add(path: Path, kind: str) -> None:
        if not path.is_dir():
            return
        n = sum(1 for _ in path.glob("*.png"))
        # Also count top-level topdown PNGs living beside maps/.
        if kind == "topdown":
            n = sum(1 for p in path.glob("topdown*.png"))
            if n <= 0:
                return
            try:
                rel = str(path.relative_to(root))
            except ValueError:
                rel = str(path)
            found.append(VizArtifactDir(rel=rel, n_png=n, kind=kind))
            return
        if n <= 0 and kind not in ("frontier_picks",):
            return
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        found.append(VizArtifactDir(rel=rel, n_png=n, kind=kind))

    bundles = root / "bundles"
    if bundles.is_dir():
        for bundle in sorted(p for p in bundles.iterdir() if p.is_dir()):
            _add(bundle / "frontier_picks", "frontier_picks")
            _add(bundle / "maps", "maps")
            _add(bundle, "topdown")
    _add(root / "figures", "figures")
    # Prefer frontier_picks → maps → figures → topdown in display order.
    kind_rank = {"frontier_picks": 0, "maps": 1, "figures": 2, "topdown": 3, "other": 9}
    found.sort(key=lambda a: (kind_rank.get(a.kind, 9), a.rel))
    return found


def format_viz_artifact_lines(out_dir: str | Path | None) -> list[str]:
    """Lines for ``emet jobs status`` / ``report`` pointing at feh-able image dirs."""
    arts = discover_out_viz_artifacts(out_dir)
    if not arts:
        if out_dir and Path(str(out_dir)).expanduser().is_dir():
            return [
                "viz:       (no PNGs under bundles/ or figures/ yet)",
                "           tip: feh needs a glob — OUT/ itself is mostly logs/jsonl",
            ]
        return []
    root = Path(str(out_dir)).expanduser()
    lines: list[str] = []
    for i, a in enumerate(arts):
        prefix = "viz:       " if i == 0 else "           "
        if a.kind == "topdown":
            lines.append(f"{prefix}{a.rel}/topdown*.png  ({a.n_png} png)")
        else:
            lines.append(f"{prefix}{a.rel}/  ({a.n_png} png, {a.kind})")
    # Prefer frontier picks, else step maps, else whatever has PNGs.
    preferred = next((a for a in arts if a.kind == "frontier_picks" and a.n_png > 0), None)
    if preferred is None:
        preferred = next((a for a in arts if a.kind == "maps" and a.n_png > 0), None)
    if preferred is None:
        preferred = next((a for a in arts if a.n_png > 0), None)
    if preferred is not None:
        if preferred.kind == "frontier_picks":
            glob = "iter_*.png"
        elif preferred.kind == "topdown":
            glob = "topdown*.png"
        else:
            glob = "*.png"
        lines.append(f"feh:       feh {root / preferred.rel}/{glob}")
    return lines


def format_job_detail(job: JobRecord) -> str:
    """Human-readable multi-line status (not JSON)."""
    prog = compute_job_progress(job)
    lines = [
        f"id:        {job.id}",
        f"name:      {job.name}",
    ]
    desc = job_description(job)
    if desc:
        lines.append(f"why:       {desc}")
    lines.extend(
        [
            f"status:    {job.status}",
            f"pid:       {job.pid if job.pid is not None else '-'}",
            f"age:       {_format_age(job.created_at)}",
            f"out_dir:   {job.out_dir or '-'}",
            f"log_path:  {job.log_path or '-'}",
        ]
    )
    lines.extend(format_viz_artifact_lines(job.out_dir))
    if prog.source != "none":
        lines.append(f"progress:  {format_progress_brief(prog)}")
        if prog.rate_s_per_unit is not None:
            lines.append(f"rate:      {_format_duration(prog.rate_s_per_unit)}/unit")
        if prog.eta_s is not None:
            lines.append(f"eta:       {_format_duration(prog.eta_s)}")
    if job.wait_pids:
        lines.append(f"wait_pids: {', '.join(str(p) for p in job.wait_pids)}")
    if job.error:
        lines.append(f"error:     {job.error}")
    if job.cmd:
        lines.append(f"cmd:       {_ellipsize(job.cmd, 120)}")
    if job.repo:
        lines.append(f"repo:      {job.repo}")
    if job.meta:
        # Avoid dumping huge blobs; show progress-related keys first.
        interesting = {
            k: job.meta[k] for k in ("units_done", "units_total", "phase", "current_id", "note") if k in job.meta
        }
        if interesting:
            lines.append(f"meta:      {interesting}")
    return "\n".join(lines)


_ACTIVE_REPORT_STATUSES: frozenset[str] = frozenset({"queued", "waiting", "running"})
_EPISODE_JSONL_RE = re.compile(
    r"^(?P<arm>classic|agentic|graph_eqa|dynagraph)_(?:q)?(?P<qid>\d+)\.jsonl$",
    re.IGNORECASE,
)
_HOLDOUT_IDS_RE = re.compile(r"HOLDOUT_IDS=([0-9,\s]+)")
_IDS_FLAG_RE = re.compile(r"--ids\s+([0-9,\s]+)")


@dataclass
class EpisodeScore:
    """One scored HM-EQA / H2H episode jsonl row."""

    arm: str
    question_id: int
    correct: bool | None = None
    predicted: str | None = None
    gold: str | None = None
    planning_steps: int | None = None
    confident: bool | None = None  # EQA VLM Confidence: (legacy alias of eqa_confident)
    verified: bool | None = None  # agentic verify / VLM-assess gate
    answerable: bool | None = None  # agentic policy ANSWER (from summary/trace)
    answer_provenance: str | None = None  # forced-answer / submit channel tag
    path: str | None = None

    @property
    def result_label(self) -> str:
        if self.correct is True:
            return "ok"
        if self.correct is False:
            return "FAIL"
        return "?"

    @property
    def eqa_confident(self) -> bool | None:
        return self.confident

    def conf_cell(self) -> str:
        """Compact verify-gate + EQA self-report for the report table."""
        parts: list[str] = []
        if self.verified is not None:
            parts.append(f"v={'Y' if self.verified else 'N'}")
        if self.answerable is not None and self.answerable != self.verified:
            parts.append(f"a={'Y' if self.answerable else 'N'}")
        if self.confident is not None:
            parts.append(f"e={'Y' if self.confident else 'N'}")
        return " ".join(parts) if parts else "-"


def _load_agentic_summary_flags(
    out_dir: Path,
    arm: str,
    qid: int,
    *,
    debug_bundle_dir: str | None = None,
) -> tuple[bool | None, bool | None]:
    """Return (verified, answerable) from ``bundles/{arm}_q{qid}/agentic_summary.json``."""
    candidates: list[Path] = []
    if debug_bundle_dir:
        candidates.append(Path(str(debug_bundle_dir)).expanduser() / "agentic_summary.json")
    candidates.extend(
        [
            out_dir / "bundles" / f"{arm}_q{qid}" / "agentic_summary.json",
            out_dir / "bundles" / f"{arm}_q{qid:04d}" / "agentic_summary.json",
        ]
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        verified = data.get("verified") if isinstance(data.get("verified"), bool) else None
        answerable = data.get("answerable") if isinstance(data.get("answerable"), bool) else None
        return verified, answerable
    return None, None


def resolve_report_job(job_id: str | None = None) -> JobRecord | None:
    """Resolve a job for ``emet jobs report``.

    With no id: prefer a single active (running/waiting/queued) job; if several,
    pick running first then most recently updated. If none active, fall back to
    the most recently updated terminal job that has an ``out_dir``.
    """
    if job_id:
        job = load_job(job_id)
        return refresh_job_liveness(job) if job is not None else None

    active = [refresh_job_liveness(j) for j in list_jobs(include_terminal=False)]
    active = [j for j in active if j.status in _ACTIVE_REPORT_STATUSES]
    if active:
        rank = {"running": 0, "waiting": 1, "queued": 2}
        active.sort(key=lambda j: (rank.get(j.status, 9), -j.updated_at))
        return active[0]

    terminal = [
        refresh_job_liveness(j) for j in list_jobs(include_terminal=True) if j.status in TERMINAL_STATUSES and j.out_dir
    ]
    if not terminal:
        return None
    terminal.sort(key=lambda j: j.updated_at, reverse=True)
    return terminal[0]


def parse_planned_question_ids(job: JobRecord) -> list[int]:
    """Best-effort planned QID list from cmd / orchestrator.log / progress."""
    blobs: list[str] = [job.cmd or ""]
    if job.out_dir:
        orch = Path(job.out_dir).expanduser() / "orchestrator.log"
        if orch.is_file():
            try:
                blobs.append(orch.read_text(encoding="utf-8", errors="replace")[:8000])
            except OSError:
                pass
        prog = read_progress_file(job.out_dir)
        ids_raw = prog.get("ids") or prog.get("holdout_ids")
        if isinstance(ids_raw, str):
            blobs.append(f"HOLDOUT_IDS={ids_raw}")
        elif isinstance(ids_raw, list):
            return [int(x) for x in ids_raw]

    for blob in blobs:
        for cre in (_HOLDOUT_IDS_RE, _IDS_FLAG_RE):
            m = cre.search(blob)
            if not m:
                continue
            ids: list[int] = []
            for part in m.group(1).split(","):
                part = part.strip()
                if part.isdigit():
                    ids.append(int(part))
            if ids:
                return ids
    return []


def collect_episode_scores(out_dir: str | Path | None) -> list[EpisodeScore]:
    """Load non-empty ``{arm}_q{id}.jsonl`` scores under an H2H OUT dir."""
    if not out_dir:
        return []
    root = Path(out_dir).expanduser()
    if not root.is_dir():
        return []
    scores: list[EpisodeScore] = []
    for path in sorted(root.glob("*_q*.jsonl")):
        m = _EPISODE_JSONL_RE.match(path.name)
        if not m:
            continue
        try:
            if path.stat().st_size <= 0:
                continue
            line = next(
                (ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()),
                "",
            )
            if not line:
                continue
            row = json.loads(line)
        except (OSError, json.JSONDecodeError, StopIteration, TypeError, ValueError):
            continue
        qid = int(row.get("question_id") or m.group("qid"))
        arm = str(m.group("arm")).lower()
        pred = row.get("predicted_answer") or row.get("parsed_answer_letter") or row.get("formatted_answer")
        gold = row.get("gold_answer_letter") or row.get("gt_answer") or row.get("answer_gt")
        steps = row.get("planning_steps")
        eqa_conf = row.get("confident")
        if not isinstance(eqa_conf, bool):
            eqa_conf = row.get("model_confident") if isinstance(row.get("model_confident"), bool) else None
        verified, answerable = _load_agentic_summary_flags(
            root,
            arm,
            qid,
            debug_bundle_dir=str(row["debug_bundle_dir"]) if row.get("debug_bundle_dir") else None,
        )
        prov_raw = row.get("answer_provenance")
        provenance = str(prov_raw).strip() if prov_raw not in (None, "") else None
        scores.append(
            EpisodeScore(
                arm=arm,
                question_id=qid,
                correct=row.get("correct") if isinstance(row.get("correct"), bool) else None,
                predicted=str(pred).strip() if pred not in (None, "") else None,
                gold=str(gold).strip() if gold not in (None, "") else None,
                planning_steps=int(steps) if isinstance(steps, (int, float)) else None,
                confident=eqa_conf if isinstance(eqa_conf, bool) else None,
                verified=verified,
                answerable=answerable,
                answer_provenance=provenance,
                path=str(path),
            )
        )
    scores.sort(key=lambda s: (s.arm, s.question_id))
    return scores


def provenance_accuracy_breakdown(
    episodes: list[EpisodeScore],
) -> dict[str, dict[str, Any]]:
    """Per ``answer_provenance`` n/correct/accuracy for scored episodes."""
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for e in episodes:
        if e.correct is None:
            continue
        key = (e.answer_provenance or "").strip() or "unset"
        counts[key] += 1
        if e.correct is True:
            correct[key] += 1
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(counts):
        n = counts[key]
        ok = correct[key]
        out[key] = {"n": n, "correct": ok, "accuracy": (float(ok) / float(n)) if n else None}
    return out


def list_crash_markers(out_dir: str | Path | None) -> list[str]:
    if not out_dir:
        return []
    root = Path(out_dir).expanduser()
    if not root.is_dir():
        return []
    names: list[str] = []
    for path in sorted(root.glob("*")):
        n = path.name
        if n.endswith(".CRASH") or n.startswith("native_crash_") or n.startswith("host_freeze_"):
            names.append(n)
    return names


def job_report_dict(job: JobRecord) -> dict[str, Any]:
    """Structured payload for ``emet jobs report --json``."""
    prog = compute_job_progress(job)
    episodes = collect_episode_scores(job.out_dir)
    planned = parse_planned_question_ids(job)
    scored_ids = {e.question_id for e in episodes}
    remaining = [q for q in planned if q not in scored_ids]
    n_ok = sum(1 for e in episodes if e.correct is True)
    n_fail = sum(1 for e in episodes if e.correct is False)
    by_prov = provenance_accuracy_breakdown(episodes)
    excl = [e for e in episodes if e.correct is not None and (e.answer_provenance or "") != "uniform_prior"]
    excl_ok = sum(1 for e in excl if e.correct is True)
    return {
        "id": job.id,
        "name": job.name,
        "status": job.status,
        "out_dir": job.out_dir,
        "progress": {
            "units_done": prog.units_done,
            "units_total": prog.units_total,
            "phase": prog.phase,
            "current_id": prog.current_id,
            "eta_s": prog.eta_s,
            "rate_s_per_unit": prog.rate_s_per_unit,
            "brief": format_progress_brief(prog),
        },
        "planned_ids": planned,
        "remaining_ids": remaining,
        "n_correct": n_ok,
        "n_incorrect": n_fail,
        "n_scored": len(episodes),
        "accuracy": (float(n_ok) / float(n_ok + n_fail)) if (n_ok + n_fail) else None,
        "by_provenance": by_prov,
        "n_excl_uniform_prior": len(excl),
        "correct_excl_uniform_prior": excl_ok,
        "accuracy_excl_uniform_prior": (float(excl_ok) / float(len(excl))) if excl else None,
        "crashes": list_crash_markers(job.out_dir),
        "episodes": [asdict(e) for e in episodes],
    }


def format_job_report(
    job: JobRecord,
    *,
    fail_only: bool = False,
) -> str:
    """Operator-facing progress + per-episode score table (default ``emet jobs report``)."""
    prog = compute_job_progress(job)
    episodes = collect_episode_scores(job.out_dir)
    planned = parse_planned_question_ids(job)
    scored_ids = {e.question_id for e in episodes}
    remaining = [q for q in planned if q not in scored_ids]
    crashes = list_crash_markers(job.out_dir)
    n_ok = sum(1 for e in episodes if e.correct is True)
    n_fail = sum(1 for e in episodes if e.correct is False)
    n_scored = n_ok + n_fail
    acc = (100.0 * n_ok / n_scored) if n_scored else None

    done_s = (
        f"{prog.units_done}/{prog.units_total}"
        if prog.units_done is not None and prog.units_total is not None
        else (str(prog.units_done) if prog.units_done is not None else f"{len(episodes)}/?")
    )
    eta_s = f"ETA {_format_duration(prog.eta_s)}" if prog.eta_s is not None else ""
    rate_s = f"~{_format_duration(prog.rate_s_per_unit)}/ep" if prog.rate_s_per_unit is not None else ""
    headline_bits = [f"{done_s} done", job.status]
    if eta_s:
        headline_bits.append(eta_s)
    if rate_s:
        headline_bits.append(rate_s)
    if n_ok or n_fail:
        acc_s = f"  ({acc:.0f}%)" if acc is not None else ""
        headline_bits.append(f"scored {n_ok} ok / {n_fail} fail{acc_s}")

    lines = [
        "  ".join(headline_bits),
        f"job:  {job.id}  ({job.name})",
        f"out:  {job.out_dir or '-'}",
    ]
    for viz_line in format_viz_artifact_lines(job.out_dir):
        # Re-indent status-style "viz:" / "feh:" lines for the report layout.
        if viz_line.startswith("viz:"):
            lines.append("viz:  " + viz_line[len("viz:") :].lstrip())
        elif viz_line.startswith("feh:"):
            lines.append("feh:  " + viz_line[len("feh:") :].lstrip())
        else:
            lines.append("      " + viz_line.strip())
    if prog.phase or prog.current_id:
        cur = ""
        if prog.current_id:
            cur = f" q{prog.current_id}" if str(prog.current_id).isdigit() else f" {prog.current_id}"
        lines.append(f"now:  {prog.phase or '-'}{cur}")

    by_prov = provenance_accuracy_breakdown(episodes)
    if by_prov:
        bits = [f"{k}={v['correct']}/{v['n']}" for k, v in by_prov.items()]
        excl = [e for e in episodes if e.correct is not None and (e.answer_provenance or "") != "uniform_prior"]
        excl_ok = sum(1 for e in excl if e.correct is True)
        excl_bit = ""
        if excl and len(excl) < n_scored:
            excl_bit = f"  excl_uniform={excl_ok}/{len(excl)}"
        lines.append(f"prov: {'  '.join(bits)}{excl_bit}")

    by_arm: dict[str, list[EpisodeScore]] = {}
    for e in episodes:
        by_arm.setdefault(e.arm, []).append(e)
    if len(by_arm) > 1 and n_scored:
        arm_bits = []
        for arm_name, rows in sorted(by_arm.items()):
            ok = sum(1 for e in rows if e.correct is True)
            tot = sum(1 for e in rows if e.correct is not None)
            if tot:
                arm_bits.append(f"{arm_name} {ok}/{tot}")
        if arm_bits:
            lines.append("arms: " + "  |  ".join(arm_bits))

    table_eps = [e for e in episodes if (e.correct is False) or not fail_only]
    if episodes:
        lines.append("")
        if fail_only:
            lines.append(f"fails only ({sum(1 for e in table_eps if e.correct is False)}):")
        lines.append(f"{'Q':>4}  {'arm':<8}  {'result':<5}  {'pred/gold':<10}  {'steps':>5}  conf")
        for e in table_eps:
            pg = f"{e.predicted or '—'}/{e.gold or '—'}"
            steps = "-" if e.planning_steps is None else str(e.planning_steps)
            lines.append(f"{e.question_id:>4}  {e.arm:<8}  {e.result_label:<5}  {pg:<10}  {steps:>5}  {e.conf_cell()}")
        if not table_eps and fail_only:
            lines.append("(no incorrect episodes)")
        if any(e.verified is not None or e.confident is not None for e in episodes):
            lines.append("conf: v=verify-gate  e=EQA Confidence: (a=answerable only if ≠ v)")
        lines.append("tip:  emet jobs report JOB --question ID --rooms")
    else:
        lines.append("")
        lines.append("(no scored episode jsonl yet)")

    lines.append("")
    if remaining:
        lines.append("next: " + ", ".join(str(q) for q in remaining))
    elif planned and len(scored_ids) >= len(planned):
        lines.append("next: (all planned ids scored)")
    elif prog.current_id and str(prog.current_id) not in {str(q) for q in scored_ids}:
        lines.append(f"next: {prog.current_id} (in flight)")
    else:
        lines.append("next: -")

    if crashes:
        lines.append("crashes: " + ", ".join(crashes))
    else:
        lines.append("crashes: none")
    return "\n".join(lines)


def _episode_row_for_qid(out_dir: str | Path | None, qid: int, arm: str | None = None) -> dict[str, Any] | None:
    """First non-empty jsonl row for a question id under an H2H OUT dir."""
    if not out_dir:
        return None
    root = Path(out_dir).expanduser()
    if not root.is_dir():
        return None
    matches: list[Path] = []
    for path in sorted(root.glob("*_q*.jsonl")):
        m = _EPISODE_JSONL_RE.match(path.name)
        if not m or int(m.group("qid")) != int(qid):
            continue
        if arm and str(m.group("arm")).lower() != arm.lower():
            continue
        matches.append(path)
    # Prefer agentic arm when no explicit arm requested.
    matches.sort(key=lambda p: (0 if p.name.lower().startswith("agentic") else 1, p.name))
    for path in matches:
        try:
            if path.stat().st_size <= 0:
                continue
            line = next(
                (ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()),
                "",
            )
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("_jsonl_path", str(path))
            return row
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


def _find_agentic_trace(out_dir: str | Path | None, qid: int, row: dict[str, Any] | None) -> Path | None:
    """Locate ``agentic_trace.jsonl`` for a question from the episode row or caches.

    Prefer the job ``OUT/bundles/`` copy over Habitat episode-cache paths — the
    cache can lag H2H bundle copies (e.g. missing ``router_room`` / ``rooms_line``).
    """
    candidates: list[Path] = []
    if out_dir:
        root = Path(out_dir).expanduser()
        candidates.append(root / "bundles" / f"agentic_q{qid}" / "agentic_trace.jsonl")
        candidates.extend(sorted(root.glob(f"bundles/agentic_q{qid}/**/agentic_trace.jsonl")))
    if row:
        bundle = row.get("debug_bundle_dir")
        if bundle:
            candidates.append(Path(bundle).expanduser() / "agentic_trace.jsonl")
    cache = Path.home() / ".cache" / "habitat_eqa" / "episodes"
    if cache.is_dir():
        candidates.append(cache / f"h2h_agentic_q{qid:04d}" / f"q{qid:04d}_dynagraph" / "agentic_trace.jsonl")
        candidates.extend(sorted(cache.glob(f"*q{qid:04d}*/**/agentic_trace.jsonl")))
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def job_record_for_out_dir(out_dir: str | Path, *, name: str | None = None) -> JobRecord:
    """Synthetic job record so ``emet jobs report --out-dir`` works without a registry id."""
    root = Path(out_dir).expanduser().resolve()
    return JobRecord(
        id="out-dir",
        name=name or root.name,
        status="done",
        out_dir=str(root),
    )


def _find_episode_map_paths(out_dir: str | Path | None, qid: int, row: dict[str, Any] | None) -> list[str]:
    """Topdown / room-label map paths for a question (OUT bundle preferred)."""
    found: list[str] = []
    roots: list[Path] = []
    if out_dir:
        roots.append(Path(out_dir).expanduser() / "bundles" / f"agentic_q{qid}")
    if row and row.get("debug_bundle_dir"):
        roots.append(Path(str(row["debug_bundle_dir"])).expanduser())
    names = ("topdown_rooms.png", "topdown_map.png", "topdown_map_overlay.png")
    seen: set[str] = set()
    for root in roots:
        for name in names:
            path = root / name
            if path.is_file():
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    found.append(key)
    return found


def _load_trace_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return rows


def _xy2(xyz: Any) -> list[float] | None:
    if not isinstance(xyz, (list, tuple)) or len(xyz) < 2:
        return None
    try:
        return [round(float(xyz[0]), 2), round(float(xyz[1]), 2)]
    except (TypeError, ValueError):
        return None


def analyze_agentic_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive verify/abstain/investigate/explore signals from an agentic trace."""
    tool_counts: dict[str, int] = {}
    for r in rows:
        key = str(r.get("tool") or r.get("event") or "?")
        tool_counts[key] = tool_counts.get(key, 0) + 1

    verifies = [r for r in rows if r.get("tool") == "verify_siglip"]
    verify_obs = [r.get("obs_id") for r in verifies if r.get("obs_id") is not None]
    capture_obs = [
        r.get("obs_id") for r in rows if r.get("tool") == "capture_and_update" and r.get("obs_id") is not None
    ]
    phrases = sorted({str(r.get("phrase")) for r in verifies if r.get("phrase")})
    det_scores = [float(r["detector_score"]) for r in verifies if isinstance(r.get("detector_score"), (int, float))]
    abstains = [str(r.get("reason") or "") for r in rows if r.get("tool") == "abstain_unverified"]
    fallback_submits = sum(1 for r in rows if r.get("tool") == "submit_answer" and r.get("picked_by") == "fallback")
    seen: set[Any] = set()
    dup_verify_obs = sorted({o for o in verify_obs if o in seen or seen.add(o)})  # type: ignore[func-returns-value]

    # First hyp board (evidence cards shown to the router).
    hypotheses: list[dict[str, Any]] = []
    for r in rows:
        hyps = r.get("hypotheses")
        if isinstance(hyps, list) and hyps:
            for h in hyps[:12]:
                if not isinstance(h, dict):
                    continue
                hypotheses.append(
                    {
                        "obs_id": h.get("obs_id"),
                        "source": h.get("source"),
                        "phrase": h.get("phrase"),
                        "xy": _xy2(h.get("xyz")),
                    }
                )
            break

    # Router picks (VLM tool choices).
    router_picks: list[str] = []
    for r in rows:
        if r.get("picked_by") != "vlm":
            continue
        calls = r.get("router_tool_calls")
        if isinstance(calls, list) and calls:
            tool = str(calls[0])
        else:
            tool = str(r.get("tool") or "?")
        args = r.get("args") if isinstance(r.get("args"), dict) else {}
        oid = args.get("obs_id")
        toward = args.get("toward")
        if oid is not None:
            router_picks.append(f"{tool}(obs={oid})")
        elif toward:
            router_picks.append(f"{tool}(toward={toward!r})")
        else:
            router_picks.append(tool)

    # Investigate / navigate outcomes (skip tool_pick rows; keep nav + station).
    investigates: list[dict[str, Any]] = []
    for r in rows:
        if r.get("tool") != "investigate" and r.get("tool") != "navigate_to_obs":
            continue
        if r.get("event") == "tool_pick":
            continue
        entry: dict[str, Any] = {
            "round": r.get("round"),
            "obs_id": r.get("obs_id"),
            "event": r.get("event"),
            "ok": r.get("ok"),
            "status": r.get("status"),
            "nav_success": r.get("nav_success"),
            "target_xy": _xy2(r.get("target_xyz")),
            "closest_m": (round(float(r["closest_m"]), 3) if isinstance(r.get("closest_m"), (int, float)) else None),
            "place_inspect": r.get("place_inspect"),
            "station_obs_id": r.get("station_obs_id"),
        }
        investigates.append(entry)

    explores: list[dict[str, Any]] = []
    for r in rows:
        if r.get("tool") != "explore_frontier" or r.get("event") == "tool_pick":
            continue
        explores.append(
            {
                "round": r.get("round"),
                "ok": r.get("ok"),
                "source": r.get("source"),
                "xy": _xy2(r.get("frontier_xyz")),
            }
        )

    assesses: list[dict[str, Any]] = []
    for r in rows:
        if r.get("tool") != "vlm_assess":
            continue
        assesses.append(
            {
                "round": r.get("round"),
                "obs_id": r.get("obs_id"),
                "present": r.get("present"),
                "answerable": r.get("answerable"),
                "phrase": r.get("phrase"),
                "reason": (str(r.get("reason") or "")[:160] or None),
            }
        )

    nav_loop_blocked = sum(
        1 for r in rows if str(r.get("status") or "") == "NAV_LOOP_BLOCKED" or r.get("tool") == "nav_loop_redirect"
    )
    station_inspects = [e for e in investigates if e.get("event") == "station_inspect"]
    closest_ms = [e["closest_m"] for e in station_inspects if e.get("closest_m") is not None]
    place_ledger = [str(e["place_inspect"]) for e in station_inspects if e.get("place_inspect")]

    salvage = next(
        (
            {
                "letter": r.get("letter"),
                "prior_answer": r.get("prior_answer"),
                "n_images": r.get("n_images"),
                "applied": True,
            }
            for r in rows
            if r.get("event") == "final_location_salvage"
        ),
        None,
    )
    if salvage is None:
        salvage = next(
            (
                {
                    "letter": r.get("letter"),
                    "prior_answer": r.get("prior_answer"),
                    "n_images": r.get("n_images"),
                    "applied": False,
                    "counterfactual": True,
                }
                for r in rows
                if r.get("event") == "final_location_salvage_counterfactual"
            ),
            None,
        )
    summary = next((r for r in rows if r.get("tool") == "summary"), None)
    summary_bits: dict[str, Any] | None = None
    if summary is not None:
        summary_bits = {
            "final_answer": summary.get("final_answer"),
            "confidence": summary.get("confidence"),
            "verified": summary.get("verified"),
            "n_rounds": summary.get("n_rounds"),
            "n_nav": summary.get("n_nav"),
            "n_explore": summary.get("n_explore"),
            "budget_hit": summary.get("budget_hit"),
        }

    present_any = any(bool(a.get("present")) for a in assesses)
    close_absent = any(
        e.get("closest_m") is not None
        and float(e["closest_m"]) <= 0.6
        and isinstance(e.get("place_inspect"), str)
        and "ABSENT" in e["place_inspect"]
        for e in station_inspects
    )

    room_turns: list[dict[str, Any]] = []
    for r in rows:
        if r.get("event") != "router_room":
            continue
        targets = r.get("question_target_rooms")
        if isinstance(targets, list):
            target_list = [str(t) for t in targets]
        elif isinstance(targets, set):
            target_list = sorted(str(t) for t in targets)
        else:
            target_list = []
        rooms_line = r.get("rooms_line")
        room_turns.append(
            {
                "round": r.get("round"),
                "merged": r.get("current_room"),
                "vlm": r.get("current_room_vlm"),
                "graph": r.get("current_room_graph"),
                "rooms_line": str(rooms_line).strip() if rooms_line else "",
                "question_target_rooms": target_list,
                "prefer_explore_reason": str(r.get("prefer_explore_reason") or ""),
                "tool_calls": list(r.get("tool_calls") or []),
                "router_ms": (float(r["router_ms"]) if isinstance(r.get("router_ms"), (int, float)) else None),
                "n_room_images": (
                    int(r["n_room_images"]) if isinstance(r.get("n_room_images"), (int, float)) else None
                ),
            }
        )

    redirects = [
        {
            "round": r.get("round"),
            "from": r.get("from") or r.get("from_tool"),
            "to": r.get("to") or r.get("to_tool"),
            "from_args": r.get("from_args") if isinstance(r.get("from_args"), dict) else {},
        }
        for r in rows
        if r.get("event") == "prefer_explore_redirect"
    ]

    target_rooms: list[str] = []
    for turn in room_turns:
        if turn.get("question_target_rooms"):
            target_rooms = list(turn["question_target_rooms"])
            break
    rooms_line_latest = ""
    for turn in reversed(room_turns):
        if turn.get("rooms_line"):
            rooms_line_latest = str(turn["rooms_line"])
            break
    merged_seq = [str(t.get("merged") or "unknown") for t in room_turns]
    n_mismatch = sum(
        1
        for t in room_turns
        if t.get("prefer_explore_reason") == "room_mismatch"
        or t.get("room_mismatch_diagnostic")
        or t.get("prefer_explore_room_mismatch")
    )
    n_disagree = sum(
        1
        for t in room_turns
        if t.get("vlm")
        and t.get("graph")
        and t["vlm"] != t["graph"]
        and t["vlm"] != "unknown"
        and t["graph"] != "unknown"
    )
    unique_rooms = sorted({m for m in merged_seq if m and m != "unknown"})
    router_ms_vals = [float(t["router_ms"]) for t in room_turns if isinstance(t.get("router_ms"), (int, float))]

    return {
        "n_rows": len(rows),
        "tool_counts": tool_counts,
        "phrases": phrases,
        "n_verify": len(verifies),
        "answerable_any": any(bool(v.get("answerable")) for v in verifies),
        "max_detector_score": max(det_scores) if det_scores else None,
        "capture_obs": capture_obs,
        "verify_obs": verify_obs,
        "duplicate_verify_obs": dup_verify_obs,
        "abstain_reasons": abstains,
        "fallback_submits": fallback_submits,
        "hypotheses": hypotheses,
        "router_picks": router_picks,
        "investigates": investigates,
        "explores": explores,
        "assesses": assesses,
        "n_investigate": sum(1 for e in investigates if e.get("event") != "station_inspect"),
        "n_station_inspect": len(station_inspects),
        "n_explore": len(explores),
        "n_nav_loop_blocked": nav_loop_blocked,
        "min_closest_m": min(closest_ms) if closest_ms else None,
        "place_ledger": place_ledger,
        "present_any": present_any,
        "close_absent": close_absent,
        "salvage": salvage,
        "summary": summary_bits,
        "rooms": {
            "turns": room_turns,
            "question_target_rooms": target_rooms,
            "rooms_line": rooms_line_latest,
            "merged_sequence": merged_seq,
            "unique_rooms": unique_rooms,
            "n_turns": len(room_turns),
            "n_mismatch": n_mismatch,
            "n_vlm_graph_disagree": n_disagree,
            "n_prefer_explore_redirect": len(redirects),
            "redirects": redirects,
            "router_ms_mean": (sum(router_ms_vals) / len(router_ms_vals)) if router_ms_vals else None,
            "router_ms_max": max(router_ms_vals) if router_ms_vals else None,
            "n_room_images_mean": (
                sum(int(t["n_room_images"]) for t in room_turns if t.get("n_room_images") is not None)
                / max(1, sum(1 for t in room_turns if t.get("n_room_images") is not None))
                if any(t.get("n_room_images") is not None for t in room_turns)
                else None
            ),
        },
    }


def question_report_dict(job: JobRecord, qid: int, arm: str | None = None) -> dict[str, Any]:
    """Structured per-question payload for ``emet jobs report --question X --json``."""
    row = _episode_row_for_qid(job.out_dir, qid, arm=arm)
    trace_path = _find_agentic_trace(job.out_dir, qid, row)
    trace = analyze_agentic_trace(_load_trace_rows(trace_path)) if trace_path else None
    maps = _find_episode_map_paths(job.out_dir, qid, row)
    payload: dict[str, Any] = {
        "id": job.id,
        "question_id": qid,
        "found": row is not None,
        "trace_path": str(trace_path) if trace_path else None,
        "maps": maps,
    }
    if row is not None:
        pred = row.get("predicted_answer") or row.get("parsed_answer_letter") or row.get("formatted_answer")
        arm_name = str(row.get("method") or row.get("arm") or arm or "agentic").lower()
        # Prefer arm from jsonl filename when present on the row helper.
        jsonl_path = row.get("_jsonl_path")
        if jsonl_path:
            m = _EPISODE_JSONL_RE.match(Path(str(jsonl_path)).name)
            if m:
                arm_name = str(m.group("arm")).lower()
        verified, answerable = _load_agentic_summary_flags(
            Path(job.out_dir).expanduser() if job.out_dir else Path("."),
            arm_name,
            qid,
            debug_bundle_dir=str(row["debug_bundle_dir"]) if row.get("debug_bundle_dir") else None,
        )
        eqa_conf = row.get("confident")
        if not isinstance(eqa_conf, bool):
            eqa_conf = row.get("model_confident") if isinstance(row.get("model_confident"), bool) else None
        conf_score = EpisodeScore(
            arm=arm_name,
            question_id=qid,
            confident=eqa_conf if isinstance(eqa_conf, bool) else None,
            verified=verified,
            answerable=answerable,
        )
        payload["episode"] = {
            "arm": row.get("method") or row.get("arm"),
            "scene": row.get("scene"),
            "question": row.get("question"),
            "choices": row.get("choices"),
            "predicted": str(pred).strip() if pred not in (None, "") else None,
            "gold": row.get("gold_answer_letter"),
            "correct": row.get("correct"),
            "confident": eqa_conf if isinstance(eqa_conf, bool) else None,
            "verified": verified,
            "answerable": answerable,
            "conf_detail": conf_score.conf_cell(),
            "planning_steps": row.get("planning_steps"),
            "observations": row.get("observations"),
            "graph_nodes": row.get("graph_nodes"),
            "eqa_action": row.get("eqa_action"),
            "eqa_iterations": row.get("eqa_iterations"),
            "error": row.get("error") or None,
            "confidence_reasoning": row.get("eqa_confidence_reasoning") or None,
            "jsonl_path": row.get("_jsonl_path"),
        }
    if trace is not None:
        rooms = trace.get("rooms")
        if isinstance(rooms, dict) and not rooms.get("question_target_rooms"):
            qtext = ""
            if row is not None:
                qtext = str(row.get("question") or "").strip()
                choices = row.get("choices")
                if isinstance(choices, (list, tuple)) and choices:
                    letters = "ABCDEFGH"
                    mcq = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(choices) if i < len(letters))
                    qtext = f"{qtext}\n{mcq}".strip()
                elif choices:
                    qtext = f"{qtext} {choices}".strip()
            if qtext:
                try:
                    from emet.memory.graph_eqa.room_clusters import question_target_rooms

                    rooms["question_target_rooms"] = sorted(question_target_rooms(qtext))
                except Exception:
                    pass
        payload["trace"] = trace
    return payload


_QUESTION_REPORT_SECTIONS = (
    "summary",
    "rooms",
    "router",
    "nav",
    "assess",
    "verify",
    "flags",
)


def _normalize_report_sections(
    sections: list[str] | tuple[str, ...] | None,
    *,
    rooms_focus: bool = False,
    brief: bool = False,
) -> set[str]:
    if rooms_focus:
        return {"summary", "rooms", "flags"}
    if sections:
        out: set[str] = set()
        for raw in sections:
            for part in str(raw).lower().replace(" ", "").split(","):
                if part in ("all", "*"):
                    return set(_QUESTION_REPORT_SECTIONS)
                if part:
                    out.add(part)
        return out or set(_QUESTION_REPORT_SECTIONS)
    if brief:
        return {"summary", "rooms", "router", "flags"}
    return set(_QUESTION_REPORT_SECTIONS)


def _format_rooms_section(rooms: dict[str, Any], *, verbose: bool = False) -> list[str]:
    """Pretty room timeline for ``emet jobs report -q …``."""

    def _turn_line(t: dict[str, Any]) -> list[str]:
        reason = str(t.get("prefer_explore_reason") or "")
        reason_s = f"  [{reason}]" if reason else ""
        tools = t.get("tool_calls") or []
        tool_s = ",".join(str(x) for x in tools) if tools else "-"
        vlm = t.get("vlm") or "?"
        graph = t.get("graph") or "?"
        merged = t.get("merged") or "?"
        disagree = " *" if (vlm != graph and vlm != "unknown" and graph != "unknown") else ""
        out = [f"  r{t.get('round')}: {merged:<12}  vlm={vlm:<12} graph={graph:<12}{disagree}  → {tool_s}{reason_s}"]
        if verbose and t.get("rooms_line"):
            out.append(f"         {t['rooms_line']}")
        return out

    lines: list[str] = ["── rooms ──"]
    targets = rooms.get("question_target_rooms") or []
    if targets:
        lines.append(f"targets: {', '.join(targets)}")
    else:
        lines.append("targets: (none / not logged)")
    rooms_line = rooms.get("rooms_line") or ""
    if rooms_line:
        lines.append(rooms_line if rooms_line.startswith("Rooms:") else f"Rooms: {rooms_line}")
    else:
        lines.append("Rooms: (not logged — re-run with newer router_room trace)")
    seq = rooms.get("merged_sequence") or []
    if seq:
        compact: list[str] = []
        for name in seq:
            if not compact or compact[-1] != name:
                compact.append(str(name))
        lines.append(f"path:   {' → '.join(compact)}")
    lines.append(
        f"turns={rooms.get('n_turns', 0)}  "
        f"mismatch={rooms.get('n_mismatch', 0)}  "
        f"vlm≠graph={rooms.get('n_vlm_graph_disagree', 0)}  "
        f"redirects={rooms.get('n_prefer_explore_redirect', 0)}"
    )
    router_ms = [
        float(t["router_ms"]) for t in (rooms.get("turns") or []) if isinstance(t.get("router_ms"), (int, float))
    ]
    # Also accept top-level if we add it later; primary is per-turn in analyze.
    if not router_ms and isinstance(rooms.get("router_ms_mean"), (int, float)):
        router_ms = [float(rooms["router_ms_mean"])]
    if router_ms:
        mean_ms = sum(router_ms) / len(router_ms)
        lines.append(
            f"router_ms: mean={mean_ms:.0f}  min={min(router_ms):.0f}  max={max(router_ms):.0f}  n={len(router_ms)}"
        )
    turns = list(rooms.get("turns") or [])
    if not verbose and len(turns) > 10:
        for t in turns[:8]:
            lines.extend(_turn_line(t))
        lines.append(f"  … (+{len(turns) - 10} turns)")
        for t in turns[-2:]:
            lines.extend(_turn_line(t))
    else:
        for t in turns:
            lines.extend(_turn_line(t))
    for red in (rooms.get("redirects") or [])[: (12 if verbose else 6)]:
        lines.append(f"  redirect r{red.get('round')}: {red.get('from')} → {red.get('to')}")
    return lines


def format_question_report(
    job: JobRecord,
    qid: int,
    arm: str | None = None,
    *,
    sections: list[str] | tuple[str, ...] | None = None,
    rooms_focus: bool = False,
    verbose: bool = False,
    brief: bool = False,
) -> str:
    """Human-readable per-question deep dive (episode row + agentic trace signals)."""
    data = question_report_dict(job, qid, arm=arm)
    want = _normalize_report_sections(sections, rooms_focus=rooms_focus, brief=brief)
    lines = [f"q{qid}  job {job.id}  ({job.name})"]
    if not data["found"]:
        lines.append(f"(no scored jsonl for q{qid} under {job.out_dir or '-'})")
        if data.get("trace_path"):
            lines.append(f"trace: {data['trace_path']}")
        else:
            lines.append("trace: (none)")
        return "\n".join(lines)

    ep = data["episode"]
    result = "ok" if ep["correct"] is True else ("FAIL" if ep["correct"] is False else "?")
    pg = f"{ep['predicted'] or '—'}/{ep['gold'] or '—'}"
    conf_s = ep.get("conf_detail") or str(ep.get("confident"))
    if "summary" in want:
        lines.append(f"result: {result}  pred/gold {pg}  steps {ep['planning_steps']}  conf {conf_s}")
        if ep.get("verified") is not None or ep.get("confident") is not None:
            lines.append(
                f"gate: verified={ep.get('verified')} answerable={ep.get('answerable')}  "
                f"eqa_Confidence={ep.get('confident')}"
            )
        if ep.get("arm") or ep.get("scene"):
            lines.append(f"arm/scene: {ep.get('arm') or '-'} / {ep.get('scene') or '-'}")
        if ep.get("question"):
            lines.append(f"Q: {str(ep['question'])[:200]}")
        if ep.get("choices") and not brief:
            lines.append(f"choices: {ep['choices']}")
        if ep.get("error"):
            lines.append(f"error: {ep['error']}")
        if ep.get("confidence_reasoning") and verbose:
            lines.append(f"reason: {str(ep['confidence_reasoning'])[:220]}")

    trace = data.get("trace")
    if not trace:
        lines.append("")
        lines.append("trace: (none found)")
        return "\n".join(lines)

    if "summary" in want:
        lines.append("")
        lines.append(f"trace: {data['trace_path']}")
        summ = trace.get("summary") or {}
        if summ:
            lines.append(
                f"summary: answer={summ.get('final_answer')} conf={summ.get('confidence')} "
                f"verified={summ.get('verified')} rounds={summ.get('n_rounds')} "
                f"nav={summ.get('n_nav')} explore={summ.get('n_explore')} "
                f"budget_hit={summ.get('budget_hit')}"
            )
        lines.append(
            f"actions: investigate={trace.get('n_investigate', 0)} "
            f"station={trace.get('n_station_inspect', 0)} "
            f"explore={trace.get('n_explore', 0)} "
            f"nav_loop_blocked={trace.get('n_nav_loop_blocked', 0)} "
            f"min_closest_m={trace.get('min_closest_m')}"
        )
        maps = data.get("maps") or []
        if maps:
            lines.append("maps:  " + "  |  ".join(maps[:3]))

    rooms = trace.get("rooms") or {}
    if "rooms" in want:
        lines.append("")
        lines.extend(_format_rooms_section(rooms, verbose=verbose))
        if not rooms.get("n_turns"):
            lines.append("  (no router_room events — older trace)")

    if "router" in want:
        lines.append("")
        lines.append("── router ──")
        if trace.get("hypotheses"):
            hyp_bits = []
            limit = 10 if verbose else 6
            for h in trace["hypotheses"][:limit]:
                hyp_bits.append(f"obs={h.get('obs_id')} {h.get('source')}:{h.get('phrase')!r} xy={h.get('xy')}")
            lines.append("hyp: " + " | ".join(hyp_bits))
        if trace.get("router_picks"):
            picks = trace["router_picks"]
            lim = 20 if verbose else 12
            shown = picks if len(picks) <= lim else picks[: lim - 2] + [f"…(+{len(picks) - lim + 2})"]
            lines.append(f"picks: {' → '.join(shown)}")

    if "nav" in want and not brief:
        lines.append("")
        lines.append("── nav ──")
        if trace.get("investigates"):
            for e in trace["investigates"]:
                if e.get("event") == "station_inspect":
                    lines.append(
                        f"  station r{e.get('round')} obs={e.get('obs_id')} "
                        f"closest={e.get('closest_m')}m  {e.get('place_inspect')}"
                    )
                elif e.get("status") == "NAV_LOOP_BLOCKED" or e.get("ok") is False:
                    lines.append(
                        f"  investigate r{e.get('round')} obs={e.get('obs_id')} BLOCKED status={e.get('status')}"
                    )
                else:
                    lines.append(
                        f"  investigate r{e.get('round')} obs={e.get('obs_id')} "
                        f"nav={e.get('nav_success')} target={e.get('target_xy')}"
                    )
        if trace.get("explores"):
            ex = []
            for e in trace["explores"][: (12 if verbose else 8)]:
                ex.append(f"r{e.get('round')}:{e.get('source')}@{e.get('xy')}")
            lines.append("explore: " + " | ".join(ex))
        if trace.get("place_ledger") and verbose:
            lines.append("ledger: " + " || ".join(trace["place_ledger"][:4]))
        if trace.get("salvage"):
            s = trace["salvage"]
            kind = "counterfactual" if s.get("counterfactual") or s.get("applied") is False else "applied"
            lines.append(
                f"salvage({kind}): letter={s.get('letter')} prior={s.get('prior_answer')} n_images={s.get('n_images')}"
            )

    if "assess" in want and not brief:
        assesses = trace.get("assesses") or []
        if assesses:
            lines.append("")
            lines.append("── assess ──")
            limit = len(assesses) if verbose else min(8, len(assesses))
            for a in assesses[:limit]:
                reason = a.get("reason") or ""
                if not verbose and len(reason) > 100:
                    reason = reason[:97] + "…"
                lines.append(
                    f"  r{a.get('round')} obs={a.get('obs_id')} "
                    f"present={a.get('present')} ans={a.get('answerable')}  {reason}"
                )
            if not verbose and len(assesses) > limit:
                lines.append(f"  … (+{len(assesses) - limit} more)")

    if "verify" in want and not brief:
        lines.append("")
        lines.append("── verify ──")
        lines.append(f"tools: {trace['tool_counts']}")
        if trace["phrases"]:
            lines.append(f"phrases: {trace['phrases']}")
        md = trace["max_detector_score"]
        lines.append(
            f"verifies: {trace['n_verify']}  answerable_any={trace['answerable_any']}  max_detector={md:.3f}"
            if md is not None
            else f"verifies: {trace['n_verify']}  answerable_any={trace['answerable_any']}  max_detector=-"
        )
        if verbose or not brief:
            lines.append(f"capture_obs: {trace['capture_obs']}")
            lines.append(f"verify_obs:  {trace['verify_obs']}")

    if "flags" in want:
        flags: list[str] = []
        if trace["duplicate_verify_obs"]:
            flags.append(f"stale re-verify obs {trace['duplicate_verify_obs']}")
        if trace["fallback_submits"] >= 3:
            flags.append(f"{trace['fallback_submits']} fallback submit picks")
        if trace["n_verify"] and not trace["answerable_any"]:
            flags.append("never answerable (presence≠sufficiency or weak detect)")
        if len(trace["phrases"]) == 1 and any(
            w in trace["phrases"][0].lower() for w in ("already", "sets ", "how many")
        ):
            flags.append(f"suspect verify phrase {trace['phrases'][0]!r}")
        if trace.get("n_nav_loop_blocked", 0) >= 2:
            flags.append(f"nav-loop blocked x{trace['n_nav_loop_blocked']} (re-investigate same place)")
        if trace.get("close_absent"):
            flags.append("close look (≤0.6m) still ABSENT — VLM miss or wrong surface")
        if trace.get("n_explore", 0) >= 3 and not trace.get("present_any"):
            flags.append("explore-heavy with no present assess")
        rooms = trace.get("rooms") or {}
        if rooms.get("n_mismatch", 0) >= 2 and rooms.get("n_prefer_explore_redirect", 0) == 0:
            # Diagnostic only after Hydra-aligned rooms (no hard leave-wrong-room).
            flags.append("room_mismatch diagnostic (informational; no hard redirect)")
        if rooms.get("n_vlm_graph_disagree", 0) >= 3:
            flags.append(f"vlm≠graph on {rooms['n_vlm_graph_disagree']} router turns")
        lines.append("")
        if flags:
            lines.append("RED FLAGS: " + "; ".join(flags))
        else:
            lines.append("red flags: none")
        if trace["abstain_reasons"]:
            lines.append("abstain: " + " | ".join(trace["abstain_reasons"][-2:]))
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

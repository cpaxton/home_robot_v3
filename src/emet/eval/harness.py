# Copyright (c) Chris Paxton 2026
"""Shared helpers for long Habitat / HM-EQA evals (affinity, crash policy, progress).

Prefer driving these via ``emet eval`` / ``emet hmeqa`` / ``emet jobs`` rather than
ad-hoc shell ``taskset`` / ``nohup`` lines.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from emet.utils.cpu_affinity import (
    apply_affinity,
    cpus_at_or_above_mhz,
    format_taskset_list,
    online_cpu_ids,
    safe_cpu_ids,
)
from emet.utils.job_registry import write_progress_file
from emet.utils.process_tree import kill_process_tree, popen_session, terminate_process_tree

CrashPolicyName = Literal["skip", "abort"]

NATIVE_SIGNAL_NAMES: dict[int, str] = {
    132: "SIGILL",
    134: "SIGABRT",
    135: "SIGBUS",
    136: "SIGFPE",
    137: "SIGKILL",
    139: "SIGSEGV",
}


@dataclass
class CrashPolicy:
    """Per-episode native-crash handling for multi-qid batches.

    - ``skip``: settle, optional retry, then continue (default).
    - ``abort``: stop the whole batch on the first native crash.
    - ``streak_abort``: under ``skip``, abort after this many *consecutive*
      native crashes (early-exit when the harness is wedged). ``0`` disables.
    """

    policy: CrashPolicyName = "skip"
    retries: int = 1
    settle_sec: float = 60.0
    streak_abort: int = 2

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> CrashPolicy:
        e = env if env is not None else os.environ
        if str(e.get("NATIVE_CRASH_ABORT", "")).strip() == "1":
            policy: CrashPolicyName = "abort"
        else:
            raw = str(e.get("NATIVE_CRASH_POLICY", "skip")).strip().lower() or "skip"
            policy = "abort" if raw == "abort" else "skip"
        retries = int(e.get("NATIVE_CRASH_RETRIES", "1") or "1")
        settle = float(e.get("NATIVE_CRASH_SETTLE_SEC", "60") or "60")
        streak = int(e.get("NATIVE_CRASH_STREAK_ABORT", "2") or "2")
        return cls(policy=policy, retries=retries, settle_sec=settle, streak_abort=streak)


@dataclass
class EpisodeResult:
    returncode: int
    timed_out: bool = False
    signal_name: str | None = None

    @property
    def is_native_crash(self) -> bool:
        if self.timed_out:
            return False
        return self.signal_name is not None or self.returncode in NATIVE_SIGNAL_NAMES

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass
class CrashStreak:
    consecutive: int = 0
    total: int = 0
    last_ids: list[str] = field(default_factory=list)

    def record_ok(self) -> None:
        self.consecutive = 0

    def record_crash(self, qid: str) -> None:
        self.consecutive += 1
        self.total += 1
        self.last_ids.append(str(qid))

    def should_abort(self, policy: CrashPolicy) -> bool:
        if policy.policy == "abort" and self.total >= 1:
            return True
        if policy.streak_abort > 0 and self.consecutive >= policy.streak_abort:
            return True
        return False


def native_signal_name(returncode: int) -> str | None:
    if returncode in NATIVE_SIGNAL_NAMES:
        return NATIVE_SIGNAL_NAMES[returncode]
    if returncode >= 128:
        sig = returncode - 128
        try:
            return signal.Signals(sig).name
        except (ValueError, AttributeError):
            return f"signal_{sig}"
    return None


def apply_eval_affinity(
    *,
    pid: int | None = None,
    exclude_min_mhz: float | None = None,
    fail_closed: bool = True,
) -> dict[str, Any]:
    """Pin ``pid`` (default current) away from turbo CPUs; optionally fail closed."""
    if exclude_min_mhz is None:
        exclude_min_mhz = float(os.environ.get("EMET_EXCLUDE_CPU_MIN_MHZ", "6000") or "6000")
    if str(os.environ.get("EMET_SKIP_CPU_AFFINITY", "0")).strip() == "1":
        return {"skipped": True, "reason": "EMET_SKIP_CPU_AFFINITY=1"}

    mhz = exclude_min_mhz if exclude_min_mhz > 0 else None
    turbo = cpus_at_or_above_mhz(mhz) if mhz else []
    kept = safe_cpu_ids(exclude_min_mhz=mhz)
    compact = format_taskset_list(kept)
    target = os.getpid() if pid is None else int(pid)
    apply_affinity(kept, pid=target)

    try:
        actual = sorted(os.sched_getaffinity(target))
    except (AttributeError, OSError, PermissionError) as exc:
        if fail_closed:
            raise RuntimeError(f"could not verify CPU affinity for pid {target}: {exc}") from exc
        return {"applied": compact, "turbo_cpus": turbo, "verified": False, "error": str(exc)}

    leaked = sorted(set(actual) & set(turbo)) if turbo else []
    summary = {
        "applied": compact,
        "turbo_cpus": turbo,
        "kept": kept,
        "actual": actual,
        "leaked_turbo": leaked,
        "verified": not leaked,
        "pid": target,
    }
    if leaked and fail_closed:
        raise RuntimeError(
            f"CPU affinity still includes turbo CPUs {leaked}; "
            f"wanted mask {compact} (exclude >={exclude_min_mhz} MHz)"
        )
    return summary


def run_timed_command(
    cmd: Sequence[str],
    *,
    timeout_sec: float,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    log_path: Path | None = None,
    grace_s: float = 30.0,
) -> EpisodeResult:
    """Run ``cmd`` in a new session; kill the process tree on timeout or return."""
    merged = os.environ.copy()
    if env:
        merged.update({str(k): str(v) for k, v in env.items()})
    log_f = None
    try:
        stdout: Any = subprocess.PIPE
        stderr: Any = subprocess.STDOUT
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = log_path.open("a", encoding="utf-8", errors="replace")
            stdout = log_f
            stderr = subprocess.STDOUT
        proc = popen_session(
            list(cmd),
            env=merged,
            cwd=str(cwd) if cwd is not None else None,
            stdout=stdout,
            stderr=stderr,
        )
        timed_out = False
        try:
            rc = proc.wait(timeout=float(timeout_sec))
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_tree(proc, grace_s=grace_s)
            kill_process_tree(proc)
            rc = 124
        else:
            if rc != 0:
                terminate_process_tree(proc, grace_s=min(grace_s, 5.0))
        if timed_out:
            sig: str | None = "TIMEOUT"
        else:
            sig = native_signal_name(int(rc))
        return EpisodeResult(returncode=int(rc), timed_out=timed_out, signal_name=sig)
    finally:
        if log_f is not None:
            log_f.close()


def write_crash_marker(
    out_dir: Path | str,
    arm: str,
    qid: str | int,
    *,
    returncode: int,
    signal_name: str | None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    out = Path(out_dir)
    path = out / f"{arm}_q{qid}.CRASH"
    payload: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arm": arm,
        "question_id": str(qid),
        "exit_code": int(returncode),
        "signal": signal_name,
    }
    if extra:
        payload.update(dict(extra))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def count_scored_units(out_dir: Path | str, arms: Sequence[str], qids: Sequence[str | int]) -> int:
    out = Path(out_dir)
    n = 0
    for arm in arms:
        for qid in qids:
            ep = out / f"{arm}_q{qid}.jsonl"
            if ep.is_file() and ep.stat().st_size > 0:
                n += 1
    return n


def count_crash_markers(out_dir: Path | str) -> int:
    return len(list(Path(out_dir).glob("*_q*.CRASH")))


def update_eval_progress(
    out_dir: Path | str,
    *,
    units_done: int,
    units_total: int,
    phase: str,
    current_id: str,
    units_failed: int | None = None,
) -> Path:
    meta: dict[str, Any] = {
        "units_done": int(units_done),
        "units_total": int(units_total),
        "phase": phase,
        "current_id": str(current_id),
    }
    if units_failed is not None:
        meta["units_failed"] = int(units_failed)
    return write_progress_file(out_dir, **meta)


def detect_host_freeze(out_dir: Path | str) -> dict[str, Any] | None:
    """If progress is mid-episode with empty jsonl (post-reboot), return a summary."""
    out = Path(out_dir)
    progress_path = out / "progress.json"
    if not progress_path.is_file():
        return None
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    phase = str(progress.get("phase") or "")
    qid = str(progress.get("current_id") or "")
    if phase not in ("classic", "agentic") or not qid or qid == "-":
        return None
    ep = out / f"{phase}_q{qid}.jsonl"
    if not ep.exists() or ep.stat().st_size > 0:
        return None
    elog = out / f"{phase}.log"
    trailing_nuls = 0
    if elog.is_file():
        data = elog.read_bytes()
        i = len(data) - 1
        while i >= 0 and data[i] == 0:
            i -= 1
        trailing_nuls = len(data) - 1 - i
    return {
        "arm": phase,
        "question_id": qid,
        "empty_jsonl": str(ep),
        "log_trailing_nuls": trailing_nuls,
        "progress": progress,
    }


def write_host_freeze_capsule(out_dir: Path | str, info: Mapping[str, Any]) -> Path:
    out = Path(out_dir)
    arm = info["arm"]
    qid = info["question_id"]
    capsule = out / f"host_freeze_{arm}_q{qid}.log"
    if capsule.exists():
        return capsule
    elog = out / f"{arm}.log"
    tail = ""
    if elog.is_file():
        data = elog.read_bytes()
        i = len(data) - 1
        while i >= 0 and data[i] == 0:
            i -= 1
        tail = data[max(0, i - 4000) : i + 1].decode("utf-8", "replace")
    capsule.write_text(
        "\n".join(
            [
                f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
                "kind=host-freeze-or-hard-reboot",
                f"arm={arm}",
                f"question_id={qid}",
                f"log_trailing_nuls={info.get('log_trailing_nuls')}",
                "",
                "----- progress.json -----",
                json.dumps(info.get("progress") or {}, indent=2),
                "",
                "----- episode log tail -----",
                tail,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return capsule


def resolve_hmeqa_out(explicit: str | Path | None = None) -> Path:
    """Resolve HM-EQA OUT dir from arg, EMET_HMEQA_OUT, or status latest symlink."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_out = os.environ.get("EMET_HMEQA_OUT", "").strip()
    if env_out:
        return Path(env_out).expanduser().resolve()
    repo = Path.cwd().name
    latest = Path.home() / "runs" / "emet" / "status" / repo / "latest"
    if latest.is_symlink() or latest.is_file():
        try:
            return latest.resolve()
        except OSError:
            pass
    runs = Path.home() / "runs" / "emet"
    if runs.is_dir():
        cands = sorted(
            [p for p in runs.glob("hmeqa_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if cands:
            return cands[0].resolve()
    raise FileNotFoundError(
        "Could not resolve HM-EQA OUT; pass an explicit path or set EMET_HMEQA_OUT"
    )


def affinity_summary_dict() -> dict[str, Any]:
    mhz = float(os.environ.get("EMET_EXCLUDE_CPU_MIN_MHZ", "6000") or "6000")
    turbo = cpus_at_or_above_mhz(mhz) if mhz > 0 else []
    kept = safe_cpu_ids(exclude_min_mhz=mhz if mhz > 0 else None)
    return {
        "exclude_min_mhz": mhz,
        "online": online_cpu_ids(),
        "turbo_cpus": turbo,
        "kept": kept,
        "taskset": format_taskset_list(kept),
    }


def settle_after_crash(
    seconds: float,
    *,
    log: Callable[[str], None] | None = None,
    sync_fs: bool = True,
) -> None:
    if sync_fs:
        try:
            os.sync()  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            subprocess.run(["sync"], check=False)
    sec = max(0.0, float(seconds))
    if sec <= 0:
        return
    if log:
        log(f"native-crash settle {sec:.0f}s")
    time.sleep(sec)


DEFAULT_BAL32_IDS = (
    "2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,"
    "32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84"
)

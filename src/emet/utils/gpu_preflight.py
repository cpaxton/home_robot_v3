# Copyright (c) Chris Paxton 2026

"""GPU preflight helpers for paper evals / overnight smokes.

Canonical implementation for ``emet eval {status,check,wait,kill-stale}``.
``scripts/gpu_preflight.sh`` delegates to that CLI when possible.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_NEED_MIB = 12000
DEFAULT_STABLE_CHECKS = 3
DEFAULT_WAIT_INTERVAL_S = 30.0
DEFAULT_SETTLE_S = 15.0

# Patterns matched against full process command lines (pgrep -f / ps args).
STALE_EVAL_PATTERNS: tuple[str, ...] = (
    r"emet\.simulation\.mujoco_server",
    r"emet\.habitat\.(zmq_server|habitat_subprocess)|emet-habitat|packages/emet_habitat",
    r"eval_dynamic_exploration\.py|run_dynagraph_dynamic_",
    r"emet run (dynagraph|graph-eqa|dynamem|agent|scene-graph)",
    r"uv run emet (run |test |serve )",
    r"emet sqa3d run",
    r"python[0-9.]* -m emet\.app\.run_(dynagraph|dynamem|agent)",
)

_GPU_CMD_HINT = re.compile(r"(home_robot|emet|habitat)", re.IGNORECASE)


@dataclass(frozen=True)
class GpuMemoryInfo:
    free_mib: int
    total_mib: int | None


@dataclass(frozen=True)
class ComputeApp:
    pid: int
    process_name: str
    used_memory: str


def env_need_mib(default: int = DEFAULT_NEED_MIB) -> int:
    raw = os.environ.get("NEED_MIB", "").strip()
    if not raw:
        return int(default)
    return int(raw)


def env_settle_s(default: float = DEFAULT_SETTLE_S) -> float:
    raw = os.environ.get("GPU_SETTLE_SEC", "").strip()
    if not raw:
        return float(default)
    return float(raw)


def env_stable_checks(default: int = DEFAULT_STABLE_CHECKS) -> int:
    raw = os.environ.get("GPU_STABLE_CHECKS", "").strip()
    if not raw:
        return int(default)
    return int(raw)


def env_wait_interval_s(default: float = DEFAULT_WAIT_INTERVAL_S) -> float:
    raw = os.environ.get("GPU_WAIT_INTERVAL", "").strip()
    if not raw:
        return float(default)
    return float(raw)


def _run_nvidia_smi(args: Sequence[str]) -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip()


def gpu_memory_info() -> GpuMemoryInfo | None:
    """Return free/total MiB for GPU 0, or None if nvidia-smi unavailable."""
    out = _run_nvidia_smi(
        [
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return None
    line = out.splitlines()[0].strip()
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 1:
        return None
    try:
        free = int(parts[0])
    except ValueError:
        return None
    total: int | None = None
    if len(parts) >= 2:
        try:
            total = int(parts[1])
        except ValueError:
            total = None
    return GpuMemoryInfo(free_mib=free, total_mib=total)


def gpu_free_mib() -> int:
    info = gpu_memory_info()
    return 0 if info is None else int(info.free_mib)


def list_compute_apps() -> list[ComputeApp]:
    out = _run_nvidia_smi(
        [
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    if not out:
        return []
    apps: list[ComputeApp] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        name = parts[1] if len(parts) > 1 else ""
        mem = parts[2] if len(parts) > 2 else ""
        apps.append(ComputeApp(pid=pid, process_name=name, used_memory=mem))
    return apps


def protected_pids(*, extra: Iterable[int] | None = None) -> set[int]:
    """PIDs that ``kill_stale`` must never signal (caller + ancestors + env extras)."""
    out: set[int] = set()
    pid = os.getpid()
    for _ in range(64):
        if pid <= 1 or pid in out:
            break
        out.add(pid)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
                # pid (comm) state ppid ...
                body = f.read()
            close = body.rfind(")")
            fields = body[close + 2 :].split()
            pid = int(fields[1])  # ppid
        except (OSError, ValueError, IndexError):
            break
    for raw in os.environ.get("EMET_GPU_PROTECT_PIDS", "").split():
        try:
            out.add(int(raw))
        except ValueError:
            continue
    if extra is not None:
        out.update(int(p) for p in extra)
    return out


def _process_args(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
        if raw:
            return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        pass
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return (proc.stdout or "").strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _pgrep_f(pattern: str) -> list[int]:
    try:
        proc = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    pids: list[int] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def _signal_pid(pid: int, sig: signal.Signals, log: Callable[[str], None]) -> None:
    cmd = _process_args(pid)
    label = "SIGTERM" if sig == signal.SIGTERM else "SIGKILL"
    log(f"[gpu] {label} pid={pid}: {cmd[:120]}")
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def kill_matching_pids(
    pids: Sequence[int],
    *,
    protected: set[int],
    log: Callable[[str], None] | None = None,
    escalate_s: float = 2.0,
) -> int:
    """SIGTERM then SIGKILL listed PIDs (skip protected). Returns number signaled."""
    _log = log or (lambda _m: None)
    targets = [int(p) for p in pids if int(p) not in protected]
    if not targets:
        return 0
    for pid in targets:
        _signal_pid(pid, signal.SIGTERM, _log)
    if escalate_s > 0:
        time.sleep(float(escalate_s))
    for pid in targets:
        if pid not in protected and _pid_alive(pid):
            _signal_pid(pid, signal.SIGKILL, _log)
    return len(targets)


def kill_stale_eval_processes(
    *,
    kill_gpu_apps: bool = True,
    settle_s: float | None = None,
    patterns: Sequence[str] = STALE_EVAL_PATTERNS,
    protected: set[int] | None = None,
    log: Callable[[str], None] | None = None,
    escalate_s: float = 2.0,
    sleep_fn: Callable[[float], None] | None = None,
) -> int:
    """Kill stale eval/sim trees. Returns approximate number of PIDs signaled."""
    _log = log or (lambda m: print(m, flush=True))
    _sleep = sleep_fn or time.sleep
    prot = protected if protected is not None else protected_pids()
    signaled = 0
    for pattern in patterns:
        pids = _pgrep_f(pattern)
        signaled += kill_matching_pids(pids, protected=prot, log=_log, escalate_s=escalate_s)
    settle = env_settle_s() if settle_s is None else float(settle_s)
    if settle > 0:
        _sleep(settle)
    if kill_gpu_apps:
        uniq: list[int] = []
        seen: set[int] = set()
        for app in list_compute_apps():
            cmd = _process_args(app.pid)
            if not (_GPU_CMD_HINT.search(cmd) or _GPU_CMD_HINT.search(app.process_name)):
                continue
            if app.pid in seen or app.pid in prot:
                continue
            seen.add(app.pid)
            uniq.append(app.pid)
        if uniq:
            for pid in uniq:
                _signal_pid(pid, signal.SIGTERM, _log)
                signaled += 1
            _sleep(5.0 if settle >= 5 else max(0.0, settle))
            for pid in uniq:
                if _pid_alive(pid):
                    _signal_pid(pid, signal.SIGKILL, _log)
            if settle > 0:
                _sleep(min(2.0, settle))
    return signaled


def format_status_lines() -> list[str]:
    info = gpu_memory_info()
    lines: list[str] = []
    if info is None:
        lines.append("GPU: nvidia-smi unavailable (free=0)")
    else:
        total = "?" if info.total_mib is None else str(info.total_mib)
        lines.append(f"GPU: {info.free_mib} MiB free / {total} MiB total")
    apps = list_compute_apps()
    if apps:
        lines.append("Compute apps:")
        for app in apps:
            lines.append(f"  pid={app.pid} {app.process_name} {app.used_memory}".rstrip())
    else:
        lines.append("Compute apps: (none)")
    return lines


# Magnum/Habitat-Sim headless failure (seen when CUDA↔EGL device map is broken).
HABITAT_EGL_FAIL_PATTERNS: tuple[str, ...] = (
    "unable to find CUDA device",
    "WindowlessContext: Unable to create windowless context",
)


def habitat_egl_error_in_text(text: str) -> bool:
    """True if log text looks like Habitat-Sim EGL/CUDA device-map failure."""
    if not text:
        return False
    lower = text.lower()
    return any(p.lower() in lower for p in HABITAT_EGL_FAIL_PATTERNS)


def recent_emet_segfault_hint() -> str | None:
    """Best-effort dmesg scan for ``emet`` / python segfaults (may need privileges)."""
    try:
        proc = subprocess.run(
            ["dmesg", "--ctime", "--color=never"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    blob = (proc.stdout or "") + (proc.stderr or "")
    if not blob.strip():
        return None
    hits = [
        ln.strip()
        for ln in blob.splitlines()
        if re.search(r"segfault|invalid opcode", ln, re.IGNORECASE) and re.search(r"\b(emet|python)", ln, re.IGNORECASE)
    ]
    if not hits:
        return None
    return hits[-1]


def diagnose_eval_environment(
    *,
    repo_root: str | None = None,
) -> tuple[bool, list[str]]:
    """Read-only preflight for Habitat/HM-EQA agents (no sim import).

    Returns ``(ok, lines)``. ``ok`` is False when NVML is missing, CUDA is hidden,
    or a recent ``emet`` segfault hint is present — empty nvidia-smi apps alone
    still yields ok=True with an explicit warning (EGL can still be broken).
    """
    lines = list(format_status_lines())
    ok = True
    root = repo_root or os.environ.get("EMET_REPO_ROOT", "").strip() or os.getcwd()
    hab = os.path.join(root, ".venv-habitat", "bin", "emet-habitat")
    if os.path.isfile(hab) and os.access(hab, os.X_OK):
        lines.append(f"Habitat wrapper: {hab}")
    else:
        ok = False
        lines.append(f"ERROR: missing executable Habitat wrapper at {hab}")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        lines.append("CUDA_VISIBLE_DEVICES: (unset)")
    else:
        lines.append(f"CUDA_VISIBLE_DEVICES={cvd!r}")
        if cvd.strip() == "":
            ok = False
            lines.append(
                "ERROR: CUDA_VISIBLE_DEVICES is empty — Torch/Habitat see no GPU; "
                "HM-EQA will fail even if nvidia-smi looks fine."
            )
        else:
            lines.append(
                "WARN: CUDA_VISIBLE_DEVICES remaps CUDA indices; Magnum EGL may still "
                "enumerate all devices → 'unable to find CUDA device 0 among N EGL devices'."
            )

    display = os.environ.get("DISPLAY")
    lines.append(f"DISPLAY={display!r}" if display is not None else "DISPLAY: (unset)")

    apps = list_compute_apps()
    if not apps:
        lines.append(
            "NOTE: empty nvidia-smi compute apps ≠ Habitat EGL healthy. "
            "Morning HM-EQA failsets have failed with WindowlessContext/"
            "'unable to find CUDA device 0' while VRAM looked free."
        )

    seg = recent_emet_segfault_hint()
    if seg:
        ok = False
        lines.append(f"ERROR: recent kernel segfault hint: {seg}")
        lines.append(
            "Cursor agent sessions die when ``emet``/Habitat-Sim segfaults in the "
            "agent tool process tree. Prefer ``emet jobs run``; after a crash use "
            "``emet jobs`` / ``~/runs/emet/`` — do not hard-kill Habitat mid-episode."
        )
    else:
        lines.append(
            "Segfault scan: no recent emet/python segfault in dmesg (or dmesg unavailable without privileges)."
        )

    lines.append(
        "Agent rules: never run Habitat/VLM as blocking Cursor commands; "
        "launch with ``emet jobs run``; cancel with ``emet jobs cancel`` "
        "(not raw kill); do not ``kill-stale`` while a managed job is starting."
    )
    return ok, lines


def check_gpu_memory(need_mib: int | None = None) -> tuple[bool, str]:
    """Return (ok, message). ok False if nvidia-smi missing or free < need."""
    need = env_need_mib() if need_mib is None else int(need_mib)
    info = gpu_memory_info()
    if info is None:
        return False, "ERROR: nvidia-smi not found or failed"
    msg = f"GPU: {info.free_mib} MiB free / {info.total_mib} MiB total (need >= {need} MiB)"
    if info.free_mib < need:
        return False, msg + "\nERROR: insufficient free GPU memory"
    return True, msg


def wait_gpu_stable(
    need_mib: int | None = None,
    *,
    stable_checks: int | None = None,
    interval_s: float | None = None,
    log: Callable[[str], None] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    max_rounds: int | None = None,
) -> bool:
    """Wait until free VRAM stays >= need for ``stable_checks`` consecutive reads."""
    need = env_need_mib() if need_mib is None else int(need_mib)
    checks = env_stable_checks() if stable_checks is None else int(stable_checks)
    interval = env_wait_interval_s() if interval_s is None else float(interval_s)
    _log = log or (lambda m: print(m, flush=True))
    _sleep = sleep_fn or time.sleep
    ok = 0
    rounds = 0
    while ok < checks:
        if max_rounds is not None and rounds >= max_rounds:
            return False
        free = gpu_free_mib()
        if free >= need:
            ok += 1
        else:
            ok = 0
        _log(f"[gpu] free={free}MiB need={need} stable={ok}/{checks}")
        if ok >= checks:
            return True
        rounds += 1
        _sleep(interval)
    return False


# --- disk preflight (episode debug bundles) --------------------------------


def _episodes_root() -> Path | None:
    try:
        from emet.habitat.episode_debug import default_episodes_root

        return Path(default_episodes_root())
    except Exception:
        return None


def _dir_gb(path: Path) -> float:
    """Directory size in GiB (shallow recursive)."""
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except Exception:
        return 0.0
    return total / (1024**3)


def disk_free_gb(path: Path | str) -> float:
    """Free space (GiB) on the filesystem containing *path*."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True) if not p.exists() else None
    try:
        shutil.disk_usage(p)
    except Exception:
        return 0.0
    return shutil.disk_usage(p).free / (1024**3)


def disk_status_lines(episodes_root: Path | None = None) -> list[str]:
    """Free-space + episode-bundle summary lines for ``emet eval status``."""
    root = episodes_root or _episodes_root()
    lines: list[str] = []
    if root is not None:
        free = disk_free_gb(root)
        lines.append(f"disk free under {root}: {free:.1f} GB")
        if root.is_dir():
            bundles = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
            total = sum(_dir_gb(p) for p in bundles)
            lines.append(f"episode bundles: {len(bundles)} dirs, {total:.1f} GB (clean with: emet eval clean-bundles)")
    return lines


def clean_episode_bundles(
    keep: int = 2,
    max_age_days: float = 0.0,
    *,
    apply: bool = False,
    root: Path | None = None,
    protect_newer_than_h: float = 12.0,
) -> list[str]:
    """Retention-prune episode debug bundles under ``~/.cache/habitat_eqa/episodes``.

    ``keep`` count-prunes only old RUNS (a run = a sweep prefix such as
    ``subset_paper113_20260813_104004_dynagraph``; per-qid H2H bundles sharing one
    run tag group together). Runs whose newest mtime is within
    ``protect_newer_than_h`` hours are never count-pruned (they are the active or
    just-finished sweep). ``max_age_days`` removes anything older than N days.
    Returns human lines; ``apply=False`` is a dry run. Never touches results/*.jsonl.
    """
    import re as _re

    root = root or _episodes_root()
    if root is None or not root.is_dir():
        return [f"no episode bundles under {root}"]
    dirs = sorted(
        (p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=lambda p: p.stat().st_mtime,
    )

    def _run_tag(name: str) -> str:
        # Sweep bundles: ``subset_paper113_20260813_104004_dynagraph_qwen3_vl``.
        m = _re.search(r"(subset_\w+_\d{8}_\d{6})", name)
        if m:
            return m.group(1)
        # Per-qid H2H bundles share one run tag, e.g.
        # ``h2h_agentic_q0012_gre_a2_grounded_fae4b89c_20260814_160721``.
        m = _re.search(r"(h2h_[a-z0-9]+_q\d{4}_\S+?_\d{8}_\d{6})", name)
        if m:
            return m.group(1)
        # Anything else (cli_episode_q*, ad-hoc) is unique — age-pruned only.
        return name

    now = time.time()
    by_run: dict[str, list[Path]] = {}
    is_run: dict[Path, bool] = {}
    for p in dirs:
        tag = _run_tag(p.name)
        by_run.setdefault(tag, []).append(p)
        is_run[p] = tag != p.name  # False => unique/ad-hoc, age-pruned only

    runs = sorted(
        by_run.values(),
        key=lambda ps: max(p.stat().st_mtime for p in ps),
        reverse=True,
    )

    doomed: list[tuple[Path, str]] = []
    for ps in runs:
        newest = max(p.stat().st_mtime for p in ps)
        # Unique/ad-hoc bundles are never count-pruned (age-only).
        if not any(is_run[p] for p in ps):
            if max_age_days > 0 and (now - newest) / 86400.0 > max_age_days:
                for p in ps:
                    doomed.append((p, f"older than {max_age_days:.0f}d"))
            continue
        protected = (now - newest) < protect_newer_than_h * 3600.0
        if max_age_days > 0 and (now - newest) / 86400.0 > max_age_days:
            for p in ps:
                doomed.append((p, f"older than {max_age_days:.0f}d"))
            continue
        if protected:
            continue
        unprotected = [r for r in runs if (now - max(p.stat().st_mtime for p in r)) / 3600.0 >= protect_newer_than_h]
        rank = unprotected.index(ps) if ps in unprotected else -1
        if rank >= keep:
            for p in ps:
                doomed.append((p, f"beyond keep={keep} runs"))

    out: list[str] = []
    freed = 0.0
    for p, why in doomed:
        sz = _dir_gb(p)
        freed += sz
        if apply:
            out.append(f"DELETE {sz:7.2f} GB  {p.name}  ({why})")
            shutil.rmtree(p, ignore_errors=True)
        else:
            out.append(f"would   {sz:7.2f} GB  {p.name}  ({why})")
    out.append(
        f"freed: {freed:.2f} GB ({len(doomed)} bundles) [{'APPLIED' if apply else 'dry-run; use --apply to delete'}]"
    )
    out.append(f"kept {len(dirs) - len(doomed)} bundles; results/*.jsonl untouched.")
    return out

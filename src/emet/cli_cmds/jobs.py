# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import click

from emet.cli_cmds.bootstrap import (
    _project_root,
)


@click.group(
    "jobs",
    invoke_without_command=True,
    short_help="List and manage queued/running eval experiments",
)
@click.pass_context
def jobs_group(ctx: click.Context) -> None:
    """Track paper evals / overnight smokes (registry + process scan).

    Queue scripts register under ``~/runs/emet/jobs/`` (``EMET_JOBS_DIR``).

    \b
    Examples:
      emet jobs
      emet jobs list --all
      emet jobs report
      emet jobs status JOB_ID
      emet jobs cancel JOB_ID
      emet jobs logs JOB_ID --tail 50
      emet jobs run --name eqa-smoke -- ./scripts/run_….sh OUT
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(jobs_list, show_all=False, as_json=False, scan=True)


@jobs_group.command("list", short_help="List registered jobs (default: non-terminal)")
@click.option("--all", "show_all", is_flag=True, help="Include done/failed/cancelled.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON array.")
@click.option(
    "--scan/--no-scan",
    default=True,
    show_default=True,
    help="Also show unmanaged eval processes from pgrep.",
)
def jobs_list(show_all: bool, as_json: bool, scan: bool) -> None:
    """List jobs from the registry; optionally scan for unmanaged eval PIDs."""
    from emet.utils.job_registry import (
        format_job_header,
        format_job_row,
        format_scanned_header,
        format_scanned_row,
        list_jobs,
        scan_eval_processes,
    )

    jobs = list_jobs(include_terminal=show_all)
    if as_json:
        payload: dict[str, Any] = {"jobs": [j.to_dict() for j in jobs]}
        if scan:
            payload["unmanaged"] = [
                {"pid": s.pid, "cmd": s.cmd, "pattern": s.matched_pattern}
                for s in scan_eval_processes()
                if s.pid not in {j.pid for j in jobs if j.pid is not None}
            ]
        click.echo(json.dumps(payload, indent=2))
        return

    if not jobs:
        click.echo("(no registered jobs)" if show_all else "(no active registered jobs)")
    else:
        click.echo(format_job_header())
        for job in jobs:
            click.echo(format_job_row(job))

    if scan:
        registered_pids = {j.pid for j in jobs if j.pid is not None}
        unmanaged = [s for s in scan_eval_processes() if s.pid not in registered_pids]
        if unmanaged:
            click.echo("")
            click.echo(f"Unmanaged eval processes ({len(unmanaged)}, not in registry):")
            click.echo(format_scanned_header())
            for s in unmanaged[:40]:
                click.echo(format_scanned_row(s))
            if len(unmanaged) > 40:
                click.echo(f"  … {len(unmanaged) - 40} more")


@jobs_group.command("status", short_help="Show one job record")
@click.argument("job_id")
@click.option("--json", "as_json", is_flag=True, help="Emit full JSON record.")
def jobs_status(job_id: str, as_json: bool) -> None:
    from emet.utils.job_registry import (
        compute_job_progress,
        format_job_detail,
        format_progress_brief,
        load_job,
        refresh_job_liveness,
    )

    job = load_job(job_id)
    if job is None:
        click.echo(f"unknown job: {job_id}", err=True)
        sys.exit(1)
    job = refresh_job_liveness(job)
    if as_json:
        payload = job.to_dict()
        prog = compute_job_progress(job)
        payload["progress"] = {
            "units_done": prog.units_done,
            "units_total": prog.units_total,
            "phase": prog.phase,
            "current_id": prog.current_id,
            "elapsed_s": prog.elapsed_s,
            "rate_s_per_unit": prog.rate_s_per_unit,
            "eta_s": prog.eta_s,
            "source": prog.source,
            "brief": format_progress_brief(prog),
        }
        click.echo(json.dumps(payload, indent=2))
    else:
        click.echo(format_job_detail(job))


@jobs_group.command(
    "report",
    short_help="Progress + per-episode scores (defaults to running job)",
)
@click.argument("job_id", required=False, default=None)
@click.option(
    "--question",
    "-q",
    "question_id",
    type=int,
    default=None,
    help="Deep-dive one question id: episode row + agentic trace (rooms, router, verify, flags).",
)
@click.option("--arm", default=None, help="Restrict --question to one arm (classic/agentic).")
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    default=None,
    help="Report this OUT dir directly (no job id / registry lookup).",
)
@click.option(
    "--rooms",
    "rooms_focus",
    is_flag=True,
    help="With --question: focus on room timeline (summary + rooms + flags).",
)
@click.option(
    "--section",
    "-s",
    "sections",
    multiple=True,
    help="With --question: include sections (summary,rooms,router,nav,assess,verify,flags). Repeatable.",
)
@click.option("--verbose", "-v", is_flag=True, help="With --question: full assess reasons + per-turn Rooms lines.")
@click.option("--brief", is_flag=True, help="With --question: summary + rooms + router + flags only.")
@click.option("--fail-only", is_flag=True, help="Scorecard: list incorrect episodes only.")
@click.option("--json", "as_json", is_flag=True, help="Emit structured JSON.")
def jobs_report(
    job_id: str | None,
    question_id: int | None,
    arm: str | None,
    out_dir: str | None,
    rooms_focus: bool,
    sections: tuple[str, ...],
    verbose: bool,
    brief: bool,
    fail_only: bool,
    as_json: bool,
) -> None:
    """Scorecard for an H2H / eval OUT dir.

    With no JOB_ID, picks the active running/waiting job (most recent if several),
    else the newest finished job with an ``out_dir``. Pass ``--question ID`` for a
    per-episode trace analysis. Use ``--rooms`` to audit graph/VLM room names.
    """
    from emet.utils.job_registry import (
        format_job_report,
        format_question_report,
        job_record_for_out_dir,
        job_report_dict,
        question_report_dict,
        resolve_report_job,
    )

    if out_dir:
        job = job_record_for_out_dir(out_dir)
    else:
        job = resolve_report_job(job_id)
    if job is None:
        if job_id:
            click.echo(f"unknown job: {job_id}", err=True)
        else:
            click.echo("no job to report (registry empty; try --out-dir)", err=True)
        sys.exit(1)
    if question_id is not None:
        if as_json:
            click.echo(json.dumps(question_report_dict(job, question_id, arm=arm), indent=2))
        else:
            click.echo(
                format_question_report(
                    job,
                    question_id,
                    arm=arm,
                    sections=list(sections) if sections else None,
                    rooms_focus=rooms_focus,
                    verbose=verbose,
                    brief=brief,
                )
            )
        return
    if as_json:
        click.echo(json.dumps(job_report_dict(job), indent=2))
    else:
        click.echo(format_job_report(job, fail_only=fail_only))


@jobs_group.command("cancel", short_help="Cancel a registered job (kill process tree)")
@click.argument("job_id")
@click.option("--grace-sec", type=float, default=10.0, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Emit full JSON record.")
def jobs_cancel(job_id: str, grace_sec: float, as_json: bool) -> None:
    from emet.utils.job_registry import (
        cancel_job,
        format_job_detail,
        scan_eval_processes,
    )

    try:
        job = cancel_job(job_id, grace_s=grace_sec)
    except KeyError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    if as_json:
        click.echo(json.dumps(job.to_dict(), indent=2))
    else:
        click.echo(f"cancelled {job.id}")
        click.echo(format_job_detail(job))
        # Habitat grandchildren can briefly outlive the wrapper PID.
        leftovers = scan_eval_processes()
        if leftovers:
            click.echo(
                "WARNING: unmanaged eval processes still visible "
                f"({len(leftovers)}). Re-check with `emet jobs` / "
                "`emet eval status`; use `emet eval kill-stale` only if "
                "nothing intentional is live.",
                err=True,
            )
        if job.out_dir:
            out = Path(job.out_dir)
            # Overnight OUT is …/base/bal32; resume the overnight base when possible.
            base = out.parent if out.name in {"bal32", "holdout8"} else out
            click.echo(
                f"resume hint: uv run emet hmeqa overnight --base {base} --job-name {job.name or 'hmeqa-overnight'}"
                if (base / "gate.json").is_file() or (base / "holdout8").is_dir()
                else f"resume hint: uv run emet hmeqa resume {out} --preset paper-router",
                err=True,
            )


@jobs_group.command("logs", short_help="Tail a job log (log_path or out_dir/*.log)")
@click.argument("job_id")
@click.option("--tail", "n_tail", type=int, default=40, show_default=True)
def jobs_logs(job_id: str, n_tail: int) -> None:
    from emet.utils.job_registry import load_job

    job = load_job(job_id)
    if job is None:
        click.echo(f"unknown job: {job_id}", err=True)
        sys.exit(1)
    candidates: list[Path] = []
    if job.log_path:
        candidates.append(Path(job.log_path))
    if job.out_dir:
        od = Path(job.out_dir)
        for name in (
            "queue.log",
            "orchestrator.log",
            "nohup.log",
            "phase1_smoke.log",
            "eqa_smoke.log",
        ):
            candidates.append(od / name)
        candidates.extend(sorted(od.glob("*.log")))
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        click.echo("no log file found for job", err=True)
        sys.exit(1)
    click.echo(f"# {path}", err=True)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-max(1, n_tail) :]:
        click.echo(line)


@jobs_group.command("register", short_help="Register a job (for scripts)")
@click.option("--name", required=True, help="Short job name.")
@click.option("--job-id", default=None, hidden=True)
@click.option(
    "--description",
    "-d",
    default=None,
    help="Human why/what for this run (shown in emet jobs list + status).",
)
@click.option("--cmd", default="", help="Command summary.")
@click.option("--out-dir", type=click.Path(), default=None)
@click.option("--log-path", type=click.Path(), default=None)
@click.option("--repo", type=click.Path(), default=None)
@click.option("--wait-pid", multiple=True, type=int, help="PIDs to wait on (repeatable).")
@click.option("--pid", type=int, default=None, help="Controller PID if already running.")
@click.option(
    "--status",
    type=click.Choice(["queued", "waiting", "running", "done", "failed", "cancelled"]),
    default="queued",
)
def jobs_register(
    name: str,
    job_id: str | None,
    description: str | None,
    cmd: str,
    out_dir: str | None,
    log_path: str | None,
    repo: str | None,
    wait_pid: tuple[int, ...],
    pid: int | None,
    status: str,
) -> None:
    """Print the new job id on stdout (scripts should capture it)."""
    from emet.utils.job_registry import register_job

    job = register_job(
        name=name,
        cmd=cmd,
        out_dir=out_dir,
        log_path=log_path,
        repo=repo or str(_project_root()),
        wait_pids=list(wait_pid),
        pid=pid,
        status=status,  # type: ignore[arg-type]
        description=description,
        job_id=job_id,
    )
    click.echo(job.id)


@jobs_group.command("update", short_help="Update job status / pid / progress (for scripts)")
@click.argument("job_id")
@click.option(
    "--status",
    type=click.Choice(["queued", "waiting", "running", "done", "failed", "cancelled"]),
    default=None,
)
@click.option("--pid", type=int, default=None)
@click.option("--cmd", default=None)
@click.option("--out-dir", type=click.Path(), default=None)
@click.option("--log-path", type=click.Path(), default=None)
@click.option("--error", default=None)
@click.option("--units-done", type=int, default=None, help="Completed work units (for ETA).")
@click.option("--units-total", type=int, default=None, help="Total work units (for ETA).")
@click.option("--phase", default=None, help="Current phase label (e.g. classic, agentic).")
@click.option("--current-id", default=None, help="Current unit id (e.g. question id).")
@click.option(
    "--description",
    "-d",
    default=None,
    help="Set/replace human why/what (shown in emet jobs list + status).",
)
@click.option(
    "--meta",
    multiple=True,
    help="Extra meta KEY=VALUE (repeatable). Progress keys also accepted here.",
)
def jobs_update(
    job_id: str,
    status: str | None,
    pid: int | None,
    cmd: str | None,
    out_dir: str | None,
    log_path: str | None,
    error: str | None,
    units_done: int | None,
    units_total: int | None,
    phase: str | None,
    current_id: str | None,
    description: str | None,
    meta: tuple[str, ...],
) -> None:
    from emet.utils.job_registry import update_job

    meta_update: dict = {}
    for item in meta:
        if "=" not in item:
            click.echo(f"ignore --meta {item!r} (want KEY=VALUE)", err=True)
            continue
        k, v = item.split("=", 1)
        meta_update[k.strip()] = v.strip()

    try:
        job = update_job(
            job_id,
            status=status,  # type: ignore[arg-type]
            pid=pid,
            cmd=cmd,
            out_dir=out_dir,
            log_path=log_path,
            error=error,
            meta_update=meta_update or None,
            units_done=units_done,
            units_total=units_total,
            phase=phase,
            current_id=current_id,
            description=description,
        )
    except KeyError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    click.echo(json.dumps(job.to_dict(), indent=2))


@jobs_group.command(
    "run",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    short_help="Start a self-registering detached job supervisor",
)
@click.option("--name", required=True, help="Short job name.")
@click.option(
    "--description",
    "-d",
    default=None,
    help="Human why/what for this run (shown in emet jobs list + status).",
)
@click.option("--out-dir", type=click.Path(), default=None, help="Artifact directory.")
@click.option(
    "--wait-pid",
    multiple=True,
    type=int,
    help="Wait for these PIDs before starting (repeatable).",
)
@click.option(
    "--wait-timeout-sec",
    type=click.FloatRange(min=0.0),
    default=21600.0,
    show_default=True,
    help="Total timeout for explicit --wait-pid prerequisites.",
)
@click.option(
    "--need-mib",
    type=int,
    default=None,
    help="If set, run emet eval wait before the command.",
)
@click.option(
    "--gpu-wait-max-rounds",
    type=click.IntRange(min=1),
    default=120,
    show_default=True,
    help="Maximum free-VRAM checks before the job fails.",
)
@click.option(
    "--cpu-safe/--no-cpu-safe",
    default=None,
    help="Pin job away from turbo P-cores (default: on when --need-mib is set).",
)
@click.option(
    "--gpu-exclusive/--no-gpu-exclusive",
    default=None,
    help="Hold the host-wide GPU lock (default: on for --need-mib or GPU-like commands).",
)
@click.option(
    "--lock-timeout-sec",
    type=click.FloatRange(min=0.0),
    default=None,
    help="Exclusive-lock timeout (default: EMET_GPU_LOCK_TIMEOUT or 21600).",
)
@click.option("--foreground", is_flag=True, help="Run in foreground (no nohup).")
@click.pass_context
def jobs_run(
    ctx: click.Context,
    name: str,
    description: str | None,
    out_dir: str | None,
    wait_pid: tuple[int, ...],
    wait_timeout_sec: float,
    need_mib: int | None,
    gpu_wait_max_rounds: int,
    cpu_safe: bool | None,
    gpu_exclusive: bool | None,
    lock_timeout_sec: float | None,
    foreground: bool,
) -> None:
    """Start a detached supervisor that registers and runs a managed job.

    GPU-like commands and jobs with ``--need-mib`` hold a host-wide ``flock``
    for their full lifetime, closing the launch race between separate checkouts.

    \b
    Example:
      emet jobs run --name improve-eqa -d "owlv2 + no confirm gate" --need-mib 14000 -- \\
        ./scripts/run_dynagraph_dynamic_improve_smokes.sh OUT
    """
    import shlex

    from emet.utils.job_registry import (
        command_looks_like_gpu_job,
        gpu_lock_path,
        load_job,
        new_job_id,
    )

    cmd_args = list(ctx.args)
    if cmd_args and cmd_args[0] == "--":
        cmd_args = cmd_args[1:]
    if not cmd_args:
        click.echo("usage: emet jobs run --name NAME [--description TEXT] -- CMD [ARGS…]", err=True)
        sys.exit(2)

    root = _project_root()
    out = Path(out_dir).expanduser() if out_dir else Path.home() / "runs" / "emet" / "jobs_runs" / name
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "job.log"
    cmd_str = " ".join(shlex.quote(a) for a in cmd_args)

    auto_gpu_job = need_mib is not None or command_looks_like_gpu_job(name, cmd_str)
    use_cpu_safe = bool(cpu_safe) if cpu_safe is not None else auto_gpu_job
    use_gpu_excl = bool(gpu_exclusive) if gpu_exclusive is not None else auto_gpu_job

    wait_pids = list(wait_pid)
    lock_timeout = lock_timeout_sec
    if lock_timeout is None:
        raw_timeout = os.environ.get("EMET_GPU_LOCK_TIMEOUT", "").strip()
        try:
            lock_timeout = float(raw_timeout) if raw_timeout else 21600.0
        except ValueError as exc:
            raise click.ClickException(f"invalid EMET_GPU_LOCK_TIMEOUT={raw_timeout!r}") from exc
        if lock_timeout < 0:
            click.echo(
                "warning: unbounded EMET_GPU_LOCK_TIMEOUT is no longer supported; using 21600 seconds",
                err=True,
            )
            lock_timeout = 21600.0

    job_id = new_job_id()
    click.echo(f"prepared    {job_id}", err=True)
    click.echo(f"name        {name}", err=True)
    if description and str(description).strip():
        click.echo(f"why         {str(description).strip()}", err=True)
    click.echo(f"out_dir     {out}", err=True)
    click.echo(f"log         {log_path}", err=True)

    wrapper = out / "job_wrapper.sh"
    register_args = [
        "jobs",
        "register",
        "--job-id",
        job_id,
        "--name",
        name,
        "--cmd",
        cmd_str,
        "--out-dir",
        str(out),
        "--log-path",
        str(log_path),
        "--repo",
        str(root),
        "--status",
        "waiting",
    ]
    if description and str(description).strip():
        register_args.extend(["--description", str(description).strip()])
    for wpid in wait_pids:
        register_args.extend(["--wait-pid", str(int(wpid))])
    register_line = '"$EMET_BIN" ' + shlex.join(register_args) + ' --pid "$$"\n'
    wait_lines = ""
    if wait_pids:
        wait_lines = (
            'pid_is_running() { local stat; stat="$(ps -o stat= -p "$1" 2>/dev/null || true)"; '
            '[[ -n "$stat" && "$stat" != Z* ]]; }\n'
            f"WAIT_PID_DEADLINE=$((SECONDS + {max(0, int(wait_timeout_sec))}))\n"
            "for WAIT_PID in " + " ".join(str(int(wpid)) for wpid in wait_pids) + "; do\n"
            '  while pid_is_running "$WAIT_PID"; do\n'
            "    if (( SECONDS >= WAIT_PID_DEADLINE )); then\n"
            '      echo "ERROR: timed out waiting for explicit prerequisite pid $WAIT_PID" >&2\n'
            '      "$EMET_BIN" jobs update "$JOB_ID" --status failed '
            '--error "explicit wait-pid timeout $WAIT_PID" >/dev/null 2>&1 || true\n'
            "      exit 4\n"
            "    fi\n"
            "    sleep 1\n"
            "  done\n"
            "done\n"
        )
    need_block = ""
    if need_mib is not None:
        need_block = (
            f'NEED_MIB={int(need_mib)} "$EMET_BIN" eval wait --need-mib {int(need_mib)} '
            f"--max-rounds {int(gpu_wait_max_rounds)}\n"
            f'"$EMET_BIN" eval status || true\n'
        )
    gpu_lock_block = ""
    if use_gpu_excl:
        default_lock = shlex.quote(str(gpu_lock_path()))
        timeout = shlex.quote(str(float(lock_timeout)))
        gpu_lock_block = (
            "GPU_LOCK_FILE=" + default_lock + "\n"
            "if ! command -v flock >/dev/null 2>&1; then\n"
            '  echo "ERROR: gpu-exclusive jobs require the flock utility" >&2\n'
            '  "$EMET_BIN" jobs update "$JOB_ID" --status failed '
            '--error "flock utility is unavailable" >/dev/null 2>&1 || true\n'
            "  exit 2\n"
            "fi\n"
            'if ! mkdir -p "$(dirname "$GPU_LOCK_FILE")"; then\n'
            '  echo "ERROR: cannot create GPU lock directory for $GPU_LOCK_FILE" >&2\n'
            '  "$EMET_BIN" jobs update "$JOB_ID" --status failed '
            '--error "cannot create GPU lock directory" >/dev/null 2>&1 || true\n'
            "  exit 2\n"
            "fi\n"
            'exec 9>"$GPU_LOCK_FILE"\n'
            'echo "waiting for exclusive GPU lock: $GPU_LOCK_FILE" >&2\n'
            "_lock_rc=0\n"
            "flock -w " + timeout + " -x 9 || _lock_rc=$?\n"
            'if [[ "$_lock_rc" -ne 0 ]]; then\n'
            '  echo "ERROR: timed out waiting for GPU lock: $GPU_LOCK_FILE" >&2\n'
            '  "$EMET_BIN" jobs update "$JOB_ID" --status failed '
            '--error "gpu lock timeout $GPU_LOCK_FILE" >/dev/null 2>&1 || true\n'
            "  exit 3\n"
            "fi\n"
            'export EMET_GPU_LOCK="$GPU_LOCK_FILE"\n'
            'export EMET_GPU_LOCK_FILE="$GPU_LOCK_FILE"\n'
            'echo "acquired exclusive GPU lock: $GPU_LOCK_FILE" >&2\n'
        )
    cpu_block = ""
    if use_cpu_safe:
        cpu_block = (
            # After a previous sim job releases the GPU lock, importing the
            # full ``emet`` CLI (MuJoCo native) segfaults. Pin via the
            # stdlib affinity module only.
            '"$EMET_PY" -m emet.utils.cpu_affinity --apply --apply-pid $$ || {\n'
            '  echo "ERROR: cpu-safe affinity failed (fail-closed)" >&2\n'
            "  exit 2\n"
            "}\n"
        )
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'cd "{root}"\n'
        f'export EMET_JOB_ID="{job_id}"\n'
        f'JOB_ID="{job_id}"\n'
        f'EMET_BIN="{root}/.venv/bin/emet"\n'
        'if [ ! -x "$EMET_BIN" ]; then EMET_BIN="emet"; fi\n'
        f'EMET_PY="{root}/.venv/bin/python"\n'
        'if [ ! -x "$EMET_PY" ]; then EMET_PY="python3"; fi\n'
        f"{register_line}"
        f"{gpu_lock_block}"
        f"{wait_lines}"
        f"{need_block}"
        f"{cpu_block}"
        f'"$EMET_BIN" jobs update "$JOB_ID" --status running --pid $$\n'
        "set +e\n"
        f"{cmd_str}\n"
        "rc=$?\n"
        "set -e\n"
        'if [ "$rc" -eq 0 ]; then\n'
        f'  "$EMET_BIN" jobs update "$JOB_ID" --status done\n'
        "else\n"
        f'  "$EMET_BIN" jobs update "$JOB_ID" --status failed --error "exit $rc"\n'
        "fi\n"
        "exit $rc\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    if foreground:
        rc = subprocess.call(["bash", str(wrapper)])
        sys.exit(rc)

    # Detach: start_new_session so cancel can killpg
    with log_path.open("a", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            ["bash", str(wrapper)],
            cwd=str(root),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + 10.0
    job = None
    while time.monotonic() < deadline:
        job = load_job(job_id)
        if job is not None:
            break
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    if job is None:
        from emet.utils.process_tree import terminate_process_tree

        terminate_process_tree(proc, grace_s=1.0)
        raise click.ClickException(f"detached supervisor failed to register job {job_id}; see {log_path}")
    if job.pid != proc.pid:
        raise click.ClickException(f"job {job_id} registered unexpected supervisor pid {job.pid} (spawned {proc.pid})")
    click.echo(f"registered  {job_id}", err=True)
    click.echo(f"pid         {proc.pid}", err=True)
    click.echo(job_id)


def register(main: click.Group) -> None:
    main.add_command(jobs_group)

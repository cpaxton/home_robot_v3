# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click

from emet.cli_cmds.bootstrap import (
    _active_project_root,
    _project_root,
)


@click.group("status", short_help="Per-checkout STATUS.log helpers (after agent death)")
def status_group() -> None:
    """Tail-able recovery log for long GPU / HM-EQA runs.

    Prefer ``emet status tail`` over ``bash scripts/status_log.sh``. Orchestrators
    still *source* ``scripts/status_log.sh`` to write records.

    \b
    Examples:
      emet status tail
      emet status path
      emet status latest
    """


def _status_log_script() -> Path:
    return _active_project_root() / "scripts" / "status_log.sh"


@status_group.command("tail", short_help="Show last N STATUS.log lines")
@click.argument("n", required=False, default="12")
def status_tail(n: str) -> None:
    script = _status_log_script()
    if not script.is_file():
        click.echo(f"missing {script}", err=True)
        sys.exit(1)
    sys.exit(subprocess.call(["bash", str(script), "tail", str(n)], cwd=str(script.parent.parent)))


@status_group.command("path", short_help="Print STATUS.log path for this checkout")
def status_path() -> None:
    script = _status_log_script()
    sys.exit(subprocess.call(["bash", str(script), "path"], cwd=str(script.parent.parent)))


@status_group.command("latest", short_help="Resolve latest OUT symlink for this checkout")
def status_latest() -> None:
    script = _status_log_script()
    sys.exit(subprocess.call(["bash", str(script), "latest"], cwd=str(script.parent.parent)))


@click.group("eval", short_help="GPU preflight and eval process cleanup")
def eval_group() -> None:
    """GPU preflight and stale-process cleanup for paper evals / overnight smokes.

    Prefer these over sourcing ``scripts/gpu_preflight.sh`` from an interactive shell.
    Overnight bash scripts may still source that file; it delegates here when possible.

    Examples:
      emet eval status
      emet eval diagnose
      emet eval check --need-mib 12000
      emet eval wait --need-mib 12000
      emet eval kill-stale
    """


@eval_group.command("status", short_help="Show free VRAM, GPU compute apps, and disk")
def eval_status() -> None:
    """Print GPU free/total MiB, nvidia-smi compute apps, and disk free (read-only)."""
    from emet.utils.gpu_preflight import disk_status_lines, format_status_lines

    for line in format_status_lines():
        click.echo(line)
    for line in disk_status_lines():
        click.echo(line)


@eval_group.command(
    "diagnose",
    short_help="Explain GPU/EGL readiness (empty nvidia-smi ≠ Habitat OK)",
)
def eval_diagnose() -> None:
    """Read-only Habitat/HM-EQA readiness notes for agents.

    Empty compute apps does **not** prove Magnum EGL can create a WindowlessContext.
    Also flags empty ``CUDA_VISIBLE_DEVICES``, missing ``.venv-habitat``, and recent
    ``emet`` segfault hints from dmesg when readable.
    """
    from pathlib import Path

    from emet.utils.gpu_preflight import diagnose_eval_environment

    ok, lines = diagnose_eval_environment(repo_root=str(Path.cwd()))
    for line in lines:
        click.echo(line)
    if not ok:
        sys.exit(1)


@eval_group.command("check", short_help="Exit 1 if free VRAM below threshold")
@click.option(
    "--need-mib",
    type=int,
    default=None,
    help="Minimum free VRAM in MiB (default: NEED_MIB or 12000).",
)
def eval_check(need_mib: int | None) -> None:
    """One-shot GPU memory gate (same role as ``gpu_preflight.sh --check``)."""
    from emet.utils.gpu_preflight import check_gpu_memory, list_compute_apps

    ok, msg = check_gpu_memory(need_mib)
    click.echo(msg)
    if not ok:
        for app in list_compute_apps():
            click.echo(
                f"  pid={app.pid} {app.process_name} {app.used_memory}".rstrip(),
                err=True,
            )
        sys.exit(1)


@eval_group.command("wait", short_help="Block until free VRAM is stably above threshold")
@click.option(
    "--need-mib",
    type=int,
    default=None,
    help="Minimum free VRAM in MiB (default: NEED_MIB or 12000).",
)
@click.option(
    "--max-rounds",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum checks (default: GPU_WAIT_MAX_ROUNDS or 120).",
)
def eval_wait(need_mib: int | None, max_rounds: int | None) -> None:
    """Wait for consecutive stable free-VRAM reads (``gpu_preflight.sh --wait``)."""
    from emet.utils.gpu_preflight import wait_gpu_stable

    ok = wait_gpu_stable(
        need_mib,
        max_rounds=max_rounds,
        log=lambda m: click.echo(m, err=True),
    )
    if not ok:
        click.echo("WARNING: GPU wait timed out; free VRAM still below threshold", err=True)
        sys.exit(1)


@eval_group.command("kill-stale", short_help="Stop orphaned eval/sim/uv GPU workers")
@click.option(
    "--no-gpu",
    "no_gpu",
    is_flag=True,
    help="Only match process patterns; do not kill nvidia-smi compute apps.",
)
@click.option(
    "--settle-sec",
    type=float,
    default=None,
    help="Sleep after pattern kills (default: GPU_SETTLE_SEC or 15).",
)
def eval_kill_stale(no_gpu: bool, settle_sec: float | None) -> None:
    """SIGTERM→SIGKILL stale mujoco/habitat/dynagraph/uv trees; skip caller ancestry.

    Protects this process and its parents, plus ``EMET_GPU_PROTECT_PIDS``.
    Patterns also match ``uv run emet {run,test,serve}`` in *other* terminals —
    do not run this while intentional GPU work is still in progress elsewhere;
    set ``EMET_GPU_PROTECT_PIDS`` or use ``emet eval wait`` instead.
    For port-only MuJoCo cleanup see ``emet kill-mujoco-server``.
    """
    from emet.utils.gpu_preflight import kill_stale_eval_processes

    click.echo(
        "kill-stale: matching sim/eval/uv emet trees "
        "(other terminals' pytest/serve may match; EMET_GPU_PROTECT_PIDS to keep)",
        err=True,
    )
    n = kill_stale_eval_processes(
        kill_gpu_apps=not no_gpu,
        settle_s=settle_sec,
        log=lambda m: click.echo(m, err=True),
    )
    click.echo(f"kill-stale done (signaled≈{n})")


@eval_group.command("clean-bundles", short_help="Retention-prune HM-EQA episode debug bundles")
@click.option("--keep", type=int, default=2, help="keep newest N runs per sweep prefix (default 2)")
@click.option("--max-age-days", type=float, default=0.0, help="also delete bundles older than N days")
@click.option("--apply", is_flag=True, help="actually delete (default: dry-run)")
def eval_clean_bundles(keep: int, max_age_days: float, apply: bool) -> None:
    """Prune large per-episode debug bundles under ~/.cache/habitat_eqa/episodes.

    Each full-113 sweep writes GB of debug frames/MP4/topdown maps per method; the
    scored results live in results/*.jsonl and are never touched. Keeps the newest
    ``--keep`` runs per sweep, deletes the rest. Dry-run by default; use --apply.
    """
    from emet.utils.gpu_preflight import clean_episode_bundles, disk_status_lines

    for line in disk_status_lines():
        click.echo(line)
    click.echo("---")
    for line in clean_episode_bundles(keep=keep, max_age_days=max_age_days, apply=apply):
        click.echo(line)


@eval_group.command("affinity", short_help="Show or apply turbo-CPU exclusion mask")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON summary.")
@click.option("--apply", "do_apply", is_flag=True, help="Pin current process (or --pid).")
@click.option("--pid", type=int, default=None, help="Target PID (default: this process).")
@click.option(
    "--fail-open",
    is_flag=True,
    help="Do not exit non-zero if turbo CPUs remain after apply.",
)
def eval_affinity(as_json: bool, do_apply: bool, pid: int | None, fail_open: bool) -> None:
    """Exclude logical CPUs whose max freq is ≥ ``EMET_EXCLUDE_CPU_MIN_MHZ`` (default 6000)."""
    from emet.utils.cpu_affinity import affinity_summary_dict, apply_eval_affinity

    if do_apply:
        try:
            summary = apply_eval_affinity(pid=pid, fail_closed=not fail_open)
        except RuntimeError as exc:
            click.echo(f"ERROR: {exc}", err=True)
            sys.exit(2)
        if as_json:
            click.echo(json.dumps(summary, indent=2))
        else:
            click.echo(
                f"affinity pid={summary.get('pid')} mask={summary.get('applied')} "
                f"turbo_excluded={summary.get('turbo_cpus')}"
            )
        return

    summary = affinity_summary_dict()
    if as_json:
        click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(
            f"taskset {summary['taskset']}  (exclude>={summary['exclude_min_mhz']} MHz turbo={summary['turbo_cpus']})"
        )


@eval_group.command("recover", short_help="status + diagnose + wait (post-crash preflight)")
@click.option(
    "--need-mib",
    type=int,
    default=None,
    help="Minimum free VRAM in MiB (default: NEED_MIB or 12000).",
)
@click.option(
    "--skip-wait",
    is_flag=True,
    help="Only status+diagnose; do not block on free VRAM.",
)
@click.option(
    "--max-rounds",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum checks (default: GPU_WAIT_MAX_ROUNDS or 120).",
)
def eval_recover(need_mib: int | None, skip_wait: bool, max_rounds: int | None) -> None:
    """One-shot recovery gate after agent death / host reboot / failed HM-EQA job."""
    from emet.utils.cpu_affinity import affinity_summary_dict
    from emet.utils.gpu_preflight import (
        diagnose_eval_environment,
        format_status_lines,
        wait_gpu_stable,
    )

    for line in format_status_lines():
        click.echo(line)
    ok, lines = diagnose_eval_environment(repo_root=str(_project_root()))
    for line in lines:
        click.echo(line)
    aff = affinity_summary_dict()
    click.echo(
        f"affinity: prefer taskset {aff['taskset']} "
        f"(turbo {aff['turbo_cpus']} excluded by emet jobs --cpu-safe / emet hmeqa)"
    )
    if not ok:
        sys.exit(1)
    if skip_wait:
        return
    if not wait_gpu_stable(
        need_mib,
        max_rounds=max_rounds,
        log=lambda m: click.echo(m, err=True),
    ):
        click.echo("WARNING: GPU wait timed out; free VRAM still below threshold", err=True)
        sys.exit(1)
    click.echo("recover: GPU ready — next: emet hmeqa resume  (or emet jobs)")


def register(main: click.Group) -> None:
    main.add_command(status_group)
    main.add_command(eval_group)

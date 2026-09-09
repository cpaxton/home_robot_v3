# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click

from emet.cli_cmds.bootstrap import (
    _jobs_run_id_from_output,
    _project_root,
    _timestamp,
)


def _run_habitat_wrapper(args: list[str]) -> int:
    """Run the emet-habitat wrapper. Returns exit code."""
    from emet.habitat.wrapper_config import build_habitat_wrapper_command, ensure_habitat_eqa_data_dir_env

    cmd = build_habitat_wrapper_command(args)
    if cmd is None:
        click.echo(
            "Habitat wrapper not found. From the project root run:\n"
            "  ./scripts/install_habitat.sh\n\n"
            "See docs/habitat/README.md.",
            err=True,
        )
        return 1
    env = os.environ.copy()
    ensure_habitat_eqa_data_dir_env(env)
    return subprocess.call(cmd, cwd=_project_root(), env=env)


@click.group("habitat", short_help="Habitat-Sim EQA harness (requires emet-habitat / .venv-habitat)")
def habitat_cmd() -> None:
    """HM-EQA / OpenEQA evaluation in Habitat driving emet GraphEQA / Dynagraph.

    Requires ``./scripts/install_habitat.sh`` (``.venv-habitat``). See docs/habitat/README.md.
    """


@habitat_cmd.command("info", short_help="Print data paths and asset status")
def habitat_info() -> None:
    sys.exit(_run_habitat_wrapper(["info"]))


@habitat_cmd.command(
    "safe-start",
    short_help="Preflight + jobs-wrapped Habitat EGL probe (safe for Cursor agents)",
)
@click.option("--need-mib", default=4000, type=int, show_default=True, help="VRAM free for EGL probe")
@click.option("--question-id", default=0, type=int, show_default=True)
@click.option(
    "--smoke-episode",
    is_flag=True,
    default=False,
    help="Also queue a mock-llm dynagraph episode (gpu-exclusive waits behind the probe)",
)
@click.option(
    "--force-inline",
    is_flag=True,
    default=False,
    help="Run probe in this process (dangerous in Cursor — can segfault the agent host)",
)
@click.option("--job-name", default="habitat-egl-probe", show_default=True)
def habitat_safe_start(
    need_mib: int,
    question_id: int,
    smoke_episode: bool,
    force_inline: bool,
    job_name: str,
) -> None:
    """Recover GPU state, then queue a detached Habitat EGL probe (never inline by default).

    Empty ``nvidia-smi`` ≠ Habitat OK. This path::

        emet eval recover → emet jobs run (detached) → emet-habitat egl-probe

    Detach is intentional: Habitat teardown often SIGSEGVs Cursor agent hosts.
    This command returning 0 only means the probe job was **queued**, not that EGL
    succeeded. Wait until ``emet jobs status JOB`` is ``done`` and logs look OK
    before ``emet hmeqa h2h`` / overnight. Do **not** pass ``--force-inline`` from
    a Cursor agent session.
    """
    import shlex

    root = _project_root()
    # 1) Preflight (read-only diagnose + wait for VRAM).
    recover_cmd = [
        sys.executable,
        "-m",
        "emet.cli",
        "eval",
        "recover",
        "--need-mib",
        str(int(need_mib)),
    ]
    click.echo(f"preflight: {' '.join(recover_cmd)}", err=True)
    rc = subprocess.call(recover_cmd, cwd=str(root))
    if rc != 0:
        click.echo(
            "preflight failed — fix GPU/EGL (see emet eval diagnose) before Habitat",
            err=True,
        )
        sys.exit(rc)

    probe_args = ["egl-probe", "--question-id", str(int(question_id)), "--json"]
    if force_inline:
        click.echo(
            "WARNING: --force-inline runs Habitat in this process; "
            "Cursor agent hosts often die on Habitat/VLM teardown SIGSEGV.",
            err=True,
        )
        sys.exit(_run_habitat_wrapper(probe_args))

    from emet.habitat.wrapper_config import build_habitat_wrapper_command, ensure_habitat_eqa_data_dir_env

    wrap = build_habitat_wrapper_command(probe_args)
    if wrap is None:
        click.echo(
            "Habitat wrapper not found. From the project root run:\n  ./scripts/install_habitat.sh\n",
            err=True,
        )
        sys.exit(1)

    # Preserve HABITAT_EQA_DATA_DIR for the job child.
    env_prefix = []
    env = os.environ.copy()
    ensure_habitat_eqa_data_dir_env(env)
    data_dir = env.get("HABITAT_EQA_DATA_DIR")
    if data_dir:
        env_prefix.append(f"HABITAT_EQA_DATA_DIR={shlex.quote(data_dir)}")

    inner = "env " + " ".join(env_prefix + [shlex.quote(c) for c in wrap])
    out_dir = Path(os.path.expanduser("~/runs/emet")) / f"habitat_egl_probe_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs_cmd = [
        sys.executable,
        "-m",
        "emet.cli",
        "jobs",
        "run",
        "--name",
        job_name,
        "--need-mib",
        str(int(need_mib)),
        "--out-dir",
        str(out_dir),
        "--",
        "bash",
        "-lc",
        inner,
    ]
    click.echo(f"queuing detached EGL probe via emet jobs: OUT={out_dir}", err=True)
    launched = subprocess.run(
        jobs_cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if launched.stderr:
        click.echo(launched.stderr.rstrip("\n"), err=True)
    if launched.stdout:
        click.echo(launched.stdout.rstrip("\n"), err=True)
    if launched.returncode != 0:
        click.echo(
            f"EGL probe job launch failed (rc={launched.returncode}). Check: uv run emet jobs",
            err=True,
        )
        sys.exit(launched.returncode)

    probe_job = _jobs_run_id_from_output(launched.stdout)
    job_ref = probe_job or "JOB"
    click.echo(
        "EGL probe job queued (detached — not finished yet).\n"
        f"  OUT={out_dir}\n"
        f"  uv run emet jobs status {job_ref}\n"
        f"  uv run emet jobs logs {job_ref} --tail 40\n"
        "Do NOT launch HM-EQA until status is done and logs show EGL OK.\n"
        "Only then:\n"
        "  uv run emet hmeqa h2h --preset paper-router …\n"
        "  # or: uv run emet hmeqa overnight",
        err=True,
    )

    if smoke_episode:
        smoke_name = f"{job_name}-smoke"
        smoke_out = Path(os.path.expanduser("~/runs/emet")) / f"habitat_mock_smoke_{_timestamp()}"
        smoke_out.mkdir(parents=True, exist_ok=True)
        smoke_wrap = build_habitat_wrapper_command(
            [
                "run-episode",
                "--question-id",
                str(int(question_id)),
                "--method",
                "dynagraph",
                "--mock-llm",
                "--max-planning-steps",
                "2",
                "--output",
                str(smoke_out / "smoke.jsonl"),
            ]
        )
        if smoke_wrap is None:
            sys.exit(1)
        smoke_inner = "env " + " ".join(env_prefix + [shlex.quote(c) for c in smoke_wrap])
        smoke_cmd = [
            sys.executable,
            "-m",
            "emet.cli",
            "jobs",
            "run",
            "--name",
            smoke_name,
            "--need-mib",
            str(max(int(need_mib), 8000)),
            "--out-dir",
            str(smoke_out),
            "--",
            "bash",
            "-lc",
            smoke_inner,
        ]
        click.echo(
            f"queuing mock-llm smoke behind probe (gpu-exclusive): OUT={smoke_out}",
            err=True,
        )
        smoke = subprocess.run(
            smoke_cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        if smoke.stderr:
            click.echo(smoke.stderr.rstrip("\n"), err=True)
        if smoke.stdout:
            click.echo(smoke.stdout.rstrip("\n"), err=True)
        if smoke.returncode != 0:
            sys.exit(smoke.returncode)
        smoke_job = _jobs_run_id_from_output(smoke.stdout)
        smoke_ref = smoke_job or "SMOKE_JOB"
        click.echo(
            "Smoke episode also queued (detached). Wait for probe done, then smoke done:\n"
            f"  uv run emet jobs status {smoke_ref}\n"
            f"  uv run emet jobs logs {smoke_ref} --tail 40\n"
            "Still do not launch HM-EQA until the EGL probe job is done + OK.",
            err=True,
        )
        sys.exit(0)
    sys.exit(0)


@habitat_cmd.command("egl-probe", short_help="Delegate to emet-habitat egl-probe (prefer safe-start)")
@click.option("--question-id", default=0, type=int)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option(
    "--force-inline",
    is_flag=True,
    default=False,
    help="Required to run in-process; otherwise redirects to safe-start",
)
def habitat_egl_probe(question_id: int, as_json: bool, force_inline: bool) -> None:
    """Thin alias. Agents should use ``emet habitat safe-start`` instead."""
    if not force_inline:
        click.echo(
            "Refusing inline Habitat EGL probe (segfaults Cursor agent hosts). "
            "Use: uv run emet habitat safe-start\n"
            "Or pass --force-inline only from a dedicated terminal.",
            err=True,
        )
        sys.exit(2)
    args = ["egl-probe", "--question-id", str(int(question_id))]
    if as_json:
        args.append("--json")
    sys.exit(_run_habitat_wrapper(args))


@habitat_cmd.command("list-questions", short_help="List HM-EQA questions from CSV")
@click.option("--limit", default=10, type=int)
def habitat_list_questions(limit: int) -> None:
    sys.exit(_run_habitat_wrapper(["list-questions", "--limit", str(limit)]))


@habitat_cmd.command("serve", short_help="Start Habitat-Sim ZMQ server (interactive)")
@click.option("--question-id", type=int, default=None)
@click.option("--scene-id", default=None)
@click.option("--floor", default=0, type=int)
@click.option("--port-offset", default=0, type=int)
def habitat_serve(
    question_id: int | None,
    scene_id: str | None,
    floor: int,
    port_offset: int,
) -> None:
    """Same as ``emet serve habitat`` — Stretch-shaped ZMQ for dynagraph / agent."""
    args = ["serve", "--port-offset", str(port_offset)]
    if question_id is not None:
        args.extend(["--question-id", str(question_id)])
    if scene_id:
        args.extend(["--scene-id", scene_id])
    if floor:
        args.extend(["--floor", str(floor)])
    sys.exit(_run_habitat_wrapper(args))


@habitat_cmd.command("run-episode", short_help="Run one HM-EQA episode")
@click.option("--question-id", default=0, type=int)
@click.option(
    "--method",
    type=click.Choice(["static_graph", "graph_eqa", "dynagraph"]),
    default="dynagraph",
    help="HM-EQA method (graph_eqa is a legacy alias for static_graph).",
)
@click.option("--mock-llm", is_flag=True, default=False)
@click.option("--max-planning-steps", default=5, type=int)
@click.option(
    "--rerun",
    "enable_rerun",
    is_flag=True,
    default=False,
    help="Live Rerun VLM-context viewer (ports 9090/9877). Also EMET_EVAL_RERUN=1. Off by default.",
)
def habitat_run_episode(
    question_id: int,
    method: str,
    mock_llm: bool,
    max_planning_steps: int,
    enable_rerun: bool,
) -> None:
    args = [
        "run-episode",
        "--question-id",
        str(question_id),
        "--method",
        method,
        "--max-planning-steps",
        str(max_planning_steps),
    ]
    if mock_llm:
        args.append("--mock-llm")
    if enable_rerun:
        args.append("--rerun")
        os.environ["EMET_EVAL_RERUN"] = "1"
    sys.exit(_run_habitat_wrapper(args))


@habitat_cmd.command("compare-batch", short_help="GraphEQA vs Dynagraph on same questions")
@click.option("--question-start", default=0, type=int)
@click.option("--question-end", default=5, type=int)
@click.option("--mock-llm", is_flag=True, default=False)
@click.option("--max-planning-steps", default=20, type=int)
def habitat_compare_batch(
    question_start: int,
    question_end: int,
    mock_llm: bool,
    max_planning_steps: int,
) -> None:
    args = [
        "compare-batch",
        "--question-start",
        str(question_start),
        "--question-end",
        str(question_end),
        "--max-planning-steps",
        str(max_planning_steps),
        "--output",
        f"{os.path.expanduser('~')}/.cache/habitat_eqa/results/compare_q{question_start}-{question_end}.json",
    ]
    if mock_llm:
        args.append("--mock-llm")
    sys.exit(_run_habitat_wrapper(args))


def register(main: click.Group) -> None:
    main.add_command(habitat_cmd)

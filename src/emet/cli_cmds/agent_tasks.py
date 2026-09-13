"""Import-light CLI for the small agent benchmark. No model/API clients."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click


@click.group("agent-tasks")
def agent_tasks_group():
    """Preflight, run, inspect, replay and export embodied-task evidence.

    Initial execution is an explicitly assisted MuJoCo fixture through shared
    CHAT tools. It makes no model calls and does not certify learned execution.
    """


def suite_option(fn):
    return click.option("--suite", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)(fn)


@agent_tasks_group.command("preflight")
@suite_option
@click.option("--out", type=click.Path(path_type=Path), default=None)
def preflight_command(suite, out):
    """Validate configuration and footprint routes without loading native libraries."""
    from emet.eval.agent_tasks.runner import preflight
    from emet.eval.agent_tasks.spec import load_suite

    try:
        result = preflight(load_suite(suite))
    except (ValueError, KeyError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
    rendered = json.dumps(result, indent=2)
    click.echo(rendered)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n")
    if not result["passed"]:
        raise click.ClickException("preflight failed")


@agent_tasks_group.command("run")
@suite_option
@click.option("--episode", "episode_ids", multiple=True)
@click.option("--out", required=True, type=click.Path(path_type=Path))
@click.option("--repeats", default=1, type=click.IntRange(1, 3))
@click.option("--timeout", default=600, type=click.IntRange(1, 600))
@click.option("--render/--no-render", default=True)
@click.option(
    "--control", default="witness", type=click.Choice(["witness", "noop", "partial", "wrong_destination", "stale_plan"])
)
def run_command(suite, episode_ids, out, repeats, timeout, render, control):
    """Run bounded, isolated assisted controls; never call a paid or local model.

    For GPU rendering, invoke through emet jobs. Set MUJOCO_GL explicitly for
    your host. --no-render is diagnostic-only and cannot pass the evidence gate.
    """
    from emet.eval.agent_tasks.recording import write_json
    from emet.eval.agent_tasks.runner import preflight
    from emet.eval.agent_tasks.spec import default_suite, fingerprint, load_suite, select_episode
    from emet.utils.process_tree import popen_session, terminate_process_tree

    suite_path = Path(suite or default_suite()).resolve()
    config = load_suite(suite_path)
    selected = [select_episode(config, eid) for eid in episode_ids] if episode_ids else config["episodes"]
    checks = preflight(config)
    if not checks["passed"]:
        raise click.ClickException("preflight failed; run preflight for details")
    if out.exists():
        raise click.ClickException("output already exists; use a fresh directory to preserve prior evidence")
    out.mkdir(parents=True)
    write_json(out / "preflight.json", checks)
    rows = []
    for episode in selected:
        for repeat in range(repeats):
            run_id = f"{episode['id']}_{repeat + 1}"
            dest = out / run_id
            cmd = [
                sys.executable,
                "-m",
                "emet.eval.agent_tasks.worker",
                "--suite",
                str(suite_path),
                "--episode",
                episode["id"],
                "--out",
                str(dest.resolve()),
                "--control",
                control,
                "--timeout",
                str(timeout),
            ]
            if not render:
                cmd.append("--no-render")
            env = os.environ.copy()
            if not render:
                env["MUJOCO_GL"] = "disable"
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("MPLCONFIGDIR", str((out / "mpl-cache").resolve()))
            click.echo(f"{run_id}: assisted fixture / {control}")
            with (out / f"{run_id}.log").open("w") as log:
                proc = popen_session(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    terminate_process_tree(proc, grace_s=2)
                    code = -1
                except BaseException:
                    terminate_process_tree(proc, grace_s=2)
                    raise
            metrics_path = dest / "metrics.json"
            metrics = (
                json.loads(metrics_path.read_text())
                if metrics_path.is_file()
                else {
                    "status": "timeout" if code == -1 else "native_failure",
                    "success": False,
                    "evidence_complete": False,
                }
            )
            row = {"episode_id": episode["id"], "repeat": repeat + 1, "path": run_id, "exit_code": code, **metrics}
            rows.append(row)
            complete = len(rows) == len(selected) * repeats
            certificate = {
                "schema_version": 1,
                "suite_fingerprint": fingerprint(config),
                "rows": rows,
                "scope": "assisted fixture / CHAT registry only; no learned or ZMQ certification",
                "certified": complete
                and repeats == 3
                and control == "witness"
                and all(r["success"] and r["evidence_complete"] and r["exit_code"] == 0 for r in rows),
            }
            write_json(out / "certificate.json", certificate)
            click.echo(f"  {metrics['status']}; evidence_complete={metrics.get('evidence_complete', False)}")
            if metrics["status"] in {"infrastructure_failed", "native_failure", "timeout"}:
                raise click.ClickException(f"infrastructure gate failed; inspect {out / f'{run_id}.log'}")
    if any(not r["success"] or not r["evidence_complete"] for r in rows):
        raise click.ClickException("task or evidence gate failed; all outcomes retained")


@agent_tasks_group.command("inspect")
@click.argument("run", type=click.Path(exists=True, file_okay=False, path_type=Path))
def inspect_command(run):
    """Validate the evidence hash chain and summarize one recorded episode."""
    from emet.eval.agent_tasks.recording import load_run

    manifest, events, metrics = load_run(run)
    click.echo(
        json.dumps(
            {
                "episode": manifest["episode"]["id"],
                "mode": manifest["mode"],
                "events": len(events),
                "metrics": metrics,
                "report": str((run / "report.html").resolve()),
            },
            indent=2,
        )
    )


@agent_tasks_group.command("replay")
@click.argument("run", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--rerun", "open_rerun", is_flag=True, help="Open the saved .rrd in the installed Rerun viewer.")
def replay_command(run, open_rerun):
    """Show the offline report path, or explicitly open the saved Rerun recording."""
    from emet.eval.agent_tasks.recording import load_run

    load_run(run)
    if open_rerun:
        executable = shutil.which("rerun")
        if executable is None or not (run / "episode.rrd").is_file():
            raise click.ClickException("Rerun executable or episode.rrd missing")
        subprocess.run([executable, str(run / "episode.rrd")], check=True)
    else:
        click.echo(str((run / "report.html").resolve()))


@agent_tasks_group.command("compare")
@click.argument("left", type=click.Path(exists=True, path_type=Path))
@click.argument("right", type=click.Path(exists=True, path_type=Path))
def compare_command(left, right):
    """Compare paired episode results; reject mismatched tasks or assistance."""
    from emet.eval.agent_tasks.recording import load_run

    a, ea, ma = load_run(left)
    b, eb, mb = load_run(right)
    keys = ("suite_fingerprint", "episode_fingerprint", "robot", "seed", "assistance")
    mismatch = [key for key in keys if a[key] != b[key]]
    if mismatch:
        raise click.ClickException("unpaired runs: " + ", ".join(mismatch))
    click.echo(
        json.dumps(
            {
                "left": ma,
                "right": mb,
                "event_counts": [len(ea), len(eb)],
                "source_revisions": [a["source_revision"], b["source_revision"]],
            },
            indent=2,
        )
    )


@agent_tasks_group.command("export")
@click.argument("run", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--paper-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None)
def export_command(run, paper_dir):
    """Regenerate report, Rerun, video and paper figures offline from recorded data."""
    if paper_dir is not None:
        from emet.eval.agent_tasks.publication import export_paper

        try:
            result = export_paper(run, paper_dir)
        except (ValueError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps(result, indent=2))
        return
    from emet.eval.agent_tasks.visualize import export_run

    result = export_run(run)
    click.echo(json.dumps(result, indent=2))
    if not result["complete"]:
        raise click.ClickException("evidence package incomplete")

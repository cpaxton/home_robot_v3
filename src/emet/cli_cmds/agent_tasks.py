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

    Run assisted MuJoCo witnesses or the shared task agent with cached local
    models. Paid providers are disabled. Evidence declares all simulation assistance.
    """


def suite_option(fn):
    return click.option("--suite", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)(fn)


@agent_tasks_group.command("list")
def list_environments_command():
    """List bundled environment layouts and their task instructions."""
    from emet.eval.agent_tasks.spec import default_suite, load_suite

    rows = []
    for path in sorted(default_suite().parent.glob("agent_tasks*.yaml")):
        suite = load_suite(path)
        rows.append(
            {
                "suite": str(path),
                "name": suite["name"],
                "rooms": [r["id"] for r in suite["scene"]["rooms"]],
                "tasks": [{"id": e["id"], "instruction": e["instruction"]} for e in suite["episodes"]],
            }
        )
    click.echo(json.dumps(rows, indent=2))


@agent_tasks_group.command("export-agents")
@click.argument("runs", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--paper-dir", required=True, type=click.Path(path_type=Path))
@click.option(
    "--bundle",
    default="agent_task_policy",
    show_default=True,
    help="Separate paper data/figure prefix for this cohort.",
)
def export_agents_command(runs, paper_dir, bundle):
    """Export actual local-policy diagnostics (including failures) into the paper."""
    from emet.eval.agent_tasks.publication import export_agent_paper

    try:
        result = export_agent_paper(list(runs), paper_dir, bundle=bundle)
    except (ValueError, KeyError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, indent=2))


@agent_tasks_group.command("agent")
@suite_option
@click.option("--episode", required=True)
@click.option("--out", required=True, type=click.Path(path_type=Path))
@click.option(
    "--model",
    default="qwen25-3B-Instruct",
    type=click.Choice(["qwen35-0.8B", "qwen35-2B", "qwen35-4B", "qwen25-3B-Instruct"]),
)
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]))
@click.option("--max-rounds", default=24, type=click.IntRange(1, 128))
@click.option("--max-tokens", default=128, type=click.IntRange(32, 1024))
@click.option("--skill-interface", default="atomic", type=click.Choice(["atomic", "plan_execute"]))
@click.option("--timeout", default=1800, type=click.IntRange(1, 7200))
def agent_command(suite, episode, out, model, device, max_rounds, max_tokens, skill_interface, timeout):
    """Run the actual shared task agent with a cached local model; paid providers disabled."""
    from emet.eval.agent_tasks.spec import default_suite, load_suite, select_episode
    from emet.utils.process_tree import popen_session, terminate_process_tree

    suite = Path(suite or default_suite()).resolve()
    select_episode(load_suite(suite), episode)
    if out.exists():
        raise click.ClickException("output already exists; use a fresh directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "emet.eval.agent_tasks.worker",
        "--suite",
        str(suite),
        "--episode",
        episode,
        "--out",
        str(out.resolve()),
        "--agent-model",
        model,
        "--device",
        device,
        "--max-rounds",
        str(max_rounds),
        "--max-tokens",
        str(max_tokens),
        "--skill-interface",
        skill_interface,
        "--timeout",
        str(timeout),
    ]
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MPLCONFIGDIR", "/tmp/emet-agent-mpl")
    log_path = out.with_suffix(".log")
    click.echo(f"Shared task agent: {model} on {device}; log: {log_path}")
    with log_path.open("x") as log:
        proc = popen_session(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process_tree(proc, grace_s=2)
            code = -1
        except BaseException:
            terminate_process_tree(proc, grace_s=2)
            raise
    path = out / "metrics.json"
    if not path.is_file() and (out / "manifest.json").is_file():
        from emet.eval.agent_tasks.recording import load_run, write_json
        from emet.eval.agent_tasks.visualize import export_run

        manifest, events, _ = load_run(out)
        last_score = next((e["evaluator"]["score"] for e in reversed(events) if "score" in e["evaluator"]), {})
        write_json(
            path,
            {
                "completed": 0,
                "total": len(manifest["episode"]["goals"]),
                **last_score,
                "success": False,
                "status": "timeout" if code == -1 else "native_failure",
                "event_hash": events[-1]["hash"] if events else "",
                "agent_ran": any(e["kind"] == "model_output" for e in events),
                "model_rounds": sum(e["kind"] == "model_output" for e in events),
                "actions": sum(e["kind"] == "tool_start" for e in events),
                "agent_status": "timeout" if code == -1 else "native_failure",
                "paid_cost_usd": 0,
                "evidence_complete": False,
                "interrupted": True,
            },
        )
        export_run(out)
    metrics = json.loads(path.read_text()) if path.is_file() else {"status": "native_failure"}
    click.echo(json.dumps(metrics, indent=2))
    if code or not metrics.get("evidence_complete"):
        raise click.ClickException(f"agent task or evidence gate failed; inspect {out}")


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


def _qa_worker(arguments, out, timeout):
    """Own the native/model worker and preserve its log on failure or deadline."""
    from emet.utils.process_tree import popen_session, terminate_process_tree

    if out.exists():
        raise click.ClickException("output already exists; choose a fresh directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".log")
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "4")
    cmd = [sys.executable, "-m", "emet.eval.agent_tasks.qa", "--out", str(out), *arguments]
    click.echo(f"QA worker log: {log_path}")
    with log_path.open("x") as log:
        proc = popen_session(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process_tree(proc, grace_s=2)
            raise click.ClickException(f"QA worker timed out; retained evidence: {out}") from None
        except BaseException:
            terminate_process_tree(proc, grace_s=2)
            raise
    if code:
        raise click.ClickException(f"QA worker failed; inspect {log_path}")
    click.echo(f"QA output: {out}")


@agent_tasks_group.command("qa-build")
@suite_option
@click.option("--out", required=True, type=click.Path(path_type=Path))
@click.option("--timeout", default=180, type=click.IntRange(1, 600))
def qa_build_command(suite, out, timeout):
    """Render ten QA cases, enforcing target readability; no model calls."""
    from emet.eval.agent_tasks.spec import default_suite

    _qa_worker(["--suite", str(suite or default_suite())], out, timeout)


@agent_tasks_group.command("qa-agent")
@click.argument("public", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out", required=True, type=click.Path(path_type=Path))
@click.option("--model", default="qwen35-0.8B", type=click.Choice(["qwen35-0.8B", "qwen35-2B", "qwen35-4B"]))
@click.option("--question", multiple=True)
@click.option("--max-rounds", default=8, type=click.IntRange(1, 32))
@click.option("--timeout", default=1800, type=click.IntRange(1, 7200))
def qa_agent_command(public, out, model, question, max_rounds, timeout):
    """Run image-scoped QA through the shared task agent and a cached CPU VLM."""
    args = ["--public", str(public), "--model", model, "--max-rounds", str(max_rounds)]
    for qid in question:
        args.extend(["--question", qid])
    _qa_worker(args, out, timeout)


@agent_tasks_group.command("qa-score")
@click.argument("pack", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--answers", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", type=click.Path(path_type=Path))
def qa_score_command(pack, answers, out):
    """Score structured answers and evidence IDs against the private key."""
    from emet.eval.agent_tasks.qa import export_gallery, load_public, score_answers

    try:
        responses = json.loads(answers.read_text())
        result = score_answers(pack, responses)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
    if out:
        report = out.with_suffix(".html")
        if out.exists() or report.exists():
            raise click.ClickException("score or report output already exists")
        with out.open("x") as f:
            json.dump(result, f, indent=2)
        export_gallery(
            pack,
            load_public(pack / "public"),
            json.loads((pack / "private_answers.json").read_text())["keys"],
            responses=responses,
            destination=report,
        )
        click.echo(f"QA result gallery: {report}")
    click.echo(json.dumps(result, indent=2))
    if not result["success"]:
        raise click.ClickException("QA answers or evidence failed")

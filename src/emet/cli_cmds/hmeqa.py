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
from click.core import ParameterSource

from emet.cli_cmds.bootstrap import (
    _project_root,
)


@click.group("hmeqa", short_help="HM-EQA classic vs agentic H2H helpers")
def hmeqa_group() -> None:
    """Dogfood entrypoints for Habitat HM-EQA head-to-head runs.

    Prefer these over hand-built ``env … taskset … ./scripts/run_hmeqa_*.sh`` lines.

    \b
    Examples:
      emet eval recover --need-mib 12000
      emet hmeqa resume
      emet hmeqa status
      emet hmeqa h2h --out OUT --resume --ids 15,68,105,17
      emet hmeqa overnight
      emet hmeqa inspect OUT --qid 105 --open rgb
      emet hmeqa significance OUT
      emet hmeqa ladder RUN_DIR --require-balanced32-gate
      emet hmeqa h2h --preset paper-router --ids 15,56,65,68
      emet status tail
    """


def _hmeqa_apply_preset(
    ctx: click.Context,
    *,
    preset: str | None,
    agentic_verifier: str,
    require_verified: bool,
    agentic_router: bool,
) -> tuple[str, bool, bool]:
    """Apply ``paper-router`` only where Click defaults were left untouched."""
    if (preset or "").strip().lower() != "paper-router":
        return agentic_verifier, require_verified, agentic_router
    from emet.eval.harness import apply_paper_router_preset

    return apply_paper_router_preset(
        agentic_verifier=agentic_verifier,
        require_verified=require_verified,
        agentic_router=agentic_router,
        verifier_source=ctx.get_parameter_source("agentic_verifier"),
        verified_source=ctx.get_parameter_source("require_verified"),
        router_source=ctx.get_parameter_source("agentic_router"),
    )


@hmeqa_group.command("status", short_help="Show OUT progress, crashes, scored counts")
@click.argument("out_dir", required=False)
def hmeqa_status(out_dir: str | None) -> None:
    from emet.eval.harness import count_crash_markers, resolve_hmeqa_out

    out = resolve_hmeqa_out(out_dir)
    click.echo(f"OUT={out}")
    progress = out / "progress.json"
    if progress.is_file():
        click.echo(progress.read_text(encoding="utf-8").rstrip())
    else:
        click.echo("(no progress.json)")
    scored_c = len([p for p in out.glob("classic_q*.jsonl") if p.stat().st_size > 0])
    scored_a = len([p for p in out.glob("agentic_q*.jsonl") if p.stat().st_size > 0])
    crashes = count_crash_markers(out)
    click.echo(f"scored classic={scored_c} agentic={scored_a} crash_markers={crashes}")
    for cap in sorted(out.glob("native_crash_*.log")) + sorted(out.glob("host_freeze_*.log")):
        click.echo(f"capsule {cap.name}")


@hmeqa_group.command("summarize", short_help="Run summarize_hmeqa_agentic_h2h.py on OUT")
@click.argument("out_dir", required=False)
def hmeqa_summarize(out_dir: str | None) -> None:
    from emet.eval.harness import resolve_hmeqa_out

    out = resolve_hmeqa_out(out_dir)
    script = _project_root() / "scripts" / "summarize_hmeqa_agentic_h2h.py"
    rc = subprocess.call([sys.executable, str(script), str(out)], cwd=str(_project_root()))
    sys.exit(rc)


@hmeqa_group.command(
    "significance",
    short_help="Paired McNemar / Wilcoxon / bootstrap on classic vs agentic H2H",
)
@click.argument("out_dir", required=False)
@click.option(
    "--from-summary",
    "from_summary",
    type=click.Path(path_type=Path),
    default=None,
    help="Load h2h_summary JSON instead of OUT/*.jsonl",
)
@click.option(
    "--json",
    "json_out",
    type=click.Path(path_type=Path),
    default=None,
    help="Write full result JSON (default: OUT/significance.json when out_dir set)",
)
@click.option("--n-boot", default=5000, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
def hmeqa_significance(
    out_dir: str | None,
    from_summary: Path | None,
    json_out: Path | None,
    n_boot: int,
    seed: int,
) -> None:
    """Dogfood wrapper around ``emet.eval.hmeqa_significance``."""
    from emet.eval.hmeqa_significance import main as significance_main

    argv: list[str] = []
    if out_dir:
        argv.append(out_dir)
    if from_summary is not None:
        argv.extend(["--from-summary", str(from_summary)])
    if json_out is not None:
        argv.extend(["--json", str(json_out)])
    argv.extend(["--n-boot", str(n_boot), "--seed", str(seed)])
    sys.exit(significance_main(argv))


@hmeqa_group.command(
    "failures",
    short_help="Attribute classic vs agentic letter failures (context gaps)",
)
@click.argument("out_dir", required=False)
@click.option(
    "--from-summary",
    "from_summary",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional h2h_summary JSON (enrich with OUT traces when out_dir set)",
)
@click.option(
    "--json",
    "json_out",
    type=click.Path(path_type=Path),
    default=None,
    help="Write failure_report.json (default: OUT/failure_report.json)",
)
def hmeqa_failures(
    out_dir: str | None,
    from_summary: Path | None,
    json_out: Path | None,
) -> None:
    """Offline classic_only / context-gap attribution from H2H OUT + traces."""
    from emet.eval.hmeqa_failures import main as failures_main

    argv: list[str] = []
    if out_dir:
        argv.append(out_dir)
    if from_summary is not None:
        argv.extend(["--from-summary", str(from_summary)])
    if json_out is not None:
        argv.extend(["--json", str(json_out)])
    sys.exit(failures_main(argv))


@hmeqa_group.command(
    "inspect",
    short_help="Episode score + assess/explore + feh/mpv paths (replaces one-off JSON dumps)",
)
@click.argument("out_dir", required=False)
@click.option("--qid", type=int, default=None, help="Question id to inspect.")
@click.option("--arm", default="agentic", show_default=True, help="classic or agentic.")
@click.option(
    "--misses",
    is_flag=True,
    help="List incorrect scored episodes (no --qid needed).",
)
@click.option(
    "--open",
    "open_kind",
    type=click.Choice(["rgb", "frames", "images", "frontier", "maps", "video"]),
    default=None,
    help="Launch feh/mpv on that media set (requires DISPLAY).",
)
@click.option("--json", "as_json", is_flag=True, help="Print full JSON payload.")
def hmeqa_inspect(
    out_dir: str | None,
    qid: int | None,
    arm: str,
    misses: bool,
    open_kind: str | None,
    as_json: bool,
) -> None:
    """Summarize one episode (or list misses) and print copy-paste viewer commands.

    \b
    Examples:
      emet hmeqa inspect OUT --qid 105
      emet hmeqa inspect OUT --misses
      emet hmeqa inspect OUT --qid 105 --open rgb
    """
    from emet.eval.harness import resolve_hmeqa_out
    from emet.eval.hmeqa_inspect import (
        format_inspect_text,
        inspect_episode,
        list_scored_episodes,
        open_media,
    )

    out = resolve_hmeqa_out(out_dir)
    if misses:
        rows = list_scored_episodes(out, arm=arm)
        bad = [r for r in rows if not r.get("correct")]
        if as_json:
            click.echo(json.dumps({"out_dir": str(out), "misses": bad}, indent=2))
        else:
            click.echo(f"OUT={out}  arm={arm}  scored={len(rows)}  misses={len(bad)}")
            for r in bad:
                q = (str(r.get("question") or ""))[:90]
                click.echo(f"  q{r.get('qid')} pred={r.get('predicted')} gold={r.get('gold')}  {q}")
        if qid is None:
            return
    if qid is None:
        raise click.UsageError("provide --qid N (or --misses alone)")
    payload = inspect_episode(out, qid, arm=arm)
    if as_json:
        click.echo(json.dumps(payload, indent=2, default=str))
    else:
        click.echo(format_inspect_text(payload))
    if open_kind:
        try:
            pid = open_media(open_kind, payload.get("media") or {})
        except (FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"opened {open_kind} (pid={pid})")


@hmeqa_group.command(
    "ladder",
    short_help="Summarize probe/holdout ladder runs; optional balanced-32 gate",
)
@click.argument("run_dirs", nargs=-1, required=True, type=click.Path(path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option(
    "--require-balanced32-gate",
    is_flag=True,
    help="Exit 2 unless probe has verified answers and zero forced submits",
)
def hmeqa_ladder(
    run_dirs: tuple[Path, ...],
    output: Path | None,
    require_balanced32_gate: bool,
) -> None:
    """Summarize agentic ladder metrics (accuracy, selective risk, fused verify, …)."""
    from emet.eval.agentic_metrics import (
        balanced32_gate,
        summarize_policy_metrics,
        summarize_run,
    )

    reports = [summarize_run(path) for path in run_dirs]
    combined_episodes = [episode for report in reports for episode in report["episodes"]]
    combined = {
        "runs": reports,
        "summary": summarize_policy_metrics(combined_episodes),
    }
    passed, reasons = balanced32_gate(combined)
    combined["balanced32_gate"] = {"passed": passed, "reasons": reasons}
    text = json.dumps(combined, indent=2) + "\n"
    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    click.echo(text, nl=False)
    if require_balanced32_gate and not passed:
        sys.exit(2)
    sys.exit(0)


def _hmeqa_nested_value(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        value = value[part]
    return value


_HMEQA_FROZEN_PARAMETER_PATHS = {
    "arms": "evaluation.arms",
    "holdout_ids": "ids.question_ids",
    "agentic_verifier": "evaluation.agentic_verifier",
    "require_verified": "evaluation.require_verified",
    "agentic_router": "evaluation.agentic_router",
    "use_hm3d_semantics": "evaluation.use_hm3d_semantics",
    "use_enrich_labels": "evaluation.use_enrich_labels",
    "decision_policy": "variant.agentic_decision_policy",
    "graph_evidence_mode": "variant.graph_evidence_mode",
    "room_history_mode": "variant.room_history_mode",
    "room_policy": "variant.room_policy",
    "room_target_hints": "variant.room_target_hints",
    "investigate_stamp": "variant.investigate_stamp",
    "attempt_ledger_mode": "variant.attempt_ledger_mode",
    "action_progress_mode": "variant.action_progress_mode",
    "variant_id": "variant.id",
    "eqa_hf_model_id": "model.requested_hf_model_id",
    "eqa_vl_family": "model.vl_family",
    "eqa_vl_quantization": "model.vl_quantization",
    "eqa_answer_max_new_tokens": "budgets.answer_max_new_tokens",
    "host": "model.host",
    "vl_endpoint": "model.vl_endpoint",
    "vl_port": "model.vl_port",
    "episode_timeout": "budgets.episode_timeout_seconds",
    "max_planning_steps": "budgets.max_planning_steps",
    "max_movement_step": "budgets.max_movement_step",
}


def _hmeqa_frozen_options(fn):
    """Shared behavior/model/budget flags for ``hmeqa h2h`` and ``resume``."""
    options = [
        click.option(
            "--agentic-verifier",
            type=click.Choice(["none", "owlv2", "yoloe"]),
            default="none",
            show_default=True,
            help="Hybrid presence backend for the agentic arm.",
        ),
        click.option(
            "--require-verified/--allow-unverified",
            default=True,
            show_default=True,
            help="Require fused evidence before submit (the exhaustion ladder may still answer).",
        ),
        click.option(
            "--agentic-router/--no-agentic-router",
            default=False,
            show_default=True,
            help="Use VLM tool routing (fallback policy is deterministic).",
        ),
        click.option(
            "--use-hm3d-semantics/--no-hm3d-semantics",
            default=False,
            show_default=True,
            help="Use HM3D semantic sensor labels (GT-derived oracle axis).",
        ),
        click.option(
            "--enrich-labels/--no-enrich-labels",
            "use_enrich_labels",
            default=False,
            show_default=True,
            help="Seed per-question GraphEQA GT object hints (separate oracle axis).",
        ),
        click.option(
            "--decision-policy",
            type=click.Choice(["legacy", "grounded_v2"]),
            default="legacy",
            show_default=True,
            help="Agentic decision implementation gate.",
        ),
        click.option(
            "--graph-evidence-mode",
            type=click.Choice(["off", "shadow", "agent"]),
            default="off",
            show_default=True,
            help="Stable graph evidence rollout mode.",
        ),
        click.option(
            "--room-history-mode",
            type=click.Choice(["off", "shadow", "agent"]),
            default="off",
            show_default=True,
            help="Room-history collection/visibility mode.",
        ),
        click.option(
            "--room-policy",
            type=click.Choice(["canonical", "llm"]),
            default="canonical",
            show_default=True,
        ),
        click.option(
            "--room-target-hints/--no-room-target-hints",
            default=True,
            show_default=True,
            help="Expose the legacy question-derived target-room hints.",
        ),
        click.option(
            "--investigate-stamp/--no-investigate-stamp",
            default=False,
            show_default=True,
            help="Write room timeline stamps after investigate (known letter-regression axis).",
        ),
        click.option(
            "--attempt-ledger-mode",
            type=click.Choice(["off", "shadow", "agent"]),
            default="off",
            show_default=True,
            help="Attempt-ledger collection/visibility mode.",
        ),
        click.option(
            "--action-progress-mode",
            type=click.Choice(["off", "shadow", "enforce"]),
            default="off",
            show_default=True,
            help=(
                "Static-world duplicate-action policy: shadow records decisions; "
                "enforce suppresses unchanged terminal/no-progress variants."
            ),
        ),
        click.option(
            "--variant-id",
            default="legacy",
            show_default=True,
            help="Stable A/B label recorded in run_manifest.json and episode environments.",
        ),
        click.option(
            "--preset",
            type=click.Choice(["paper-router"]),
            default=None,
            help=(
                "paper-router: none verifier + allow-unverified + agentic-router. "
                "It never changes the explicit A/B axes; explicit flags still win."
            ),
        ),
        click.option(
            "--eqa-hf-model-id",
            default=None,
            help="Override HF VLM id (sets EQA_HF_MODEL_ID in the Habitat child).",
        ),
        click.option(
            "--eqa-vl-family",
            default=None,
            help="Override VL family (sets EQA_VL_FAMILY in the Habitat child).",
        ),
        click.option(
            "--eqa-vl-quantization",
            default=None,
            help="Freeze the effective VL quantization label (default: config int4).",
        ),
        click.option(
            "--eqa-answer-max-new-tokens",
            type=int,
            default=384,
            show_default=True,
            help="Per-answer VLM decode cap.",
        ),
        click.option("--episode-timeout", type=int, default=7200, show_default=True),
        click.option("--max-planning-steps", type=int, default=20, show_default=True),
        click.option("--max-movement-step", type=int, default=10, show_default=True),
        click.option(
            "--host",
            default=None,
            help=(
                "LAN LLM host. Injects EMET_LLM_HOST, EMET_OPENAI_BASE_URL, and EMET_VL_ENDPOINT into the managed job."
            ),
        ),
        click.option(
            "--vl-endpoint",
            default=None,
            help="Override EMET_VL_ENDPOINT (wins over --host's default).",
        ),
        click.option(
            "--vl-port",
            type=int,
            default=None,
            help="With --host: VL OpenAI port (default 8000; dual-2b uses 8001).",
        ),
    ]
    for option in reversed(options):
        fn = option(fn)
    return fn


def _hmeqa_reuse_frozen_defaults(
    ctx: click.Context,
    *,
    out: Path,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Reuse frozen values for omitted resume flags; explicit mismatches fail later."""
    from emet.eval.hmeqa_launch import (
        HmeqaRunManifestError,
        load_hmeqa_run_manifest,
        normalize_hmeqa_run_config,
    )

    try:
        manifest = load_hmeqa_run_manifest(out)
        config = normalize_hmeqa_run_config(manifest["config"])
    except (KeyError, HmeqaRunManifestError) as exc:
        raise click.ClickException(str(exc)) from exc
    result = dict(values)
    for parameter, path in _HMEQA_FROZEN_PARAMETER_PATHS.items():
        if ctx.get_parameter_source(parameter) is not ParameterSource.DEFAULT:
            continue
        value = _hmeqa_nested_value(config, path)
        if parameter in {"arms", "holdout_ids"}:
            value = ",".join(str(item) for item in value)
        elif parameter == "eqa_hf_model_id" and value is None:
            value = config["model"].get("hf_model_id")
        result[parameter] = value
    return result


def _hmeqa_config_sources(ctx: click.Context, *, preset: str | None) -> dict[str, str]:
    """Record where each frozen effective value came from on the first launch."""
    source_labels = {
        ParameterSource.COMMANDLINE: "command_line",
        ParameterSource.ENVIRONMENT: "environment",
        ParameterSource.DEFAULT_MAP: "default_map",
        ParameterSource.DEFAULT: "cli_default",
        ParameterSource.PROMPT: "prompt",
    }
    result: dict[str, str] = {}
    for parameter, path in _HMEQA_FROZEN_PARAMETER_PATHS.items():
        source = ctx.get_parameter_source(parameter)
        label = source_labels.get(source, str(source or "unknown").lower())
        if source is ParameterSource.DEFAULT and parameter in {
            "eqa_hf_model_id",
            "eqa_vl_family",
            "eqa_vl_quantization",
            "eqa_answer_max_new_tokens",
        }:
            label = "config_default"
        if (
            preset == "paper-router"
            and source is ParameterSource.DEFAULT
            and parameter in {"agentic_verifier", "require_verified", "agentic_router"}
        ):
            label = "preset:paper-router"
        result[path] = label
        if parameter == "eqa_hf_model_id":
            result["model.hf_model_id"] = label
    if (
        ctx.get_parameter_source("vl_endpoint") is ParameterSource.DEFAULT
        and ctx.get_parameter_source("host") is not ParameterSource.DEFAULT
    ):
        result["model.vl_endpoint"] = "derived:model.host"
    if (
        ctx.get_parameter_source("host") is not ParameterSource.DEFAULT
        or ctx.get_parameter_source("vl_endpoint") is not ParameterSource.DEFAULT
    ):
        result["model.hf_model_id"] = "derived:remote_vl"
    result["model.llm_port"] = "config_default"
    result["budgets.agentic_max_tool_rounds"] = "config_default"
    result["budgets.agentic_max_nav_steps"] = "config_default"
    result["inputs.data_dir"] = (
        "environment:HABITAT_EQA_DATA_DIR" if os.environ.get("HABITAT_EQA_DATA_DIR", "").strip() else "config_default"
    )
    result["inputs.hm3d_root"] = (
        "environment:HM3D_SCENE_DIR" if os.environ.get("HM3D_SCENE_DIR", "").strip() else "config_default"
    )
    return result


def _hmeqa_apply_variant_config(
    ctx: click.Context,
    *,
    path: Path | None,
    values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Apply a strict variant file beneath explicit CLI flags."""
    if path is None:
        return dict(values), {}
    from emet.eval.hmeqa_launch import HmeqaRunManifestError, load_hmeqa_variant_config

    try:
        configured, source = load_hmeqa_variant_config(path)
    except HmeqaRunManifestError as exc:
        raise click.ClickException(str(exc)) from exc
    result = dict(values)
    source_overrides: dict[str, str] = {}
    for parameter, value in configured.items():
        if ctx.get_parameter_source(parameter) is not ParameterSource.DEFAULT:
            continue
        result[parameter] = value
        source_overrides[_HMEQA_FROZEN_PARAMETER_PATHS[parameter]] = source
    return result, source_overrides


def _hmeqa_frozen_values(local_values: dict[str, Any]) -> dict[str, Any]:
    """Select only manifest-frozen callback values from ``locals()``."""
    return {name: local_values[name] for name in _HMEQA_FROZEN_PARAMETER_PATHS}


def _hmeqa_launch(
    *,
    out: Path,
    resume: bool,
    frozen_values: dict[str, Any],
    coverage_qids: str,
    cooldown: int,
    crash_policy: str,
    streak_abort: int,
    job_name: str,
    need_mib: int,
    foreground: bool,
    description: str | None = None,
    config_sources: dict[str, str] | None = None,
) -> None:
    """Register H2H via ``emet jobs run`` (cpu-safe + gpu-exclusive defaults)."""
    from emet.eval.hmeqa_launch import (
        HmeqaRunManifestError,
        build_hmeqa_run_config,
        load_hmeqa_run_manifest,
        prepare_hmeqa_run_manifest,
    )

    root = _project_root()
    values = dict(frozen_values)
    arms = str(values["arms"])
    ids = str(values["holdout_ids"])
    data_dir = os.environ.get("HABITAT_EQA_DATA_DIR", "").strip() or None
    hm3d_root = os.environ.get("HM3D_SCENE_DIR", "").strip() or None
    try:
        if resume and (out / "run_manifest.json").is_file():
            frozen_inputs = load_hmeqa_run_manifest(out)["config"]["inputs"]
            data_dir = data_dir or frozen_inputs["data_dir"]
            hm3d_root = hm3d_root or frozen_inputs["hm3d_root"]
        run_config = build_hmeqa_run_config(
            arms=arms,
            ids=ids,
            agentic_verifier=values["agentic_verifier"],
            require_verified=values["require_verified"],
            agentic_router=values["agentic_router"],
            use_hm3d_semantics=values["use_hm3d_semantics"],
            use_enrich_labels=values["use_enrich_labels"],
            decision_policy=values["decision_policy"],
            graph_evidence_mode=values["graph_evidence_mode"],
            room_history_mode=values["room_history_mode"],
            room_policy=values["room_policy"],
            room_target_hints=values["room_target_hints"],
            investigate_stamp=values["investigate_stamp"],
            attempt_ledger_mode=values["attempt_ledger_mode"],
            action_progress_mode=values["action_progress_mode"],
            variant_id=values["variant_id"],
            eqa_hf_model_id=values["eqa_hf_model_id"],
            eqa_vl_family=values["eqa_vl_family"],
            eqa_vl_quantization=values["eqa_vl_quantization"],
            eqa_answer_max_new_tokens=values["eqa_answer_max_new_tokens"],
            host=values["host"],
            vl_endpoint=values["vl_endpoint"],
            vl_port=values["vl_port"],
            episode_timeout_seconds=values["episode_timeout"],
            max_planning_steps=values["max_planning_steps"],
            max_movement_step=values["max_movement_step"],
            data_dir=data_dir,
            hm3d_root=hm3d_root,
        )
        manifest = prepare_hmeqa_run_manifest(
            out,
            project_root=root,
            config=run_config,
            sources=config_sources,
            resume=resume,
        )
    except HmeqaRunManifestError as exc:
        raise click.ClickException(str(exc)) from exc
    run_config = manifest["config"]
    vl_ep = str(run_config["model"].get("vl_endpoint") or "")
    # Re-enter CLI so jobs run applies mutex/affinity wrapper.
    cmd = [
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
        str(out),
    ]
    if description and str(description).strip():
        cmd.extend(["--description", str(description).strip()])
    if foreground:
        cmd.append("--foreground")
    cmd.extend(
        [
            "--",
            sys.executable,
            "-m",
            "emet.eval.hmeqa_launch",
            "run-child",
            "--out",
            str(out),
            "--resume",
            str(int(resume)),
            "--coverage-qids",
            coverage_qids,
            "--cooldown",
            str(int(cooldown)),
            "--crash-policy",
            crash_policy,
            "--streak-abort",
            str(int(streak_abort)),
        ]
    )
    click.echo(
        f"launching via emet jobs: OUT={out} resume={int(resume)} arms={arms} "
        f"variant={run_config['variant']['id']} digest={manifest['config_digest']}",
        err=True,
    )
    if vl_ep:
        click.echo(f"EQA VL endpoint (injected into job env): {vl_ep}", err=True)
    elif values["host"] or values["vl_endpoint"]:
        click.echo("warning: host/vl-endpoint set but EMET_VL_ENDPOINT missing from env parts", err=True)
    rc = subprocess.call(cmd, cwd=str(root))
    sys.exit(rc)


@hmeqa_group.command("h2h", short_help="Launch classic vs agentic H2H via emet jobs")
@click.argument("out_dir", required=False)
@click.option("--resume", is_flag=True, help="Skip hash-validated COMPLETE markers.")
@click.option("--arms", default="classic,agentic", show_default=True)
@click.option(
    "--ids",
    "holdout_ids",
    default=None,
    help="Comma-separated question ids (default: bal-32 list).",
)
@click.option("--coverage-qids", default="15,28,47", show_default=True)
@click.option("--cooldown", type=int, default=20, show_default=True, help="EPISODE_COOLDOWN_SEC")
@click.option(
    "--crash-policy",
    type=click.Choice(["skip", "abort"]),
    default="skip",
    show_default=True,
    help="skip=continue after settle; abort=stop batch on first native crash.",
)
@click.option(
    "--streak-abort",
    type=int,
    default=2,
    show_default=True,
    help="Under skip: abort after N consecutive native crashes (0=never).",
)
@_hmeqa_frozen_options
@click.option(
    "--variant-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Strict YAML file for the complete variant axes. Explicit variant flags override "
        "the file; effective values and the file digest are frozen in run_manifest.json."
    ),
)
@click.option("--job-name", default="hmeqa-h2h", show_default=True)
@click.option(
    "--description",
    "-d",
    default=None,
    help="Human why/what for this run (stored on the job; shown in emet jobs).",
)
@click.option("--need-mib", type=int, default=12000, show_default=True)
@click.option("--foreground", is_flag=True)
@click.pass_context
def hmeqa_h2h(
    ctx: click.Context,
    out_dir: str | None,
    resume: bool,
    arms: str,
    holdout_ids: str | None,
    coverage_qids: str,
    cooldown: int,
    crash_policy: str,
    streak_abort: int,
    agentic_verifier: str,
    require_verified: bool,
    agentic_router: bool,
    use_hm3d_semantics: bool,
    use_enrich_labels: bool,
    decision_policy: str,
    graph_evidence_mode: str,
    room_history_mode: str,
    room_policy: str,
    room_target_hints: bool,
    investigate_stamp: bool,
    attempt_ledger_mode: str,
    action_progress_mode: str,
    variant_id: str,
    preset: str | None,
    eqa_hf_model_id: str | None,
    eqa_vl_family: str | None,
    eqa_vl_quantization: str | None,
    eqa_answer_max_new_tokens: int,
    episode_timeout: int,
    max_planning_steps: int,
    max_movement_step: int,
    host: str | None,
    vl_endpoint: str | None,
    vl_port: int | None,
    variant_config: Path | None,
    job_name: str,
    description: str | None,
    need_mib: int,
    foreground: bool,
) -> None:
    from emet.eval.harness import DEFAULT_BAL32_IDS

    if resume and variant_config is not None:
        raise click.ClickException("--variant-config is first-launch only; resume reuses the frozen run manifest")
    if out_dir:
        out = Path(out_dir).expanduser().resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = Path.home() / "runs" / "emet" / f"hmeqa_agentic_h2h_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    frozen, variant_sources = _hmeqa_apply_variant_config(
        ctx,
        path=variant_config,
        values=_hmeqa_frozen_values(locals()),
    )
    if resume:
        frozen = _hmeqa_reuse_frozen_defaults(ctx, out=out, values=frozen)
    (
        frozen["agentic_verifier"],
        frozen["require_verified"],
        frozen["agentic_router"],
    ) = _hmeqa_apply_preset(
        ctx,
        preset=preset,
        agentic_verifier=frozen["agentic_verifier"],
        require_verified=frozen["require_verified"],
        agentic_router=frozen["agentic_router"],
    )
    frozen["holdout_ids"] = frozen["holdout_ids"] or DEFAULT_BAL32_IDS
    config_sources = _hmeqa_config_sources(ctx, preset=preset)
    config_sources.update(variant_sources)
    _hmeqa_launch(
        out=out,
        resume=resume,
        frozen_values=frozen,
        coverage_qids=coverage_qids,
        cooldown=cooldown,
        crash_policy=crash_policy,
        streak_abort=streak_abort,
        job_name=job_name,
        need_mib=need_mib,
        foreground=foreground,
        description=description,
        config_sources=config_sources,
    )


@hmeqa_group.command("resume", short_help="Resume latest (or given) H2H OUT under safe defaults")
@click.argument("out_dir", required=False)
@click.option("--arms", default="classic,agentic", show_default=True)
@click.option("--ids", "holdout_ids", default=None, help="Override ids (default: from STATUS / bal-32).")
@click.option("--coverage-qids", default="15,28,47", show_default=True)
@click.option("--cooldown", type=int, default=30, show_default=True)
@click.option("--crash-policy", type=click.Choice(["skip", "abort"]), default="skip", show_default=True)
@click.option("--streak-abort", type=int, default=2, show_default=True)
@_hmeqa_frozen_options
@click.option("--job-name", default="hmeqa-h2h-resume", show_default=True)
@click.option(
    "--description",
    "-d",
    default=None,
    help="Human why/what for this run (stored on the job; shown in emet jobs).",
)
@click.option("--need-mib", type=int, default=12000, show_default=True)
@click.option("--foreground", is_flag=True)
@click.pass_context
def hmeqa_resume(
    ctx: click.Context,
    out_dir: str | None,
    arms: str,
    holdout_ids: str | None,
    coverage_qids: str,
    cooldown: int,
    crash_policy: str,
    streak_abort: int,
    agentic_verifier: str,
    require_verified: bool,
    agentic_router: bool,
    use_hm3d_semantics: bool,
    use_enrich_labels: bool,
    decision_policy: str,
    graph_evidence_mode: str,
    room_history_mode: str,
    room_policy: str,
    room_target_hints: bool,
    investigate_stamp: bool,
    attempt_ledger_mode: str,
    action_progress_mode: str,
    variant_id: str,
    preset: str | None,
    eqa_hf_model_id: str | None,
    eqa_vl_family: str | None,
    eqa_vl_quantization: str | None,
    eqa_answer_max_new_tokens: int,
    episode_timeout: int,
    max_planning_steps: int,
    max_movement_step: int,
    host: str | None,
    vl_endpoint: str | None,
    vl_port: int | None,
    job_name: str,
    description: str | None,
    need_mib: int,
    foreground: bool,
) -> None:
    from emet.eval.harness import (
        DEFAULT_BAL32_IDS,
        detect_host_freeze,
        resolve_hmeqa_out,
        write_host_freeze_capsule,
    )

    out = resolve_hmeqa_out(out_dir)
    frozen = _hmeqa_reuse_frozen_defaults(
        ctx,
        out=out,
        values=_hmeqa_frozen_values(locals()),
    )
    (
        frozen["agentic_verifier"],
        frozen["require_verified"],
        frozen["agentic_router"],
    ) = _hmeqa_apply_preset(
        ctx,
        preset=preset,
        agentic_verifier=frozen["agentic_verifier"],
        require_verified=frozen["require_verified"],
        agentic_router=frozen["agentic_router"],
    )
    freeze = detect_host_freeze(out)
    if freeze:
        cap = write_host_freeze_capsule(out, freeze)
        click.echo(f"host-freeze capsule → {cap}", err=True)
    frozen["holdout_ids"] = frozen["holdout_ids"] or DEFAULT_BAL32_IDS
    _hmeqa_launch(
        out=out,
        resume=True,
        frozen_values=frozen,
        coverage_qids=coverage_qids,
        cooldown=cooldown,
        crash_policy=crash_policy,
        streak_abort=streak_abort,
        job_name=job_name,
        need_mib=need_mib,
        foreground=foreground,
        description=description,
        config_sources=_hmeqa_config_sources(ctx, preset=preset),
    )


@hmeqa_group.command("overnight", short_help="Holdout-8 → gate → bal-32 via one emet jobs run")
@click.option(
    "--base",
    "base_dir",
    default=None,
    help=(
        "Overnight base dir (default: ~/runs/emet/hmeqa_overnight_<stamp>). "
        "Re-pass the same --base after emet jobs cancel to resume: skips DONE "
        "phases and RESUME=1 on partial H2H dirs."
    ),
)
@click.option("--holdout-ids", default=None, help="Default: paper holdout-8.")
@click.option("--bal32-ids", default=None, help="Default: balanced-32.")
@click.option("--gate-min-acc", type=float, default=0.25, show_default=True)
@click.option("--skip-bal32", is_flag=True, help="Stop after holdout (+ optional retune).")
@click.option(
    "--agentic-verifier",
    type=click.Choice(["none", "owlv2", "yoloe"]),
    default="none",
    show_default=True,
)
@click.option(
    "--require-verified/--allow-unverified",
    default=False,
    show_default=True,
    help="Overnight default: allow-unverified (require-verified abstains too often on bal-32).",
)
@click.option(
    "--agentic-router/--no-agentic-router",
    default=True,
    show_default=True,
    help="Overnight default: VLM tool routing on.",
)
@click.option("--cooldown", type=int, default=20, show_default=True)
@click.option("--crash-policy", type=click.Choice(["skip", "abort"]), default="skip", show_default=True)
@click.option("--streak-abort", type=int, default=2, show_default=True)
@click.option("--egl-fail-abort", type=int, default=2, show_default=True)
@click.option("--job-name", default="hmeqa-overnight", show_default=True)
@click.option(
    "--description",
    "-d",
    default=None,
    help="Human why/what for this run (stored on the job; shown in emet jobs).",
)
@click.option("--need-mib", type=int, default=12000, show_default=True)
@click.option("--foreground", is_flag=True)
def hmeqa_overnight(
    base_dir: str | None,
    holdout_ids: str | None,
    bal32_ids: str | None,
    gate_min_acc: float,
    skip_bal32: bool,
    agentic_verifier: str,
    require_verified: bool,
    agentic_router: bool,
    cooldown: int,
    crash_policy: str,
    streak_abort: int,
    egl_fail_abort: int,
    job_name: str,
    description: str | None,
    need_mib: int,
    foreground: bool,
) -> None:
    """Launch (or run in-process) the overnight holdout→bal32 ladder.

    When already inside ``emet jobs`` (``EMET_JOB_ID`` set), runs the orchestrator
    in-process so nested jobs are not created. Otherwise wraps one ``emet jobs run``.

    Pause with ``emet jobs cancel JOB_ID``. Resume by re-running this command with
    the same ``--base`` (skips ``DONE`` phases; keeps scored per-qid jsonl).
    """
    from emet.eval.harness import DEFAULT_BAL32_IDS, DEFAULT_HOLDOUT8_IDS
    from emet.eval.hmeqa_overnight import run_overnight

    if base_dir:
        base = Path(base_dir).expanduser().resolve()
    else:
        env_base = os.environ.get("OVERNIGHT_BASE", "").strip()
        if env_base:
            base = Path(env_base).expanduser().resolve()
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            base = Path.home() / "runs" / "emet" / f"hmeqa_overnight_{stamp}"
    base.mkdir(parents=True, exist_ok=True)

    ids_h = holdout_ids or os.environ.get("HOLDOUT8_IDS", "").strip() or DEFAULT_HOLDOUT8_IDS
    ids_b = bal32_ids or os.environ.get("BAL32_IDS", "").strip() or DEFAULT_BAL32_IDS

    # Already under a verified live job supervisor — do not nest.
    from emet.utils.job_registry import validated_current_job_id

    managed_job_id = validated_current_job_id()
    if managed_job_id:
        click.echo(f"overnight in-process (managed job {managed_job_id}): BASE={base}", err=True)
        rc = run_overnight(
            base=base,
            holdout_ids=ids_h,
            bal32_ids=ids_b,
            gate_min_acc=gate_min_acc,
            skip_bal32=skip_bal32,
            agentic_verifier=agentic_verifier,
            require_verified=require_verified,
            agentic_router=agentic_router,
            cooldown=cooldown,
            crash_policy=crash_policy,
            streak_abort=streak_abort,
            egl_fail_abort=egl_fail_abort,
        )
        sys.exit(rc)

    root = _project_root()
    inner_parts = [
        sys.executable,
        "-m",
        "emet.eval.hmeqa_overnight",
        "--base",
        str(base),
        "--holdout-ids",
        ids_h,
        "--bal32-ids",
        ids_b,
        "--gate-min-acc",
        str(gate_min_acc),
        "--agentic-verifier",
        agentic_verifier,
        "--cooldown",
        str(int(cooldown)),
        "--crash-policy",
        crash_policy,
        "--streak-abort",
        str(int(streak_abort)),
        "--egl-fail-abort",
        str(int(egl_fail_abort)),
    ]
    if skip_bal32:
        inner_parts.append("--skip-bal32")
    if require_verified:
        inner_parts.append("--require-verified")
    else:
        inner_parts.append("--allow-unverified")
    if agentic_router:
        inner_parts.append("--agentic-router")
    else:
        inner_parts.append("--no-agentic-router")

    cmd = [
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
        str(base),
    ]
    if description and str(description).strip():
        cmd.extend(["--description", str(description).strip()])
    if foreground:
        cmd.append("--foreground")
    cmd.extend(["--", *inner_parts])
    click.echo(f"launching overnight via emet jobs: BASE={base}", err=True)
    rc = subprocess.call(cmd, cwd=str(root))
    sys.exit(rc)


def register(main: click.Group) -> None:
    main.add_command(hmeqa_group)

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI: offline RoboVista robot-centric MCQ-VQA (HuggingFace ``sy-xie/robovista``)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click


@click.group("robovista", invoke_without_command=True)
@click.pass_context
def robovista_group(ctx: click.Context) -> None:
    """RoboVista offline MCQ-VQA (static images; not comparable to HM-EQA)."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@robovista_group.command("info", short_help="Print RoboVista Hub id and domain counts")
@click.option(
    "--max-questions",
    type=int,
    default=None,
    help="Optional cap when loading from Hub (still downloads full cache first).",
)
@click.option(
    "--domain",
    "domains",
    multiple=True,
    help="Filter domain (repeatable). Examples: domestic, industrial.",
)
def robovista_info(max_questions: int | None, domains: tuple[str, ...]) -> None:
    """Load RoboVista (first run ~1.1 GB HF cache) and print domain counts."""
    from emet.benchmarks.robovista.datasets import (
        ROBOVISTA_DOMAINS,
        ROBOVISTA_HF_ID,
        count_by_domain,
        load_robovista,
    )

    click.echo(f"HF dataset: {ROBOVISTA_HF_ID} (split=train)")
    click.echo(f"Known domains: {', '.join(ROBOVISTA_DOMAINS)}")
    click.echo("Loading… (first download embeds images, ~1.1 GB)")
    questions = load_robovista(domains=domains or None, max_questions=max_questions)
    counts = count_by_domain(questions)
    click.echo(f"Loaded {len(questions)} questions")
    for domain, n in counts.items():
        click.echo(f"  {domain}: {n}")


@robovista_group.command("run-batch", short_help="Score RoboVista with a local VLM or mock LLM")
@click.option(
    "--eqa-vl-family",
    default=None,
    help="VLM family (e.g. qwen3_vl), same naming as SQA3D/Habitat.",
)
@click.option(
    "--eqa-hf-model-id",
    default=None,
    help="Override HuggingFace model id for the EQA VLM.",
)
@click.option(
    "--device",
    default=None,
    help="Torch device (cuda / cpu). Default: auto from VLM builders.",
)
@click.option(
    "--domain",
    "domains",
    multiple=True,
    help="Filter domain (repeatable), e.g. --domain domestic.",
)
@click.option(
    "--ability-type",
    "ability_types",
    multiple=True,
    help="Filter ability_type (repeatable).",
)
@click.option("--max-questions", type=int, default=None, help="Cap number of questions.")
@click.option(
    "--mock-llm",
    is_flag=True,
    default=False,
    help="Skip VLM; answer with gold letter (CI / wiring smoke).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Default: ~/runs/emet/robovista/<timestamp>/",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip question ids already present in predictions.jsonl.",
)
def robovista_run_batch(
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    device: str | None,
    domains: tuple[str, ...],
    ability_types: tuple[str, ...],
    max_questions: int | None,
    mock_llm: bool,
    output_dir: Path | None,
    resume: bool,
) -> None:
    """Run offline RoboVista MCQ-VQA and write predictions.jsonl + summary.json."""
    from emet.benchmarks.robovista.runner import default_output_dir, run_robovista_batch

    if not mock_llm and eqa_vl_family is None and eqa_hf_model_id is None:
        raise click.UsageError("Provide --eqa-vl-family and/or --eqa-hf-model-id, or pass --mock-llm.")

    out = output_dir or default_output_dir(datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    summary = run_robovista_batch(
        domains=domains or None,
        ability_types=ability_types or None,
        max_questions=max_questions,
        mock_llm=mock_llm,
        eqa_vl_family=eqa_vl_family,
        eqa_hf_model_id=eqa_hf_model_id,
        device=device,
        output_dir=out,
        resume=resume,
    )
    click.echo(json.dumps(summary, indent=2))

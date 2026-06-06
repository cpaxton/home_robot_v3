# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""``emet-habitat`` CLI (runs inside ``.venv-habitat``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from emet.habitat.config import (
    default_habitat_eqa_data_dir,
    default_hm3d_scene_dir,
    questions_csv_path,
    scene_init_poses_csv_path,
)
from emet.habitat.datasets import load_hmeqa_questions
from emet.habitat.metrics import summarize_episodes, write_episode_jsonl


@click.group()
def main() -> None:
    """Habitat EQA harness for emet GraphEQA / Dynagraph."""


@main.command("list-questions")
@click.option("--limit", default=10, type=int, help="Max rows to print")
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
def list_questions(limit: int, data_dir: Path | None) -> None:
    """Print HM-EQA questions (requires downloaded CSV)."""
    path = (data_dir / "questions.csv") if data_dir else questions_csv_path()
    qs = load_hmeqa_questions(path)
    click.echo(f"{len(qs)} questions in {path}")
    for q in qs[:limit]:
        click.echo(f"  [{q.index}] {q.scene} floor={q.floor} answer={q.answer_letter} | {q.question[:80]}")


@main.command("info")
def info_cmd() -> None:
    """Print default data paths and env vars."""
    click.echo(f"HABITAT_EQA_DATA_DIR={default_habitat_eqa_data_dir()}")
    click.echo(f"HM3D_SCENE_DIR={default_hm3d_scene_dir()}")
    click.echo(f"questions.csv exists: {questions_csv_path().is_file()}")
    click.echo(f"scene_init_poses.csv exists: {scene_init_poses_csv_path().is_file()}")


def _eqa_cli_options(fn):
    opts = [
        click.option(
            "--eqa-vl-family",
            default=None,
            help="EQA VLM family: gemma4, qwen3_vl, qwen2_5_vl (default: dynav_config.yaml)",
        ),
        click.option("--eqa-hf-model-id", default=None, help="Override HF model id (e.g. google/gemma-3-4b-it)"),
        click.option("--device", default="cuda", help="VLM device (cuda, cpu, mps)"),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


@main.command("run-episode")
@click.option("--dataset", type=click.Choice(["hmeqa"]), default="hmeqa")
@click.option("--question-id", default=0, type=int)
@click.option("--method", type=click.Choice(["graph_eqa", "dynagraph"]), default="dynagraph")
@click.option("--mock-llm", is_flag=True, default=False, help="Use mocked EQA responses (smoke / CI)")
@click.option("--max-planning-steps", default=20, type=int, help="EQA planning iterations (GraphEQA ref: 20)")
@click.option("--max-movement-step", default=10, type=int, help="Nav substeps per planning iteration")
@click.option("--hm3d-root", type=click.Path(path_type=Path), default=None)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Write episode JSONL")
@click.option("--rotate-in-place/--no-rotate-in-place", default=True, help="Sweep heading before EQA")
@click.option(
    "--use-hm3d-semantics/--no-hm3d-semantics",
    default=None,
    help="Use HM3D semantic sensor for graph labels (default: auto if assets exist)",
)
@_eqa_cli_options
def run_episode(
    dataset: str,
    question_id: int,
    method: str,
    mock_llm: bool,
    max_planning_steps: int,
    max_movement_step: int,
    hm3d_root: Path | None,
    data_dir: Path | None,
    output: Path | None,
    rotate_in_place: bool,
    use_hm3d_semantics: bool | None,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    device: str,
) -> None:
    """Run one HM-EQA episode in Habitat-Sim."""
    if dataset != "hmeqa":
        raise click.ClickException(f"Unsupported dataset {dataset!r}")

    questions_path = (data_dir / "questions.csv") if data_dir else None
    init_poses_path = (data_dir / "scene_init_poses.csv") if data_dir else None

    from emet_habitat.runner import run_hmeqa_episode

    try:
        metrics = run_hmeqa_episode(
            question_id=question_id,
            method=method,
            mock_llm=mock_llm,
            max_planning_steps=max_planning_steps,
            max_movement_step=max_movement_step,
            hm3d_root=hm3d_root,
            questions_path=questions_path,
            init_poses_path=init_poses_path,
            rotate_in_place=rotate_in_place,
            use_hm3d_semantics=use_hm3d_semantics,
            eqa_vl_family=eqa_vl_family,
            eqa_hf_model_id=eqa_hf_model_id,
            device=device,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"{exc}\n\nDownload assets: uv run python scripts/download_habitat_eqa_data.py --instructions"
        ) from exc

    click.echo(json.dumps(metrics.to_dict(), indent=2))
    summary = summarize_episodes([metrics])
    click.echo(f"summary: {summary}")

    if output:
        write_episode_jsonl(output, [metrics])
        click.echo(f"wrote {output}")


@main.command("run-batch")
@click.option("--method", type=click.Choice(["graph_eqa", "dynagraph"]), default="graph_eqa")
@click.option("--question-start", default=0, type=int)
@click.option("--question-end", default=-1, type=int, help="Inclusive; -1 = last question in CSV")
@click.option("--mock-llm", is_flag=True, default=False)
@click.option("--max-planning-steps", default=20, type=int, help="EQA planning iterations (GraphEQA ref: 20)")
@click.option("--max-movement-step", default=10, type=int, help="Nav substeps per planning iteration")
@click.option("--hm3d-root", type=click.Path(path_type=Path), default=None)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
@click.option("--output", type=click.Path(path_type=Path), required=True, help="Write all episodes JSONL")
@click.option(
    "--use-hm3d-semantics/--no-hm3d-semantics",
    default=None,
    help="Use HM3D semantic sensor for graph labels (default: auto if assets exist)",
)
@_eqa_cli_options
def run_batch(
    method: str,
    question_start: int,
    question_end: int,
    mock_llm: bool,
    max_planning_steps: int,
    max_movement_step: int,
    hm3d_root: Path | None,
    data_dir: Path | None,
    output: Path,
    use_hm3d_semantics: bool | None,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    device: str,
) -> None:
    """Run a slice of HM-EQA (GraphEQA paper: 113 questions, method=graph_eqa)."""
    from emet_habitat.runner import run_hmeqa_batch

    questions_path = (data_dir / "questions.csv") if data_dir else None
    init_poses_path = (data_dir / "scene_init_poses.csv") if data_dir else None
    qs = load_hmeqa_questions(questions_path)
    end = len(qs) - 1 if question_end < 0 else min(question_end, len(qs) - 1)
    ids = list(range(max(0, question_start), end + 1))
    click.echo(f"Running {len(ids)} HM-EQA episodes ({method}, mock_llm={mock_llm})")

    try:
        metrics = run_hmeqa_batch(
            question_ids=ids,
            method=method,
            mock_llm=mock_llm,
            max_planning_steps=max_planning_steps,
            max_movement_step=max_movement_step,
            hm3d_root=hm3d_root,
            questions_path=questions_path,
            init_poses_path=init_poses_path,
            eqa_vl_family=eqa_vl_family or ("gemma4" if not mock_llm else None),
            eqa_hf_model_id=eqa_hf_model_id,
            device=device,
            use_hm3d_semantics=use_hm3d_semantics,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    write_episode_jsonl(output, metrics)
    summary = summarize_episodes(metrics)
    click.echo(f"wrote {output}")
    click.echo(f"summary: {summary}")


if __name__ == "__main__":
    try:
        main(standalone_mode=True)
    except click.ClickException as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

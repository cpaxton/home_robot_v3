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


@main.command("run-episode")
@click.option("--dataset", type=click.Choice(["hmeqa"]), default="hmeqa")
@click.option("--question-id", default=0, type=int)
@click.option("--method", type=click.Choice(["graph_eqa", "dynagraph"]), default="dynagraph")
@click.option("--mock-llm", is_flag=True, default=False, help="Use mocked EQA responses (smoke / CI)")
@click.option("--max-planning-steps", default=3, type=int)
@click.option("--hm3d-root", type=click.Path(path_type=Path), default=None)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Write episode JSONL")
@click.option("--rotate-in-place", is_flag=True, default=False)
def run_episode(
    dataset: str,
    question_id: int,
    method: str,
    mock_llm: bool,
    max_planning_steps: int,
    hm3d_root: Path | None,
    data_dir: Path | None,
    output: Path | None,
    rotate_in_place: bool,
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
            hm3d_root=hm3d_root,
            questions_path=questions_path,
            init_poses_path=init_poses_path,
            rotate_in_place=rotate_in_place,
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


if __name__ == "__main__":
    try:
        main(standalone_mode=True)
    except click.ClickException as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

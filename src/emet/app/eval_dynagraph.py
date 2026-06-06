# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""CLI: unified Dynagraph episode evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import click

from emet.memory.graph_eqa.dynagraph_eval import compute_dynagraph_eval


@click.command("eval-dynagraph")
@click.option("--episode", "episode_dir", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", default=None, type=click.Path())
@click.option("--match-xy-m", default=0.55, type=float, show_default=True)
@click.option("--bounds-iou-min", default=0.08, type=float, show_default=True)
@click.option(
    "--questions",
    "questions_path",
    default=None,
    type=click.Path(exists=True),
    help="Question bank YAML (scores eqa_results.json when present)",
)
@click.option("--question-env", default=None, help="Filter question bank environment tag")
def main(
    episode_dir: str,
    output_path: str | None,
    match_xy_m: float,
    bounds_iou_min: float,
    questions_path: str | None,
    question_env: str | None,
) -> None:
    """Score a Dynagraph export: explore, graph, fusion, GT, and EQA sections."""
    metrics = compute_dynagraph_eval(
        episode_dir,
        match_xy_m=match_xy_m,
        bounds_iou_min=bounds_iou_min,
        questions_path=questions_path,
        question_env=question_env,
    )
    text = json.dumps(metrics, indent=2)
    click.echo(text)
    if output_path:
        Path(output_path).write_text(text + "\n", encoding="utf-8")
        click.echo(f"Wrote -> {output_path}")

    g = metrics.get("graph", {})
    e = metrics.get("explore", {})
    eqa = metrics.get("eqa", {})
    click.echo(
        f"\nSummary: nodes={int(g.get('node_count', 0))} "
        f"explored={e.get('explored_area_m2', 0):.2f}m² "
        f"eqa_acc={eqa.get('accuracy', 'n/a')}"
    )


if __name__ == "__main__":
    main()

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

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
from emet.habitat.hmeqa_enrich_labels import HMEQA_PAPER_QUESTION_COUNT, hmeqa_paper_question_ids
from emet.habitat.metrics import compare_method_results, summarize_episodes, write_episode_jsonl
from emet.habitat.hm3d_semantics import compute_hmeqa_semantics_coverage, format_hmeqa_semantics_coverage_report


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
    if questions_csv_path().is_file():
        try:
            cov = compute_hmeqa_semantics_coverage()
            click.echo(
                "HM3D semantics (paper Q 0–112): "
                f"{len(cov.questions_with_semantics)}/{cov.paper_question_count} questions, "
                f"{len(cov.scenes_with_semantics)}/{cov.unique_paper_scenes} scenes "
                f"(train annotated {cov.train_scenes_with_semantics}/{cov.train_scene_count})"
            )
            click.echo(
                "  Report: uv run python scripts/download_habitat_eqa_data.py --report-hmeqa-semantics"
            )
        except FileNotFoundError:
            pass


def _eqa_cli_options(fn):
    opts = [
        click.option(
            "--eqa-vl-family",
            default=None,
            help="EQA VLM family: qwen3_vl, qwen3_5, qwen2_5_vl, gemma4 (default: dynav_config.yaml eqa.vl_family)",
        ),
        click.option("--eqa-hf-model-id", default=None, help="Override HF model id (e.g. google/gemma-3-4b-it)"),
        click.option("--device", default="cuda", help="VLM device (cuda, cpu, mps)"),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


def _frontier_cli_options(fn):
    opts = [
        click.option(
            "--frontier-nodes/--no-frontier-nodes",
            default=None,
            help="Mirror voxel frontiers as graph nodes for EQA (default: dynav_config.yaml)",
        ),
        click.option(
            "--frontier-keyword-weight",
            default=None,
            type=float,
            help="Keyword bias on fluid frontier sampling (0 = time heuristic only)",
        ),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


def _diagnostics_cli_options(fn):
    opts = [
        click.option("--export-map/--no-export-map", default=None, help="Write topdown_map.png per episode"),
        click.option("--export-video/--no-export-video", default=None, help="Write episode_rgb.mp4 per episode"),
        click.option("--map-stride", default=None, type=int, help="Save maps/step_NNNN.png every N updates"),
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
@_frontier_cli_options
@_eqa_cli_options
@_diagnostics_cli_options
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
    frontier_nodes: bool | None,
    frontier_keyword_weight: float | None,
    export_map: bool | None,
    export_video: bool | None,
    map_stride: int | None,
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
            frontier_nodes_enabled=frontier_nodes,
            frontier_keyword_weight=frontier_keyword_weight,
            debug_run_tag=f"cli_episode_q{question_id:04d}",
            save_debug_bundle=True,
            export_map=export_map,
            export_video=export_video,
            map_stride=map_stride,
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
@click.option(
    "--question-end",
    default=-1,
    type=int,
    help=f"Inclusive; -1 = last question in CSV (paper subset: 0–{HMEQA_PAPER_QUESTION_COUNT - 1})",
)
@click.option(
    "--paper-subset/--all-questions",
    default=True,
    help=f"Limit to GraphEQA HM-EQA paper questions (indices 0–{HMEQA_PAPER_QUESTION_COUNT - 1})",
)
@click.option(
    "--question-ids",
    default=None,
    help="Comma-separated explicit question ids (overrides --question-start/--question-end)",
)
@click.option("--resume", is_flag=True, default=False, help="Skip question ids already in --output JSONL")
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
@_frontier_cli_options
@_eqa_cli_options
@_diagnostics_cli_options
def run_batch(
    method: str,
    question_start: int,
    question_end: int,
    paper_subset: bool,
    question_ids: str | None,
    resume: bool,
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
    frontier_nodes: bool | None,
    frontier_keyword_weight: float | None,
    export_map: bool | None,
    export_video: bool | None,
    map_stride: int | None,
) -> None:
    """Run a slice of HM-EQA (GraphEQA paper: 113 questions, method=graph_eqa)."""
    from emet_habitat.runner import run_hmeqa_batch

    questions_path = (data_dir / "questions.csv") if data_dir else None
    init_poses_path = (data_dir / "scene_init_poses.csv") if data_dir else None
    qs = load_hmeqa_questions(questions_path)
    if question_ids:
        ids = [int(x) for x in question_ids.split(",") if x.strip()]
    elif paper_subset:
        paper_ids = set(hmeqa_paper_question_ids())
        end = HMEQA_PAPER_QUESTION_COUNT - 1 if question_end < 0 else min(question_end, HMEQA_PAPER_QUESTION_COUNT - 1)
        ids = [qid for qid in range(max(0, question_start), end + 1) if qid in paper_ids]
    else:
        end = len(qs) - 1 if question_end < 0 else min(question_end, len(qs) - 1)
        ids = list(range(max(0, question_start), end + 1))
    click.echo(
        f"Running {len(ids)} HM-EQA episodes ({method}, mock_llm={mock_llm}, "
        f"paper_subset={paper_subset}, resume={resume})"
    )

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
            eqa_vl_family=eqa_vl_family,
            eqa_hf_model_id=eqa_hf_model_id,
            device=device,
            use_hm3d_semantics=use_hm3d_semantics,
            output_jsonl=output,
            resume=resume,
            frontier_nodes_enabled=frontier_nodes,
            frontier_keyword_weight=frontier_keyword_weight,
            export_map=export_map,
            export_video=export_video,
            map_stride=map_stride,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    summary = summarize_episodes(metrics)
    click.echo(f"results: {output}")
    click.echo(f"batch summary (this run): {summary}")


@main.command("compare-batch")
@click.option("--question-start", default=0, type=int)
@click.option("--question-end", default=5, type=int, help="Inclusive; -1 = last question in CSV")
@click.option("--mock-llm", is_flag=True, default=False)
@click.option("--max-planning-steps", default=20, type=int)
@click.option("--max-movement-step", default=10, type=int)
@click.option("--hm3d-root", type=click.Path(path_type=Path), default=None)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Write comparison JSON")
@click.option(
    "--use-hm3d-semantics/--no-hm3d-semantics",
    default=None,
    help="Use HM3D semantic sensor for graph labels (default: auto if assets exist)",
)
@_eqa_cli_options
def compare_batch(
    question_start: int,
    question_end: int,
    mock_llm: bool,
    max_planning_steps: int,
    max_movement_step: int,
    hm3d_root: Path | None,
    data_dir: Path | None,
    output: Path | None,
    use_hm3d_semantics: bool | None,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    device: str,
) -> None:
    """Run graph_eqa and dynagraph on the same questions; print side-by-side summary.

    Both methods use identical graph-memory settings on HM-EQA; expect matching accuracy.
    """
    from emet_habitat.runner import run_hmeqa_compare

    questions_path = (data_dir / "questions.csv") if data_dir else None
    init_poses_path = (data_dir / "scene_init_poses.csv") if data_dir else None
    qs = load_hmeqa_questions(questions_path)
    end = len(qs) - 1 if question_end < 0 else min(question_end, len(qs) - 1)
    ids = list(range(max(0, question_start), end + 1))
    click.echo(f"Comparing graph_eqa vs dynagraph on {len(ids)} questions (mock_llm={mock_llm})")

    try:
        graph, dyna = run_hmeqa_compare(
            question_ids=ids,
            mock_llm=mock_llm,
            max_planning_steps=max_planning_steps,
            max_movement_step=max_movement_step,
            hm3d_root=hm3d_root,
            questions_path=questions_path,
            init_poses_path=init_poses_path,
            eqa_vl_family=eqa_vl_family,
            eqa_hf_model_id=eqa_hf_model_id,
            device=device,
            use_hm3d_semantics=use_hm3d_semantics,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    comparison = compare_method_results(graph, dyna)
    click.echo(f"graph_eqa:  {comparison['graph_eqa']}")
    click.echo(f"dynagraph:  {comparison['dynagraph']}")
    click.echo(
        f"agreement: both={comparison['both_correct']} graph_only={comparison['graph_only']} "
        f"dynagraph_only={comparison['dynagraph_only']} neither={comparison['neither']}"
    )
    for row in comparison["per_question"]:
        click.echo(
            f"  Q{row['question_id']:02d} gold={row['gold']} "
            f"graph={row['graph_eqa_pred']}({'ok' if row['graph_eqa_correct'] else 'x'}) "
            f"dyna={row['dynagraph_pred']}({'ok' if row['dynagraph_correct'] else 'x'})"
        )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "comparison": comparison,
            "graph_eqa_episodes": [e.to_dict() for e in graph],
            "dynagraph_episodes": [e.to_dict() for e in dyna],
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        click.echo(f"wrote {output}")


@main.command("run-ovmm-find-episode")
@click.option(
    "--episodes",
    type=click.Path(path_type=Path),
    default=None,
    help="Episode YAML (default: configs/ovmm/habitat_find_phase_episodes.yaml)",
)
@click.option("--episode-id", required=True, help="Episode id from registry")
@click.option(
    "--backend",
    type=click.Choice(["dynamem", "graph_eqa", "dynagraph", "ground_truth"]),
    default="dynagraph",
)
@click.option("--merge-xy-m", type=float, default=None)
@click.option("--staleness-horizon", type=int, default=None)
@click.option("--cpu-only", is_flag=True, default=False)
@click.option("--not-rotate", is_flag=True, default=False)
@click.option("--hm3d-root", type=click.Path(path_type=Path), default=None)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Write JSON metrics")
def run_ovmm_find_episode(
    episodes: Path | None,
    episode_id: str,
    backend: str,
    merge_xy_m: float | None,
    staleness_horizon: int | None,
    cpu_only: bool,
    not_rotate: bool,
    hm3d_root: Path | None,
    data_dir: Path | None,
    output: Path | None,
) -> None:
    """Run one Habitat find-phase episode (FindObj / FindRec)."""
    from emet.eval.ovmm_find_phase import FindPhaseRunConfig
    from emet_habitat.ovmm_find_runner import load_habitat_find_phase_episodes, run_habitat_find_phase_episode

    repo = Path(__file__).resolve().parents[3]
    ep_path = episodes or (repo / "configs" / "ovmm" / "habitat_find_phase_episodes.yaml")
    init_poses_path = (data_dir / "scene_init_poses.csv") if data_dir else scene_init_poses_csv_path()
    rows = load_habitat_find_phase_episodes(ep_path)
    ep = next((e for e in rows if e.id == episode_id), None)
    if ep is None:
        raise click.ClickException(f"Unknown episode_id {episode_id!r} in {ep_path}")

    run_cfg = FindPhaseRunConfig(
        backend=backend,  # type: ignore[arg-type]
        merge_xy_m=merge_xy_m,
        staleness_horizon=staleness_horizon,
        cpu_only=cpu_only,
        not_rotate=not_rotate,
    )
    metrics = run_habitat_find_phase_episode(
        ep,
        run_cfg,
        hm3d_root=hm3d_root,
        init_poses_path=init_poses_path,
    )
    text = json.dumps(metrics, indent=2)
    click.echo(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


@main.command("run-ovmm-find-batch")
@click.option("--episodes", type=click.Path(path_type=Path), default=None)
@click.option("--episode-id", multiple=True, help="Subset of episode ids")
@click.option(
    "--backend",
    type=click.Choice(["dynamem", "graph_eqa", "dynagraph", "ground_truth"]),
    default="dynagraph",
)
@click.option("--merge-xy-m", type=float, default=None)
@click.option("--staleness-horizon", type=int, default=None)
@click.option("--cpu-only", is_flag=True, default=False)
@click.option("--not-rotate", is_flag=True, default=False)
@click.option("--hm3d-root", type=click.Path(path_type=Path), default=None)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--run-tag", default=None, help="Episode bundle tag under ~/.cache/habitat_eqa/episodes/")
@click.option("--export-map/--no-export-map", default=None)
@click.option("--export-video/--no-export-video", default=None)
def run_ovmm_find_batch(
    episodes: Path | None,
    episode_id: tuple[str, ...],
    backend: str,
    merge_xy_m: float | None,
    staleness_horizon: int | None,
    cpu_only: bool,
    not_rotate: bool,
    hm3d_root: Path | None,
    data_dir: Path | None,
    output_dir: Path,
    run_tag: str | None,
    export_map: bool | None,
    export_video: bool | None,
) -> None:
    """Batch Habitat find-phase evaluation."""
    from emet.eval.ovmm_find_phase import FindPhaseRunConfig
    from emet_habitat.ovmm_find_runner import load_habitat_find_phase_episodes, run_habitat_find_phase_episode

    repo = Path(__file__).resolve().parents[3]
    ep_path = episodes or (repo / "configs" / "ovmm" / "habitat_find_phase_episodes.yaml")
    init_poses_path = (data_dir / "scene_init_poses.csv") if data_dir else scene_init_poses_csv_path()
    rows = load_habitat_find_phase_episodes(ep_path)
    if episode_id:
        id_set = set(episode_id)
        rows = [e for e in rows if e.id in id_set]
    run_cfg = FindPhaseRunConfig(
        backend=backend,  # type: ignore[arg-type]
        merge_xy_m=merge_xy_m,
        staleness_horizon=staleness_horizon,
        cpu_only=cpu_only,
        not_rotate=not_rotate,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_tag = run_tag or output_dir.name
    for ep in rows:
        click.echo(f"Running {ep.id} …", err=True)
        metrics = run_habitat_find_phase_episode(
            ep,
            run_cfg,
            hm3d_root=hm3d_root,
            init_poses_path=init_poses_path,
            debug_run_tag=bundle_tag,
            export_map=export_map,
            export_video=export_video,
        )
        out = output_dir / f"{ep.id}_{backend}.json"
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        click.echo(f"  partial={metrics.get('find_partial_success')} -> {out}", err=True)


if __name__ == "__main__":
    try:
        main(standalone_mode=True)
    except click.ClickException as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""``emet-habitat`` CLI (runs inside ``.venv-habitat``)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

from emet.core.parameters import Parameters
from emet.habitat.config import (
    default_habitat_eqa_data_dir,
    default_hm3d_scene_dir,
    questions_csv_path,
    scene_init_poses_csv_path,
)
from emet.habitat.datasets import load_hmeqa_questions
from emet.habitat.hm3d_semantics import compute_hmeqa_semantics_coverage
from emet.habitat.hmeqa_enrich_labels import HMEQA_PAPER_QUESTION_COUNT, hmeqa_paper_question_ids
from emet.habitat.metrics import compare_method_results, summarize_episodes, write_episode_jsonl


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
            click.echo("  Report: uv run python scripts/download_habitat_eqa_data.py --report-hmeqa-semantics")
        except FileNotFoundError:
            pass


@main.command("egl-probe")
@click.option("--question-id", default=0, type=int, help="HM-EQA question whose scene to open")
@click.option("--json", "as_json", is_flag=True, default=False, help="Print one JSON object")
def egl_probe_cmd(question_id: int, as_json: bool) -> None:
    """Open Habitat WindowlessContext + one RGB frame (no VLM).

    Prefer launching via ``emet habitat safe-start`` (jobs-wrapped) from Cursor —
    never as a blocking agent shell that can segfault the IDE host.
    """
    from emet_habitat.egl_probe import run_egl_probe

    result = run_egl_probe(question_id=int(question_id))
    payload = {
        "ok": result.ok,
        "message": result.message,
        "scene_glb": result.scene_glb,
        "rgb_shape": list(result.rgb_shape) if result.rgb_shape else None,
        "error": result.error,
    }
    if as_json:
        click.echo(json.dumps(payload))
    else:
        click.echo(result.message)
        if result.scene_glb:
            click.echo(f"scene_glb={result.scene_glb}")
        if result.rgb_shape:
            click.echo(f"rgb_shape={result.rgb_shape}")
        if result.error:
            click.echo(result.error, err=True)
    sys.exit(0 if result.ok else 2)


def _eqa_cli_options(fn):
    opts = [
        click.option(
            "--eqa-vl-family",
            default=None,
            help="EQA VLM family: qwen3_vl, qwen3_5, qwen2_5_vl, gemma4, internvl (default: dynav_config.yaml eqa.vl_family)",
        ),
        click.option("--eqa-hf-model-id", default=None, help="Override HF model id (e.g. OpenGVLab/InternVL3-14B-hf)"),
        click.option(
            "--eqa-vl-quantization",
            default=None,
            help="Override EQA VLM quantization (for example int4 or none).",
        ),
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


def _habitat_nav_cli_options(fn):
    opts = [
        click.option(
            "--habitat-perfect-nav/--no-habitat-perfect-nav",
            "habitat_perfect_nav",
            default=None,
            help="Habitat EQA nav: navmesh pathing (default on); off exercises voxel A*",
        ),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


def _configure_dynagraph_harness(
    parameters: Parameters,
    *,
    memory_summary: bool | None = None,
    mcq_debias: bool | None = None,
    explore_when_uncovered: str | None = None,
) -> None:
    from emet.eval.benchmark_dynagraph import apply_dynagraph_harness_overrides

    apply_dynagraph_harness_overrides(
        parameters,
        memory_summary=memory_summary,
        mcq_debias=mcq_debias,
        explore_when_uncovered=explore_when_uncovered,
    )


def _dynagraph_harness_cli_options(fn):
    opts = [
        click.option(
            "--mcq-debias/--no-mcq-debias",
            default=None,
            help="Dynagraph end-of-episode MCQ debias (default: harness profile)",
        ),
        click.option(
            "--memory-summary/--no-memory-summary",
            default=None,
            help="Dynagraph CONFIRMED_MEMORY prompt block (default: harness profile)",
        ),
        click.option(
            "--explore-when-uncovered",
            type=click.Choice(["off", "on", "conservative"]),
            default=None,
            help="Frontier override when graph lacks question objects (default: harness profile)",
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
        click.option(
            "--debug-run-tag",
            default=None,
            type=str,
            help="Episode bundle parent tag under ~/.cache/habitat_eqa/episodes/ "
            "(default: cli_episode_qNNNN). Use distinct tags for H2H arms so maps do not overwrite.",
        ),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


@main.command("run-episode")
@click.option("--dataset", type=click.Choice(["hmeqa"]), default="hmeqa")
@click.option("--question-id", default=0, type=int)
@click.option(
    "--method",
    type=click.Choice(["static_graph", "graph_eqa", "dynagraph"]),
    default="dynagraph",
    help="HM-EQA method (graph_eqa is a legacy alias for static_graph).",
)
@click.option("--mock-llm", is_flag=True, default=False, help="Use mocked EQA responses (smoke / CI)")
@click.option(
    "--mock-llm-explore",
    is_flag=True,
    default=False,
    help="With --mock-llm, return confidence:false so the agent navigates each planning step",
)
@click.option("--max-planning-steps", default=20, type=int, help="EQA planning iterations (GraphEQA ref: 20)")
@click.option("--max-movement-step", default=10, type=int, help="Nav substeps per planning iteration")
@click.option("--hm3d-root", type=click.Path(path_type=Path), default=None)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Write episode JSONL")
@click.option("--rotate-in-place/--no-rotate-in-place", default=True, help="Sweep heading before EQA")
@click.option(
    "--rerun",
    "enable_rerun",
    is_flag=True,
    default=False,
    help="Live Rerun VLM-context viewer (ports 9090/9877). Also EMET_EVAL_RERUN=1. Off by default.",
)
@click.option(
    "--extra-instruction",
    default=None,
    type=str,
    help="Optional text appended to the EQA question (same compose path as emet run agent --eqa-eval)",
)
@click.option(
    "--use-hm3d-semantics/--no-hm3d-semantics",
    default=None,
    help="Use HM3D semantic sensor for graph labels (default: auto if assets exist)",
)
@click.option(
    "--enrich-labels/--no-enrich-labels",
    "use_enrich_labels",
    default=False,
    help="Seed GraphEQA per-question GT object hints (separate oracle axis; default off)",
)
@_frontier_cli_options
@_habitat_nav_cli_options
@_eqa_cli_options
@_dynagraph_harness_cli_options
@_diagnostics_cli_options
def run_episode(
    dataset: str,
    question_id: int,
    method: str,
    mock_llm: bool,
    mock_llm_explore: bool,
    max_planning_steps: int,
    max_movement_step: int,
    hm3d_root: Path | None,
    data_dir: Path | None,
    output: Path | None,
    rotate_in_place: bool,
    enable_rerun: bool,
    extra_instruction: str | None,
    use_hm3d_semantics: bool | None,
    use_enrich_labels: bool,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    eqa_vl_quantization: str | None,
    device: str,
    frontier_nodes: bool | None,
    frontier_keyword_weight: float | None,
    habitat_perfect_nav: bool | None,
    export_map: bool | None,
    export_video: bool | None,
    map_stride: int | None,
    debug_run_tag: str | None,
    mcq_debias: bool | None,
    memory_summary: bool | None,
    explore_when_uncovered: str | None,
) -> None:
    """Run one HM-EQA episode in Habitat-Sim."""
    if enable_rerun:
        os.environ["EMET_EVAL_RERUN"] = "1"
    if dataset != "hmeqa":
        raise click.ClickException(f"Unsupported dataset {dataset!r}")
    if mock_llm_explore and not mock_llm:
        raise click.ClickException("--mock-llm-explore requires --mock-llm")

    questions_path = (data_dir / "questions.csv") if data_dir else None
    init_poses_path = (data_dir / "scene_init_poses.csv") if data_dir else None
    run_tag = (debug_run_tag or "").strip() or f"cli_episode_q{question_id:04d}"

    from emet_habitat.runner import run_hmeqa_episode

    try:
        metrics = run_hmeqa_episode(
            question_id=question_id,
            method=method,
            mock_llm=mock_llm,
            mock_llm_explore=mock_llm_explore,
            max_planning_steps=max_planning_steps,
            max_movement_step=max_movement_step,
            hm3d_root=hm3d_root,
            questions_path=questions_path,
            init_poses_path=init_poses_path,
            rotate_in_place=rotate_in_place,
            use_hm3d_semantics=use_hm3d_semantics,
            use_enrich_labels=use_enrich_labels,
            eqa_vl_family=eqa_vl_family,
            eqa_hf_model_id=eqa_hf_model_id,
            eqa_vl_quantization=eqa_vl_quantization,
            device=device,
            frontier_nodes_enabled=frontier_nodes,
            frontier_keyword_weight=frontier_keyword_weight,
            habitat_perfect_nav=habitat_perfect_nav,
            memory_summary=memory_summary,
            mcq_debias=mcq_debias,
            explore_when_uncovered=explore_when_uncovered,
            debug_run_tag=run_tag,
            save_debug_bundle=True,
            export_map=export_map,
            export_video=export_video,
            map_stride=map_stride,
            extra_instruction=extra_instruction,
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
@click.option(
    "--method",
    type=click.Choice(["static_graph", "graph_eqa", "dynagraph"]),
    default="static_graph",
    help="HM-EQA method (graph_eqa is a legacy alias for static_graph).",
)
@click.option("--question-start", default=0, type=int)
@click.option(
    "--question-end",
    default=-1,
    type=int,
    help=f"Inclusive; -1 = last question in CSV (legacy fixed slice: 0–{HMEQA_PAPER_QUESTION_COUNT - 1})",
)
@click.option(
    "--paper-subset/--all-questions",
    default=True,
    help=(
        f"Limit to the historical emet q0–{HMEQA_PAPER_QUESTION_COUNT - 1} slice; "
        "this is not GraphEQA's semantic-filtered HM-EQA set"
    ),
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
@click.option(
    "--enrich-labels/--no-enrich-labels",
    "use_enrich_labels",
    default=False,
    help="Seed GraphEQA per-question GT object hints (separate oracle axis; default off)",
)
@_frontier_cli_options
@_habitat_nav_cli_options
@_eqa_cli_options
@_dynagraph_harness_cli_options
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
    use_enrich_labels: bool,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    eqa_vl_quantization: str | None,
    device: str,
    frontier_nodes: bool | None,
    frontier_keyword_weight: float | None,
    habitat_perfect_nav: bool | None,
    export_map: bool | None,
    export_video: bool | None,
    map_stride: int | None,
    mcq_debias: bool | None,
    memory_summary: bool | None,
    explore_when_uncovered: str | None,
    debug_run_tag: str | None = None,
) -> None:
    """Run an HM-EQA slice (default: historical q0–112; method=static_graph)."""
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
            eqa_vl_quantization=eqa_vl_quantization,
            device=device,
            use_hm3d_semantics=use_hm3d_semantics,
            use_enrich_labels=use_enrich_labels,
            output_jsonl=output,
            resume=resume,
            frontier_nodes_enabled=frontier_nodes,
            frontier_keyword_weight=frontier_keyword_weight,
            habitat_perfect_nav=habitat_perfect_nav,
            memory_summary=memory_summary,
            mcq_debias=mcq_debias,
            explore_when_uncovered=explore_when_uncovered,
            debug_run_tag=debug_run_tag,
            export_map=export_map,
            export_video=export_video,
            map_stride=map_stride,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
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
@click.option(
    "--enrich-labels/--no-enrich-labels",
    "use_enrich_labels",
    default=False,
    help="Seed GraphEQA per-question GT object hints (separate oracle axis; default off)",
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
    use_enrich_labels: bool,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    eqa_vl_quantization: str | None,
    device: str,
) -> None:
    """Run static_graph and dynagraph on the same questions; print side-by-side summary.

    ``static_graph`` (legacy ``graph_eqa``) uses zero merge/staleness; ``dynagraph`` uses
    ``unified_eqa`` (0.45 m merge) plus tuned extras. Accuracy need not match.
    """
    from emet_habitat.runner import run_hmeqa_compare

    questions_path = (data_dir / "questions.csv") if data_dir else None
    init_poses_path = (data_dir / "scene_init_poses.csv") if data_dir else None
    qs = load_hmeqa_questions(questions_path)
    end = len(qs) - 1 if question_end < 0 else min(question_end, len(qs) - 1)
    ids = list(range(max(0, question_start), end + 1))
    click.echo(f"Comparing static_graph vs dynagraph on {len(ids)} questions (mock_llm={mock_llm})")

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
            eqa_vl_quantization=eqa_vl_quantization,
            device=device,
            use_hm3d_semantics=use_hm3d_semantics,
            use_enrich_labels=use_enrich_labels,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    comparison = compare_method_results(graph, dyna)
    click.echo(f"static_graph:  {comparison['static_graph']}")
    click.echo(f"dynagraph:     {comparison['dynagraph']}")
    click.echo(
        f"agreement: both={comparison['both_correct']} static_only={comparison['static_only']} "
        f"dynagraph_only={comparison['dynagraph_only']} neither={comparison['neither']}"
    )
    for row in comparison["per_question"]:
        click.echo(
            f"  Q{row['question_id']:02d} gold={row['gold']} "
            f"static={row['static_graph_pred']}({'ok' if row['static_graph_correct'] else 'x'}) "
            f"dyna={row['dynagraph_pred']}({'ok' if row['dynagraph_correct'] else 'x'})"
        )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "comparison": comparison,
            "static_graph_episodes": [e.to_dict() for e in graph],
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
    type=click.Choice(["dynamem", "static_graph", "graph_eqa", "dynagraph", "ground_truth"]),
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


@main.command("explore-frontiers")
@click.option("--question-id", default=14, type=int, help="HM-EQA question id (selects scene + spawn)")
@click.option("--scene-id", default=None, help="Override HM3D scene id (requires matching init pose CSV row)")
@click.option("--max-steps", default=40, type=int, help="Exploration iterations (update + nav)")
@click.option("--warmup-updates", default=5, type=int)
@click.option("--seed", default=0, type=int)
@click.option("--hm3d-root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--output-dir", type=click.Path(path_type=Path), default=None, help="Write frontier_explore.json + trajectory"
)
@click.option("--rotate-in-place/--no-rotate-in-place", default=True)
@click.option(
    "--no-frontier-nodes", is_flag=True, default=False, help="Disable graph frontier nodes (voxel sample only)"
)
def explore_frontiers(
    question_id: int,
    scene_id: str | None,
    max_steps: int,
    warmup_updates: int,
    seed: int,
    hm3d_root: Path | None,
    output_dir: Path | None,
    rotate_in_place: bool,
    no_frontier_nodes: bool,
) -> None:
    """VLM-free frontier exploration smoke (mapping + navmesh coverage only)."""
    from emet_habitat.frontier_explore import run_frontier_exploration

    if output_dir is None:
        output_dir = Path.home() / ".cache/habitat_eqa/explore" / f"q{question_id:04d}_s{seed}"
    result = run_frontier_exploration(
        scene_id=scene_id,
        question_id=question_id,
        hm3d_root=hm3d_root,
        max_steps=max_steps,
        warmup_updates=warmup_updates,
        rotate_in_place=rotate_in_place,
        output_dir=output_dir,
        seed=seed,
        frontier_nodes_enabled=not no_frontier_nodes,
    )
    click.echo(json.dumps(result.to_dict(), indent=2))
    click.echo(f"wrote {result.output_dir}")


@main.command("run-ovmm-find-batch")
@click.option("--episodes", type=click.Path(path_type=Path), default=None)
@click.option("--episode-id", multiple=True, help="Subset of episode ids")
@click.option(
    "--backend",
    type=click.Choice(["dynamem", "static_graph", "graph_eqa", "dynagraph", "ground_truth"]),
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
        click.echo(
            f"  task=move {metrics.get('object_query')} "
            f"from {metrics.get('start_recep')} to {metrics.get('goal_recep')} | "
            f"FindObj={metrics.get('find_object_success')} "
            f"FindRec={metrics.get('find_recep_success')} "
            f"partial={metrics.get('find_partial_success')} -> {out}",
            err=True,
        )


@main.command("serve")
@click.option("--question-id", type=int, default=None, help="HM-EQA question id (loads scene + init pose)")
@click.option("--scene-id", default=None, help="HM3D scene id for free play (e.g. Y8Y6ukxGMvn)")
@click.option("--floor", default=0, type=int, help="Floor index when resolving init pose from CSV")
@click.option("--port-offset", default=0, type=int, help="Add to default ZMQ ports (4401–4404)")
@click.option("--hm3d-root", type=click.Path(path_type=Path), default=None)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None, help="HABITAT_EQA_DATA_DIR override")
@click.option(
    "--use-hm3d-semantics/--no-hm3d-semantics",
    default=None,
    help="Load HM3D semantic meshes when available (default: auto)",
)
@click.option("--verbose", is_flag=True, default=False)
def serve_cmd(
    question_id: int | None,
    scene_id: str | None,
    floor: int,
    port_offset: int,
    hm3d_root: Path | None,
    data_dir: Path | None,
    use_hm3d_semantics: bool | None,
    verbose: bool,
) -> None:
    """Start Habitat-Sim as a Stretch-compatible ZMQ server (interactive play).

    Examples::

        emet-habitat serve --scene-id Y8Y6ukxGMvn
        emet-habitat serve --question-id 17

    Then in another terminal::

        emet run dynagraph --no-rerun --question "where is the couch?"
        emet run agent -c "describe what you see"
    """
    from emet_habitat.habitat_serve_session import resolve_habitat_serve_config
    from emet_habitat.zmq_server import run_habitat_zmq_server

    questions_path = (data_dir / "questions.csv") if data_dir else None
    init_poses_path = (data_dir / "scene_init_poses.csv") if data_dir else None
    try:
        cfg = resolve_habitat_serve_config(
            question_id=question_id,
            scene_id=scene_id,
            floor=floor,
            hm3d_root=hm3d_root,
            questions_path=questions_path,
            init_poses_path=init_poses_path,
            use_hm3d_semantics=use_hm3d_semantics,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    run_habitat_zmq_server(cfg, port_offset=port_offset, verbose=verbose)


if __name__ == "__main__":
    try:
        main(standalone_mode=True)
    except click.ClickException as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

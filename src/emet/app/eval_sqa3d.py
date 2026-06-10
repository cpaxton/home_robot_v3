# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI: score SQA3D QA predictions (EM@1) and optional localization."""

from __future__ import annotations

import json
from pathlib import Path

import click

from emet.benchmarks.sqa3d.config import SQA3D_SPLITS, default_sqa3d_data_dir
from emet.benchmarks.sqa3d.datasets import load_sqa3d_questions
from emet.benchmarks.sqa3d.metrics import (
    is_episode_metrics_jsonl,
    load_predictions,
    score_sqa3d_episode_jsonl,
    score_sqa3d_predictions,
)


@click.group("sqa3d", invoke_without_command=True)
@click.pass_context
def sqa3d_group(ctx: click.Context) -> None:
    """SQA3D situated 3D QA benchmark (dataset loaders + scoring)."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@sqa3d_group.command("verify", short_help="Check benchmark data + embodied smoke connections")
@click.option("--split", type=click.Choice(SQA3D_SPLITS), default="val", show_default=True)
@click.option("--scannet-smoke-scenes", default="scene0380_00,scene0249_00", show_default=True)
@click.option("--run-embodied-smoke", is_flag=True, default=False, help="Run one mock-LLM episode")
def sqa3d_verify(split: str, scannet_smoke_scenes: str, run_embodied_smoke: bool) -> None:
    """Verify SQA3D JSON, optional localization, and ScanNet mesh availability."""
    from emet.benchmarks.sqa3d.config import (
        annotations_json_path,
        balanced_dir,
        localization_json_path,
        questions_json_path,
    )
    from emet.benchmarks.sqa3d.datasets import load_sqa3d_localization, load_sqa3d_questions
    from emet.benchmarks.sqa3d.scannet.config import (
        collect_sqa3d_scene_ids,
        count_scannet_scenes_on_disk,
        default_scannet_root,
        scene_assets_present,
        scene_mesh_path,
    )

    data_dir = default_sqa3d_data_dir()
    scannet_root = default_scannet_root()
    ok = True

    click.echo(f"SQA3D_DATA_DIR={data_dir}")
    for sp in SQA3D_SPLITS:
        q_ok = questions_json_path(sp, data_dir).is_file()
        a_ok = annotations_json_path(sp, data_dir).is_file()
        l_ok = localization_json_path(sp, data_dir).is_file()
        click.echo(f"  {sp}: questions={q_ok} annotations={a_ok} localization={l_ok}")
        if not (q_ok and a_ok):
            ok = False

    try:
        n_qa = len(load_sqa3d_questions(split, data_dir=data_dir))
        n_loc = len(load_sqa3d_localization(split, data_dir=data_dir))
        click.echo(f"Loaded {split}: {n_qa} QA, {n_loc} localization rows")
        if n_qa == 0:
            ok = False
    except Exception as exc:
        click.echo(f"FAIL load {split}: {exc}", err=True)
        ok = False

    scenes = collect_sqa3d_scene_ids(split, data_dir=data_dir)
    present, total = count_scannet_scenes_on_disk(scenes, scannet_root)
    click.echo(f"SCANNET_ROOT={scannet_root}")
    click.echo(f"  {split} scenes on disk: {present}/{total}")

    smoke = [s.strip() for s in scannet_smoke_scenes.split(",") if s.strip()]
    for scene_id in smoke:
        mesh_ok = scene_assets_present(scene_id, scannet_root)
        click.echo(f"  smoke mesh {scene_id}: {mesh_ok} ({scene_mesh_path(scene_id, scannet_root)})")
        if not mesh_ok:
            ok = False

    if run_embodied_smoke and smoke:
        from emet.benchmarks.sqa3d.scannet.runner import run_sqa3d_episode

        qs = load_sqa3d_questions(split, data_dir=data_dir)
        target = smoke[0]
        q = next((x for x in qs if x.scene_id == target), None)
        if q is None and split != "train":
            qs_train = load_sqa3d_questions("train", data_dir=data_dir)
            q = next((x for x in qs_train if x.scene_id == target), None)
            split = "train"
        if q is None:
            click.echo(f"FAIL: no SQA3D question for smoke scene {target}", err=True)
            raise SystemExit(1)
        click.echo(f"Running embodied smoke: split={split} question_id={q.question_id} scene={q.scene_id}")
        row = run_sqa3d_episode(
            question_id=q.question_id,
            split=split,
            mock_llm=True,
            max_planning_steps=2,
            scannet_root=scannet_root,
        )
        click.echo(f"  em={row.em} predicted={row.predicted_answer!r}")
        if not row.em:
            ok = False

    if not ok:
        click.echo("\nVERIFY FAILED — see docs/sqa3d.md for download commands", err=True)
        raise SystemExit(1)
    click.echo("\nVERIFY OK")


@sqa3d_group.command("info", short_help="Print SQA3D data paths and file status")
def sqa3d_info() -> None:
    from emet.benchmarks.sqa3d.config import (
        annotations_json_path,
        answer_dict_path,
        balanced_dir,
        localization_json_path,
        questions_json_path,
    )

    data_dir = default_sqa3d_data_dir()
    click.echo(f"SQA3D_DATA_DIR={data_dir}")
    click.echo(f"balanced_dir exists={balanced_dir(data_dir).is_dir()}")
    for split in SQA3D_SPLITS:
        q = questions_json_path(split, data_dir)
        a = annotations_json_path(split, data_dir)
        loc = localization_json_path(split, data_dir)
        click.echo(
            f"  {split}: questions={q.is_file()} annotations={a.is_file()} localization={loc.is_file()}"
        )
    click.echo(f"answer_dict={answer_dict_path(data_dir).is_file()}")
    from emet.benchmarks.sqa3d.scannet.config import default_scannet_root, scene_mesh_path

    scannet_root = default_scannet_root()
    click.echo(f"SCANNET_ROOT={scannet_root}")
    click.echo(f"  example mesh scene0380_00={scene_mesh_path('scene0380_00', scannet_root).is_file()}")
    if not balanced_dir(data_dir).is_dir():
        click.echo("\nDownload: uv run python scripts/download_sqa3d_data.py --fetch-annotations")
    if not scene_mesh_path("scene0380_00", scannet_root).is_file():
        click.echo(
            "ScanNet: uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00"
        )


@sqa3d_group.command("list-questions", short_help="List questions from a split")
@click.option("--split", type=click.Choice(SQA3D_SPLITS), default="val", show_default=True)
@click.option("--limit", default=10, type=int, show_default=True)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
def sqa3d_list_questions(split: str, limit: int, data_dir: Path | None) -> None:
    questions = load_sqa3d_questions(split, data_dir=data_dir)
    click.echo(f"split={split} n={len(questions)}")
    for q in questions[:limit]:
        click.echo(
            f"  id={q.question_id} scene={q.scene_id} "
            f"answer={q.primary_answer!r} | {q.question[:72]}"
        )


@sqa3d_group.command("run-episode", short_help="SQA3D episode on ScanNet mesh (DynaMem or Dynagraph)")
@click.option("--question-id", required=True, type=int)
@click.option("--split", type=click.Choice(SQA3D_SPLITS), default="val", show_default=True)
@click.option(
    "--method",
    type=click.Choice(["dynamem", "dynagraph"]),
    default="dynagraph",
    show_default=True,
    help="dynagraph=DynaMem voxel map + GraphEQA graph EQA (default); dynamem=voxel EQA only",
)
@click.option("--mock-llm", is_flag=True, default=False)
@click.option(
    "--profile",
    type=click.Choice(["smoke", "tuned"]),
    default=None,
    help="smoke=fast CI defaults; tuned=real-VLM defaults (auto when not --mock-llm)",
)
@click.option("--max-planning-steps", default=None, type=int, help="Default: 8 smoke / 15 tuned")
@click.option("--eqa-vl-family", default=None, help="Override eqa.vl_family (qwen3_vl, gemma4, …)")
@click.option("--eqa-hf-model-id", default=None, help="Override eqa.vl_hf_model_id")
@click.option("--device", default=None, help="VLM device (cuda, cpu). Default: cuda when not --mock-llm")
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
@click.option("--scannet-root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--replay-mode",
    type=click.Choice(["auto", "sens", "mesh"]),
    default="auto",
    show_default=True,
    help="auto=posed .sens RGB when on disk, else mesh; sens=require .sens; mesh=Open3D only",
)
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Append JSONL result")
def sqa3d_run_episode(
    question_id: int,
    split: str,
    method: str,
    mock_llm: bool,
    profile: str | None,
    max_planning_steps: int | None,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    device: str | None,
    data_dir: Path | None,
    scannet_root: Path | None,
    replay_mode: str,
    output: Path | None,
) -> None:
    """Run DynaMem or Dynagraph at the SQA3D annotated pose in a ScanNet scene."""
    from emet.benchmarks.sqa3d.episode_metrics import append_sqa3d_jsonl
    from emet.benchmarks.sqa3d.scannet.runner import run_sqa3d_episode

    row = run_sqa3d_episode(
        question_id=question_id,
        method=method,
        mock_llm=mock_llm,
        max_planning_steps=max_planning_steps,
        split=split,
        data_dir=data_dir,
        scannet_root=scannet_root,
        eqa_vl_family=eqa_vl_family,
        eqa_hf_model_id=eqa_hf_model_id,
        device=device if device else ("cuda" if not mock_llm else None),
        profile=profile,
        replay_mode=replay_mode,
    )
    text = json.dumps(row.to_dict(), indent=2)
    click.echo(text)
    if output:
        append_sqa3d_jsonl(output, row)
        click.echo(f"Appended -> {output}")
    click.echo(f"\nSummary: em={row.em} predicted={row.predicted_answer!r} gold={row.gold_answers}")


@sqa3d_group.command("run-batch", short_help="Batch SQA3D episodes")
@click.option(
    "--question-start",
    default=0,
    type=int,
    help="Start index into the split question list (not question_id)",
)
@click.option(
    "--question-end",
    default=5,
    type=int,
    help="End index into the split question list (exclusive)",
)
@click.option("--split", type=click.Choice(SQA3D_SPLITS), default="val", show_default=True)
@click.option("--method", type=click.Choice(["dynamem", "dynagraph"]), default="dynagraph", show_default=True)
@click.option("--mock-llm", is_flag=True, default=False)
@click.option("--profile", type=click.Choice(["smoke", "tuned"]), default=None)
@click.option("--max-planning-steps", default=None, type=int)
@click.option("--eqa-vl-family", default=None, help="Override eqa.vl_family (default: dynav_config.yaml)")
@click.option("--eqa-hf-model-id", default=None, help="Override eqa.vl_hf_model_id")
@click.option("--device", default=None, help="VLM device (cuda, cpu). Default: cuda when not --mock-llm")
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
@click.option("--scannet-root", type=click.Path(path_type=Path), default=None)
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True)
@click.option("--resume", is_flag=True, default=False)
@click.option(
    "--skip-missing-scenes/--no-skip-missing-scenes",
    default=True,
    show_default=True,
    help="Skip questions whose ScanNet mesh is not on disk",
)
@click.option(
    "--replay-mode",
    type=click.Choice(["auto", "sens", "mesh"]),
    default="auto",
    show_default=True,
    help="auto=posed .sens RGB when on disk, else mesh; sens=require .sens; mesh=Open3D only",
)
@click.option(
    "--isolate-episodes/--no-isolate-episodes",
    default=False,
    show_default=True,
    help="Real-VLM: run each episode in a fresh subprocess (frees GPU between episodes)",
)
def sqa3d_run_batch(
    question_start: int,
    question_end: int,
    split: str,
    method: str,
    mock_llm: bool,
    profile: str | None,
    max_planning_steps: int | None,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    device: str | None,
    data_dir: Path | None,
    scannet_root: Path | None,
    output: Path,
    resume: bool,
    skip_missing_scenes: bool,
    replay_mode: str,
    isolate_episodes: bool,
) -> None:
    from emet.benchmarks.sqa3d.datasets import load_sqa3d_questions
    from emet.benchmarks.sqa3d.episode_metrics import summarize_sqa3d_episodes
    from emet.benchmarks.sqa3d.scannet.config import default_scannet_root, filter_questions_with_scannet
    from emet.benchmarks.sqa3d.scannet.runner import run_sqa3d_batch

    questions = load_sqa3d_questions(split, data_dir=data_dir)
    subset = questions[question_start:question_end]
    if skip_missing_scenes:
        root = scannet_root or default_scannet_root()
        filtered = filter_questions_with_scannet(subset, root, replay_mode=replay_mode)
        skipped = len(subset) - len(filtered)
        if skipped:
            click.echo(
                f"Skipping {skipped} question(s) without ScanNet replay assets "
                f"(replay_mode={replay_mode}) under {root}"
            )
        subset = filtered
    if not subset:
        raise click.ClickException(
            "No questions to run (empty slice or no ScanNet meshes). "
            "Download: uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00"
        )
    ids = [q.question_id for q in subset]
    rows = run_sqa3d_batch(
        question_ids=ids,
        method=method,
        mock_llm=mock_llm,
        max_planning_steps=max_planning_steps,
        split=split,
        data_dir=data_dir,
        scannet_root=scannet_root,
        output_jsonl=output,
        resume=resume,
        eqa_vl_family=eqa_vl_family,
        eqa_hf_model_id=eqa_hf_model_id,
        device=device if device else ("cuda" if not mock_llm else None),
        profile=profile,
        replay_mode=replay_mode,
        isolate_episodes=isolate_episodes,
    )
    summary = summarize_sqa3d_episodes(rows)
    click.echo(json.dumps(summary, indent=2))


@sqa3d_group.command("run-real-sweep", short_help="Download meshes + real-VLM batch on a question slice")
@click.option("--question-start", default=0, type=int)
@click.option("--question-end", default=5, type=int)
@click.option("--split", type=click.Choice(SQA3D_SPLITS), default="val", show_default=True)
@click.option("--method", type=click.Choice(["dynamem", "dynagraph"]), default="dynagraph", show_default=True)
@click.option("--max-planning-steps", default=None, type=int)
@click.option("--eqa-vl-family", default=None)
@click.option("--eqa-hf-model-id", default=None)
@click.option("--device", default=None, help="VLM device (cuda, cpu). Default: cuda")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Sweep output directory (default: configs/sqa3d/benchmark.yaml → ~/runs/emet/sqa3d)",
)
@click.option("--download/--no-download", default=True, show_default=True)
@click.option(
    "--with-sens",
    is_flag=True,
    default=False,
    help="Also download ScanNet .sens posed RGB-D (large; ~hundreds of MB per scene)",
)
@click.option(
    "--replay-mode",
    type=click.Choice(["auto", "sens", "mesh"]),
    default="auto",
    show_default=True,
)
@click.option(
    "--isolate-episodes/--no-isolate-episodes",
    default=True,
    show_default=True,
    help="Run each episode in a fresh subprocess (recommended for real VLM; frees GPU)",
)
def sqa3d_run_real_sweep(
    question_start: int,
    question_end: int,
    split: str,
    method: str,
    max_planning_steps: int | None,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    device: str | None,
    output_dir: Path,
    download: bool,
    with_sens: bool,
    replay_mode: str,
    isolate_episodes: bool,
) -> None:
    """Download ScanNet meshes for the slice, run real-VLM batch, score EM@1."""
    import subprocess
    import sys

    from emet.benchmarks.sqa3d.benchmark_config import load_sqa3d_benchmark_config
    from emet.benchmarks.sqa3d.scannet.config import default_scannet_root

    if output_dir is None:
        output_dir = load_sqa3d_benchmark_config().paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{method}_{split}_q{question_start}-{question_end}"
    jsonl = output_dir / f"{tag}.jsonl"
    eval_json = output_dir / f"{tag}_eval.json"

    if download:
        sens_note = " + .sens" if with_sens else ""
        click.echo(
            f"Downloading ScanNet meshes{sens_note} for {split} questions "
            f"[{question_start}:{question_end})..."
        )
        dl_cmd = [
            sys.executable,
            "scripts/download_scannet_data.py",
            "--accept-tos",
            "--scenes-from-sqa3d",
            "--split",
            split,
            "--question-start",
            str(question_start),
            "--question-end",
            str(question_end),
            "--scannet-root",
            str(default_scannet_root()),
        ]
        if with_sens:
            dl_cmd.append("--with-sens")
        dl = subprocess.run(
            dl_cmd,
            cwd=Path(__file__).resolve().parents[3],
        )
        if dl.returncode != 0:
            raise SystemExit(dl.returncode)

    ctx = click.get_current_context()
    ctx.invoke(
        sqa3d_run_batch,
        question_start=question_start,
        question_end=question_end,
        split=split,
        method=method,
        mock_llm=False,
        profile="tuned",
        max_planning_steps=max_planning_steps,
        eqa_vl_family=eqa_vl_family,
        eqa_hf_model_id=eqa_hf_model_id,
        device=device,
        data_dir=None,
        scannet_root=None,
        output=jsonl,
        resume=False,
        skip_missing_scenes=True,
        replay_mode=replay_mode,
        isolate_episodes=isolate_episodes,
    )

    ctx.invoke(
        eval_sqa3d_main,
        predictions=jsonl,
        split=split,
        data_dir=None,
        questions_path=None,
        annotations_path=None,
        output=eval_json,
        require_all=False,
    )
    click.echo(f"\nReal-VLM sweep complete:\n  episodes: {jsonl}\n  eval: {eval_json}")


@sqa3d_group.command("plot-results", short_help="TP/FP/FN breakdown + paper figures from episode JSONL")
@click.option("--predictions", "-p", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", "-o", required=True, type=click.Path(path_type=Path))
@click.option("--split", type=click.Choice(SQA3D_SPLITS), default="val", show_default=True)
@click.option("--method", default=None, help="Filter episodes by method tag (dynamem, dynagraph)")
@click.option("--top-k", default=12, type=int, help="Top gold labels for confusion matrix")
@click.option("--no-plots", is_flag=True, default=False, help="Write JSON/JSONL only (no PNG)")
def sqa3d_plot_results(
    predictions: Path,
    output_dir: Path,
    split: str,
    method: str | None,
    top_k: int,
    no_plots: bool,
) -> None:
    """Classify TP/FP/FN/infra and write matplotlib figures for the paper."""
    from emet.benchmarks.sqa3d.analysis import generate_sqa3d_figure_bundle

    bundle = generate_sqa3d_figure_bundle(
        predictions,
        output_dir,
        split=split,
        method=method,
        top_k=top_k,
        write_plots=not no_plots,
    )
    outcomes = bundle["outcomes"]
    click.echo(
        f"TP={outcomes['tp']} FP={outcomes['fp']} FN={outcomes['fn']} "
        f"infra={outcomes['n_infra']} EM@1={outcomes['em@1']:.3f}"
    )
    click.echo(json.dumps(bundle["artifacts"], indent=2))


@click.command("eval-sqa3d")
@click.option("--predictions", "-p", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--split", type=click.Choice(SQA3D_SPLITS), default="val", show_default=True)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
@click.option("--questions-path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--annotations-path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--require-all", is_flag=True, default=False, help="Fail if any question lacks a prediction")
def eval_sqa3d_main(
    predictions: Path,
    split: str,
    data_dir: Path | None,
    questions_path: Path | None,
    annotations_path: Path | None,
    output: Path | None,
    require_all: bool,
) -> None:
    """Score SQA3D predictions (JSONL, episode JSONL, or eqa_results.json) with EM@1."""
    if is_episode_metrics_jsonl(predictions):
        metrics = score_sqa3d_episode_jsonl(predictions)
        payload = {
            "benchmark": "sqa3d",
            "split": split,
            "predictions": str(predictions),
            "format": "episode_jsonl",
            "qa": metrics,
        }
    else:
        questions = load_sqa3d_questions(
            split,
            data_dir=data_dir,
            questions_path=questions_path,
            annotations_path=annotations_path,
        )
        preds = load_predictions(predictions)
        metrics = score_sqa3d_predictions(questions, preds, require_all=require_all)
        payload = {
            "benchmark": "sqa3d",
            "split": split,
            "predictions": str(predictions),
            "data_dir": str(data_dir or default_sqa3d_data_dir()),
            "format": "predictions",
            "qa": metrics,
        }
    text = json.dumps(payload, indent=2)
    click.echo(text)
    if output:
        output.write_text(text + "\n", encoding="utf-8")
        click.echo(f"Wrote -> {output}")
    click.echo(
        f"\nSummary: em@1={metrics['em@1']:.4f} "
        f"em@1_refined={metrics['em@1_refined']:.4f} "
        f"n={int(metrics['n_scored'])}/{int(metrics['n_questions'])}"
    )


if __name__ == "__main__":
    eval_sqa3d_main()

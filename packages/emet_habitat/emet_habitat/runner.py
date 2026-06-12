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

"""Run HM-EQA episodes with emet GraphEQA / Dynagraph controllers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from emet.controller.controller_dynagraph import DynagraphController
from emet.controller.controller_graph_eqa import GraphEQAController
from emet.controller.task.dynamem import EQAExecuter
from emet.core.parameters import Parameters, get_parameters
from emet.habitat.config import default_hm3d_scene_dir
from emet.habitat.datasets import get_question, load_hmeqa_questions, load_scene_init_poses
from emet.habitat.hmeqa_enrich_labels import enrich_labels_for_question
from emet.habitat.episode_debug import (
    enrich_episode_metrics,
    run_tag_from_output_jsonl,
    save_episode_debug_bundle,
    save_error_episode_bundle,
    write_run_manifest,
)
from emet.habitat.metrics import (
    EpisodeMetrics,
    append_episode_jsonl,
    extract_mcq_letter,
    extract_mcq_letter_from_raw_eqa,
    grade_mcq_answer,
)
from emet_habitat.robot_client import HabitatRobotClient
from emet_habitat.simulator import HabitatEQASimulator


def _release_gpu_memory() -> None:
    """Best-effort VRAM cleanup between Habitat episodes (semantic meshes + VLM)."""
    try:
        from emet.llms.graph_eqa_vlm import release_shared_graph_eqa_vlm

        release_shared_graph_eqa_vlm()
    except Exception:
        pass
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _mock_eqa_response(gold_letter: str) -> str:
    return (
        "reasoning: mock habitat harness\n"
        f"answer: {gold_letter}\n"
        "confidence: true\n"
        "action:\n"
        "confidence_reasoning: mocked for smoke test\n"
    )


def _apply_method_parameters(parameters: Parameters | dict, method: str) -> Parameters:
    if isinstance(parameters, dict):
        params = Parameters(**parameters)
    else:
        params = parameters
    # HM-EQA episodes are short (~20 planning steps). Use identical graph-memory
    # settings so Dynagraph is a same-stack regression check vs GraphEQA, not a
    # competing merge/staleness config (those apply on long real-robot runs).
    if method in ("graph_eqa", "dynagraph"):
        params["dynagraph_merge_xy_m"] = 0.0
        params["dynagraph_staleness_horizon"] = 0
    else:
        raise ValueError(f"Unknown method {method!r}; use graph_eqa or dynagraph")
    return params


def _cfg_hf_id_matches_family(hf_model_id: str, family: str) -> bool:
    from emet.llms.eqa_vl_settings import _hf_id_matches_family

    return _hf_id_matches_family(hf_model_id, family)


def _configure_frontier_parameters(
    parameters: Parameters,
    *,
    frontier_nodes_enabled: bool | None = None,
    frontier_keyword_weight: float | None = None,
) -> None:
    """Override ``graph_eqa_frontier_nodes`` for HM-EQA ablations."""
    if frontier_nodes_enabled is None and frontier_keyword_weight is None:
        return
    blk = dict(parameters.get("graph_eqa_frontier_nodes") or {})
    if frontier_nodes_enabled is not None:
        blk["enabled"] = bool(frontier_nodes_enabled)
    if frontier_keyword_weight is not None:
        blk["keyword_score_weight"] = float(frontier_keyword_weight)
    parameters.set("graph_eqa_frontier_nodes", blk)


def _configure_eqa_parameters(
    parameters: Parameters,
    *,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    device: str = "cuda",
) -> None:
    if eqa_vl_family is None and eqa_hf_model_id is None:
        return
    from emet.llms.eqa_vl_settings import resolve_vl_hf_model_id
    from emet.llms.vllm_registry import default_hf_model_id, normalize_vl_family

    eqa = dict(parameters.get("eqa", {}) or {})
    if eqa_vl_family is not None:
        eqa["backend"] = "qwen_vl"
        eqa["vl_family"] = eqa_vl_family
        eqa.setdefault("vl_quantization", "int4")
        fam = normalize_vl_family(eqa_vl_family)
        if eqa_hf_model_id is None:
            existing = str(eqa.get("vl_hf_model_id") or "")
            wrong_family_id = bool(existing) and not _cfg_hf_id_matches_family(existing, fam)
            if wrong_family_id or not existing:
                if fam == "gemma4":
                    eqa["vl_hf_model_id"] = resolve_vl_hf_model_id(fam, parameters, device=device)
                else:
                    eqa["vl_hf_model_id"] = default_hf_model_id(fam)
    if eqa_hf_model_id is not None:
        eqa["vl_hf_model_id"] = eqa_hf_model_id
    eqa.setdefault("prompt_variant", "hmeqa")
    parameters.set("eqa", eqa)


def _make_controller(
    robot: HabitatRobotClient,
    parameters: Parameters,
    *,
    method: str,
    mock_llm: bool,
    gold_letter: str,
    no_rerun: bool,
    use_real_vlm: bool,
    device: str | None,
    use_hm3d_semantics: bool | None = None,
):
    params = _apply_method_parameters(parameters, method)
    hm3d_sem = robot.uses_hm3d_semantics if use_hm3d_semantics is None else use_hm3d_semantics
    # HM3D semantic sensor supplies graph labels; reserve VLM for EQA queries only.
    graph_perception = use_real_vlm and not hm3d_sem
    # Habitat: depth voxel map for nav only — no SigLIP/YoloE reload per episode.
    common = {
        "robot": robot,
        "parameters": params,
        "save_rerun": False if no_rerun else False,
        "cpu_only": not use_real_vlm,
        "use_sensor_perception": graph_perception,
        "use_instance_graph": False,
        "manipulation_only": True,
    }
    if method == "dynagraph":
        agent = DynagraphController(**common)
    else:
        agent = GraphEQAController(**common)

    if mock_llm and agent.graph_memory is not None:
        agent.graph_memory.eqa_client = lambda _q: _mock_eqa_response(gold_letter)
        agent.graph_memory.image_description_client = lambda _x: "object"
    elif agent.graph_memory is not None:
        from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients

        keyword_client, eqa_client = build_graph_eqa_vlm_clients(parameters=params, device=device)
        agent.graph_memory.image_description_client = keyword_client
        agent.graph_memory.eqa_client = eqa_client
        if agent.sensor_builder is not None:
            agent.sensor_builder._perception = keyword_client
            agent.sensor_builder._lazy_vl_client = keyword_client
            agent.sensor_builder.cpu_only = False
    return agent


def run_hmeqa_episode(
    *,
    question_id: int,
    method: str = "dynagraph",
    mock_llm: bool = True,
    max_planning_steps: int = 20,
    max_movement_step: int = 10,
    hm3d_root: Path | None = None,
    questions_path: Path | None = None,
    init_poses_path: Path | None = None,
    no_rerun: bool = True,
    rotate_in_place: bool = True,
    use_hm3d_semantics: bool | None = None,
    eqa_vl_family: str | None = None,
    eqa_hf_model_id: str | None = None,
    device: str | None = "cuda",
    frontier_nodes_enabled: bool | None = None,
    frontier_keyword_weight: float | None = None,
    debug_run_tag: str | None = None,
    save_debug_bundle: bool = True,
) -> EpisodeMetrics:
    questions = load_hmeqa_questions(questions_path)
    q = get_question(questions, question_id=question_id)
    poses = load_scene_init_poses(init_poses_path)
    init_pose = poses.get((q.scene, q.floor))
    if init_pose is None:
        raise KeyError(f"No init pose for scene={q.scene!r} floor={q.floor}")

    hm3d = hm3d_root or default_hm3d_scene_dir()
    sim = HabitatEQASimulator.from_scene_id(
        q.scene,
        hm3d_root=hm3d,
        use_hm3d_semantics=use_hm3d_semantics,
    )
    use_real_vlm = not mock_llm
    try:
        sim.set_init_pose(init_pose)
        robot = HabitatRobotClient(sim)
        if sim.uses_hm3d_semantics:
            print(f"HM3D semantics enabled for scene {q.scene}", flush=True)
        parameters = get_parameters("dynav_config.yaml")
        _configure_eqa_parameters(
            parameters,
            eqa_vl_family=eqa_vl_family,
            eqa_hf_model_id=eqa_hf_model_id,
            device=device or "cuda",
        )
        _configure_frontier_parameters(
            parameters,
            frontier_nodes_enabled=frontier_nodes_enabled,
            frontier_keyword_weight=frontier_keyword_weight,
        )
        agent = _make_controller(
            robot,
            parameters,
            method=method,
            mock_llm=mock_llm,
            gold_letter=q.answer_letter,
            no_rerun=no_rerun,
            use_real_vlm=use_real_vlm,
            device=device,
            use_hm3d_semantics=use_hm3d_semantics,
        )
        agent._eqa_question = q.question_formatted
        agent.start()
        if agent.graph_memory is not None:
            hints = enrich_labels_for_question(question_id, q.scene)
            if hints:
                agent.graph_memory.seed_object_hints(hints)
            agent.graph_memory.extract_relevant_objects(q.question_formatted)
        executor = EQAExecuter(agent)
        if rotate_in_place:
            executor.rotate_in_place()
        for _ in range(5):
            agent.update()
            if agent.graph_memory is not None and hasattr(agent, "_sync_graph_frontier_nodes"):
                agent._sync_graph_frontier_nodes()

        discord_text, _images = agent.run_eqa(
            q.question_formatted,
            max_planning_steps=max_planning_steps,
            max_movement_step=max_movement_step,
        )
        raw_eqa = ""
        parsed_letter = ""
        model_confident = False
        formatted_answer = ""
        eqa_action = ""
        eqa_confidence_reasoning = ""
        if agent.graph_memory is not None:
            raw_eqa = agent.graph_memory.last_eqa_raw
            _reasoning, answer, model_confident, eqa_action, eqa_confidence_reasoning = (
                agent.graph_memory.last_eqa_parsed
            )
            formatted_answer = str(answer or "")
            # Prefer raw mLLM ``answer:`` field; human formatting can replace letters with prose.
            parsed_letter = extract_mcq_letter_from_raw_eqa(raw_eqa, q.choices)
            if not parsed_letter:
                parsed_letter = extract_mcq_letter(answer, q.choices)
        predicted = parsed_letter
        if not predicted:
            tail = discord_text.split("---")[-1].strip() if "---" in discord_text else discord_text
            predicted = extract_mcq_letter(tail, q.choices)
        correct = grade_mcq_answer(predicted, q.answer_letter, choices=q.choices) if predicted else False

        eqa_cfg = dict(parameters.get("eqa", {}) or {})
        metrics = EpisodeMetrics(
            dataset="hmeqa",
            method=method,
            question_id=question_id,
            scene=q.scene,
            floor=q.floor,
            question=q.question,
            gold_answer_letter=q.answer_letter,
            predicted_answer=str(predicted)[:200],
            correct=correct,
            confident=model_confident,
            planning_steps=getattr(agent, "obs_count", 0),
            success=correct,
            parsed_answer_letter=parsed_letter,
            model_confident=model_confident,
            raw_eqa_output=raw_eqa[:8000],
        )
        enrich_episode_metrics(
            metrics,
            agent=agent,
            choices=q.choices,
            formatted_answer=formatted_answer,
            eqa_action=str(eqa_action or ""),
            eqa_confidence_reasoning=str(eqa_confidence_reasoning or ""),
            vl_family=str(eqa_cfg.get("vl_family") or eqa_vl_family or ""),
            vl_hf_model_id=str(eqa_cfg.get("vl_hf_model_id") or eqa_hf_model_id or ""),
        )
        if save_debug_bundle and debug_run_tag:
            try:
                save_episode_debug_bundle(
                    run_tag=debug_run_tag,
                    metrics=metrics,
                    agent=agent,
                    raw_eqa_full=raw_eqa,
                )
            except Exception as exc:
                print(
                    f"question_id={question_id} debug bundle save failed (metrics kept): {exc}",
                    flush=True,
                )
        return metrics
    finally:
        sim.close()
        _release_gpu_memory()


def run_hmeqa_batch(
    *,
    question_ids: list[int],
    method: str = "graph_eqa",
    mock_llm: bool = False,
    max_planning_steps: int = 20,
    max_movement_step: int = 10,
    hm3d_root: Path | None = None,
    questions_path: Path | None = None,
    init_poses_path: Path | None = None,
    eqa_vl_family: str | None = "gemma4",
    eqa_hf_model_id: str | None = None,
    device: str | None = "cuda",
    continue_on_error: bool = True,
    use_hm3d_semantics: bool | None = None,
    output_jsonl: Path | None = None,
    resume: bool = False,
    frontier_nodes_enabled: bool | None = None,
    frontier_keyword_weight: float | None = None,
) -> list[EpisodeMetrics]:
    from emet.habitat.metrics import read_completed_question_ids

    results: list[EpisodeMetrics] = []
    questions = load_hmeqa_questions(questions_path)
    done: set[int] = set()
    run_tag = run_tag_from_output_jsonl(output_jsonl)
    parameters = get_parameters("dynav_config.yaml")
    _configure_frontier_parameters(
        parameters,
        frontier_nodes_enabled=frontier_nodes_enabled,
        frontier_keyword_weight=frontier_keyword_weight,
    )
    if output_jsonl is not None:
        if resume and output_jsonl.exists():
            done = read_completed_question_ids(output_jsonl)
        elif output_jsonl.exists():
            output_jsonl.unlink()
        manifest = write_run_manifest(
            output_jsonl=output_jsonl,
            method=method,
            question_ids=question_ids,
            mock_llm=mock_llm,
            max_planning_steps=max_planning_steps,
            max_movement_step=max_movement_step,
            eqa_vl_family=eqa_vl_family,
            eqa_hf_model_id=eqa_hf_model_id,
            device=device,
            resume=resume,
            parameters=parameters,
        )
        print(f"run manifest: {manifest}", flush=True)
    for qid in question_ids:
        if qid in done:
            print(f"question_id={qid} skip (already in {output_jsonl})", flush=True)
            continue
        try:
            row = run_hmeqa_episode(
                question_id=qid,
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
                frontier_nodes_enabled=frontier_nodes_enabled,
                frontier_keyword_weight=frontier_keyword_weight,
                debug_run_tag=run_tag if output_jsonl is not None else None,
            )
            results.append(row)
            if output_jsonl is not None:
                append_episode_jsonl(output_jsonl, row)
                bundle = row.debug_bundle_dir or "(no bundle)"
                print(
                    f"question_id={qid} done correct={row.correct} "
                    f"frontier_nodes={row.frontier_nodes} bundle={bundle} (appended {output_jsonl})",
                    flush=True,
                )
            _release_gpu_memory()
        except Exception as exc:
            if not continue_on_error:
                raise
            q = get_question(questions, question_id=qid)
            print(f"question_id={qid} failed: {exc}", flush=True)
            err_row = EpisodeMetrics(
                dataset="hmeqa",
                method=method,
                question_id=qid,
                scene=q.scene,
                floor=q.floor,
                question=q.question,
                gold_answer_letter=q.answer_letter,
                predicted_answer=f"ERROR: {exc}"[:200],
                correct=False,
                confident=False,
                planning_steps=0,
                success=False,
                choices=list(q.choices or []),
                error=str(exc),
            )
            if output_jsonl is not None:
                save_error_episode_bundle(run_tag=run_tag, metrics=err_row)
            results.append(err_row)
            if output_jsonl is not None:
                append_episode_jsonl(output_jsonl, err_row)
    return results


def run_hmeqa_compare(
    *,
    question_ids: list[int],
    mock_llm: bool = False,
    max_planning_steps: int = 20,
    max_movement_step: int = 10,
    hm3d_root: Path | None = None,
    questions_path: Path | None = None,
    init_poses_path: Path | None = None,
    eqa_vl_family: str | None = "gemma4",
    eqa_hf_model_id: str | None = None,
    device: str | None = "cuda",
    use_hm3d_semantics: bool | None = None,
) -> tuple[list[EpisodeMetrics], list[EpisodeMetrics]]:
    """Run the same HM-EQA questions with graph_eqa then dynagraph."""
    common: Any = {
        "mock_llm": mock_llm,
        "max_planning_steps": max_planning_steps,
        "max_movement_step": max_movement_step,
        "hm3d_root": hm3d_root,
        "questions_path": questions_path,
        "init_poses_path": init_poses_path,
        "eqa_vl_family": eqa_vl_family,
        "eqa_hf_model_id": eqa_hf_model_id,
        "device": device,
        "use_hm3d_semantics": use_hm3d_semantics,
    }
    graph = run_hmeqa_batch(question_ids=question_ids, method="graph_eqa", **cast(Any, common))
    _release_gpu_memory()
    dyna = run_hmeqa_batch(question_ids=question_ids, method="dynagraph", **cast(Any, common))
    _release_gpu_memory()
    return graph, dyna

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Run HM-EQA episodes with emet GraphEQA / Dynagraph controllers."""

from __future__ import annotations

from pathlib import Path

from emet.controller.controller_dynagraph import DynagraphController
from emet.controller.controller_graph_eqa import GraphEQAController
from emet.controller.task.dynamem import EQAExecuter
from emet.core.parameters import Parameters, get_parameters
from emet.habitat.config import default_hm3d_scene_dir
from emet.habitat.datasets import get_question, load_hmeqa_questions, load_scene_init_poses
from emet.habitat.hmeqa_enrich_labels import enrich_labels_for_question
from emet.habitat.metrics import EpisodeMetrics, append_episode_jsonl, extract_mcq_letter, grade_mcq_answer

from emet_habitat.robot_client import HabitatRobotClient
from emet_habitat.simulator import HabitatEQASimulator


def _release_gpu_memory() -> None:
    """Best-effort VRAM cleanup between Habitat episodes (semantic meshes + VLM)."""
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


def _configure_eqa_parameters(
    parameters: Parameters,
    *,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
) -> None:
    if eqa_vl_family is None and eqa_hf_model_id is None:
        return
    from emet.llms.vllm_registry import default_hf_model_id, normalize_vl_family

    eqa = dict(parameters.get("eqa", {}) or {})
    if eqa_vl_family is not None:
        eqa["backend"] = "qwen_vl"
        eqa["vl_family"] = eqa_vl_family
        if eqa_hf_model_id is None:
            fam = normalize_vl_family(eqa_vl_family)
            # E4B bf16 OOMs on 24GB with multi-image EQA; gemma-3-4b-it is the stable HM-EQA default.
            # Override with --eqa-hf-model-id google/gemma-4-e2b-it for Gemma 4 checkpoints.
            if fam == "gemma4":
                eqa["vl_hf_model_id"] = "google/gemma-3-4b-it"
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
    common = dict(
        robot=robot,
        parameters=params,
        save_rerun=False if no_rerun else False,
        cpu_only=not use_real_vlm,
        use_sensor_perception=graph_perception,
        use_instance_graph=False,
        # Habitat: depth voxel map for nav only — no SigLIP/YoloE reload per episode.
        manipulation_only=True,
    )
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
        agent.start()
        if agent.graph_memory is not None:
            hints = enrich_labels_for_question(question_id, q.scene)
            if hints:
                agent.graph_memory.seed_object_hints(hints)
        executor = EQAExecuter(agent)
        if rotate_in_place:
            executor.rotate_in_place()
        for _ in range(5):
            agent.update()

        discord_text, _images = agent.run_eqa(
            q.question_formatted,
            max_planning_steps=max_planning_steps,
            max_movement_step=max_movement_step,
        )
        raw_eqa = ""
        parsed_letter = ""
        model_confident = False
        if agent.graph_memory is not None:
            raw_eqa = agent.graph_memory.last_eqa_raw
            _reasoning, answer, model_confident, _action, _cr = agent.graph_memory.last_eqa_parsed
            parsed_letter = extract_mcq_letter(answer, q.choices)
            if not parsed_letter:
                parsed_letter = extract_mcq_letter(raw_eqa, q.choices)
        predicted = parsed_letter or (
            discord_text.split("---")[-1].strip() if "---" in discord_text else discord_text
        )
        correct = grade_mcq_answer(predicted, q.answer_letter, choices=q.choices)

        return EpisodeMetrics(
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
            raw_eqa_output=raw_eqa[:2000],
        )
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
) -> list[EpisodeMetrics]:
    from emet.habitat.metrics import read_completed_question_ids

    results: list[EpisodeMetrics] = []
    questions = load_hmeqa_questions(questions_path)
    done: set[int] = set()
    if output_jsonl is not None:
        if resume and output_jsonl.exists():
            done = read_completed_question_ids(output_jsonl)
        elif output_jsonl.exists():
            output_jsonl.unlink()
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
                )
            results.append(row)
            if output_jsonl is not None:
                append_episode_jsonl(output_jsonl, row)
                print(f"question_id={qid} done correct={row.correct} (appended {output_jsonl})", flush=True)
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
                )
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
    common = dict(
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
    graph = run_hmeqa_batch(question_ids=question_ids, method="graph_eqa", **common)
    _release_gpu_memory()
    dyna = run_hmeqa_batch(question_ids=question_ids, method="dynagraph", **common)
    _release_gpu_memory()
    return graph, dyna

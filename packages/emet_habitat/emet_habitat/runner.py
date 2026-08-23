# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Run HM-EQA episodes with emet GraphEQA / Dynagraph controllers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

from emet.controller.controller_dynagraph import DynagraphController
from emet.controller.controller_graph_eqa import GraphEQAController
from emet.controller.task.dynamem import EQAExecuter
from emet.core.parameters import Parameters, get_parameters
from emet.eval.benchmark_dynagraph import apply_dynagraph_harness_overrides
from emet.eval.episode_diagnostics import (
    EpisodeDiagnosticsConfig,
    EpisodeDiagnosticsRecorder,
    bind_diagnostics_recorder,
    habitat_export_voxel_history_default,
    unbind_diagnostics_recorder,
)
from emet.habitat.config import default_hm3d_scene_dir, hm3d_scene_glb_path
from emet.habitat.datasets import get_question, load_hmeqa_questions, load_scene_init_poses
from emet.habitat.episode_debug import (
    enrich_episode_metrics,
    run_tag_from_output_jsonl,
    save_episode_debug_bundle,
    save_error_episode_bundle,
    write_run_manifest,
)
from emet.habitat.hm3d_semantics import (
    hm3d_annotated_scene_dataset_config,
    hm3d_semantic_glb_for_basis,
    resolve_hm3d_semantics_enabled,
)
from emet.habitat.hmeqa_enrich_labels import enrich_labels_for_dataset_question
from emet.habitat.metrics import (
    EpisodeMetrics,
    append_episode_jsonl,
    choices_are_location_mcq,
    extract_mcq_letter,
    extract_mcq_letter_from_raw_eqa,
    grade_mcq_answer,
    should_abstain_location_mcq,
)
from emet.memory.graph_eqa.mcq_debias import match_freeform_to_choice
from emet_habitat.robot_client import HabitatRobotClient
from emet_habitat.simulator import HabitatEQASimulator


def _semantic_choice_letter(answer_text: str, choices: list[str]) -> str:
    """Resolve semantic answer text to the benchmark's letter encoding."""
    idx = match_freeform_to_choice(str(answer_text or ""), choices)
    return chr(ord("A") + idx) if idx is not None and 0 <= idx < min(len(choices), 5) else ""


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


def _mock_eqa_response(gold_letter: str, *, confident: bool = True) -> str:
    conf = "true" if confident else "false"
    mode = "smoke test" if confident else "explore harness (forces nav each planning step)"
    return (
        "reasoning: mock habitat harness\n"
        f"answer: {gold_letter}\n"
        f"confidence: {conf}\n"
        "action:\n"
        f"confidence_reasoning: mocked for {mode}\n"
    )


def _apply_method_parameters(parameters: Parameters | dict, method: str) -> Parameters:
    from emet.eval.benchmark_dynagraph import apply_habitat_eqa_method_parameters

    return apply_habitat_eqa_method_parameters(parameters, method)


def _normalize_hmeqa_method(method: str) -> str:
    from emet.eval.memory_backends import normalize_hmeqa_method

    return normalize_hmeqa_method(method)


def _cfg_hf_id_matches_family(hf_model_id: str, family: str) -> bool:
    from emet.llms.eqa_vl_settings import _hf_id_matches_family

    return _hf_id_matches_family(hf_model_id, family)


def _configure_habitat_nav(
    parameters: Parameters,
    *,
    habitat_perfect_nav: bool | None = None,
) -> None:
    """Habitat HM-EQA: navmesh pathing on by default; disable to exercise voxel A*."""
    eqa = dict(parameters.get("eqa", {}) or {})
    if habitat_perfect_nav is not None:
        eqa["habitat_perfect_nav"] = bool(habitat_perfect_nav)
    else:
        eqa.setdefault("habitat_perfect_nav", True)
    eqa.setdefault("habitat_explore_frontiers", True)
    eqa.setdefault("image_nav_min_approach_m", 0.35)
    parameters.set("eqa", eqa)


def _configure_habitat_mapping(parameters: Parameters) -> None:
    """Exploration-friendly mapping defaults for HM-EQA open floorplans.

    Default dynav clamps (``max_depth=2.5``, ``pad_obstacles=2``,
    ``smooth_kernel_size=3``) keep the reachable frontier hugging a ~2.5 m ring
    and seal thin corridors under dilation + morphological opening. Habitat-only
    overrides — real-robot dynav defaults stay unchanged.

    Temporary (2026-07): obstacle dilation is **off** (``pad_obstacles=0``,
    ``smooth_kernel_size=0``) so investigate/explore approach samples can sit on
    the indoor side of doorways instead of only the patio-side free ring.
    Override with ``EMET_HABITAT_PAD_OBSTACLES`` (grid cells; default ``0``).
    """
    import os

    parameters.set("max_depth", 4.5)
    pad_raw = os.environ.get("EMET_HABITAT_PAD_OBSTACLES", "0").strip()
    try:
        pad = max(0, int(pad_raw))
    except ValueError:
        pad = 0
    parameters.set("pad_obstacles", pad)
    # min_pad is unused by SparseVoxelMap.from_parameters / Dynamem ctor today, but
    # keep it consistent so future readers do not re-inflate Habitat maps.
    parameters.set("min_pad_obstacles", 0 if pad == 0 else max(1, pad))
    filters = dict(parameters.get("filters") or {})
    # Morphological open/close on explored (and obstacles in classic SparseVoxelMap)
    # also acts like soft dilation — disable while probing doorway entry.
    filters["smooth_kernel_size"] = 0 if pad == 0 else int(filters.get("smooth_kernel_size") or 1)
    parameters.set("filters", filters)


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


def _validate_requested_hm3d_semantics(
    *,
    requested: bool | None,
    question_ids: list[int],
    questions: list[Any],
    hm3d_root: Path | None,
) -> None:
    """Fail the batch before scoring rows when an explicit GT arm lacks assets."""
    if requested is not True:
        return
    root = Path(hm3d_root or default_hm3d_scene_dir()).expanduser().resolve()
    hm3d_data_root = root.parent.parent.parent
    annotated_config = hm3d_annotated_scene_dataset_config(
        hm3d_data_root,
        split=root.name,
    )
    for question_id in question_ids:
        question = get_question(questions, question_id=question_id)
        scene_glb = hm3d_scene_glb_path(question.scene, root)
        resolve_hm3d_semantics_enabled(
            True,
            semantic_glb=hm3d_semantic_glb_for_basis(scene_glb),
            annotated_config=annotated_config,
        )


def _configure_eqa_parameters(
    parameters: Parameters,
    *,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    eqa_vl_quantization: str | None,
    device: str = "cuda",
) -> None:
    """HM-EQA always uses the MCQ system prompt; optional VL family/HF id overrides."""
    eqa = dict(parameters.get("eqa", {}) or {})
    # Must run even when no VL CLI override — otherwise GraphEQAMemory sees an empty
    # variant, loads the caption-heavy EQA_PROMPT, and skips Reasoning: prefill.
    eqa.setdefault("prompt_variant", "hmeqa")
    if eqa_vl_family is None and eqa_hf_model_id is None and eqa_vl_quantization is None:
        parameters.set("eqa", eqa)
        return
    from emet.llms.eqa_vl_settings import resolve_vl_hf_model_id
    from emet.llms.vllm_registry import default_hf_model_id, normalize_vl_family

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
    if eqa_vl_quantization is not None:
        quantization = str(eqa_vl_quantization).strip().lower()
        eqa["vl_quantization"] = None if quantization == "none" else quantization
    parameters.set("eqa", eqa)


def _make_controller(
    robot: HabitatRobotClient,
    parameters: Parameters,
    *,
    method: str,
    mock_llm: bool,
    mock_llm_explore: bool,
    gold_letter: str,
    no_rerun: bool,
    use_real_vlm: bool,
    device: str | None,
    use_hm3d_semantics: bool | None = None,
    memory_summary: bool | None = None,
    mcq_debias: bool | None = None,
    explore_when_uncovered: str | None = None,
):
    from emet.eval.memory_backends import DYNAGRAPH

    method = _normalize_hmeqa_method(method)
    params = _apply_method_parameters(parameters, method)
    apply_dynagraph_harness_overrides(
        params,
        memory_summary=memory_summary,
        mcq_debias=mcq_debias,
        explore_when_uncovered=explore_when_uncovered,
    )
    harness_opts = dict(params.get("dynagraph_harness") or {})
    hm3d_sem = robot.uses_hm3d_semantics
    if use_hm3d_semantics is True and not hm3d_sem:
        raise RuntimeError("HM3D semantics were requested but the simulator did not enable them")
    # HM3D semantic sensor supplies graph labels; reserve VLM for EQA queries only.
    graph_perception = use_real_vlm and not hm3d_sem
    # Habitat: depth voxel map for nav only — no SigLIP/YoloE reload per episode.
    common = {
        "robot": robot,
        "parameters": params,
        "save_rerun": not no_rerun,
        "cpu_only": not use_real_vlm,
        "use_sensor_perception": graph_perception,
        "use_instance_graph": bool(harness_opts.get("use_instance_graph", False)),
        "manipulation_only": bool(harness_opts.get("manipulation_only", True)),
    }
    if method == DYNAGRAPH:
        agent = DynagraphController(**common)
    else:
        agent = GraphEQAController(**common)

    if mock_llm and agent.graph_memory is not None:
        confident = not mock_llm_explore
        agent.graph_memory.eqa_client = lambda _q: _mock_eqa_response(gold_letter, confident=confident)
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


def _seed_hmeqa_enrich_labels(
    agent,
    *,
    question_id: int,
    scene: str,
    questions_path: Path | None,
    enabled: bool,
) -> None:
    """Seed optional GT object hints independently of semantic perception."""
    if not enabled or agent.graph_memory is None:
        return
    hints = enrich_labels_for_dataset_question(
        question_id,
        scene,
        questions_path=questions_path,
    )
    if hints:
        agent.graph_memory.seed_object_hints(hints)


def run_hmeqa_episode(
    *,
    question_id: int,
    method: str = "dynagraph",
    mock_llm: bool = True,
    mock_llm_explore: bool = False,
    max_planning_steps: int = 20,
    max_movement_step: int = 10,
    hm3d_root: Path | None = None,
    questions_path: Path | None = None,
    init_poses_path: Path | None = None,
    no_rerun: bool = True,
    rotate_in_place: bool = True,
    use_hm3d_semantics: bool | None = None,
    use_enrich_labels: bool = False,
    eqa_vl_family: str | None = None,
    eqa_hf_model_id: str | None = None,
    eqa_vl_quantization: str | None = None,
    device: str | None = "cuda",
    frontier_nodes_enabled: bool | None = None,
    frontier_keyword_weight: float | None = None,
    habitat_perfect_nav: bool | None = None,
    memory_summary: bool | None = None,
    mcq_debias: bool | None = None,
    explore_when_uncovered: str | None = None,
    debug_run_tag: str | None = None,
    save_debug_bundle: bool = True,
    export_map: bool | None = None,
    export_video: bool | None = None,
    map_stride: int | None = None,
    extra_instruction: str | None = None,
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
    agent = None
    diag_recorder: EpisodeDiagnosticsRecorder | None = None
    try:
        sim.set_init_pose(init_pose)
        spawn_record = sim.last_init_pose_record
        robot = HabitatRobotClient(sim)
        if sim.uses_hm3d_semantics:
            print(f"HM3D semantics enabled for scene {q.scene}", flush=True)
        parameters = get_parameters("dynav_config.yaml")
        _configure_eqa_parameters(
            parameters,
            eqa_vl_family=eqa_vl_family,
            eqa_hf_model_id=eqa_hf_model_id,
            eqa_vl_quantization=eqa_vl_quantization,
            device=device or "cuda",
        )
        _configure_frontier_parameters(
            parameters,
            frontier_nodes_enabled=frontier_nodes_enabled,
            frontier_keyword_weight=frontier_keyword_weight,
        )
        _configure_habitat_nav(parameters, habitat_perfect_nav=habitat_perfect_nav)
        _configure_habitat_mapping(parameters)
        if memory_summary is None:
            raw = os.environ.get("EMET_DYNAGRAPH_MEMORY_SUMMARY", "").strip().lower()
            if raw in ("1", "true", "yes", "on"):
                memory_summary = True
            elif raw in ("0", "false", "no", "off"):
                memory_summary = False
        if mcq_debias is None:
            raw = os.environ.get("EMET_DYNAGRAPH_MCQ_DEBIAS", "").strip().lower()
            if raw in ("1", "true", "yes", "on"):
                mcq_debias = True
            elif raw in ("0", "false", "no", "off"):
                mcq_debias = False
        if explore_when_uncovered is None:
            raw = os.environ.get("EMET_DYNAGRAPH_EXPLORE_UNCOVERED", "").strip().lower()
            if raw:
                explore_when_uncovered = raw
        agent = _make_controller(
            robot,
            parameters,
            method=method,
            mock_llm=mock_llm,
            mock_llm_explore=mock_llm_explore,
            gold_letter=q.answer_letter,
            no_rerun=no_rerun,
            use_real_vlm=use_real_vlm,
            device=device,
            use_hm3d_semantics=use_hm3d_semantics,
            memory_summary=memory_summary,
            mcq_debias=mcq_debias,
            explore_when_uncovered=explore_when_uncovered,
        )
        diag_cfg = EpisodeDiagnosticsConfig.from_env(
            parameters,
            export_map=export_map,
            export_video=export_video,
            export_map_stride=map_stride if map_stride is not None else None,
            export_voxel_history=habitat_export_voxel_history_default(),
        )
        diag_recorder = EpisodeDiagnosticsRecorder(cfg=diag_cfg)
        bind_diagnostics_recorder(
            agent,
            diag_recorder,
            spawn_record=spawn_record,
            habitat_pathfinder=sim.pathfinder,
            habitat_floor_y=sim.floor_y,
        )
        from emet.eval.stack import compose_eqa_question

        eqa_question = compose_eqa_question(q.question_formatted, extra_instruction)
        agent._eqa_question = eqa_question
        agent.start()
        if agent.graph_memory is not None:
            # GraphEQA enrich labels are a separate GT-derived oracle axis. Keep
            # them opt-in rather than coupling them to the semantic sensor.
            _seed_hmeqa_enrich_labels(
                agent,
                question_id=question_id,
                scene=q.scene,
                questions_path=questions_path,
                enabled=use_enrich_labels,
            )
            agent.graph_memory.extract_relevant_objects(eqa_question)
        executor = EQAExecuter(agent)
        if rotate_in_place:
            executor.rotate_in_place()
        for _ in range(5):
            agent.update()
            if agent.graph_memory is not None and hasattr(agent, "_sync_graph_frontier_nodes"):
                agent._sync_graph_frontier_nodes()

        from emet.eval.dynagraph_vram import prepare_dynagraph_vram_for_eqa
        from emet.memory.graph_eqa.graph_stats import format_graph_node_breakdown

        if method == "dynagraph":
            prepare_dynagraph_vram_for_eqa(agent)
        if agent.graph_memory is not None:
            print(format_graph_node_breakdown(agent.graph_memory), flush=True)

        # Open the episode bundle dir *before* EQA so agentic frontier-pick panels
        # land under frontier_picks/ instead of only in ~/.cache.
        if save_debug_bundle and debug_run_tag:
            from emet.habitat.episode_debug import default_episodes_root

            ep_dir = default_episodes_root() / debug_run_tag / f"q{int(question_id):04d}_{method}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            agent._episode_debug_dir = str(ep_dir)
            (ep_dir / "frontier_picks").mkdir(exist_ok=True)

        discord_text, _images = agent.run_eqa(
            eqa_question,
            max_planning_steps=max_planning_steps,
            max_movement_step=max_movement_step,
        )
        raw_eqa = ""
        parsed_letter = ""
        model_confident = False
        formatted_answer = ""
        eqa_action = ""
        eqa_confidence_reasoning = ""
        summary = getattr(agent, "_agentic_eqa_summary", None)
        grounded_decision: dict[str, Any] | None = None
        if isinstance(summary, dict) and summary.get("decision_policy") == "grounded_v2":
            candidate = summary.get("final_decision")
            if isinstance(candidate, dict):
                grounded_decision = candidate
        if agent.graph_memory is not None:
            raw_eqa = agent.graph_memory.last_eqa_raw
            _reasoning, answer, model_confident, eqa_action, eqa_confidence_reasoning = (
                agent.graph_memory.last_eqa_parsed
            )
            formatted_answer = str(answer or "")
            # New runs preserve semantic answer text until this scoring boundary.
            parsed_letter = _semantic_choice_letter(formatted_answer, q.choices)
            if not parsed_letter and len(formatted_answer) <= 16:
                # Read-only compatibility for historical letter-first backends.
                parsed_letter = extract_mcq_letter(formatted_answer, q.choices)
            if not parsed_letter and not formatted_answer:
                parsed_letter = extract_mcq_letter_from_raw_eqa(raw_eqa, q.choices)
        predicted = parsed_letter
        if not predicted and grounded_decision is None:
            tail = discord_text.split("---")[-1].strip() if "---" in discord_text else discord_text
            predicted = extract_mcq_letter(tail, q.choices)
        if grounded_decision is not None:
            grounded_text = str(grounded_decision.get("answer_text") or "")
            grounded_letter = _semantic_choice_letter(grounded_text, q.choices)
            if not grounded_letter and not grounded_text:
                # Historical grounded summaries carried only a canonical letter.
                grounded_letter = extract_mcq_letter(str(grounded_decision.get("answer") or ""), q.choices)
            if grounded_letter:
                predicted = grounded_letter
                parsed_letter = grounded_letter
            else:
                predicted = ""
                parsed_letter = ""
        # Location MCQ where the target never appeared in the attached views. Prefer a
        # geometric equipment letter when we have one, but keep the model's letter
        # otherwise: blanking it here scored a guaranteed zero where a guess scores
        # 0.25 in expectation. The episode is flagged so calibration can separate
        # these from grounded answers.
        unverified_location_guess = False
        if (
            grounded_decision is None
            and agent.graph_memory is not None
            and q.choices
            and choices_are_location_mcq(q.choices)
            and not model_confident
        ):
            obs_ids = list(getattr(agent.graph_memory, "last_eqa_obs_ids", []) or [])
            visible_fn = getattr(agent.graph_memory, "_target_visible_in_obs_ids", None)
            visible = bool(callable(visible_fn) and visible_fn(obs_ids))
            if not visible:
                unverified_location_guess = True
                equip_fn = getattr(agent.graph_memory, "_equipment_letter_from_target_distances", None)
                equip = equip_fn(q.choices) if callable(equip_fn) else ""
                if equip:
                    predicted = equip
                    parsed_letter = equip
        predebias_letter = ""
        debias_votes = ""
        if (
            grounded_decision is None
            and agent.graph_memory is not None
            and getattr(agent.graph_memory, "mcq_debias_enabled", False)
            and q.choices
            and predicted
            and not should_abstain_location_mcq(raw_eqa, q.choices)
        ):
            vote_letter = agent.graph_memory.vote_mcq_letter(q.question, q.choices)
            debias_votes = json.dumps(getattr(agent.graph_memory, "last_mcq_debias", {}))
            if vote_letter:
                predebias_letter = predicted
                predicted = vote_letter
                parsed_letter = vote_letter
        correct = grade_mcq_answer(predicted, q.answer_letter, choices=q.choices) if predicted else False

        salvage_pred = ""
        if isinstance(summary, dict):
            salvage_pred = str(summary.get("salvage_counterfactual_letter") or "").strip().upper()[:1]
        if not salvage_pred and agent.graph_memory is not None:
            salvage_pred = (
                str(getattr(agent.graph_memory, "last_salvage_counterfactual_letter", "") or "").strip().upper()[:1]
            )
        if salvage_pred and salvage_pred not in "ABCD":
            salvage_pred = ""
        salvage_correct = grade_mcq_answer(salvage_pred, q.answer_letter, choices=q.choices) if salvage_pred else False
        scored_policy = ""
        if isinstance(summary, dict) and summary.get("scored_policy"):
            scored_policy = str(summary.get("scored_policy") or "")
        elif grounded_decision is not None:
            scored_policy = "grounded_v2"
        elif salvage_pred or (isinstance(summary, dict) and "salvage_counterfactual_letter" in summary):
            scored_policy = "no_salvage"

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
            salvage_pred=salvage_pred,
            salvage_correct=bool(salvage_correct),
            scored_policy=scored_policy,
        )
        # 4000 comfortably exceeds the bounded payload (4 x 200-char replies + 300-char
        # freeform + JSON escaping) so the stored JSON is never truncated mid-string.
        metrics.predebias_letter = predebias_letter
        metrics.debias_votes = debias_votes[:4000]
        metrics.unverified_location_guess = unverified_location_guess
        if agent.graph_memory is not None:
            metrics.eqa_answer_field_missing = not agent.graph_memory.last_eqa_answer_field_emitted
            metrics.eqa_salvage_used = agent.graph_memory.last_eqa_salvage_used
        if isinstance(summary, dict):
            metrics.decision_rounds = int(summary.get("decision_rounds") or 0)
            metrics.budget_hit = bool(summary.get("budget_hit"))
            metrics.answer_provenance = str(summary.get("answer_provenance") or "")
            try:
                metrics.answer_confidence = float(summary.get("answer_confidence") or 0.0)
            except (TypeError, ValueError):
                metrics.answer_confidence = 0.0
        from emet.llms.eqa_vl_settings import resolve_vl_endpoint

        vl_ep = resolve_vl_endpoint(parameters) or str(eqa_cfg.get("vl_endpoint") or "").strip()
        enrich_episode_metrics(
            metrics,
            agent=agent,
            choices=q.choices,
            formatted_answer=formatted_answer,
            eqa_action=str(eqa_action or ""),
            eqa_confidence_reasoning=str(eqa_confidence_reasoning or ""),
            vl_family=str(eqa_cfg.get("vl_family") or eqa_vl_family or ""),
            vl_hf_model_id=str(eqa_cfg.get("vl_hf_model_id") or eqa_hf_model_id or ""),
            vl_endpoint=vl_ep or "",
        )
        if save_debug_bundle and debug_run_tag:
            try:
                save_episode_debug_bundle(
                    run_tag=debug_run_tag,
                    metrics=metrics,
                    agent=agent,
                    raw_eqa_full=raw_eqa,
                    recorder=diag_recorder,
                    diagnostics_cfg=diag_cfg,
                )
            except Exception as exc:
                print(
                    f"question_id={question_id} debug bundle save failed (metrics kept): {exc}",
                    flush=True,
                )
        return metrics
    finally:
        if agent is not None and diag_recorder is not None:
            unbind_diagnostics_recorder(agent, diag_recorder)
        sim.close()
        _release_gpu_memory()


def run_hmeqa_batch(
    *,
    question_ids: list[int],
    method: str = "static_graph",
    mock_llm: bool = False,
    max_planning_steps: int = 20,
    max_movement_step: int = 10,
    hm3d_root: Path | None = None,
    questions_path: Path | None = None,
    init_poses_path: Path | None = None,
    eqa_vl_family: str | None = None,
    eqa_hf_model_id: str | None = None,
    eqa_vl_quantization: str | None = None,
    device: str | None = "cuda",
    continue_on_error: bool = True,
    use_hm3d_semantics: bool | None = None,
    use_enrich_labels: bool = False,
    output_jsonl: Path | None = None,
    resume: bool = False,
    frontier_nodes_enabled: bool | None = None,
    frontier_keyword_weight: float | None = None,
    habitat_perfect_nav: bool | None = None,
    memory_summary: bool | None = None,
    mcq_debias: bool | None = None,
    explore_when_uncovered: str | None = None,
    debug_run_tag: str | None = None,
    export_map: bool | None = None,
    export_video: bool | None = None,
    map_stride: int | None = None,
) -> list[EpisodeMetrics]:
    from emet.habitat.metrics import read_completed_question_ids

    results: list[EpisodeMetrics] = []
    questions = load_hmeqa_questions(questions_path)
    _validate_requested_hm3d_semantics(
        requested=use_hm3d_semantics,
        question_ids=question_ids,
        questions=questions,
        hm3d_root=hm3d_root,
    )
    done: set[int] = set()
    explicit_tag = (debug_run_tag or "").strip() or None
    run_tag = explicit_tag or run_tag_from_output_jsonl(output_jsonl)
    parameters = get_parameters("dynav_config.yaml")
    _configure_frontier_parameters(
        parameters,
        frontier_nodes_enabled=frontier_nodes_enabled,
        frontier_keyword_weight=frontier_keyword_weight,
    )
    _configure_habitat_nav(parameters, habitat_perfect_nav=habitat_perfect_nav)
    _configure_habitat_mapping(parameters)
    manifest_parameters = _apply_method_parameters(parameters, method)
    apply_dynagraph_harness_overrides(
        manifest_parameters,
        memory_summary=memory_summary,
        mcq_debias=mcq_debias,
        explore_when_uncovered=explore_when_uncovered,
    )
    _configure_eqa_parameters(
        manifest_parameters,
        eqa_vl_family=eqa_vl_family,
        eqa_hf_model_id=eqa_hf_model_id,
        eqa_vl_quantization=eqa_vl_quantization,
        device=device or "cuda",
    )
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = output_jsonl.parent / f"{output_jsonl.stem}_manifest.json"
        if not output_jsonl.exists() and not (resume and manifest_path.is_file()):
            # Leave a durable empty JSONL beside a newly-created manifest so a
            # crash before the first episode append can still resume safely.
            output_jsonl.touch()
        if resume and output_jsonl.exists():
            done = read_completed_question_ids(output_jsonl)
        manifest = write_run_manifest(
            output_jsonl=output_jsonl,
            method=method,
            question_ids=question_ids,
            mock_llm=mock_llm,
            max_planning_steps=max_planning_steps,
            max_movement_step=max_movement_step,
            eqa_vl_family=eqa_vl_family,
            eqa_hf_model_id=eqa_hf_model_id,
            eqa_vl_quantization=eqa_vl_quantization,
            device=device,
            use_hm3d_semantics=use_hm3d_semantics,
            use_enrich_labels=use_enrich_labels,
            resume=resume,
            parameters=manifest_parameters,
            hm3d_root=hm3d_root,
            questions_path=questions_path,
            init_poses_path=init_poses_path,
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
                eqa_vl_quantization=eqa_vl_quantization,
                device=device,
                use_hm3d_semantics=use_hm3d_semantics,
                use_enrich_labels=use_enrich_labels,
                frontier_nodes_enabled=frontier_nodes_enabled,
                frontier_keyword_weight=frontier_keyword_weight,
                habitat_perfect_nav=habitat_perfect_nav,
                memory_summary=memory_summary,
                mcq_debias=mcq_debias,
                explore_when_uncovered=explore_when_uncovered,
                debug_run_tag=run_tag,
                export_map=export_map,
                export_video=export_video,
                map_stride=map_stride,
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
    eqa_vl_family: str | None = None,
    eqa_hf_model_id: str | None = None,
    eqa_vl_quantization: str | None = None,
    device: str | None = "cuda",
    use_hm3d_semantics: bool | None = None,
    use_enrich_labels: bool = False,
) -> tuple[list[EpisodeMetrics], list[EpisodeMetrics]]:
    """Run the same HM-EQA questions with static_graph then dynagraph."""
    common: Any = {
        "mock_llm": mock_llm,
        "max_planning_steps": max_planning_steps,
        "max_movement_step": max_movement_step,
        "hm3d_root": hm3d_root,
        "questions_path": questions_path,
        "init_poses_path": init_poses_path,
        "eqa_vl_family": eqa_vl_family,
        "eqa_hf_model_id": eqa_hf_model_id,
        "eqa_vl_quantization": eqa_vl_quantization,
        "device": device,
        "use_hm3d_semantics": use_hm3d_semantics,
        "use_enrich_labels": use_enrich_labels,
    }
    graph = run_hmeqa_batch(question_ids=question_ids, method="static_graph", **cast(Any, common))
    _release_gpu_memory()
    dyna = run_hmeqa_batch(question_ids=question_ids, method="dynagraph", **cast(Any, common))
    _release_gpu_memory()
    return graph, dyna

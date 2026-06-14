# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Run SQA3D episodes: DynaMem voxel mapping + optional GraphEQA (Dynagraph) EQA."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from emet.benchmarks.sqa3d.datasets import get_sqa3d_question, load_sqa3d_questions
from emet.eval.memory_backends import SQA3D_MEMORY_BACKEND
from emet.benchmarks.sqa3d.episode_metrics import (
    SQA3DEpisodeMetrics,
    append_sqa3d_jsonl,
    read_completed_sqa3d_question_ids,
)
from emet.benchmarks.sqa3d.metrics import answer_match, clean_answer, extract_answer_from_eqa_row
from emet.benchmarks.sqa3d.scannet.config import default_scannet_root
from emet.benchmarks.sqa3d.scannet.robot_client import ScanNetRobotClient
from emet.benchmarks.sqa3d.scannet.simulator import ScanNetReplayMode, create_scannet_simulator
from emet.controller.controller_dynamem import DynamemController
from emet.controller.controller_dynagraph import DynagraphController
from emet.controller.task.dynamem import EQAExecuter
from emet.core.parameters import Parameters, get_parameters
from emet.eval.episode_diagnostics import EpisodeDiagnosticsRecorder, attach_diagnostics_recorder

SQA3DMethod = SQA3D_MEMORY_BACKEND
SQA3DProfile = Literal["smoke", "tuned"]

_INFRA_RE = re.compile(r"(?i)(cuda out of memory|out of memory|^error:)")
_COORD_ANSWER_RE = re.compile(r"(?i)(approximately\s*\(|at\s*\(-?\d|i also provide relevant images)")
_EQA_ANSWER_RE = re.compile(
    r"(?is)(?:^|\n)\s*answer:\s*(.+?)(?:\n\s*(?:confidence|action|confidence_reasoning):|\Z)"
)


def _release_gpu_memory() -> None:
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _teardown_agent(agent: object | None) -> None:
    if agent is None:
        return
    for attr in ("graph_memory", "voxel_map", "detection_model", "rerun_visualizer"):
        try:
            setattr(agent, attr, None)
        except Exception:
            pass
    _release_gpu_memory()


def _mock_eqa_response(gold_answer: str) -> str:
    return (
        "reasoning: mock sqa3d harness\n"
        f"answer: {gold_answer}\n"
        "confidence: true\n"
        "action:\n"
        "confidence_reasoning: mocked for smoke test\n"
    )


def _resolve_profile(*, mock_llm: bool, profile: SQA3DProfile | None) -> SQA3DProfile:
    if profile is not None:
        return profile
    return "smoke" if mock_llm else "tuned"


def _simulator_render_kwargs(*, profile: SQA3DProfile) -> dict[str, int]:
    if profile == "smoke":
        return {"image_width": 480, "image_height": 360}
    return {"image_width": 640, "image_height": 480}


def _apply_profile_parameters(
    parameters: Parameters | dict,
    *,
    method: SQA3DMethod,
    profile: SQA3DProfile,
) -> Parameters:
    from emet.eval.benchmark_dynagraph import apply_sqa3d_dynagraph

    return apply_sqa3d_dynagraph(parameters, method=method, profile=profile)


def _configure_eqa_parameters(
    parameters: Parameters,
    *,
    method: SQA3DMethod,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    device: str | None = None,
) -> None:
    eqa = dict(parameters.get("eqa", {}) or {})
    eqa.setdefault("prompt_variant", "sqa3d")
    if method == "dynagraph":
        eqa["sqa3d_allow_partial_graph"] = True
    if device == "cpu":
        eqa["vl_quantization"] = None
    if eqa_vl_family is not None:
        from emet.llms.vllm_registry import default_hf_model_id, normalize_vl_family

        eqa["backend"] = "qwen_vl"
        eqa["vl_family"] = eqa_vl_family
        if eqa_hf_model_id is None:
            fam = normalize_vl_family(eqa_vl_family)
            if fam == "gemma4":
                eqa["vl_hf_model_id"] = "google/gemma-3-4b-it"
            else:
                eqa["vl_hf_model_id"] = default_hf_model_id(fam)
    if eqa_hf_model_id is not None:
        eqa["vl_hf_model_id"] = eqa_hf_model_id
    parameters.set("eqa", eqa)


def _attach_graph_eqa_clients(
    agent: DynamemController,
    *,
    parameters: Parameters,
    mock_llm: bool,
    gold_answer: str,
    use_real_vlm: bool,
    device: str | None,
) -> None:
    if agent.graph_memory is None:
        raise RuntimeError("dynagraph agent has no graph_memory")
    if mock_llm:
        agent.graph_memory.eqa_client = lambda _q: _mock_eqa_response(gold_answer)
        agent.graph_memory.image_description_client = lambda _x: "object"
    elif use_real_vlm:
        from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients

        keyword_client, eqa_client = build_graph_eqa_vlm_clients(parameters=parameters, device=device)
        agent.graph_memory.image_description_client = keyword_client
        agent.graph_memory.eqa_client = eqa_client
        if agent.sensor_builder is not None:
            agent.sensor_builder._perception = keyword_client
            agent.sensor_builder._lazy_vl_client = keyword_client
            agent.sensor_builder.cpu_only = False


def _make_agent(
    robot: ScanNetRobotClient,
    parameters: Parameters,
    *,
    method: SQA3DMethod,
    mock_llm: bool,
    gold_answer: str,
    use_real_vlm: bool,
    device: str | None,
) -> DynamemController:
    if method == "dynagraph":
        agent = DynagraphController(
            robot=robot,
            parameters=parameters,
            save_rerun=False,
            cpu_only=not use_real_vlm,
            use_sensor_perception=use_real_vlm,
            use_instance_graph=False,
            manipulation_only=True,
        )
        _attach_graph_eqa_clients(
            agent,
            parameters=parameters,
            mock_llm=mock_llm,
            gold_answer=gold_answer,
            use_real_vlm=use_real_vlm,
            device=device,
        )
        return agent

    agent = DynamemController(
        robot=robot,
        parameters=parameters,
        save_rerun=False,
        manipulation_only=False,
        cpu_only=not use_real_vlm,
        eqa=True,
        defer_eqa_vllm=True,
    )
    vm = agent.voxel_map
    if mock_llm:
        vm.eqa_client = lambda _q: _mock_eqa_response(gold_answer)
        vm.image_description_client = lambda _x: "object"
        vm._eqa_pending = None
    elif use_real_vlm:
        from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients

        keyword_client, eqa_client = build_graph_eqa_vlm_clients(parameters=parameters, device=device)
        vm.image_description_client = keyword_client
        vm.eqa_client = eqa_client
        vm._eqa_pending = None
    return agent


def _looks_like_graph_coord_dump(text: str) -> bool:
    return bool(_COORD_ANSWER_RE.search(text or ""))


def _parse_structured_eqa_answer(raw_eqa: str) -> str:
    if not raw_eqa:
        return ""
    m = _EQA_ANSWER_RE.search(raw_eqa)
    if m:
        return m.group(1).strip()
    return ""


def _is_infra_failure_text(*texts: str) -> bool:
    return any(_INFRA_RE.search((text or "").strip()) for text in texts)


def _sanitize_prediction_text(text: str) -> str:
    raw = (text or "").strip()
    if not raw or _INFRA_RE.search(raw):
        return ""
    if _looks_like_graph_coord_dump(raw):
        return ""
    if raw.lower().startswith("unknown i am not fully confident"):
        m = re.search(r"(?i)answer:\s*(.+?)(?:\n|$)", raw)
        if m:
            return m.group(1).strip()
        return ""
    if raw.lower().startswith("caption:"):
        m = re.search(r"(?i)answer:\s*(.+?)(?:\n|$)", raw)
        if m:
            return m.group(1).strip()
        return ""
    return raw


def _extract_open_answer(raw_eqa: str, parsed_answer: str) -> str:
    candidates: list[str] = []
    if parsed_answer and parsed_answer.strip():
        candidates.append(parsed_answer.strip())
    structured = _parse_structured_eqa_answer(raw_eqa)
    if structured:
        candidates.append(structured)
    if raw_eqa:
        row = extract_answer_from_eqa_row({"raw_eqa_output": raw_eqa, "discord_text": raw_eqa})
        if row:
            candidates.append(row)
    for candidate in candidates:
        cleaned = _sanitize_prediction_text(candidate)
        if cleaned:
            return cleaned
    return ""


def _run_dynamem_eqa(
    agent: DynamemController,
    prompt: str,
    *,
    max_planning_steps: int,
    max_movement_step: int,
) -> tuple[str, str, bool]:
    answer = ""
    discord_text = ""
    confident = False
    raw_eqa = ""
    for _ in range(max_planning_steps):
        answer, discord_text, _images, confident = agent.run_eqa_one_iter(
            prompt,
            max_movement_step=max_movement_step,
        )
        raw_eqa = str(getattr(agent.voxel_map, "_last_eqa_raw", "") or raw_eqa)
        if confident:
            break
    predicted = _extract_open_answer(raw_eqa, str(answer or ""))
    if not predicted and discord_text:
        predicted = _extract_open_answer(discord_text, "")
    return predicted, raw_eqa, confident


def _run_dynagraph_eqa(
    agent: DynamemController,
    prompt: str,
    *,
    max_planning_steps: int,
    max_movement_step: int,
) -> tuple[str, str, bool]:
    discord_text, _images = agent.run_eqa(
        prompt,
        max_planning_steps=max_planning_steps,
        max_movement_step=max_movement_step,
    )
    raw_eqa = ""
    parsed_answer = ""
    model_confident = False
    if agent.graph_memory is not None:
        raw_eqa = agent.graph_memory.last_eqa_raw
        _reasoning, answer, model_confident, _action, _cr = agent.graph_memory.last_eqa_parsed
        parsed_answer = str(answer or "")
    predicted = _extract_open_answer(raw_eqa, parsed_answer)
    if not predicted and discord_text:
        predicted = _extract_open_answer(discord_text, "")
    return predicted, raw_eqa, model_confident


def _replay_metadata_from_sim(sim) -> dict[str, object]:
    return {
        "replay_backend": str(getattr(sim, "replay_backend", "") or ""),
        "sens_frame_index": getattr(sim, "anchor_sens_frame_index", None),
        "sens_match_xy_m": getattr(sim, "anchor_sens_match_xy_m", None),
    }


def _score_episode(
    q,
    *,
    method: SQA3DMethod,
    predicted: str,
    raw_eqa: str,
    model_confident: bool,
    planning_steps: int,
    infra_failure: bool = False,
    replay_backend: str = "",
    sens_frame_index: int | None = None,
    sens_match_xy_m: float | None = None,
    split: str = "",
    profile: SQA3DProfile = "smoke",
    replay_mode: ScanNetReplayMode = "auto",
    export_dir: str = "",
) -> SQA3DEpisodeMetrics:
    pred_clean = clean_answer(predicted)
    gts = [clean_answer(a) for a in q.answers if a.strip()]
    em, em_refined = answer_match(pred_clean, gts) if gts and pred_clean else (False, False)
    infra = bool(infra_failure) or _is_infra_failure_text(raw_eqa, predicted)
    return SQA3DEpisodeMetrics(
        dataset="sqa3d",
        method=method,
        question_id=q.question_id,
        scene_id=q.scene_id,
        question=q.question,
        situation=q.situation,
        gold_answers=list(q.answers),
        predicted_answer=predicted[:200],
        em=em,
        em_refined=em_refined,
        confident=model_confident,
        planning_steps=planning_steps,
        success=em,
        raw_eqa_output=raw_eqa[:2000],
        infra_failure=infra,
        replay_backend=replay_backend,
        sens_frame_index=sens_frame_index,
        sens_match_xy_m=sens_match_xy_m,
        split=split,
        profile=profile,
        replay_mode=replay_mode,
        question_type=str(getattr(q, "question_type", "") or ""),
        export_dir=export_dir,
    )


def run_sqa3d_episode(
    *,
    question_id: int,
    method: SQA3DMethod = "dynagraph",
    mock_llm: bool = True,
    max_planning_steps: int | None = None,
    max_movement_step: int | None = None,
    split: str = "val",
    data_dir: Path | None = None,
    scannet_root: Path | None = None,
    rotate_in_place: bool = True,
    eqa_vl_family: str | None = None,
    eqa_hf_model_id: str | None = None,
    device: str | None = "cuda",
    profile: SQA3DProfile | None = None,
    post_rotate_updates: int | None = None,
    replay_mode: ScanNetReplayMode = "auto",
    export_root: Path | None = None,
) -> SQA3DEpisodeMetrics:
    prof = _resolve_profile(mock_llm=mock_llm, profile=profile)
    if max_planning_steps is None:
        max_planning_steps = 8 if prof == "smoke" else 15
    if max_movement_step is None:
        max_movement_step = 0 if prof == "smoke" else 3
    if post_rotate_updates is None:
        post_rotate_updates = 5 if prof == "smoke" else 6

    questions = load_sqa3d_questions(split, data_dir=data_dir)
    q = get_sqa3d_question(questions, question_id=question_id)
    gold = q.primary_answer

    sim = create_scannet_simulator(
        q.scene_id,
        scannet_root=scannet_root or default_scannet_root(),
        replay_mode=replay_mode,
        **_simulator_render_kwargs(profile=prof),
    )
    use_real_vlm = not mock_llm
    agent = None
    robot = None
    replay_meta: dict[str, object] = {}
    try:
        sim.set_sqa3d_pose(q)
        replay_meta = _replay_metadata_from_sim(sim)
        robot = ScanNetRobotClient(sim)
        parameters = get_parameters("dynav_config.yaml")
        parameters = _apply_profile_parameters(parameters, method=method, profile=prof)
        _configure_eqa_parameters(
            parameters,
            method=method,
            eqa_vl_family=eqa_vl_family,
            eqa_hf_model_id=eqa_hf_model_id,
            device=device,
        )
        agent = _make_agent(
            robot,
            parameters,
            method=method,
            mock_llm=mock_llm,
            gold_answer=gold,
            use_real_vlm=use_real_vlm,
            device=device,
        )
        diag_recorder = EpisodeDiagnosticsRecorder()
        attach_diagnostics_recorder(agent, diag_recorder)
        agent.start()
        executor = EQAExecuter(agent)
        if rotate_in_place:
            executor.rotate_in_place()
        for _ in range(post_rotate_updates):
            agent.update()

        _release_gpu_memory()
        prompt = q.formatted_prompt()
        if method == "dynagraph":
            predicted, raw_eqa, model_confident = _run_dynagraph_eqa(
                agent,
                prompt,
                max_planning_steps=max_planning_steps,
                max_movement_step=max_movement_step,
            )
        else:
            predicted, raw_eqa, model_confident = _run_dynamem_eqa(
                agent,
                prompt,
                max_planning_steps=max_planning_steps,
                max_movement_step=max_movement_step,
            )

        export_dir = ""
        if export_root is not None and agent is not None:
            from emet.benchmarks.sqa3d.export_episode import export_sqa3d_episode_artifacts

            ep_path = export_sqa3d_episode_artifacts(
                agent,
                q,
                method=method,
                profile=prof,
                replay_mode=replay_mode,
                replay_meta=replay_meta,
                predicted=predicted,
                raw_eqa=raw_eqa,
                model_confident=model_confident,
                planning_steps=int(getattr(agent, "obs_count", 0)),
                export_root=export_root,
                split=split,
                infra_failure=_is_infra_failure_text(raw_eqa, predicted),
                recorder=diag_recorder,
            )
            export_dir = str(ep_path)

        return _score_episode(
            q,
            method=method,
            predicted=predicted,
            raw_eqa=raw_eqa,
            model_confident=model_confident,
            planning_steps=getattr(agent, "obs_count", 0),
            infra_failure=_is_infra_failure_text(raw_eqa, predicted),
            split=split,
            profile=prof,
            replay_mode=replay_mode,
            export_dir=export_dir,
            **replay_meta,
        )
    finally:
        sim.close()
        _teardown_agent(agent)
        del robot
        _release_gpu_memory()


def _run_sqa3d_batch_isolated(
    *,
    question_ids: list[int],
    method: SQA3DMethod,
    max_planning_steps: int | None,
    max_movement_step: int | None,
    split: str,
    data_dir: Path | None,
    scannet_root: Path | None,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    device: str | None,
    output_jsonl: Path | None,
    resume: bool,
    profile: SQA3DProfile,
    replay_mode: ScanNetReplayMode,
    continue_on_error: bool,
    export_root: Path | None = None,
) -> list[SQA3DEpisodeMetrics]:
    """One subprocess per episode so the VLM and maps are fully released between runs."""
    import subprocess
    import sys

    results: list[SQA3DEpisodeMetrics] = []
    done: set[int] = set()
    if output_jsonl is not None:
        if resume and output_jsonl.exists():
            done = read_completed_sqa3d_question_ids(output_jsonl)
        elif output_jsonl.exists():
            output_jsonl.unlink()

    for qid in question_ids:
        if qid in done:
            print(f"question_id={qid} skip (already in {output_jsonl})", flush=True)
            continue
        cmd = [
            sys.executable,
            "-m",
            "emet.cli",
            "sqa3d",
            "run-episode",
            "--question-id",
            str(qid),
            "--split",
            split,
            "--method",
            method,
            "--profile",
            profile,
            "--replay-mode",
            replay_mode,
        ]
        if max_planning_steps is not None:
            cmd.extend(["--max-planning-steps", str(max_planning_steps)])
        if data_dir is not None:
            cmd.extend(["--data-dir", str(data_dir)])
        if scannet_root is not None:
            cmd.extend(["--scannet-root", str(scannet_root)])
        if eqa_vl_family is not None:
            cmd.extend(["--eqa-vl-family", eqa_vl_family])
        if eqa_hf_model_id is not None:
            cmd.extend(["--eqa-hf-model-id", eqa_hf_model_id])
        if device is not None:
            cmd.extend(["--device", device])
        if export_root is not None:
            cmd.extend(["--export-root", str(export_root)])
        if output_jsonl is not None:
            cmd.extend(["--output", str(output_jsonl)])
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            if not continue_on_error:
                raise RuntimeError(f"isolated run-episode failed for question_id={qid} exit={proc.returncode}")
            questions = load_sqa3d_questions(split, data_dir=data_dir)
            q = get_sqa3d_question(questions, question_id=qid)
            err_row = SQA3DEpisodeMetrics(
                dataset="sqa3d",
                method=method,
                question_id=qid,
                scene_id=q.scene_id,
                question=q.question,
                situation=q.situation,
                gold_answers=list(q.answers),
                predicted_answer="",
                em=False,
                em_refined=False,
                confident=False,
                planning_steps=0,
                success=False,
                infra_failure=True,
            )
            results.append(err_row)
            if output_jsonl is not None:
                append_sqa3d_jsonl(output_jsonl, err_row)
            continue
        if output_jsonl is not None and output_jsonl.exists():
            row = None
            for line in reversed(output_jsonl.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if int(payload.get("question_id", -1)) == qid:
                    row = SQA3DEpisodeMetrics(**payload)
                    break
            if row is not None:
                results.append(row)
                print(f"question_id={qid} em={row.em} (appended {output_jsonl})", flush=True)
    return results


def run_sqa3d_batch(
    *,
    question_ids: list[int],
    method: SQA3DMethod = "dynagraph",
    mock_llm: bool = True,
    max_planning_steps: int | None = None,
    max_movement_step: int | None = None,
    split: str = "val",
    data_dir: Path | None = None,
    scannet_root: Path | None = None,
    eqa_vl_family: str | None = None,
    eqa_hf_model_id: str | None = None,
    device: str | None = "cuda",
    continue_on_error: bool = True,
    output_jsonl: Path | None = None,
    resume: bool = False,
    profile: SQA3DProfile | None = None,
    replay_mode: ScanNetReplayMode = "auto",
    isolate_episodes: bool = False,
    export_root: Path | None = None,
) -> list[SQA3DEpisodeMetrics]:
    results: list[SQA3DEpisodeMetrics] = []
    questions = load_sqa3d_questions(split, data_dir=data_dir)
    prof = _resolve_profile(mock_llm=mock_llm, profile=profile)
    if isolate_episodes and not mock_llm:
        return _run_sqa3d_batch_isolated(
            question_ids=question_ids,
            method=method,
            max_planning_steps=max_planning_steps,
            max_movement_step=max_movement_step,
            split=split,
            data_dir=data_dir,
            scannet_root=scannet_root,
            eqa_vl_family=eqa_vl_family,
            eqa_hf_model_id=eqa_hf_model_id,
            device=device,
            output_jsonl=output_jsonl,
            resume=resume,
            profile=prof,
            replay_mode=replay_mode,
            continue_on_error=continue_on_error,
            export_root=export_root,
        )
    if not mock_llm:
        parameters = get_parameters("dynav_config.yaml")
        parameters = _apply_profile_parameters(parameters, method=method, profile=prof)
        _configure_eqa_parameters(
            parameters,
            method=method,
            eqa_vl_family=eqa_vl_family,
            eqa_hf_model_id=eqa_hf_model_id,
            device=device,
        )
        from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients

        build_graph_eqa_vlm_clients(parameters=parameters, device=device)
    done: set[int] = set()
    if output_jsonl is not None:
        if resume and output_jsonl.exists():
            done = read_completed_sqa3d_question_ids(output_jsonl)
        elif output_jsonl.exists():
            output_jsonl.unlink()

    for qid in question_ids:
        if qid in done:
            print(f"question_id={qid} skip (already in {output_jsonl})", flush=True)
            continue
        try:
            row = run_sqa3d_episode(
                question_id=qid,
                method=method,
                mock_llm=mock_llm,
                max_planning_steps=max_planning_steps,
                max_movement_step=max_movement_step,
                split=split,
                data_dir=data_dir,
                scannet_root=scannet_root,
                eqa_vl_family=eqa_vl_family,
                eqa_hf_model_id=eqa_hf_model_id,
                device=device,
                profile=prof,
                replay_mode=replay_mode,
                export_root=export_root,
            )
            results.append(row)
            if output_jsonl is not None:
                append_sqa3d_jsonl(output_jsonl, row)
                print(f"question_id={qid} em={row.em} (appended {output_jsonl})", flush=True)
            _release_gpu_memory()
        except Exception as exc:
            if not continue_on_error:
                raise
            q = get_sqa3d_question(questions, question_id=qid)
            print(f"question_id={qid} failed: {exc}", flush=True)
            exc_text = str(exc)
            err_row = SQA3DEpisodeMetrics(
                dataset="sqa3d",
                method=method,
                question_id=qid,
                scene_id=q.scene_id,
                question=q.question,
                situation=q.situation,
                gold_answers=list(q.answers),
                predicted_answer=_sanitize_prediction_text(exc_text),
                em=False,
                em_refined=False,
                confident=False,
                planning_steps=0,
                success=False,
                infra_failure=_is_infra_failure_text(exc_text),
            )
            results.append(err_row)
            if output_jsonl is not None:
                append_sqa3d_jsonl(output_jsonl, err_row)
    return results

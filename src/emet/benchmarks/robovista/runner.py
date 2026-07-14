# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Offline RoboVista MCQ-VQA batch runner (static images + VLM, no sim)."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from emet.benchmarks.robovista.datasets import (
    RoboVistaQuestion,
    load_robovista,
)
from emet.benchmarks.robovista.metrics import summarize_robovista_rows
from emet.benchmarks.robovista.prompts import build_robovista_prompt
from emet.core.parameters import Parameters, get_parameters
from emet.habitat.metrics import (
    extract_mcq_letter,
    extract_mcq_letter_from_raw_eqa,
    grade_mcq_answer,
)
from emet.utils.logger import Logger

logger = Logger(__name__)

EqaClient = Callable[..., str]


def default_output_dir(run_id: str | None = None) -> Path:
    tag = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path.home() / "runs" / "emet" / "robovista" / tag


def _configure_eqa_parameters(
    parameters: Parameters,
    *,
    eqa_vl_family: str | None,
    eqa_hf_model_id: str | None,
    device: str | None,
) -> None:
    eqa = dict(parameters.get("eqa", {}) or {})
    eqa["prompt_variant"] = "mcq"
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


def _make_mock_client(gold_letter: str) -> EqaClient:
    def _mock(command: str | list, **_kwargs: Any) -> str:
        return f"Answer: {gold_letter}"

    return _mock


def _make_eqa_client(
    *,
    parameters: Parameters,
    mock_llm: bool,
    device: str | None,
) -> EqaClient | None:
    if mock_llm:
        return None
    from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients

    _, eqa_client = build_graph_eqa_vlm_clients(parameters=parameters, device=device)
    return eqa_client


def _read_completed_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.is_file():
        return set()
    done: set[str] = set()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        qid = row.get("id")
        if qid is not None:
            done.add(str(qid))
    return done


def _grade_prediction(raw: str, question: RoboVistaQuestion) -> tuple[str, bool]:
    letter = extract_mcq_letter_from_raw_eqa(raw, question.choices)
    if not letter:
        letter = extract_mcq_letter(raw, question.choices)
    ok = grade_mcq_answer(letter or raw, question.gold_letter, choices=question.choices)
    return letter, ok


def run_robovista_batch(
    *,
    questions: Sequence[RoboVistaQuestion] | None = None,
    domains: Sequence[str] | None = None,
    ability_types: Sequence[str] | None = None,
    max_questions: int | None = None,
    mock_llm: bool = False,
    eqa_vl_family: str | None = None,
    eqa_hf_model_id: str | None = None,
    device: str | None = None,
    output_dir: Path | str | None = None,
    resume: bool = False,
    parameters: Parameters | None = None,
    eqa_client: EqaClient | None = None,
) -> dict[str, Any]:
    """Run RoboVista MCQ-VQA and write JSONL + summary.json under ``output_dir``."""
    out_dir = Path(output_dir) if output_dir else default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    if questions is None:
        questions = load_robovista(
            domains=domains,
            ability_types=ability_types,
            max_questions=max_questions,
        )
    else:
        questions = list(questions)
        if domains:
            wanted = {d.strip().lower() for d in domains}
            questions = [q for q in questions if q.domain.strip().lower() in wanted]
        if ability_types:
            wanted_a = {a.strip().lower() for a in ability_types}
            questions = [q for q in questions if q.ability_type.strip().lower() in wanted_a]
        if max_questions is not None:
            questions = questions[: int(max_questions)]

    done_ids = _read_completed_ids(jsonl_path) if resume else set()
    if not resume and jsonl_path.is_file():
        jsonl_path.unlink()

    params = parameters if parameters is not None else get_parameters("dynav_config.yaml")
    _configure_eqa_parameters(
        params,
        eqa_vl_family=eqa_vl_family,
        eqa_hf_model_id=eqa_hf_model_id,
        device=device,
    )
    shared_client = eqa_client
    if shared_client is None and not mock_llm:
        shared_client = _make_eqa_client(parameters=params, mock_llm=False, device=device)

    rows: list[dict[str, Any]] = []
    if resume and jsonl_path.is_file():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    with jsonl_path.open("a" if resume else "w", encoding="utf-8") as fh:
        for q in questions:
            if q.id in done_ids:
                continue
            prompt = build_robovista_prompt(q)
            client = shared_client if shared_client is not None else _make_mock_client(q.gold_letter)
            command: list[Any] = [prompt, *q.images]
            raw = client(command)
            letter, ok = _grade_prediction(raw, q)
            row = {
                "id": q.id,
                "domain": q.domain,
                "task": q.task,
                "ability_type": q.ability_type,
                "ability_subcategory": q.ability_subcategory,
                "gold_letter": q.gold_letter,
                "predicted_letter": letter,
                "correct": ok,
                "raw_response": raw,
                "n_images": len(q.images),
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)
            logger.info(
                f"robovista id={q.id} domain={q.domain} "
                f"pred={letter or '?'} gold={q.gold_letter} ok={ok}"
            )

    summary = summarize_robovista_rows(rows)
    summary["output_dir"] = str(out_dir)
    summary["n_questions_scheduled"] = len(questions)
    summary["mock_llm"] = bool(mock_llm)
    if eqa_vl_family:
        summary["eqa_vl_family"] = eqa_vl_family
    if eqa_hf_model_id:
        summary["eqa_hf_model_id"] = eqa_hf_model_id
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.alert(
        f"RoboVista done: {summary['correct']}/{summary['n']} "
        f"acc={summary['accuracy']:.3f} -> {summary_path}"
    )
    return summary

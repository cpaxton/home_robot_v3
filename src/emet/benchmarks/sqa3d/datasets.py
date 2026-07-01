# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""SQA3D dataset loaders (questions + annotations joined by ``question_id``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from emet.benchmarks.sqa3d.config import (
    annotations_json_path,
    localization_json_path,
    questions_json_path,
)
from emet.benchmarks.sqa3d.question_types import question_type_index


@dataclass(frozen=True)
class SQA3DQuestion:
    """One situated QA sample from the SQA3D benchmark."""

    question_id: int
    scene_id: str
    situation: str
    question: str
    answers: tuple[str, ...]
    position: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    answer_type: str = ""
    question_type: str = ""
    question_type_index: int = 5
    alternative_situations: tuple[str, ...] = ()

    @property
    def primary_answer(self) -> str:
        return self.answers[0] if self.answers else ""

    def formatted_prompt(self) -> str:
        """Situation + question text for GraphEQA / Dynagraph EQA."""
        from emet.benchmarks.sqa3d.prompts import format_sqa3d_prompt

        return format_sqa3d_prompt(self.situation, self.question)


def _vec3(raw: dict[str, Any]) -> tuple[float, float, float]:
    return (float(raw["x"]), float(raw["y"]), float(raw.get("z", 0.0)))


def _quat_xyzw(raw: dict[str, Any]) -> tuple[float, float, float, float]:
    # SQA3D stores keys _x, _y, _z, _w
    return (
        float(raw.get("_x", raw.get("x", 0.0))),
        float(raw.get("_y", raw.get("y", 0.0))),
        float(raw.get("_z", raw.get("z", 0.0))),
        float(raw.get("_w", raw.get("w", 1.0))),
    )


def _answers_from_annotation(row: dict[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    for item in row.get("answers", []) or []:
        if isinstance(item, dict):
            ans = str(item.get("answer", "")).strip()
            if ans:
                out.append(ans)
        elif isinstance(item, str) and item.strip():
            out.append(item.strip())
    return tuple(out)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"SQA3D file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return raw


def load_sqa3d_questions(
    split: str = "val",
    *,
    data_dir: Path | None = None,
    questions_path: Path | None = None,
    annotations_path: Path | None = None,
) -> list[SQA3DQuestion]:
    """Load SQA3D QA split, joining questions and annotations on ``question_id``."""
    q_path = questions_path or questions_json_path(split, data_dir)
    a_path = annotations_path or annotations_json_path(split, data_dir)
    qdoc = _load_json(q_path)
    adoc = _load_json(a_path)
    questions = qdoc.get("questions", [])
    annotations = adoc.get("annotations", [])
    if not isinstance(questions, list) or not isinstance(annotations, list):
        raise ValueError("SQA3D JSON must contain list fields 'questions' and 'annotations'")

    by_id: dict[int, dict[str, Any]] = {}
    for row in questions:
        if not isinstance(row, dict):
            continue
        qid = int(row["question_id"])
        by_id[qid] = row

    out: list[SQA3DQuestion] = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        qid = int(ann["question_id"])
        qrow = by_id.get(qid)
        if qrow is None:
            continue
        question_text = str(qrow.get("question", "")).strip()
        qtype_idx = question_type_index(question_text)
        answers = _answers_from_annotation(ann)
        alt = qrow.get("alternative_situation") or []
        if not isinstance(alt, list):
            alt = []
        out.append(
            SQA3DQuestion(
                question_id=qid,
                scene_id=str(ann.get("scene_id", qrow.get("scene_id", ""))).strip(),
                situation=str(qrow.get("situation", "")).strip(),
                question=question_text,
                answers=answers,
                position=_vec3(ann["position"]),
                rotation_xyzw=_quat_xyzw(ann["rotation"]),
                answer_type=str(ann.get("answer_type", "")).strip(),
                question_type=str(ann.get("question_type", "")).strip(),
                question_type_index=qtype_idx,
                alternative_situations=tuple(str(x).strip() for x in alt if str(x).strip()),
            )
        )
    out.sort(key=lambda q: q.question_id)
    return out


def load_sqa3d_localization(
    split: str = "val",
    *,
    data_dir: Path | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load SQA3D localization split (situation → pose)."""
    loc_path = path or localization_json_path(split, data_dir)
    doc = _load_json(loc_path)
    rows = doc.get("annotations", [])
    if not isinstance(rows, list):
        raise ValueError(f"localization JSON must contain 'annotations' list: {loc_path}")
    return [r for r in rows if isinstance(r, dict)]


def get_sqa3d_question(
    questions: list[SQA3DQuestion],
    *,
    question_id: int | None = None,
    index: int | None = None,
) -> SQA3DQuestion:
    if question_id is not None:
        for q in questions:
            if q.question_id == question_id:
                return q
        raise KeyError(f"question_id {question_id} not in split (n={len(questions)})")
    if index is not None:
        if index < 0 or index >= len(questions):
            raise IndexError(f"index {index} out of range (n={len(questions)})")
        return questions[index]
    raise ValueError("Provide question_id or index")

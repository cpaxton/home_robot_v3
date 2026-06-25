# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""SQA3D evaluation metrics (EM@1 QA and localization thresholds)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from emet.benchmarks.sqa3d.question_types import QUESTION_TYPE_NAMES


def clean_answer(data: str) -> str:
    """Normalize answers for SQA3D exact-match (LEO / LLaVA-3D protocol)."""
    data = data.lower()
    data = re.sub(r"[ ]+$", "", data)
    data = re.sub(r"^[ ]+", "", data)
    data = re.sub(r" {2,}", " ", data)
    data = re.sub(r"\.[ ]{2,}", ". ", data)
    data = re.sub(r"[^a-zA-Z0-9,'\s\-:]+", "", data)
    data = re.sub(r"ç", "c", data)
    data = re.sub(r"’", "'", data)
    data = re.sub(r"\bletf\b", "left", data)
    data = re.sub(r"\blet\b", "left", data)
    data = re.sub(r"\btehre\b", "there", data)
    data = re.sub(r"\brigth\b", "right", data)
    data = re.sub(r"\brght\b", "right", data)
    data = re.sub(r"\bbehine\b", "behind", data)
    data = re.sub(r"\btv\b", "TV", data)
    data = re.sub(r"\bchai\b", "chair", data)
    data = re.sub(r"\bwasing\b", "washing", data)
    data = re.sub(r"\bwaslked\b", "walked", data)
    data = re.sub(r"\boclock\b", "o'clock", data)
    data = re.sub(r"\bo'[ ]+clock\b", "o'clock", data)

    for digit, word in (
        ("0", "zero"),
        ("1", "one"),
        ("2", "two"),
        ("3", "three"),
        ("4", "four"),
        ("5", "five"),
        ("6", "six"),
        ("7", "seven"),
        ("8", "eight"),
        ("9", "nine"),
        ("10", "ten"),
        ("11", "eleven"),
        ("12", "twelve"),
        ("13", "thirteen"),
        ("14", "fourteen"),
        ("15", "fifteen"),
        ("16", "sixteen"),
        ("17", "seventeen"),
        ("18", "eighteen"),
        ("19", "nineteen"),
        ("20", "twenty"),
        ("23", "twenty-three"),
    ):
        data = re.sub(rf"\b{digit}\b", word, data)
    data = re.sub(r"\bnone\b", "zero", data)
    data = re.sub(r"\b([a-zA-Z]+)([0-9])\b", r"\g<1>", data)
    data = re.sub(r"\ba\b ([a-zA-Z]+)", r"\g<1>", data)
    data = re.sub(r"\ban\b ([a-zA-Z]+)", r"\g<1>", data)
    data = re.sub(r"\bthe\b ([a-zA-Z]+)", r"\g<1>", data)
    data = re.sub(r"\bbackwards\b", "backward", data)
    return data


def answer_match(pred: str, gts: list[str]) -> tuple[bool, bool]:
    """Return (exact_match, refined_match) for one prediction vs GT answer list."""
    if pred in gts:
        return True, True
    for gt in gts:
        pred_compact = "".join(pred.split())
        gt_compact = "".join(gt.split())
        if pred_compact in gt_compact or gt_compact in pred_compact:
            return False, True
    return False, False


def extract_answer_from_eqa_row(row: dict[str, Any]) -> str:
    """Pull answer text from dynagraph ``eqa_results.json`` row or prediction JSONL."""
    for key in ("answer", "answer_text", "text", "predicted_answer", "prediction"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    discord = row.get("discord_text") or row.get("reply") or row.get("raw_eqa_output") or ""
    if isinstance(discord, str):
        m = re.search(r"(?i)answer:\s*(.+?)(?:\n|$)", discord)
        if m:
            return m.group(1).strip()
        if discord.strip():
            return discord.strip()
    return ""


def is_episode_metrics_jsonl(path: Path) -> bool:
    """True when JSONL rows look like ``run-batch`` / ``run-episode`` output."""
    if path.suffix.lower() != ".jsonl" or not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return False
        return isinstance(row, dict) and "em" in row and "gold_answers" in row
    return False


def load_episode_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Episode JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def score_sqa3d_episode_jsonl(path: Path) -> dict[str, Any]:
    """Score embodied ``run-batch`` JSONL (uses embedded ``em`` flags + optional re-score)."""
    episodes = load_episode_metrics_jsonl(path)
    if not episodes:
        return {"n_questions": 0.0, "em@1": 0.0, "em@1_refined": 0.0, "questions": []}

    scored: list[dict[str, Any]] = []
    em_total = em_refined_total = 0
    for row in episodes:
        qid = int(row.get("question_id", -1))
        pred = str(row.get("predicted_answer", ""))
        if pred.startswith("ERROR:"):
            pred = ""
        golds = [str(a) for a in row.get("gold_answers", []) if str(a).strip()]
        pred_clean = clean_answer(pred) if pred else ""
        gts = [clean_answer(a) for a in golds]
        em, em_refined = answer_match(pred_clean, gts) if gts and pred_clean else (False, False)
        if not pred_clean and "em" in row:
            em = bool(row.get("em"))
            em_refined = bool(row.get("em_refined", em))
        em_total += int(em)
        em_refined_total += int(em_refined)
        scored.append(
            {
                "question_id": qid,
                "scene_id": row.get("scene_id", ""),
                "question": row.get("question", ""),
                "gold_answers": golds,
                "prediction": pred,
                "em": em,
                "em_refined": em_refined,
                "planning_steps": row.get("planning_steps"),
            }
        )

    n = max(1, len(scored))
    return {
        "n_questions": float(len(scored)),
        "n_scored": float(len(scored)),
        "n_missing": 0.0,
        "em@1": em_total / n,
        "em@1_refined": em_refined_total / n,
        "source": "episode_jsonl",
        "questions": scored,
    }


def load_predictions(path: Path) -> dict[int, str]:
    """Load predictions keyed by ``question_id`` from JSONL or ``eqa_results.json``."""
    if not path.is_file():
        raise FileNotFoundError(f"Predictions not found: {path}")

    by_id: dict[int, str] = {}
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            qid = row.get("question_id")
            if qid is None:
                continue
            ans = extract_answer_from_eqa_row(row)
            if not ans and row.get("predicted_answer"):
                ans = str(row["predicted_answer"]).strip()
                if ans.startswith("ERROR:"):
                    ans = ""
            by_id[int(qid)] = ans
        return by_id

    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]]
    if isinstance(raw, list):
        rows = [r for r in raw if isinstance(r, dict)]
    elif isinstance(raw, dict) and isinstance(raw.get("questions"), list):
        rows = [r for r in raw["questions"] if isinstance(r, dict)]
    else:
        raise ValueError(f"Unsupported predictions JSON format: {path}")

    for row in rows:
        qid = row.get("question_id")
        if qid is None:
            continue
        by_id[int(qid)] = extract_answer_from_eqa_row(row)
    return by_id


def score_sqa3d_predictions(
    questions: list[Any],
    predictions: dict[int, str],
    *,
    require_all: bool = False,
) -> dict[str, Any]:
    """Score SQA3D QA predictions with EM@1 and per-question-type breakdown."""
    type_counts = dict.fromkeys(range(6), 0)
    type_em = dict.fromkeys(range(6), 0)
    type_em_refined = dict.fromkeys(range(6), 0)
    rows: list[dict[str, Any]] = []
    em_total = 0
    em_refined_total = 0
    n_scored = 0

    for q in questions:
        pred_raw = predictions.get(q.question_id)
        if pred_raw is None:
            if require_all:
                raise KeyError(f"Missing prediction for question_id={q.question_id}")
            rows.append(
                {
                    "question_id": q.question_id,
                    "scene_id": q.scene_id,
                    "question": q.question,
                    "gold_answers": list(q.answers),
                    "prediction": "",
                    "em": False,
                    "em_refined": False,
                    "missing": True,
                    "question_type": QUESTION_TYPE_NAMES[q.question_type_index],
                }
            )
            continue

        pred = clean_answer(pred_raw)
        gts = [clean_answer(a) for a in q.answers if a.strip()]
        em, em_refined = answer_match(pred, gts) if gts else (False, False)
        em_total += int(em)
        em_refined_total += int(em_refined)
        n_scored += 1
        tidx = q.question_type_index
        type_counts[tidx] += 1
        type_em[tidx] += int(em)
        type_em_refined[tidx] += int(em_refined)
        rows.append(
            {
                "question_id": q.question_id,
                "scene_id": q.scene_id,
                "question": q.question,
                "situation": q.situation,
                "gold_answers": list(q.answers),
                "prediction": pred_raw,
                "prediction_clean": pred,
                "em": em,
                "em_refined": em_refined,
                "question_type": QUESTION_TYPE_NAMES[tidx],
            }
        )

    n = max(1, n_scored)
    by_type: dict[str, dict[str, float]] = {}
    for i, name in enumerate(QUESTION_TYPE_NAMES):
        cnt = type_counts[i]
        if cnt == 0:
            continue
        by_type[name] = {
            "em@1": type_em[i] / cnt,
            "em@1_refined": type_em_refined[i] / cnt,
            "n": float(cnt),
        }

    return {
        "n_questions": float(len(questions)),
        "n_scored": float(n_scored),
        "n_missing": float(len(questions) - n_scored),
        "em@1": em_total / n,
        "em@1_refined": em_refined_total / n,
        "by_question_type": by_type,
        "questions": rows,
    }


def _pos_distance_xy_m(pos1: np.ndarray, pos2: np.ndarray) -> float:
    return float(math.sqrt(sum((pos1[:2] - pos2[:2]) ** 2)))


def _rot_distance_z_deg(rot1: np.ndarray, rot2: np.ndarray) -> float:
    r1 = R.from_quat(rot1).as_rotvec()[-1]
    r2 = R.from_quat(rot2).as_rotvec()[-1]
    diff = min(abs(r1 - r2), 2 * math.pi - abs(r1 - r2))
    return diff / math.pi * 180.0


def summarize_localization(
    gt_positions: list[tuple[float, float, float]],
    gt_rotations_xyzw: list[tuple[float, float, float, float]],
    pred_positions: list[list[tuple[float, float, float]]],
    pred_rotations_xyzw: list[list[tuple[float, float, float, float]]],
) -> dict[str, float]:
    """SQA3D localization metrics (best-of-K position + rotation per sample)."""
    if not gt_positions:
        return {
            "acc@0.5m": 0.0,
            "acc@1.0m": 0.0,
            "acc@15deg": 0.0,
            "acc@30deg": 0.0,
            "n": 0.0,
        }
    if not (len(gt_positions) == len(gt_rotations_xyzw) == len(pred_positions) == len(pred_rotations_xyzw)):
        raise ValueError("GT and prediction lists must have equal length")

    cnt_pos_0_5 = cnt_pos_1 = cnt_rot_15 = cnt_rot_30 = 0
    for gt_p, gt_r, pred_ps, pred_rs in zip(
        gt_positions, gt_rotations_xyzw, pred_positions, pred_rotations_xyzw, strict=True
    ):
        gt_p_arr = np.asarray(gt_p, dtype=np.float64)
        gt_r_arr = np.asarray(gt_r, dtype=np.float64)
        posdiff = min(_pos_distance_xy_m(gt_p_arr, np.asarray(p, dtype=np.float64)) for p in pred_ps)
        rotdiff = min(_rot_distance_z_deg(gt_r_arr, np.asarray(r, dtype=np.float64)) for r in pred_rs)
        if posdiff < 0.5:
            cnt_pos_0_5 += 1
        if posdiff < 1.0:
            cnt_pos_1 += 1
        if rotdiff < 15.0:
            cnt_rot_15 += 1
        if rotdiff < 30.0:
            cnt_rot_30 += 1

    total = len(gt_positions)
    return {
        "acc@0.5m": cnt_pos_0_5 / total,
        "acc@1.0m": cnt_pos_1 / total,
        "acc@15deg": cnt_rot_15 / total,
        "acc@30deg": cnt_rot_30 / total,
        "n": float(total),
    }

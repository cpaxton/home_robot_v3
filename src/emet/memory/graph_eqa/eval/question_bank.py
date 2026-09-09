# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Dynagraph benchmark question bank loader and EQA scorer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from emet.utils.config import resolve_config_yaml_path


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def load_question_bank(path: str | Path, *, env_filter: str | None = None) -> list[dict[str, Any]]:
    full = Path(resolve_config_yaml_path(str(path)))
    raw = yaml.safe_load(full.read_text(encoding="utf-8")) or {}
    out: list[dict[str, Any]] = []
    for block in raw.get("environments", []):
        env = str(block.get("env", ""))
        if env_filter and env != env_filter:
            continue
        for q in block.get("questions", []):
            if not isinstance(q, dict):
                continue
            out.append({**q, "env": env})
    return out


def _answer_text(row: dict[str, Any]) -> str:
    for key in ("answer", "answer_text", "raw_answer"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v
    discord = row.get("discord_text") or row.get("reply") or ""
    if isinstance(discord, str):
        m = re.search(r"(?i)answer:\s*(.+?)(?:\n|$)", discord)
        if m:
            return m.group(1).strip()
        return discord
    return ""


def _tokens_match(answer: str, expected_tokens: list[str]) -> bool:
    a = _norm(answer)
    return all(_norm(tok) in a for tok in expected_tokens)


def _xyz_error_m(answer_xyz: list[float], gt_xyz: list[float]) -> float:
    a = np.asarray(answer_xyz, dtype=np.float64).reshape(3)
    g = np.asarray(gt_xyz, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(a[:2] - g[:2]))


def score_single_question(
    row: dict[str, Any],
    *,
    episode_dir: Path | None = None,
    placements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one EQA result row against expected_tokens / optional GT body."""
    question = str(row.get("question", ""))
    answer = _answer_text(row)
    expected = [str(t) for t in row.get("expected_tokens", [])]
    token_pass = _tokens_match(answer, expected) if expected else None

    spatial_pass: bool | None = None
    xyz_err_m: float | None = None
    gt_body = row.get("gt_body_key") or row.get("gt_body")
    max_xy = float(row.get("max_xy_error_m", 0.8))

    if gt_body and episode_dir is not None:
        from emet.memory.format import SIM_GT_PLACEMENTS_FILENAME
        from emet.memory.graph_eqa.eval.sim_ground_truth_graph import read_sim_object_placements

        if placements is None:
            p = episode_dir / SIM_GT_PLACEMENTS_FILENAME
            if p.is_file():
                placements = read_sim_object_placements(
                    {"sim_object_placements": json.loads(p.read_text(encoding="utf-8"))}
                )
        if placements and str(gt_body) in placements:
            gpos = np.asarray(placements[str(gt_body)].get("pos", [0, 0, 0]), dtype=np.float64).reshape(3)
            cited = row.get("cited_xyz") or row.get("target_xyz")
            if cited is not None:
                xyz_err_m = _xyz_error_m(list(cited), gpos.tolist())
                spatial_pass = xyz_err_m <= max_xy
            elif token_pass is not None:
                spatial_pass = token_pass

    passed = token_pass if spatial_pass is None else (bool(token_pass) and bool(spatial_pass))
    if token_pass is None and spatial_pass is not None:
        passed = spatial_pass
    if token_pass is None and spatial_pass is None:
        passed = bool(answer.strip())

    return {
        "question": question,
        "answer": answer,
        "expected_tokens": expected,
        "token_pass": token_pass,
        "spatial_pass": spatial_pass,
        "xyz_err_m": xyz_err_m,
        "pass": bool(passed),
        "confidence": row.get("confidence"),
    }


def score_eqa_results(
    rows: list[dict[str, Any]],
    *,
    episode_dir: str | Path | None = None,
) -> dict[str, Any]:
    ep = Path(episode_dir) if episode_dir else None
    scored = [score_single_question(r, episode_dir=ep) for r in rows]
    n = max(1, len(scored))
    acc = sum(1 for s in scored if s["pass"]) / n
    confs = [float(s["confidence"]) for s in scored if s.get("confidence") is not None]
    return {
        "accuracy": float(acc),
        "n_questions": float(len(scored)),
        "mean_confidence": float(np.mean(confs)) if confs else None,
        "questions": scored,
    }


def write_eqa_results(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"questions": rows}
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest

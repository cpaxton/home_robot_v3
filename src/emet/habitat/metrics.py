# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Episode metrics for Habitat EQA evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EpisodeMetrics:
    dataset: str
    method: str
    question_id: int
    scene: str
    floor: int
    question: str
    gold_answer_letter: str
    predicted_answer: str
    correct: bool
    confident: bool
    planning_steps: int
    success: bool
    parsed_answer_letter: str = ""
    model_confident: bool = False
    raw_eqa_output: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def extract_mcq_letter(predicted: str, choices: list[str] | None = None) -> str:
    """Extract A–D letter from model output; optionally match choice text."""
    text = (predicted or "").strip()
    if not text:
        return ""
    compact = text.replace(" ", "").upper()
    if len(compact) == 1 and compact in "ABCD":
        return compact
    m = re.search(r"(?:^answer\s*:\s*|^|\b)([A-D])\b", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).upper()
    if choices:
        lowered = text.lower()
        for idx, choice in enumerate(choices[:4]):
            choice_l = choice.strip().lower()
            if choice_l and choice_l in lowered:
                return chr(ord("A") + idx)
    return ""


def grade_mcq_answer(
    predicted: str,
    gold_letter: str,
    *,
    choices: list[str] | None = None,
) -> bool:
    """Return True if ``predicted`` matches MCQ letter ``gold_letter`` (A–D)."""
    gold = gold_letter.strip().upper()
    if not gold:
        return False
    letter = extract_mcq_letter(predicted, choices)
    if letter:
        return letter == gold
    text = predicted.strip()
    if not text:
        return False
    if len(text) == 1 and text.upper() == gold:
        return True
    return text.upper().startswith(gold)


def write_episode_jsonl(path: Path, episodes: list[EpisodeMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep.to_dict()) + "\n")


def summarize_episodes(episodes: list[EpisodeMetrics]) -> dict[str, float]:
    if not episodes:
        return {"accuracy": 0.0, "mean_steps": 0.0, "success_rate": 0.0, "n": 0.0}
    n = len(episodes)
    return {
        "accuracy": sum(1 for e in episodes if e.correct) / n,
        "mean_steps": sum(e.planning_steps for e in episodes) / n,
        "success_rate": sum(1 for e in episodes if e.success) / n,
        "n": float(n),
    }


def compare_method_results(
    graph_eqa: list[EpisodeMetrics],
    dynagraph: list[EpisodeMetrics],
) -> dict:
    """Side-by-side summary for GraphEQA vs Dynagraph on the same question ids."""
    by_q_graph = {e.question_id: e for e in graph_eqa}
    by_q_dyna = {e.question_id: e for e in dynagraph}
    qids = sorted(set(by_q_graph) | set(by_q_dyna))
    rows: list[dict] = []
    for qid in qids:
        g = by_q_graph.get(qid)
        d = by_q_dyna.get(qid)
        rows.append(
            {
                "question_id": qid,
                "gold": (g or d).gold_answer_letter if (g or d) else "",
                "graph_eqa_pred": g.parsed_answer_letter or g.predicted_answer[:1] if g else "",
                "graph_eqa_correct": g.correct if g else False,
                "dynagraph_pred": d.parsed_answer_letter or d.predicted_answer[:1] if d else "",
                "dynagraph_correct": d.correct if d else False,
                "graph_eqa_steps": g.planning_steps if g else 0,
                "dynagraph_steps": d.planning_steps if d else 0,
            }
        )
    return {
        "graph_eqa": summarize_episodes(graph_eqa),
        "dynagraph": summarize_episodes(dynagraph),
        "both_correct": sum(1 for r in rows if r["graph_eqa_correct"] and r["dynagraph_correct"]),
        "graph_only": sum(1 for r in rows if r["graph_eqa_correct"] and not r["dynagraph_correct"]),
        "dynagraph_only": sum(1 for r in rows if r["dynagraph_correct"] and not r["graph_eqa_correct"]),
        "neither": sum(1 for r in rows if not r["graph_eqa_correct"] and not r["dynagraph_correct"]),
        "per_question": rows,
    }

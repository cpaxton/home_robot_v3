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

    def to_dict(self) -> dict:
        return asdict(self)


def grade_mcq_answer(predicted: str, gold_letter: str) -> bool:
    """Return True if ``predicted`` matches MCQ letter ``gold_letter`` (A–D)."""
    gold = gold_letter.strip().upper()
    if not gold:
        return False
    text = predicted.strip()
    if not text:
        return False
    # Exact single letter
    if len(text) == 1 and text.upper() == gold:
        return True
    # "Answer: B" / "B)" patterns
    m = re.search(r"\b([A-D])\b", text.upper())
    if m and m.group(1) == gold:
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

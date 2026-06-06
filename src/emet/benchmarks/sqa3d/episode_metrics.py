# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Episode result rows for embodied SQA3D evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SQA3DEpisodeMetrics:
    dataset: str
    method: str
    question_id: int
    scene_id: str
    question: str
    situation: str
    gold_answers: list[str]
    predicted_answer: str
    em: bool
    em_refined: bool
    confident: bool
    planning_steps: int
    success: bool
    raw_eqa_output: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def append_sqa3d_jsonl(path: Path, episode: SQA3DEpisodeMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(episode.to_dict()) + "\n")


def read_completed_sqa3d_question_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    done: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        qid = row.get("question_id")
        if isinstance(qid, int):
            done.add(qid)
    return done


def summarize_sqa3d_episodes(episodes: list[SQA3DEpisodeMetrics]) -> dict[str, float]:
    if not episodes:
        return {"em@1": 0.0, "mean_steps": 0.0, "success_rate": 0.0, "n": 0.0}
    n = len(episodes)
    return {
        "em@1": sum(1 for e in episodes if e.em) / n,
        "em@1_refined": sum(1 for e in episodes if e.em_refined) / n,
        "mean_steps": sum(e.planning_steps for e in episodes) / n,
        "success_rate": sum(1 for e in episodes if e.success) / n,
        "n": float(n),
    }

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Episode metrics for Habitat EQA evaluation.

These helpers grade multiple-choice answers and serialize per-episode JSONL rows
compatible with GraphEQA-style Habitat benchmarks.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EpisodeMetrics:
    """One graded HM-EQA / Habitat EQA episode.

    Attributes:
        dataset: Dataset name (e.g. ``hmeqa``).
        method: Method label (e.g. ``graph_eqa``, ``dynagraph``).
        question_id: Zero-based question index in the CSV.
        scene: HM3D scene id.
        floor: Floor index for the episode.
        question: Question text (for logging).
        gold_answer_letter: Gold MCQ letter ``A``–``D``.
        predicted_answer: Raw model output string.
        correct: Whether the prediction matches gold (see :func:`grade_mcq_answer`).
        confident: Harness-level confidence flag (planner reached an answer).
        planning_steps: Number of navigation / EQA steps taken.
        success: Episode completed without fatal error.
        parsed_answer_letter: Letter extracted from ``predicted_answer``.
        model_confident: Optional model-reported confidence.
        raw_eqa_output: Full EQA trace / reasoning text when available.
    """

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
        """Serialize to a JSON-friendly dict (for JSONL export)."""
        return asdict(self)


def extract_mcq_letter(predicted: str, choices: list[str] | None = None) -> str:
    """Extract A–D letter from free-form model output.

    Args:
        predicted: Raw EQA / VLM answer text.
        choices: Optional choice strings; used to match embedded choice text.

    Returns:
        Uppercase letter ``A``–``D``, or ``""`` if none found.
    """
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
    """Return True if ``predicted`` matches MCQ letter ``gold_letter`` (A–D).

    Args:
        predicted: Model output (letter or prose containing a letter / choice text).
        gold_letter: Gold answer letter from the dataset.
        choices: Optional choice list for substring matching fallback.
    """
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


def write_episode_jsonl(path: Path, episodes: list[EpisodeMetrics], *, append: bool = False) -> None:
    """Write episode metrics as JSONL (one :class:`EpisodeMetrics` per line).

    Args:
        path: Output file path (parent directories are created).
        episodes: Rows to write.
        append: When True, append to an existing file instead of truncating.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep.to_dict()) + "\n")


def append_episode_jsonl(path: Path, episode: EpisodeMetrics) -> None:
    """Append one episode row to an existing JSONL file."""
    write_episode_jsonl(path, [episode], append=True)


def read_completed_question_ids(path: Path) -> set[int]:
    """Question ids already present in a JSONL results file (for ``--resume``).

    Args:
        path: Existing results JSONL from :func:`write_episode_jsonl`.

    Returns:
        Set of ``question_id`` integers found in the file (ignores malformed lines).
    """
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


def summarize_episodes(episodes: list[EpisodeMetrics]) -> dict[str, float]:
    """Aggregate accuracy, mean planning steps, and success rate.

    Args:
        episodes: Graded episodes (may be empty).

    Returns:
        Dict with keys ``accuracy``, ``mean_steps``, ``success_rate``, ``n``.
    """
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
    """Side-by-side summary for GraphEQA vs Dynagraph on the same question ids.

    Args:
        graph_eqa: Episodes from a GraphEQA Habitat run.
        dynagraph: Episodes from a Dynagraph Habitat run.

    Returns:
        Dict with per-method summaries, disagreement counts, and ``per_question`` rows.
    """
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

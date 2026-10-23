# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Official EQA metadata adapters; not active runners or scoring substitutes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationQuestion:
    benchmark: str
    question_id: str
    variant: str
    question: str
    category: str
    scene: str | None
    reference_answer: str
    frame_indices: tuple[int, ...] = ()
    evidence_id: str | None = None

    def model_input(self):
        """Only the question enters model text; refs/categories/labels are scorer-only."""
        return {"question": self.question}


def _text(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing/non-string {key}")
    return value


def _nonempty_rows(rows):
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Expected nonempty list of question records")
    return rows


def express_questions(rows):
    """Preserve official episode IDs; never expose reference actions/goal poses."""
    questions, seen = [], set()
    for row in _nonempty_rows(rows):
        qid = row.get("episode_id")
        if isinstance(qid, bool) or not isinstance(qid, int) or qid < 0 or qid in seen:
            raise ValueError("Invalid/duplicate EXPRESS episode_id")
        seen.add(qid)
        questions.append(
            EvaluationQuestion(
                "express",
                str(qid),
                "original",
                _text(row, "question"),
                _text(row, "type"),
                _text(row, "scene_id"),
                _text(row, "answer"),
            )
        )
    return questions


def abstain_questions(answerable_rows, unanswerable_rows):
    """Keep both variants under their original ID; reject mismatched evidence pairs."""
    by_variant = {}
    for variant, rows in (("answerable", answerable_rows), ("unanswerable", unanswerable_rows)):
        questions = {}
        for row in _nonempty_rows(rows):
            qid = _text(row, "question_id")
            if qid in questions:
                raise ValueError(f"Duplicate AbstainEQA ID: {qid}")
            indices = row.get("frame_list")
            if (
                not isinstance(indices, list)
                or not indices
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices)
            ):
                raise ValueError(f"Invalid frame_list: {qid}")
            questions[qid] = EvaluationQuestion(
                "abstaineqa",
                qid,
                variant,
                _text(row, "question"),
                row["category"] if isinstance(row.get("category"), str) and row["category"].strip() else "unlabeled",
                None,  # episode_history identifies evidence, not a certified scene split.
                _text(row, "answer"),
                tuple(indices),
                _text(row, "episode_history"),
            )
        by_variant[variant] = questions
    positive, negative = by_variant["answerable"], by_variant["unanswerable"]
    if positive.keys() != negative.keys():
        raise ValueError("AbstainEQA answerable/unanswerable ID sets differ")
    output = []
    for qid in sorted(positive):
        a, b = positive[qid], negative[qid]
        if a.evidence_id != b.evidence_id or a.frame_indices != b.frame_indices:
            raise ValueError(f"AbstainEQA pair evidence differs: {qid}")
        output.extend((a, b))
    return output


def development_pilot(questions, per_category=2):
    """Deterministic internal-dev selection, paired when abstention variants exist."""
    if isinstance(per_category, bool) or not isinstance(per_category, int) or per_category < 1:
        raise ValueError("Positive integer per_category required")
    if not questions:
        raise ValueError("No questions to select")
    if len({q.benchmark for q in questions}) != 1:
        raise ValueError("Pilot selection requires a homogeneous benchmark")
    paired = questions[0].benchmark == "abstaineqa"
    candidates = [q for q in questions if q.category != "unlabeled" and (not paired or q.variant == "unanswerable")]
    if not candidates:
        raise ValueError("No categorized questions for pilot")
    selected, counts = set(), {}
    for question in sorted(candidates, key=lambda q: (q.category, q.question_id)):
        if counts.get(question.category, 0) < per_category:
            selected.add(question.question_id)
            counts[question.category] = counts.get(question.category, 0) + 1
    if any(count < per_category for count in counts.values()):
        raise ValueError("Insufficient examples for category-balanced pilot")
    return [q for q in questions if q.question_id in selected]

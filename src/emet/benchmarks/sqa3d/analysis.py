# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Outcome breakdown and figure generation for SQA3D episode JSONL results."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from emet.benchmarks.sqa3d.metrics import (
    answer_match,
    clean_answer,
    load_episode_metrics_jsonl,
    score_sqa3d_episode_jsonl,
)
from emet.benchmarks.sqa3d.question_types import QUESTION_TYPE_NAMES, question_type_index

OutcomeKind = Literal["tp", "fp", "fn", "infra"]

_INFRA_PATTERNS = (
    re.compile(r"(?i)^error:"),
    re.compile(r"(?i)cuda out of memory"),
    re.compile(r"(?i)out of memory"),
)

_ABSTAIN_TOKENS = frozenset({"", "unknown", "none", "n/a", "na"})


@dataclass(frozen=True)
class ClassifiedEpisode:
    question_id: int
    scene_id: str
    question: str
    situation: str
    gold_answers: list[str]
    predicted_answer: str
    prediction_clean: str
    gold_clean: str
    outcome: OutcomeKind
    em: bool
    em_refined: bool
    confident: bool
    question_type: str
    method: str
    planning_steps: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "scene_id": self.scene_id,
            "question": self.question,
            "situation": self.situation,
            "gold_answers": self.gold_answers,
            "predicted_answer": self.predicted_answer,
            "prediction_clean": self.prediction_clean,
            "gold_clean": self.gold_clean,
            "outcome": self.outcome,
            "em": self.em,
            "em_refined": self.em_refined,
            "confident": self.confident,
            "question_type": self.question_type,
            "method": self.method,
            "planning_steps": self.planning_steps,
        }


def _is_infra_failure(prediction: str) -> bool:
    text = prediction.strip()
    if not text:
        return False
    return any(p.search(text) for p in _INFRA_PATTERNS)


def _is_abstain(prediction_clean: str) -> bool:
    return prediction_clean.strip().lower() in _ABSTAIN_TOKENS


def classify_outcome(
    *,
    em: bool,
    prediction: str,
    prediction_clean: str,
    infra_failure: bool = False,
) -> OutcomeKind:
    """Map one episode to TP / FP / FN / infra for paper figures.

    - **tp**: EM@1 correct.
    - **fp**: substantive wrong answer (model committed to an incorrect string).
    - **fn**: abstained (unknown/empty) or could not match gold after normalization.
    - **infra**: runtime failure (OOM, ERROR: …) — excluded from accuracy numerator/denominator
      in ``summarize_outcomes`` unless ``include_infra=True``.
    """
    if em:
        return "tp"
    if infra_failure or _is_infra_failure(prediction):
        return "infra"
    if _is_abstain(prediction_clean):
        return "fn"
    return "fp"


def dedupe_episodes_by_question_id(
    episodes: list[dict[str, Any]],
    *,
    keep: Literal["first", "last"] = "last",
) -> list[dict[str, Any]]:
    """Drop duplicate ``question_id`` rows (e.g. merged batch logs)."""
    if not episodes:
        return []
    if keep == "first":
        seen: set[int] = set()
        out: list[dict[str, Any]] = []
        for row in episodes:
            qid = row.get("question_id")
            if isinstance(qid, int) and qid not in seen:
                seen.add(qid)
                out.append(row)
        return out
    last_idx: dict[int, int] = {}
    for i, row in enumerate(episodes):
        qid = row.get("question_id")
        if isinstance(qid, int):
            last_idx[qid] = i
    return [episodes[i] for i in sorted(last_idx.values())]


def classify_episodes(
    episodes: list[dict[str, Any]],
    *,
    dedupe: bool = True,
) -> list[ClassifiedEpisode]:
    rows = dedupe_episodes_by_question_id(episodes) if dedupe else list(episodes)
    classified: list[ClassifiedEpisode] = []
    for row in rows:
        pred = str(row.get("predicted_answer", "") or "")
        if pred.startswith("ERROR:"):
            pred = ""
        golds = [str(a) for a in row.get("gold_answers", []) if str(a).strip()]
        pred_clean = clean_answer(pred) if pred.strip() else ""
        gts = [clean_answer(a) for a in golds]
        gold_clean = gts[0] if gts else ""
        em, em_refined = answer_match(pred_clean, gts) if gts and pred_clean else (False, False)
        if not pred_clean and "em" in row:
            em = bool(row.get("em"))
            em_refined = bool(row.get("em_refined", em))
        qtext = str(row.get("question", ""))
        raw_eqa = str(row.get("raw_eqa_output", "") or "")
        infra_failure = bool(row.get("infra_failure", False)) or _is_infra_failure(raw_eqa)
        classified.append(
            ClassifiedEpisode(
                question_id=int(row.get("question_id", -1)),
                scene_id=str(row.get("scene_id", "")),
                question=qtext,
                situation=str(row.get("situation", "")),
                gold_answers=golds,
                predicted_answer=pred,
                prediction_clean=pred_clean,
                gold_clean=gold_clean,
                outcome=classify_outcome(
                    em=em,
                    prediction=pred,
                    prediction_clean=pred_clean,
                    infra_failure=infra_failure,
                ),
                em=em,
                em_refined=em_refined,
                confident=bool(row.get("confident", False)),
                question_type=QUESTION_TYPE_NAMES[question_type_index(qtext)],
                method=str(row.get("method", "")),
                planning_steps=row.get("planning_steps"),
            )
        )
    return classified


def summarize_outcomes(
    episodes: list[dict[str, Any]],
    *,
    dedupe: bool = True,
    include_infra: bool = False,
) -> dict[str, Any]:
    """Aggregate TP/FP/FN/infra counts and rates."""
    rows = classify_episodes(episodes, dedupe=dedupe)
    counts = Counter(r.outcome for r in rows)
    scored = [r for r in rows if include_infra or r.outcome != "infra"]
    n_scored = max(1, len(scored))
    tp = sum(1 for r in scored if r.outcome == "tp")
    fp = sum(1 for r in scored if r.outcome == "fp")
    fn = sum(1 for r in scored if r.outcome == "fn")
    infra = counts.get("infra", 0)
    by_type: dict[str, dict[str, float | int]] = {}
    for name in QUESTION_TYPE_NAMES:
        subset = [r for r in scored if r.question_type == name]
        if not subset:
            continue
        n = len(subset)
        by_type[name] = {
            "n": n,
            "tp": sum(1 for r in subset if r.outcome == "tp"),
            "fp": sum(1 for r in subset if r.outcome == "fp"),
            "fn": sum(1 for r in subset if r.outcome == "fn"),
            "em@1": sum(1 for r in subset if r.em) / n,
        }
    confident_wrong = sum(1 for r in scored if r.outcome == "fp" and r.confident)
    return {
        "n_episodes": len(rows),
        "n_scored": len(scored),
        "n_infra": infra,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn_note": "TN not defined for generative situated QA (one gold answer per question).",
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "em@1": tp / n_scored,
        "fp_confident": confident_wrong,
        "counts": dict(counts),
        "by_question_type": by_type,
        "examples": {
            "tp": [r.to_dict() for r in rows if r.outcome == "tp"][:20],
            "fp": [r.to_dict() for r in rows if r.outcome == "fp"][:20],
            "fn": [r.to_dict() for r in rows if r.outcome == "fn"][:20],
            "infra": [r.to_dict() for r in rows if r.outcome == "infra"][:20],
        },
    }


def build_label_confusion(
    episodes: list[dict[str, Any]],
    *,
    top_k: int = 12,
    dedupe: bool = True,
) -> dict[str, Any]:
    """Gold vs predicted label counts for categorical confusion-matrix figures."""
    rows = classify_episodes(episodes, dedupe=dedupe)
    gold_counter = Counter(r.gold_clean for r in rows if r.gold_clean)
    top_labels = [label for label, _ in gold_counter.most_common(top_k)]
    label_set = set(top_labels) | {"other", "abstain", "infra"}
    matrix: dict[str, dict[str, int]] = {g: dict.fromkeys(label_set, 0) for g in top_labels}
    for row in rows:
        gold = row.gold_clean if row.gold_clean in top_labels else "other"
        if row.outcome == "infra":
            pred_bucket = "infra"
        elif row.outcome == "fn" or not row.prediction_clean:
            pred_bucket = "abstain"
        elif row.prediction_clean in top_labels:
            pred_bucket = row.prediction_clean
        else:
            pred_bucket = "other"
        if gold not in matrix:
            matrix[gold] = dict.fromkeys(label_set, 0)
        matrix[gold][pred_bucket] = matrix[gold].get(pred_bucket, 0) + 1
    return {
        "top_k": top_k,
        "labels": top_labels + ["other", "abstain", "infra"],
        "matrix": matrix,
        "pairs": [{"gold": r.gold_clean, "pred": r.prediction_clean, "outcome": r.outcome, "em": r.em} for r in rows],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def generate_sqa3d_figure_bundle(
    predictions: Path,
    output_dir: Path,
    *,
    split: str = "val",
    method: str | None = None,
    top_k: int = 12,
    write_plots: bool = True,
) -> dict[str, Any]:
    """Write JSON summaries, example lists, and optional matplotlib figures."""
    episodes = load_episode_metrics_jsonl(predictions)
    if method:
        episodes = [e for e in episodes if str(e.get("method", "")) == method]
    scored = score_sqa3d_episode_jsonl(predictions)
    outcomes = summarize_outcomes(episodes)
    confusion = build_label_confusion(episodes, top_k=top_k)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "benchmark": "sqa3d",
        "split": split,
        "predictions": str(predictions),
        "method_filter": method,
        "qa_metrics": scored,
        "outcomes": outcomes,
        "confusion": confusion,
        "artifacts": {},
    }

    summary_path = output_dir / "sqa3d_outcomes_summary.json"
    summary_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    bundle["artifacts"]["summary_json"] = str(summary_path)

    for key in ("tp", "fp", "fn", "infra"):
        ex_path = output_dir / f"examples_{key}.jsonl"
        _write_jsonl(ex_path, outcomes["examples"].get(key, []))
        bundle["artifacts"][f"examples_{key}"] = str(ex_path)

    if write_plots:
        plot_paths = _render_matplotlib_figures(
            output_dir,
            outcomes=outcomes,
            confusion=confusion,
            title_suffix=f"{split}" + (f" / {method}" if method else ""),
        )
        bundle["artifacts"].update(plot_paths)

    return bundle


def _render_matplotlib_figures(
    output_dir: Path,
    *,
    outcomes: dict[str, Any],
    confusion: dict[str, Any],
    title_suffix: str,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    paths: dict[str, str] = {}

    # Outcome bar (TP / FP / FN / infra)
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["TP", "FP", "FN", "infra"]
    keys = ["tp", "fp", "fn", "infra"]
    values = [outcomes.get(k, 0) for k in keys]
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#7f7f7f"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Count")
    ax.set_title(f"SQA3D outcomes ({title_suffix})")
    for i, v in enumerate(values):
        ax.text(i, v + 0.05, str(v), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    p = output_dir / "outcomes_bar.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths["outcomes_bar_png"] = str(p)

    # EM@1 by question type
    by_type = outcomes.get("by_question_type") or {}
    if by_type:
        fig, ax = plt.subplots(figsize=(7, 4))
        names = list(by_type.keys())
        ems = [by_type[n]["em@1"] for n in names]
        ax.bar(names, ems, color="#1f77b4")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("EM@1")
        ax.set_title(f"EM@1 by question type ({title_suffix})")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        p = output_dir / "em_by_question_type.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths["em_by_question_type_png"] = str(p)

        fig, ax = plt.subplots(figsize=(8, 4))
        tp_vals = [by_type[n]["tp"] for n in names]
        fp_vals = [by_type[n]["fp"] for n in names]
        fn_vals = [by_type[n]["fn"] for n in names]
        x = np.arange(len(names))
        ax.bar(x, tp_vals, label="TP", color="#2ca02c")
        ax.bar(x, fp_vals, bottom=tp_vals, label="FP", color="#d62728")
        ax.bar(x, fn_vals, bottom=np.array(tp_vals) + np.array(fp_vals), label="FN", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title(f"Outcomes by question type ({title_suffix})")
        fig.tight_layout()
        p = output_dir / "outcomes_by_question_type.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths["outcomes_by_question_type_png"] = str(p)

    # Confusion matrix (gold rows x pred cols)
    matrix = confusion.get("matrix") or {}
    labels = confusion.get("labels") or []
    gold_rows = [
        g for g in labels if g in matrix and g not in ("other", "abstain", "infra") and sum(matrix[g].values()) > 0
    ][: int(confusion.get("top_k", 12))]
    pred_cols = labels
    if gold_rows and pred_cols:
        data = np.zeros((len(gold_rows), len(pred_cols)), dtype=int)
        for i, g in enumerate(gold_rows):
            row = matrix.get(g, {})
            for j, p in enumerate(pred_cols):
                data[i, j] = int(row.get(p, 0))
        fig, ax = plt.subplots(figsize=(max(6, len(pred_cols) * 0.6), max(5, len(gold_rows) * 0.5)))
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(pred_cols)))
        ax.set_yticks(range(len(gold_rows)))
        ax.set_xticklabels(pred_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(gold_rows, fontsize=8)
        ax.set_xlabel("Predicted (normalized)")
        ax.set_ylabel("Gold (normalized)")
        ax.set_title(f"Label confusion ({title_suffix})")
        for i in range(len(gold_rows)):
            for j in range(len(pred_cols)):
                val = data[i, j]
                if val:
                    ax.text(j, i, str(val), ha="center", va="center", color="black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        p = output_dir / "confusion_matrix.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths["confusion_matrix_png"] = str(p)

    return paths

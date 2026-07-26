# Copyright (c) Chris Paxton 2026

"""Metrics for comparing heterogeneous open-vocabulary presence verifiers."""

from __future__ import annotations

from typing import Any


def binary_metrics(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
    threshold: float,
    label_key: str = "gt_in_view",
) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for row in rows:
        label = row.get(label_key)
        score = row.get(score_key)
        if label is None or score is None:
            continue
        pred = float(score) >= float(threshold)
        truth = bool(label)
        tp += int(pred and truth)
        fp += int(pred and not truth)
        tn += int(not pred and not truth)
        fn += int(not pred and truth)
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "n_labeled": float(n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def sweep_thresholds(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
    thresholds: list[float],
    label_key: str = "gt_in_view",
) -> list[dict[str, float]]:
    return [
        binary_metrics(rows, score_key=score_key, threshold=t, label_key=label_key)
        for t in thresholds
    ]


def best_operating_point(
    sweep: list[dict[str, float]],
    *,
    min_recall: float = 0.8,
) -> dict[str, float] | None:
    """Prefer high recall, then precision/F1: proposal gates should lean FP."""
    labeled = [row for row in sweep if row["n_labeled"] > 0]
    if not labeled:
        return None
    high_recall = [row for row in labeled if row["recall"] >= min_recall]
    candidates = high_recall or labeled
    return max(candidates, key=lambda row: (row["precision"], row["f1"], row["recall"]))


def summarize_backend(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
    thresholds: list[float],
    label_key: str = "gt_in_view",
) -> dict[str, Any]:
    sweep = sweep_thresholds(
        rows,
        score_key=score_key,
        thresholds=thresholds,
        label_key=label_key,
    )
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    return {
        "score_key": score_key,
        "n_rows": len(rows),
        "n_labeled": sum(row.get(label_key) is not None for row in rows),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "best_high_recall": best_operating_point(sweep),
        "threshold_sweep": sweep,
    }

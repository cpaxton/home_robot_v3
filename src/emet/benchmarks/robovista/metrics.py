# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Aggregate RoboVista MCQ accuracy overall and by domain / ability."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _bucket_accuracy(rows: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[bool]] = {}
    for row in rows:
        name = str(row.get(key) or "unknown")
        buckets.setdefault(name, []).append(bool(row.get("correct")))
    out: dict[str, dict[str, float | int]] = {}
    for name, flags in sorted(buckets.items()):
        n = len(flags)
        correct = sum(1 for f in flags if f)
        out[name] = {
            "n": n,
            "correct": correct,
            "accuracy": (correct / n) if n else 0.0,
        }
    return out


def summarize_robovista_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return overall + per-domain + per-ability_type accuracy."""
    n = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    return {
        "n": n,
        "correct": correct,
        "accuracy": (correct / n) if n else 0.0,
        "by_domain": _bucket_accuracy(rows, "domain"),
        "by_ability_type": _bucket_accuracy(rows, "ability_type"),
    }

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""GraphEQA per-question object hints (``explore_eqa_dataset_enrich_labels.yaml``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

# GraphEQA HM-EQA paper evaluates the first 113 Explore-EQA HM3D questions (indices 0–112).
HMEQA_PAPER_QUESTION_COUNT = 113

_DEFAULT_LABELS_PATH = Path(__file__).resolve().parent / "hmeqa_enrich_labels.yaml.bundled"


def hmeqa_paper_question_ids() -> list[int]:
    """Question indices used in the GraphEQA HM-EQA paper (0 .. 112)."""
    return list(range(HMEQA_PAPER_QUESTION_COUNT))


def parse_enrich_label_text(labels: str) -> list[str]:
    """Split GraphEQA enrich label string into object hint tokens."""
    out: list[str] = []
    for token in labels.replace(".", ",").split(","):
        t = token.strip().lower()
        if t and t != "unknown":
            out.append(t)
    return out


@lru_cache(maxsize=1)
def load_hmeqa_enrich_labels(path: Path | None = None) -> dict[str, str]:
    """Load ``{questionId_scene: labels}`` mapping from bundled GraphEQA YAML."""
    yaml_path = path or _DEFAULT_LABELS_PATH
    if not yaml_path.is_file():
        return {}
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v.get("labels", "") if isinstance(v, dict) else v) for k, v in raw.items()}


def enrich_labels_for_question(
    question_id: int,
    scene: str,
    *,
    labels_path: Path | None = None,
) -> str:
    """Return enrich label string for ``{question_id}_{scene}``, or empty."""
    table = load_hmeqa_enrich_labels(labels_path)
    return table.get(f"{question_id}_{scene}", "")

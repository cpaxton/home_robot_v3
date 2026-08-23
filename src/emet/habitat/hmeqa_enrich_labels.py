# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""GraphEQA per-question object hints (``explore_eqa_dataset_enrich_labels.yaml``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

# Historical emet runs used a fixed q0–112 slice. The GraphEQA repository does
# not define that slice: its HM-EQA runner filters the 500-row CSV by semantic
# scene availability, yielding the 114-entry enrich sequence bundled below.
HMEQA_LEGACY_SLICE_COUNT = 113
HMEQA_PAPER_QUESTION_COUNT = HMEQA_LEGACY_SLICE_COUNT
GRAPHEQA_HMEQA_QUESTION_COUNT = 114

_DEFAULT_LABELS_PATH = Path(__file__).resolve().parent / "hmeqa_enrich_labels.yaml.bundled"


def hmeqa_paper_question_ids() -> list[int]:
    """Return the historical emet q0–112 slice (not GraphEQA's filtered set)."""
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


def enrich_labels_for_dataset_question(
    question_id: int,
    scene: str,
    *,
    questions_path: Path | None = None,
    labels_path: Path | None = None,
) -> str:
    """Map a questions.csv row id to the GraphEQA enrich-set ordinal."""
    selected_ids = grapheqa_baseline_question_ids(
        questions_path=questions_path,
        labels_path=labels_path,
    )
    try:
        enrich_id = selected_ids.index(int(question_id))
    except ValueError:
        return ""
    return enrich_labels_for_question(enrich_id, scene, labels_path=labels_path)


def grapheqa_baseline_question_ids(
    *,
    questions_path: Path | None = None,
    labels_path: Path | None = None,
) -> list[int]:
    """Map the upstream GraphEQA semantic-filtered sequence to CSV row ids.

    Upstream ``load_eqa_data`` filters HM-EQA questions to scenes with semantic
    annotations before enumeration. The bundled enrich file records that filtered
    order as contiguous ``{ordinal}_{scene}`` keys. The mapping is accepted only
    when filtering the supplied CSV by those scenes reproduces the exact sequence.

    The default bundled sequence has 114 rows. The often-cited 113 count belongs
    to GraphEQA's OpenEQA HM3D subset, not this HM-EQA filtered launcher.
    """
    from emet.habitat.datasets import load_hmeqa_questions

    enrich = load_hmeqa_enrich_labels(labels_path)
    ordered: list[tuple[int, str]] = []
    for key in enrich:
        ordinal_text, separator, scene = str(key).partition("_")
        if not separator:
            raise ValueError(f"invalid GraphEQA enrich key: {key!r}")
        try:
            ordinal = int(ordinal_text)
        except ValueError as exc:
            raise ValueError(f"invalid GraphEQA enrich ordinal: {key!r}") from exc
        ordered.append((ordinal, scene))
    ordered.sort()
    ordinals = [ordinal for ordinal, _scene in ordered]
    if ordinals != list(range(len(ordered))):
        raise ValueError("GraphEQA enrich ordinals must be contiguous from zero")
    if labels_path is None and len(ordered) != GRAPHEQA_HMEQA_QUESTION_COUNT:
        raise ValueError(
            f"bundled GraphEQA enrich set has {len(ordered)} rows; expected {GRAPHEQA_HMEQA_QUESTION_COUNT}"
        )

    ge_scenes = {scene for _ordinal, scene in ordered}
    questions = load_hmeqa_questions(questions_path)
    selected = [q for q in questions if q.scene in ge_scenes]
    expected_scenes = [scene for _ordinal, scene in ordered]
    actual_scenes = [q.scene for q in selected]
    if actual_scenes != expected_scenes:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(zip(actual_scenes, expected_scenes, strict=False))
                if actual != expected
            ),
            min(len(actual_scenes), len(expected_scenes)),
        )
        raise ValueError(
            "questions.csv does not match the upstream GraphEQA enrich sequence "
            f"at filtered ordinal {mismatch} ({len(actual_scenes)} rows vs "
            f"{len(expected_scenes)} expected)"
        )
    return [q.index for q in selected]

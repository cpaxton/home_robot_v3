# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Load and normalize RoboVista from HuggingFace ``sy-xie/robovista`` (~1.1 GB first download)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

ROBOVISTA_HF_ID = "sy-xie/robovista"
ROBOVISTA_SPLIT = "train"

# Official domain names from the RoboVista README / HF dataset.
ROBOVISTA_DOMAINS = (
    "agriculture",
    "driving",
    "domestic",
    "industrial",
    "surgical",
    "open datasets",
)


@dataclass(frozen=True)
class RoboVistaQuestion:
    """One normalized RoboVista MCQ row."""

    id: str
    question: str
    choices: list[str]
    gold_letter: str
    domain: str
    task: str
    ability_type: str
    ability_subcategory: str = ""
    reasoning: str = ""
    publication_source: str = ""
    images: list[Image.Image] = field(default_factory=list)


def _as_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if hasattr(image, "convert"):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def _normalize_choice_text(choice: Any) -> str:
    text = str(choice or "").strip()
    # Strip a leading "A. " / "A) " prefix if present so prompts stay consistent.
    if len(text) >= 2 and text[0].upper() in "ABCDE" and text[1] in ".)]:":
        return text[2:].strip()
    return text


def _normalize_row(row: dict[str, Any]) -> RoboVistaQuestion:
    choices_raw = row.get("choices") or []
    choices = [_normalize_choice_text(c) for c in list(choices_raw)]
    gold = str(row.get("correct_answer") or "").strip().upper()
    if gold and gold[0] in "ABCDE":
        gold = gold[0]
    images_raw = row.get("images") or []
    images = [_as_pil(img) for img in images_raw]
    return RoboVistaQuestion(
        id=str(row.get("id") or ""),
        question=str(row.get("question") or "").strip(),
        choices=choices,
        gold_letter=gold,
        domain=str(row.get("domain") or "").strip(),
        task=str(row.get("task") or "").strip(),
        ability_type=str(row.get("ability_type") or "").strip(),
        ability_subcategory=str(row.get("ability_subcategory") or "").strip(),
        reasoning=str(row.get("reasoning") or "").strip(),
        publication_source=str(row.get("publication_source") or "").strip(),
        images=images,
    )


def _domain_match(value: str, wanted: Sequence[str]) -> bool:
    if not wanted:
        return True
    lowered = value.strip().lower()
    return any(lowered == w.strip().lower() for w in wanted)


def load_robovista(
    *,
    hf_id: str = ROBOVISTA_HF_ID,
    split: str = ROBOVISTA_SPLIT,
    domains: Sequence[str] | None = None,
    ability_types: Sequence[str] | None = None,
    max_questions: int | None = None,
    rows: Iterable[dict[str, Any]] | None = None,
) -> list[RoboVistaQuestion]:
    """Load RoboVista questions from HuggingFace or an in-memory row iterable.

    Pass ``rows`` to skip the Hub download (unit tests / fixtures). First Hub
    download is ~1.1 GB with embedded images under the HuggingFace cache.
    """
    if rows is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "RoboVista requires the `datasets` package. Install with: uv sync"
            ) from exc
        ds = load_dataset(hf_id, split=split)
        rows = ds

    domain_filter = list(domains or [])
    ability_filter = list(ability_types or [])
    out: list[RoboVistaQuestion] = []
    for row in rows:
        item = dict(row) if not isinstance(row, dict) else row
        q = _normalize_row(item)
        if not _domain_match(q.domain, domain_filter):
            continue
        if ability_filter and not _domain_match(q.ability_type, ability_filter):
            continue
        if not q.id or not q.question or not q.choices or not q.gold_letter:
            continue
        out.append(q)
        if max_questions is not None and len(out) >= int(max_questions):
            break
    return out


def count_by_domain(questions: Sequence[RoboVistaQuestion]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for q in questions:
        key = q.domain or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

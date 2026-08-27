# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Dataclasses for the agentic GraphEQA loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from emet.memory.graph_eqa.agentic_config import PLACE_APPROACH_SAMPLES, PLACE_INSPECT_RECENT_K


@dataclass
class PlaceInspectVisit:
    """One completed investigate() station look for a place card this query."""

    round: int
    closest_m: float
    verify: str = ""
    assess_present: bool | None = None
    assess_answerable: bool | None = None
    suggested: str = ""
    approach_index: int | None = None


@dataclass(frozen=True)
class AnswerEvidenceRecord:
    """One answer proposal tied to the exact view that supports it."""

    letter: str
    source: str
    answer_text: str = ""
    obs_id: int | None = None
    obs_revision: int = 0
    view_id: str = ""
    present: bool = False
    answerable: bool = False
    need_more_views: bool = False
    confidence: float = 0.0
    raw: str = ""
    evidence_event_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        choice_index = ord(self.letter) - ord("A") if len(self.letter) == 1 and self.letter in "ABCDE" else None
        return {
            "answer_text": self.answer_text or None,
            "choice_index": choice_index,
            "source": self.source,
            "obs_id": self.obs_id,
            "obs_revision": int(self.obs_revision),
            "view_id": self.view_id or None,
            "present": bool(self.present),
            "answerable": bool(self.answerable),
            "need_more_views": bool(self.need_more_views),
            "confidence": float(self.confidence),
            "raw": self.raw or None,
            "evidence_event_ids": list(self.evidence_event_ids),
        }


@dataclass(frozen=True)
class FinalAnswerDecision:
    """Atomic scored answer and its aligned evidence provenance."""

    answer: str
    source: str
    confidence: float
    evidence: AnswerEvidenceRecord | None = None
    answer_text: str = ""
    choice_index: int | None = None
    evidence_event_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        evidence_event_ids = self.evidence_event_ids
        if not evidence_event_ids and self.evidence is not None:
            evidence_event_ids = self.evidence.evidence_event_ids
        return {
            "answer": self.answer,
            "answer_text": self.answer_text or None,
            "choice_index": self.choice_index,
            "source": self.source,
            "confidence": float(self.confidence),
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
            "evidence_event_ids": list(evidence_event_ids),
        }


@dataclass
class PlaceInspectRecord:
    """Per-place investigate history for the current question episode."""

    investigate_count: int = 0
    closest_m: float | None = None
    recent: list[PlaceInspectVisit] = field(default_factory=list)
    last_verify: str = ""
    last_assess_present: bool | None = None
    last_assess_answerable: bool | None = None
    last_suggested: str = ""
    tried_approaches: list[int] = field(default_factory=list)
    tried_xy: list[tuple[float, float]] = field(default_factory=list)
    coverage: str = "unknown"  # open | closed | unknown
    local_frontier_cells: int = 0
    close_map_reason: str = ""
    close_map_resolved: bool | None = None

    @property
    def approached_close(self) -> bool:
        return self.closest_m is not None and float(self.closest_m) <= 1.0

    @property
    def approaches_left(self) -> int:
        tried = {int(i) for i in self.tried_approaches}
        return max(0, PLACE_APPROACH_SAMPLES - len(tried))

    @property
    def coverage_complete(self) -> bool:
        return self.coverage == "closed"

    def card_bits(self) -> str:
        """Compact state-card suffix for the router (includes local frontier completeness)."""
        ap = f"approaches={len(self.tried_approaches)}/{PLACE_APPROACH_SAMPLES}"
        cov = f"coverage={self.coverage}"
        if self.coverage == "open":
            cov += f" local_frontier={int(self.local_frontier_cells)}"
        if self.investigate_count <= 0:
            return f"investigated=0 closest=none {ap} {cov} recent=none"
        close = "[close]" if self.approached_close else "[not_close]"
        closest = f"{float(self.closest_m):.1f}m" if self.closest_m is not None else "none"
        recent_bits: list[str] = []
        for v in self.recent[-PLACE_INSPECT_RECENT_K:]:
            bit = f"r{int(v.round)}@{float(v.closest_m):.1f}m"
            if v.approach_index is not None:
                bit += f" ap={int(v.approach_index)}"
            if v.verify:
                bit += f" verify={v.verify}"
            if v.assess_present is False:
                bit += " assess=absent"
            elif v.assess_present is True:
                bit += " assess=present"
            if v.assess_answerable:
                bit += " answerable"
            if v.suggested:
                bit += f" sug={v.suggested}"
            recent_bits.append(bit)
        recent = "; ".join(reversed(recent_bits)) if recent_bits else "none"
        if self.approaches_left > 0:
            more = " more_views"
        else:
            more = " views_exhausted"
        cm = ""
        if self.close_map_reason:
            resolved = (
                "" if self.close_map_resolved is None else (" resolved" if self.close_map_resolved else " unresolved")
            )
            cm = f" close_map={self.close_map_reason}{resolved}"
        return f"investigated={self.investigate_count} closest={closest} {close} {ap} {cov}{more}{cm} recent: {recent}"


@dataclass
class AgenticEQAResult:
    discord_text: str
    answer: str
    confidence: bool
    relevant_images: list[Any] = field(default_factory=list)
    tool_log: list[str] = field(default_factory=list)
    verified: bool = False
    verified_obs_id: int | None = None
    n_rounds: int = 0
    n_nav: int = 0
    n_explore: int = 0
    wall_s: float = 0.0
    budget_hit: bool = False
    # Counterfactual location-MCQ salvage letter (not applied to scored answer).
    salvage_counterfactual_letter: str = ""
    # Which answer channel produced ``answer`` and how much to trust it.
    answer_provenance: str = ""
    answer_confidence: float = 0.0
    decision_rounds: int = 0
    # Object-phrase ``localize_text`` hit from the loop (not a furniture wrap).
    # OVMM scores this after submit releases SigLIP — do not re-query the map.
    voxel_xyz: tuple[float, float, float] | None = None
    voxel_phrase: str | None = None
    voxel_from_pin: bool | None = None

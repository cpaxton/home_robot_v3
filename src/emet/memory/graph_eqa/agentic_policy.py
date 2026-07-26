# Copyright (c) Chris Paxton 2026

"""Explicit evidence-state policy for graph-driven embodied question answering."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AgenticState(str, Enum):
    SEARCH = "SEARCH"
    APPROACH = "APPROACH"
    VERIFY = "VERIFY"
    ASSESS = "ASSESS"
    REPLAN = "REPLAN"
    ANSWER = "ANSWER"


@dataclass(frozen=True)
class EvidenceRecord:
    """Channel-preserving evidence from exactly one fresh observation."""

    hypothesis_id: str
    obs_id: int
    phrase: str
    timestamp_s: float = field(default_factory=time.time)
    full_frame_sim: float | None = None
    dense_sim: float | None = None
    voxel_sim: float | None = None
    detector_score: float | None = None
    crop_siglip_sim: float | None = None
    graph_label_match: bool = False
    geometry_support: bool | None = None
    vlm_present_probability: float | None = None
    detector_backend: str | None = None
    bbox_xyxy: tuple[int, int, int, int] | None = None
    provenance: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HypothesisBelief:
    hypothesis_id: str
    phrase: str
    prior_probability: float = 0.25
    presence_probability: float = 0.25
    answerability_probability: float = 0.0
    evidence: list[EvidenceRecord] = field(default_factory=list)
    attempted_obs_ids: set[int] = field(default_factory=set)
    relation_sufficient: bool = False


@dataclass(frozen=True)
class Assessment:
    hypothesis_id: str
    presence_probability: float
    answerability_probability: float
    positive_channels: tuple[str, ...]
    contradiction_channels: tuple[str, ...]
    verified: bool
    answerable: bool


def _logit(probability: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, float(probability)))
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class EvidencePolicy:
    """State transitions and channel fusion independent of VLM tool routing."""

    def __init__(
        self,
        *,
        verify_probability: float = 0.55,
        answerability_probability: float = 0.55,
        detector_threshold: float = 0.10,
    ) -> None:
        self.state = AgenticState.SEARCH
        self.verify_probability = float(verify_probability)
        self.answerability_threshold = float(answerability_probability)
        self.detector_threshold = float(detector_threshold)
        self.beliefs: dict[str, HypothesisBelief] = {}
        self.active_hypothesis_id: str | None = None
        self._fresh_obs_id: int | None = None
        self._globally_scored_obs_ids: set[int] = set()

    def register_hypothesis(
        self,
        hypothesis_id: str,
        phrase: str,
        *,
        prior_probability: float = 0.25,
    ) -> HypothesisBelief:
        belief = self.beliefs.get(hypothesis_id)
        if belief is None:
            belief = HypothesisBelief(
                hypothesis_id=hypothesis_id,
                phrase=phrase,
                prior_probability=prior_probability,
                presence_probability=prior_probability,
            )
            self.beliefs[hypothesis_id] = belief
        return belief

    def choose(self, hypothesis_id: str) -> None:
        if hypothesis_id not in self.beliefs:
            raise KeyError(f"unknown hypothesis {hypothesis_id}")
        self.active_hypothesis_id = hypothesis_id
        self._fresh_obs_id = None
        self.state = AgenticState.APPROACH

    def approached(self, fresh_obs_id: int) -> None:
        if self.state != AgenticState.APPROACH or self.active_hypothesis_id is None:
            raise RuntimeError(f"approached is invalid in state {self.state}")
        obs_id = int(fresh_obs_id)
        if obs_id in self._globally_scored_obs_ids:
            raise ValueError(f"observation {obs_id} was already scored")
        self._fresh_obs_id = obs_id
        self.state = AgenticState.VERIFY

    def add_evidence(self, record: EvidenceRecord) -> None:
        if self.state != AgenticState.VERIFY:
            raise RuntimeError(f"add_evidence is invalid in state {self.state}")
        if record.hypothesis_id != self.active_hypothesis_id:
            raise ValueError("evidence does not belong to the active hypothesis")
        if self._fresh_obs_id is None or int(record.obs_id) != self._fresh_obs_id:
            raise ValueError("evidence must score the fresh observation produced by APPROACH")
        if int(record.obs_id) in self._globally_scored_obs_ids:
            raise ValueError(f"observation {record.obs_id} was already scored")
        belief = self.beliefs[record.hypothesis_id]
        belief.evidence.append(record)
        belief.attempted_obs_ids.add(int(record.obs_id))
        self._globally_scored_obs_ids.add(int(record.obs_id))
        self.state = AgenticState.ASSESS

    def assess(self, *, relation_sufficient: bool = False) -> Assessment:
        if self.state != AgenticState.ASSESS or self.active_hypothesis_id is None:
            raise RuntimeError(f"assess is invalid in state {self.state}")
        belief = self.beliefs[self.active_hypothesis_id]
        record = belief.evidence[-1]
        log_odds = _logit(belief.presence_probability)
        positives: list[str] = []
        contradictions: list[str] = []

        image_scores = [
            score for score in (record.full_frame_sim, record.dense_sim) if score is not None
        ]
        if image_scores:
            image_score = max(image_scores)
            if image_score >= 0.12:
                log_odds += 0.8
                positives.append("siglip_image")
            elif image_score < 0.10:
                log_odds -= 0.5
                contradictions.append("siglip_image")
        if record.voxel_sim is not None:
            if record.voxel_sim >= 0.21:
                log_odds += 1.8
                positives.append("siglip_voxel")
            elif record.voxel_sim < 0.10:
                log_odds -= 0.5
                contradictions.append("siglip_voxel")
        if record.detector_score is not None:
            if record.detector_score >= self.detector_threshold:
                log_odds += 1.5
                positives.append(record.detector_backend or "detector")
            else:
                log_odds -= 0.4
                contradictions.append(record.detector_backend or "detector")
        if record.crop_siglip_sim is not None and record.crop_siglip_sim >= 0.12:
            log_odds += 0.9
            positives.append("crop_siglip")
        if record.graph_label_match:
            log_odds += 1.2
            positives.append("graph_label")
        if record.geometry_support is True:
            log_odds += 0.5
            positives.append("geometry")
        elif record.geometry_support is False:
            log_odds -= 0.8
            contradictions.append("geometry")
        if record.vlm_present_probability is not None:
            log_odds += 0.8 * _logit(record.vlm_present_probability)
            (positives if record.vlm_present_probability >= 0.5 else contradictions).append(
                "vlm"
            )

        belief.presence_probability = _sigmoid(log_odds)
        belief.relation_sufficient = bool(relation_sufficient)
        evidence_diversity = len(set(positives))
        strong_channel = any(
            channel in positives
            for channel in ("siglip_voxel", "graph_label", "owlv2", "yoloe", "vlm")
        )
        verified = belief.presence_probability >= self.verify_probability and (
            evidence_diversity >= 2 or strong_channel
        )
        # Cheap OWL/SigLIP/voxel fusion is proposal-only: never open ANSWER here.
        # :meth:`apply_vlm_assessment` is the answerability gate (pixels + inventory).
        belief.answerability_probability = 0.0
        self.state = AgenticState.REPLAN
        return Assessment(
            hypothesis_id=belief.hypothesis_id,
            presence_probability=belief.presence_probability,
            answerability_probability=belief.answerability_probability,
            positive_channels=tuple(positives),
            contradiction_channels=tuple(contradictions),
            verified=verified,
            answerable=False,
        )

    def apply_vlm_assessment(
        self,
        *,
        present: bool,
        answerable: bool,
        need_more_views: bool = False,
    ) -> Assessment:
        """Multimodal VLM decides presence/answerability for the active hypothesis."""
        if self.active_hypothesis_id is None:
            raise RuntimeError("apply_vlm_assessment requires an active hypothesis")
        if self.state not in (AgenticState.ASSESS, AgenticState.REPLAN, AgenticState.VERIFY):
            raise RuntimeError(f"apply_vlm_assessment is invalid in state {self.state}")
        belief = self.beliefs[self.active_hypothesis_id]
        vlm_p = 0.85 if present else 0.15
        log_odds = _logit(belief.presence_probability) + 0.8 * _logit(vlm_p)
        belief.presence_probability = _sigmoid(log_odds)
        belief.relation_sufficient = bool(answerable)
        if answerable:
            belief.answerability_probability = max(self.answerability_threshold, 0.9)
        elif present:
            belief.answerability_probability = min(self.answerability_threshold - 0.05, 0.45)
        else:
            belief.answerability_probability = 0.1
        if need_more_views and not answerable:
            belief.answerability_probability = min(belief.answerability_probability, 0.35)
        verified = bool(present) or belief.presence_probability >= self.verify_probability
        self.state = AgenticState.ANSWER if answerable else AgenticState.REPLAN
        positives = ("vlm",) if present or answerable else ()
        contradictions = () if present or answerable else ("vlm",)
        return Assessment(
            hypothesis_id=belief.hypothesis_id,
            presence_probability=belief.presence_probability,
            answerability_probability=belief.answerability_probability,
            positive_channels=positives,
            contradiction_channels=contradictions,
            verified=verified,
            answerable=bool(answerable),
        )

    def replan(self) -> None:
        if self.state != AgenticState.REPLAN:
            raise RuntimeError(f"replan is invalid in state {self.state}")
        self.active_hypothesis_id = None
        self._fresh_obs_id = None
        self.state = AgenticState.SEARCH

    @property
    def scored_obs_ids(self) -> frozenset[int]:
        return frozenset(self._globally_scored_obs_ids)

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""VLM view assessment and answer-evidence persistence for agentic GraphEQA."""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from emet.memory.graph_eqa.agentic.config import (
    _PROVENANCE_CONFIDENCE,
    ESCAPE_MIN_TRAVEL_M,
    NOT_PRESENT_ESCAPE_STREAK,
    question_requires_close_look_keywords,
)
from emet.memory.graph_eqa.agentic.types import AnswerEvidenceRecord
from emet.memory.graph_eqa.eqa.query_images import dump_query_rgb
from emet.memory.graph_eqa.graph_memory import question_stem_for_keywords
from emet.utils.logger import Logger

_logger = Logger(__name__)



def _extract_vlm_target(self) -> None:
    """Text-only VLM: pick the seek/verify phrase + close-look flag once per episode."""
    if not self._close_look:
        self._close_look_required = False
        self._close_look_source = "disabled"
    if self.mode != "answer" or not self.question:
        return
    gm = self.graph_memory
    client = getattr(gm, "eqa_client", None) if gm is not None else None
    from emet.memory.graph_eqa.graph_memory import heuristic_relevant_phrases

    phrases = heuristic_relevant_phrases(self.question)
    fallback = phrases[0] if phrases else question_stem_for_keywords(self.question)
    if client is None:
        self._target_phrase = (fallback or self.question).strip()
        self._question_type = "other"
        self._apply_close_look_fallback()
        self._append_trace(
            {
                "event": "vlm_target_extract",
                "target_phrase": self._target_phrase,
                "question_type": self._question_type,
                "requires_close_look": self._close_look_required,
                "close_look_source": self._close_look_source,
                "source": "heuristic",
            }
        )
        return
    from emet.eval.agentic_vlm_assess import extract_target_from_question

    extracted = extract_target_from_question(client, self.question, fallback_phrase=str(fallback or ""))
    self._target_phrase = extracted.target_phrase
    self._question_type = extracted.question_type
    self._close_look_required = bool(extracted.requires_close_look)
    self._close_look_source = "vlm"
    self._append_trace({"event": "vlm_target_extract", "source": "vlm", **extracted.to_dict()})

def _apply_close_look_fallback(self) -> None:
    """Keyword heuristic when no VLM is available (or the classifier is off)."""
    if not self._close_look:
        self._close_look_required = False
        self._close_look_source = "disabled"
        return
    if question_requires_close_look_keywords(self.question):
        self._close_look_required = True
        self._close_look_source = "keyword"
    else:
        self._close_look_required = False
        self._close_look_source = "none"

def _escape_min_travel_m(self) -> float:
    """Distance the next frontier must clear once the target keeps not showing up."""
    if self.decision_policy == "grounded_v2":
        return 0.0
    if self._not_present_streak < NOT_PRESENT_ESCAPE_STREAK:
        return 0.0
    return ESCAPE_MIN_TRAVEL_M

def _update_escape_streak(self, *, present: bool) -> None:
    """Track consecutive not-visible views and publish the escape floor to the picker."""
    if present:
        self._not_present_streak = 0
    else:
        self._not_present_streak += 1
    self.agent._explore_min_travel_m = 0.0 if self.decision_policy == "grounded_v2" else self._escape_min_travel_m()

def _mcq_letter_from_suggested(self, raw: Any) -> str:
    """Map semantic VLM answer text to the benchmark choice encoding."""
    return self._mcq_letter_from_text(str(raw or ""))

def _view_identity_for_obs(self, obs_id: int | None) -> tuple[int, str]:
    if obs_id is None:
        return 0, ""
    gm = self.graph_memory
    revision = 0
    view_id = ""
    revision_fn = getattr(gm, "obs_revision", None) if gm is not None else None
    if callable(revision_fn):
        try:
            revision = int(revision_fn(int(obs_id)))
        except (TypeError, ValueError):
            revision = 0
    view_fn = getattr(gm, "view_id_for_obs", None) if gm is not None else None
    if callable(view_fn):
        try:
            view_id = str(view_fn(int(obs_id)) or "")
        except (TypeError, ValueError):
            view_id = ""
    return revision, view_id

def _persist_agentic_evidence(
    self,
    *,
    stage: str,
    outcome: str,
    obs_id: int,
    phrase: str,
    confidence: float,
    source: str,
    score: float | None = None,
    threshold: float | None = None,
    supporting_event_ids: tuple[str, ...] = (),
    payload: dict[str, Any] | None = None,
) -> str:
    gm = self.graph_memory
    record = getattr(gm, "record_agentic_evidence", None) if gm is not None else None
    if not callable(record):
        return ""
    try:
        return str(
            record(
                stage=stage,
                outcome=outcome,
                obs_id=int(obs_id),
                phrase=phrase,
                confidence=float(confidence),
                source=source,
                agent_round=int(self._round) + 1,
                score=score,
                threshold=threshold,
                supporting_event_ids=supporting_event_ids,
                payload=payload,
            )
            or ""
        )
    except (TypeError, ValueError) as exc:
        _logger.warning(f"persist {stage} evidence failed: {exc}")
        return ""

def _record_answer_evidence(
    self,
    *,
    letter: str,
    source: str,
    obs_id: int | None,
    present: bool,
    answerable: bool,
    need_more_views: bool,
    confidence: float,
    answer_text: str = "",
    raw: str = "",
    evidence_event_ids: tuple[str, ...] = (),
) -> AnswerEvidenceRecord | None:
    canonical = self._mcq_letter_from_text(letter)
    if not canonical:
        return None
    semantic = str(answer_text or "").strip() or self._choice_text_for_letter(canonical)
    revision, view_id = self._view_identity_for_obs(obs_id)
    record = AnswerEvidenceRecord(
        letter=canonical,
        source=str(source or "unknown"),
        answer_text=semantic,
        obs_id=int(obs_id) if obs_id is not None else None,
        obs_revision=revision,
        view_id=view_id,
        present=bool(present),
        answerable=bool(answerable),
        need_more_views=bool(need_more_views),
        confidence=float(confidence),
        raw=str(raw or "")[:1000],
        evidence_event_ids=tuple(dict.fromkeys(str(item) for item in evidence_event_ids if str(item))),
    )
    # One current assessment per source/view revision; older immutable world-view
    # evidence remains in GraphEQAMemory once world_evidence dual-write is enabled.
    self._answer_evidence = [
        item
        for item in self._answer_evidence
        if not (
            item.source == record.source
            and item.obs_id == record.obs_id
            and item.obs_revision == record.obs_revision
        )
    ]
    self._answer_evidence.append(record)
    return record

def _best_vlm_answer_evidence(self, *, letter: str = "") -> AnswerEvidenceRecord | None:
    expected = self._mcq_letter_from_text(letter)
    candidates = [
        item
        for item in self._answer_evidence
        if item.source == "vlm_suggested"
        and item.present
        and item.answerable
        and not item.need_more_views
        and (not expected or item.letter == expected)
    ]
    if not candidates:
        # Compatibility for tests and legacy callers that directly seed history.
        for oid, history in self._assess_history.items():
            if not bool(history.get("present")) or not bool(history.get("answerable")):
                continue
            if bool(history.get("need_more_views")):
                continue
            candidate = self._record_answer_evidence(
                letter=str(history.get("suggested_answer") or ""),
                source="vlm_suggested",
                answer_text=str(history.get("suggested_answer") or ""),
                obs_id=int(oid),
                present=True,
                answerable=True,
                need_more_views=False,
                confidence=_PROVENANCE_CONFIDENCE["vlm_suggested"],
            )
            if candidate is not None and (not expected or candidate.letter == expected):
                candidates.append(candidate)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            int(item.obs_id == self._verified_obs_id),
            item.confidence,
            item.obs_revision,
        ),
    )

def _confirmed_vlm_answer_evidence(self, *, letter: str = "") -> AnswerEvidenceRecord | None:
    """Return the latest VLM answer that opened the ANSWER gate."""
    record = self._confirmed_answer_evidence
    expected = self._mcq_letter_from_text(letter)
    if (
        record is None
        or record.source != "vlm_suggested"
        or not record.present
        or not record.answerable
        or record.need_more_views
        or (expected and record.letter != expected)
    ):
        return None
    return record

def _question_is_mcq(self) -> bool:
    """True for HM-EQA-style A–D questions; False for open find/localize questions."""
    from emet.habitat.metrics import parse_mcq_choices_from_question

    if parse_mcq_choices_from_question(self.question):
        return True
    return bool(getattr(self, "_mcq_choices", None))

def _answerable_phrase_hit(self, *, obs_id: int, phrase: str) -> bool:
    """True when target/stem tokens appear in inventory or labels near obs."""
    needle = str(phrase or self._target_phrase or "").strip().lower()
    if not needle:
        needle = question_stem_for_keywords(self.question or "").lower()
    tokens = [t for t in re.findall(r"[a-z0-9]+", needle) if len(t) >= 3]
    # Drop ultra-common stems that don't corroborate objects.
    stop = {"the", "and", "where", "what", "how", "many", "did", "leave", "there", "this", "that"}
    tokens = [t for t in tokens if t not in stop]
    if not tokens:
        return False
    labels: list[str] = list(self._inventory_labels(limit=32))
    gm = self.graph_memory
    if gm is not None:
        near_fn = getattr(gm, "labels_near_obs", None)
        if callable(near_fn):
            try:
                labels.extend(str(x) for x in (near_fn(int(obs_id)) or []))
            except Exception:
                pass
        for node in list(getattr(gm, "_nodes", None) or []):
            if int(getattr(node, "obs_id", -1) or -1) != int(obs_id):
                continue
            labels.extend(str(x) for x in (getattr(node, "labels", None) or []))
    blob = " ".join(labels).lower()
    if not blob.strip():
        return False
    return any(t in blob for t in tokens)

def _maybe_confirm_answerable(
    self,
    *,
    obs_id: int,
    present: bool,
    answerable: bool,
    need_more_views: bool,
    suggested_answer: Any,
    phrase: str,
) -> tuple[bool, str]:
    """Hybrid unlock: corroborated single view or two-view same letter.

    Returns ``(confirmed, reason)``.
    """
    if not answerable:
        if present is False:
            # Keep pending for a later agreeing view; absent does not clear it.
            pass
        return False, "not_answerable"
    if not self._answerable_confirm:
        # Legacy: raw answerable unlocks (ignore need_more_views for parity).
        return True, "confirm_disabled"
    if not self._question_is_mcq():
        # Open-ended find / localize (OVMM "Where is the table?"): no MCQ letter
        # set exists, so a fresh view that actually shows the target is enough.
        # The assess prompt is open-aware, so answerable means "visible/localizable".
        # For location questions the VLM conservatively sets need_more_views=True
        # even when the target is clearly in view — presence alone confirms here.
        if bool(present):
            self._pending_answerable = None
            return True, "open_view_present"
        return False, "open_not_present"
    if need_more_views:
        letter = self._mcq_letter_from_suggested(suggested_answer)
        self._pending_answerable = {
            "letter": letter,
            "answer_text": str(suggested_answer or "").strip(),
            "obs_id": int(obs_id),
            "phrase": str(phrase or self._target_phrase or ""),
            "present": bool(present),
        }
        return False, "need_more_views"
    letter = self._mcq_letter_from_suggested(suggested_answer)
    # Single-view present-confirm: a view that saw the target and offered a
    # letter is enough (keeps the present guard that fixed q28/q39 absence
    # answers). This raised verification from ~1-4/30 to ~5-6x in the field
    # data; verified answers score ~86% vs ~35% forced guesses.
    if self._single_view_confirm and bool(present) and bool(letter):
        self._pending_answerable = None
        return True, "single_view_present"
    phrase_hit = bool(present) and bool(letter) and self._answerable_phrase_hit(obs_id=int(obs_id), phrase=phrase)
    pending = self._pending_answerable
    # Two views that both failed to see the target are not corroboration. Without
    # this guard q28/q39 confirmed an absence letter twice and scored it 0/5.
    two_view = False
    if (
        pending
        and letter
        and bool(present)
        and bool(pending.get("present"))
        and str(pending.get("letter") or "") == letter
        and int(pending.get("obs_id", -1)) != int(obs_id)
    ):
        two_view = True
    if phrase_hit:
        self._pending_answerable = None
        return True, "phrase_corroborated"
    if two_view:
        self._pending_answerable = None
        return True, "two_view_agree"
    # Defer — stash / refresh pending for a later confirm.
    self._pending_answerable = {
        "letter": letter,
        "answer_text": str(suggested_answer or "").strip(),
        "obs_id": int(obs_id),
        "phrase": str(phrase or self._target_phrase or ""),
        "present": bool(present),
    }
    return False, "deferred"

def _run_vlm_view_assess(
    self,
    *,
    rgb: np.ndarray | None,
    phrase: str,
    obs_id: int,
    proposal: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Multimodal answerability gate. One assess per obs_id.

    Qwen looks at pixels + neutral inventory (obs counts, graph labels).
    SigLIP/OWL are where-next scores for navigation / which place to grow
    the graph — logged on the verify proposal, never fed into this prompt.
    Hybrid confirm (phrase hit / two-view) gates ``_verified``.
    """
    oid = int(obs_id)
    if oid in self._vlm_assessed_obs_ids:
        return {
            "ok": False,
            "status": "SKIPPED_SAME_VIEW",
            "obs_id": oid,
            "error": f"obs_id {oid} already VLM-assessed",
        }
    gm = self.graph_memory
    client = getattr(gm, "eqa_client", None) if gm is not None else None
    if client is None:
        self._append_trace(
            {
                "tool": "vlm_assess",
                "obs_id": oid,
                "ok": False,
                "error": "no eqa_client",
                "answerable": False,
            }
        )
        return {"ok": False, "error": "no eqa_client", "answerable": False, "obs_id": oid}

    from emet.eval.agentic_vlm_assess import assess_view_with_vlm, build_inventory_brief

    # Do not pass SigLIP/OWL proposal into inventory — ABSENT colors answers.
    inventory = build_inventory_brief(
        n_observations=len(list(getattr(gm, "_observations", None) or [])),
        graph_labels=self._inventory_labels(),
        tried_obs_ids=sorted(self._tried.keys()),
        n_rounds=self._round,
        n_nav=self._n_nav + self._n_explore,
    )
    assessment = assess_view_with_vlm(
        client,
        question=self.question,
        rgb=rgb,
        inventory=inventory,
        target_phrase=self._target_phrase or phrase,
        is_mcq=self._question_is_mcq(),
    )
    query_image_paths = dump_query_rgb(self, oid, rgb, kind="vlm_assess")
    vlm_event_id = self._persist_agentic_evidence(
        stage="vlm_assessment",
        outcome="present" if assessment.present else "absent",
        obs_id=oid,
        phrase=str(phrase or self._target_phrase or ""),
        confidence=0.9,
        source="vlm_view_assess",
        supporting_event_ids=tuple(
            item for item in (str((proposal or {}).get("evidence_event_id") or ""),) if item
        ),
        payload={
            "answerable": bool(assessment.answerable),
            "need_more_views": bool(assessment.need_more_views),
            "suggested_answer": str(assessment.suggested_answer or "")[:160],
            "reason": str(assessment.reason or "")[:240],
        },
    )
    self._vlm_assessed_obs_ids.add(oid)
    # Per-view evidence ledger: the final EQA pins the best assessed view as
    # Image 1 when nothing was corroborated (see _best_evidence_obs_id).
    self._assess_history[oid] = {
        "present": bool(assessment.present),
        "answerable": bool(assessment.answerable),
        "need_more_views": bool(assessment.need_more_views),
        "suggested_answer": assessment.suggested_answer,
        "phrase": str(phrase or self._target_phrase or ""),
    }
    if gm is not None and bool(assessment.present) and hasattr(gm, "record_close_look_label"):
        looked = str(phrase or self._target_phrase or "").strip()
        if looked:
            gm.record_close_look_label(oid, looked)
    _logger.info(
        "agentic vlm_assess obs=%d present=%s answerable=%s need_more=%s mcq=%s phrase=%r suggest=%r reason=%r",
        oid,
        bool(assessment.present),
        bool(assessment.answerable),
        bool(assessment.need_more_views),
        self._question_is_mcq(),
        str(phrase or self._target_phrase or "")[:60],
        str(assessment.suggested_answer or "")[:60],
        str(assessment.reason or "")[:80],
    )
    # Trust the VLM assess. Cheap detector status is nav/debug only.
    proposal_status = str(
        (proposal or {}).get("decision") or getattr(self._last_verify, "status", "") or ""
    ).upper()
    vlm_assessment = None
    try:
        vlm_assessment = self._evidence_policy.apply_vlm_assessment(
            present=assessment.present,
            answerable=assessment.answerable,
            need_more_views=assessment.need_more_views,
        )
    except (RuntimeError, ValueError) as exc:
        if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
            _logger.warning(
                "evidence-policy VLM assess rejected: %s (state=%s, present=%s, answerable=%s)",
                exc,
                self._evidence_policy.state,
                assessment.present,
                assessment.answerable,
            )
        else:
            _logger.warning(f"evidence-policy VLM assess rejected: {exc}")
    confirmed = False
    confirm_reason = "no_vlm"
    if vlm_assessment is not None:
        confirmed, confirm_reason = self._maybe_confirm_answerable(
            obs_id=oid,
            present=bool(assessment.present),
            answerable=bool(assessment.answerable),
            need_more_views=bool(assessment.need_more_views),
            suggested_answer=assessment.suggested_answer,
            phrase=phrase,
        )
    if confirmed:
        supporting_event_ids = tuple(
            item
            for item in (
                str((proposal or {}).get("evidence_event_id") or ""),
                vlm_event_id,
            )
            if item
        )
        fused_event_id = self._persist_agentic_evidence(
            stage="fused_confirmation",
            outcome="confirmed",
            obs_id=oid,
            phrase=str(phrase or self._target_phrase or ""),
            confidence=1.0,
            source="agentic_policy",
            supporting_event_ids=supporting_event_ids,
            payload={
                "confirm_reason": confirm_reason,
                "suggested_answer": str(assessment.suggested_answer or "")[:160],
            },
        )
        persisted_event_ids = tuple(
            dict.fromkeys(
                (
                    *supporting_event_ids,
                    *((fused_event_id,) if fused_event_id else ()),
                )
            )
        )
        if self.decision_policy == "grounded_v2":
            durable_fn = getattr(self.graph_memory, "durable_confirmation_event_ids", None)
            durable_ids: tuple[str, ...] = ()
            if callable(durable_fn):
                try:
                    durable_ids = tuple(
                        str(item)
                        for item in durable_fn(
                            obs_id=oid,
                            phrase=str(phrase or self._target_phrase or ""),
                        )
                        if str(item)
                    )
                except (TypeError, ValueError) as exc:
                    _logger.warning(f"read durable confirmation evidence failed: {exc}")
            confirmed = bool(
                vlm_event_id and fused_event_id and vlm_event_id in durable_ids and fused_event_id in durable_ids
            )
            self._verified_evidence_event_ids = durable_ids if confirmed else ()
            if not confirmed:
                confirm_reason = "durable_evidence_unavailable"
                self._pending_answerable = {
                    "letter": self._mcq_letter_from_suggested(assessment.suggested_answer),
                    "answer_text": str(assessment.suggested_answer or "").strip(),
                    "obs_id": oid,
                    "phrase": str(phrase or self._target_phrase or ""),
                    "present": bool(assessment.present),
                }
        else:
            self._verified_evidence_event_ids = persisted_event_ids
        if confirmed:
            try:
                self._evidence_policy.confirm_answerable()
            except (RuntimeError, ValueError) as exc:
                _logger.warning(f"evidence-policy confirm_answerable rejected: {exc}")
            self._verified = True
            self._verified_obs_id = oid
        if confirmed:
            self._append_trace(
                {
                    "event": "answerable_confirmed",
                    "obs_id": oid,
                    "reason": confirm_reason,
                    "suggested_answer": assessment.suggested_answer,
                    "evidence_event_ids": list(self._verified_evidence_event_ids),
                }
            )
    if not confirmed and assessment.answerable:
        self._append_trace(
            {
                "event": "answerable_deferred",
                "obs_id": oid,
                "reason": confirm_reason,
                "suggested_answer": assessment.suggested_answer,
                "pending": dict(self._pending_answerable or {}),
            }
        )
    # Qwen says target not in this view — prefer coverage before the next investigate.
    if self.decision_policy != "grounded_v2" and assessment.present is False and not assessment.answerable:
        self._prefer_explore = True
        self._prefer_explore_reason = "absent"
    self._update_escape_streak(present=assessment.present)
    payload = {
        "tool": "vlm_assess",
        "obs_id": oid,
        "phrase": phrase,
        "target": assessment.target,
        "present": assessment.present,
        "answerable": assessment.answerable,
        "need_more_views": assessment.need_more_views,
        "suggested_answer": assessment.suggested_answer,
        "reason": assessment.reason,
        "policy_state": str(self._evidence_policy.state),
        "verified": self._verified,
        "vlm_answerable": bool(self._verified),
        "answerable_confirm_reason": confirm_reason,
        "pending_answerable": dict(self._pending_answerable or {}) or None,
        "proposal_status": proposal_status or None,
        "not_present_streak": self._not_present_streak,
        "explore_min_travel_m": self._escape_min_travel_m(),
        "inventory": inventory,
        "vlm_evidence_event_id": vlm_event_id or None,
        "verified_evidence_event_ids": list(self._verified_evidence_event_ids),
    }
    if query_image_paths:
        payload.update(query_image_paths)
    self._last_vlm_assess = payload
    answer_evidence = None
    if assessment.present:
        positive = self._mcq_letter_from_suggested(assessment.suggested_answer)
        if positive:
            self._last_positive_letter = positive
            self._last_positive_obs_id = oid
            answer_evidence = self._record_answer_evidence(
                letter=positive,
                source="vlm_suggested",
                answer_text=str(assessment.suggested_answer or ""),
                obs_id=oid,
                present=True,
                answerable=bool(assessment.answerable),
                need_more_views=bool(assessment.need_more_views),
                confidence=_PROVENANCE_CONFIDENCE["vlm_suggested"],
                raw=assessment.raw,
                evidence_event_ids=(
                    self._verified_evidence_event_ids
                    if confirmed
                    else tuple(item for item in (vlm_event_id,) if item)
                ),
            )
    if confirmed and answer_evidence is not None:
        self._confirmed_answer_evidence = answer_evidence
    self._append_trace(payload)
    return {
        "ok": True,
        "obs_id": oid,
        "present": assessment.present,
        "answerable": assessment.answerable,
        "need_more_views": assessment.need_more_views,
        "suggested_answer": assessment.suggested_answer,
        "verified": self._verified,
        "vlm_answerable": bool(self._verified),
        "answerable_confirm_reason": confirm_reason,
        "policy_state": str(self._evidence_policy.state),
        "reason": assessment.reason,
        "vlm_evidence_event_id": vlm_event_id or None,
        "verified_evidence_event_ids": list(self._verified_evidence_event_ids),
    }

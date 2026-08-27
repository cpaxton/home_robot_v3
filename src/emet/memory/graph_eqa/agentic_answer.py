# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Submit / finish / forced-answer for the agentic GraphEQA executor."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from emet.habitat.metrics import (
    choices_are_count_mcq,
    extract_mcq_letter,
    parse_mcq_choices_from_question,
    should_abstain_location_mcq,
)
from emet.memory.graph_eqa.agentic_config import (
    _COORD_DUMP_RE,
    _PROVENANCE_CONFIDENCE,
)
from emet.memory.graph_eqa.agentic_policy import AgenticState
from emet.memory.graph_eqa.agentic_types import AnswerEvidenceRecord, FinalAnswerDecision
from emet.memory.graph_eqa.mcq_debias import (
    answer_is_unknownish,
    count_answer_is_none_or_zero,
    match_freeform_to_choice,
    valid_choice_indices,
)
from emet.utils.logger import Logger

_logger = Logger(__name__)


class AgenticAnswerMixin:
    """Submit, finish, and forced-answer ladder."""

    def _tool_submit_answer(self, answer: str) -> dict[str, Any]:
        if self.mode == "explore":
            return {"ok": False, "error": "submit_answer unavailable in explore mode — use finish"}
        nav_exhausted = self._n_nav + self._n_explore >= self.max_nav_steps
        if self._verified and self._evidence_policy.state != AgenticState.ANSWER:
            if nav_exhausted or self._round >= self.max_rounds - 1:
                return self._forced_answer_fallback(
                    reason="target evidence did not establish answer sufficiency",
                    prefer_answer=answer,
                )
            return {
                "ok": False,
                "error": "target present but answer relation/count is unresolved — replan for a disambiguating view",
            }
        if self._require_verified and not self._verified:
            # Exhausted candidates → best guess with provenance (do not burn rounds
            # on rejected submits, and do not throw away the four-image EQA).
            if nav_exhausted or self._round >= self.max_rounds - 1:
                return self._forced_answer_fallback(prefer_answer=answer)
            return {
                "ok": False,
                "error": "not verified — require_verified=1; VLM assess must mark answerable before submit_answer",
            }
        # Allow submit once nav budget is spent so EQA can emit Action:N for follow-up,
        # even if SigLIP never hit PRESENT (holdout q104/q105).
        if not self._verified and self._round < self.max_rounds - 1 and not nav_exhausted:
            return {
                "ok": False,
                "error": "not verified — call verify_siglip (+ VLM assess) or exhaust budget before submit_answer",
            }
        return self._do_submit_answer(prefer_answer=answer)

    def _tool_finish(self, summary: str) -> dict[str, Any]:
        if self.mode != "explore":
            return {"ok": False, "error": "finish unavailable in answer mode — use submit_answer"}
        if not self._explore_done() and self._round < self.max_rounds - 1:
            return {
                "ok": False,
                "error": "exploration not finished — frontiers remain and nav budget left",
            }
        return self._do_finish(summary)

    def _do_finish(self, summary: str = "") -> dict[str, Any]:
        text = (summary or "").strip() or (
            f"Explored {self._n_explore + self._n_nav} waypoints; {self._frontier_count()} frontiers remain."
        )
        self._append_trace(
            {
                "tool": "finish",
                "summary": text,
                "n_explore": self._n_explore,
                "n_nav": self._n_nav,
                "frontiers_left": self._frontier_count(),
            }
        )
        return {
            "ok": True,
            "answer": text,
            "discord_text": f"Explore finished: {text}",
            "confidence": True,
            "relevant_images": [],
        }

    def _pending_answerable_letter(self) -> str:
        """Letter from a deferred answerable assess, only if that view saw the target."""
        pending = self._pending_answerable or {}
        if not pending.get("present"):
            return ""
        return str(pending.get("letter") or "").strip().upper()

    def _uniform_prior_letter(self, n_choices: int) -> str:
        """Last-resort guess when no channel produced a letter.

        Deterministic per question so reruns stay comparable, and spread across the
        option set so a whole benchmark is not biased toward one letter.
        """
        from emet.habitat.metrics import parse_mcq_choices_from_question

        n = max(1, min(int(n_choices), len("ABCDE")))
        choices = parse_mcq_choices_from_question(self.question)
        if not choices:
            choices = list(getattr(self, "_mcq_choices", None) or [])
        valid = valid_choice_indices(choices[:n]) if choices else list(range(n))
        if not valid:
            valid = list(range(n))
        digest = hashlib.sha1((self.question or "").encode("utf-8")).hexdigest()
        return chr(ord("A") + valid[int(digest[:8], 16) % len(valid)])

    def _confidence_for_provenance(self, provenance: str) -> float:
        """Coarse calibrated confidence so a robot can say "I think X, but I'm unsure".

        Ordered by channel reliability in the 2026-07 trace audit; these are priors,
        not fitted values. Refit them from the per-channel accuracy report before
        quoting them anywhere.
        """
        score = _PROVENANCE_CONFIDENCE.get(provenance, 0.4)
        confirmed_vlm = provenance == "vlm_suggested" and self._confirmed_vlm_answer_evidence() is not None
        if self._verified or confirmed_vlm:
            score += 0.15
        return round(min(score, 0.95), 2)

    def _forced_answer_fallback(
        self,
        *,
        reason: str = "require_verified and no fused verification before budget/rounds exhausted",
        prefer_answer: str = "",
    ) -> dict[str, Any]:
        """Budget or verification exhausted: still commit to a best guess.

        The previous behavior returned a bare ``Unknown`` without ever calling the
        four-image EQA, so every budget-exhausted episode scored zero even when the
        graph held the evidence. Now we run the EQA, walk the answer channels in
        reliability order, and fall back to a uniform-prior option, recording which
        rung fired so accuracy and calibration can be measured separately.
        """
        from emet.habitat.metrics import parse_mcq_choices_from_question

        if not self._force_answer and self._confirmed_vlm_answer_evidence() is None:
            return self._abstain_unverified(reason=reason)

        out = self._do_submit_answer(prefer_answer=prefer_answer)
        answer = str(out.get("answer") or "")
        provenance = str(out.get("answer_source") or "query")
        choices = parse_mcq_choices_from_question(self.question)
        resolved_letter = self._mcq_letter_from_text(answer) if choices else ""
        raw_eqa_answer = self._eqa_self_answer_text() if choices else ""
        raw_eqa_letter = self._eqa_self_answer_letter() if choices else ""
        grounded_decision = (
            self.decision_policy == "grounded_v2"
            and self._final_answer_decision is not None
            and bool(self._mcq_letter_from_text(self._final_answer_decision.answer))
            and self._final_answer_decision.source in {"eqa_answer", "prefer", "vlm_suggested"}
        )
        decision_evidence = self._final_answer_decision.evidence if self._final_answer_decision is not None else None
        evidence_backed_decision = bool(
            decision_evidence is not None
            and decision_evidence.present
            and decision_evidence.answerable
            and not decision_evidence.need_more_views
            and self._mcq_letter_from_text(self._final_answer_decision.answer)
        )
        answer_verified = bool(self._verified or evidence_backed_decision)
        # Unverified forced answers show an option-position bias (the 2026-08 bal-32
        # audit: wrong forced choices were overwhelmingly last). Run semantic
        # freeform + capped rotation voting before the fallback ladder.
        debias_letter = ""
        debias_detail: dict[str, Any] = {}
        if (
            self._mcq_debias
            and choices
            and len(choices) >= 2
            and not grounded_decision
            and not evidence_backed_decision
        ):
            gm = self.graph_memory
            vote_fn = getattr(gm, "vote_mcq_letter", None) if gm is not None else None
            if callable(vote_fn):
                try:
                    debias_letter = str(vote_fn(self.question, choices, max_votes=2) or "").strip().upper()
                except Exception as e:
                    _logger.warning(f"forced-answer mcq debias failed ({e})")
                    debias_letter = ""
                if debias_letter and not self._mcq_letter_from_text(debias_letter):
                    debias_letter = ""
                if gm is not None:
                    debias_detail = dict(getattr(gm, "last_mcq_debias", None) or {})
        if debias_letter:
            answer, provenance = self._choice_text_for_letter(debias_letter), "mcq_debias"
        elif choices and not resolved_letter:
            # Keep channel tags distinct so calibration / H2H summaries can separate a
            # view that saw the target from a deferred assess from a uniform prior.
            trusted = self._trusted_vlm_letter()
            if trusted:
                answer, provenance = self._choice_text_for_letter(trusted), "vlm_suggested"
            else:
                pending = self._pending_answerable_letter()
                if pending:
                    answer, provenance = self._choice_text_for_letter(pending), "pending_letter"
                else:
                    prior = self._uniform_prior_letter(len(choices))
                    answer, provenance = self._choice_text_for_letter(prior), "uniform_prior"
        resolved_letter = self._mcq_letter_from_text(answer) if choices else ""
        evidence = (
            self._final_answer_decision.evidence
            if self._final_answer_decision is not None and self._final_answer_decision.answer == answer
            else None
        )
        answer_text = (
            self._final_answer_decision.answer_text
            if self._final_answer_decision is not None and self._final_answer_decision.answer == answer
            else answer
        )
        self._final_answer_decision = FinalAnswerDecision(
            answer=answer,
            source=provenance,
            confidence=self._confidence_for_provenance(provenance),
            evidence=evidence,
            answer_text=answer_text,
            choice_index=(ord(resolved_letter) - ord("A") if resolved_letter else None),
            evidence_event_ids=(
                evidence.evidence_event_ids
                if evidence is not None
                else (self._verified_evidence_event_ids if self._verified else ())
            ),
        )
        self._answer_provenance = provenance
        confidence_score = self._confidence_for_provenance(provenance)
        self._append_trace(
            {
                "tool": "forced_answer",
                "reason": reason,
                "answer": answer,
                "answer_provenance": provenance,
                "answer_confidence": confidence_score,
                "raw_eqa_answer": raw_eqa_answer or None,
                "raw_eqa_choice_index": (ord(raw_eqa_letter) - ord("A") if raw_eqa_letter else None),
                "resolved_choice_index": (ord(resolved_letter) - ord("A") if resolved_letter else None),
                "mcq_debias": debias_detail or None,
                "agentic_mcq_debias_enabled": bool(self._mcq_debias),
                "evidence_backed_decision": bool(evidence_backed_decision),
                "final_decision": (
                    self._final_answer_decision.to_dict() if self._final_answer_decision is not None else None
                ),
                "verified": answer_verified,
                "n_nav": self._n_nav,
                "n_explore": self._n_explore,
                "last_verify": (
                    None
                    if self._last_verify is None
                    else {
                        "status": self._last_verify.status,
                        "sim": float(self._last_verify.sim),
                        "obs_id": int(self._last_verify.obs_id),
                        "phrase": self._last_verify.phrase,
                    }
                ),
            }
        )
        out = dict(out)
        out.update(
            {
                "ok": True,
                "answer": answer,
                "answer_source": provenance,
                "answer_provenance": provenance,
                "answer_confidence": confidence_score,
                "final_decision": (
                    self._final_answer_decision.to_dict() if self._final_answer_decision is not None else None
                ),
                "confidence": answer_verified,
                "discord_text": (
                    f"Answer:{answer}\nConfidence:{confidence_score}\n"
                    f"[answer_provenance:{provenance}] [forced: {reason}]"
                ),
                "verified": answer_verified,
            }
        )
        return out

    def _abstain_unverified(
        self,
        reason: str = "require_verified and no fused verification before budget/rounds exhausted",
    ) -> dict[str, Any]:
        """Legacy no-guess path, kept behind ``EMET_EQA_FORCE_ANSWER=0`` for A/B."""
        msg = f"Unknown (unverified — {reason})"
        self._append_trace(
            {
                "tool": "abstain_unverified",
                "reason": reason,
                "n_nav": self._n_nav,
                "n_explore": self._n_explore,
                "last_verify": (
                    None
                    if self._last_verify is None
                    else {
                        "status": self._last_verify.status,
                        "sim": float(self._last_verify.sim),
                        "obs_id": int(self._last_verify.obs_id),
                        "phrase": self._last_verify.phrase,
                    }
                ),
            }
        )
        return {
            "ok": True,
            "answer": "Unknown",
            "discord_text": f"Answer:{msg}",
            "confidence": False,
            "relevant_images": [],
            "verified": False,
        }

    @staticmethod
    def _looks_like_coordinate_dump(text: str) -> bool:
        """True for nearest-furniture XYZ dumps (``The fan is at approximately (x,y,z) m``)."""
        return bool(_COORD_DUMP_RE.search(text or ""))

    def _mcq_letter_from_text(self, text: str) -> str:
        """Extract a canonical A–D letter when the question is MCQ-shaped."""
        from emet.habitat.metrics import extract_mcq_letter, parse_mcq_choices_from_question

        raw = (text or "").strip()
        if not raw:
            return ""
        choices = parse_mcq_choices_from_question(self.question)
        if not choices:
            choices = list(getattr(self, "_mcq_choices", None) or [])
        if not choices:
            return ""
        letter = extract_mcq_letter(raw, choices)
        if letter:
            return letter
        if choices:
            idx = match_freeform_to_choice(raw, choices)
            if idx is not None and 0 <= idx < min(len(choices), 5):
                return chr(ord("A") + idx)
        if len(raw) == 1 and raw.upper() in "ABCDE":
            return raw.upper()
        return ""

    def _choice_text_for_letter(self, letter: str) -> str:
        """Return semantic option text for an internal benchmark letter."""
        from emet.habitat.metrics import parse_mcq_choices_from_question

        canonical = str(letter or "").strip().upper()
        choices = parse_mcq_choices_from_question(self.question)
        if not choices:
            choices = list(getattr(self, "_mcq_choices", None) or [])
        if len(canonical) != 1 or canonical not in "ABCDE":
            return ""
        idx = ord(canonical) - ord("A")
        return str(choices[idx]).strip() if 0 <= idx < len(choices) else ""

    def _semantic_answer_text(self, raw: str, letter: str) -> str:
        """Keep semantic text, replacing legacy letter-only forms with option text."""
        text = str(raw or "").strip()
        labeled = re.fullmatch(
            r"(?:answer\s*[:=-]\s*)?([A-E])\s*[\).:-]\s*(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if labeled and labeled.group(1).upper() == str(letter or "").strip().upper():
            return self._choice_text_for_letter(letter) or labeled.group(2).strip()
        if text and not re.fullmatch(
            r"(?:answer\s*[:=-]\s*)?[A-E](?:\s*[\).}]|\s*)",
            text,
            flags=re.IGNORECASE,
        ):
            return text
        return self._choice_text_for_letter(letter) or text

    def _decision_for_letter(
        self,
        letter: str,
        source: str,
        *,
        evidence: AnswerEvidenceRecord | None = None,
        answer_text: str = "",
    ) -> FinalAnswerDecision:
        """Build a semantic decision while retaining its resolved choice index."""
        canonical = self._mcq_letter_from_text(letter)
        semantic = (
            evidence.answer_text
            if evidence is not None and evidence.answer_text
            else self._semantic_answer_text(answer_text, canonical)
        )
        idx = ord(canonical) - ord("A") if canonical else None
        return FinalAnswerDecision(
            answer=semantic or str(answer_text or "").strip() or "Unknown",
            source=source,
            confidence=self._confidence_for_provenance(source),
            evidence=evidence,
            answer_text=semantic,
            choice_index=idx,
        )

    def _trusted_vlm_letter(self) -> str:
        """MCQ letter from the most recent assess that actually saw the target.

        Absence in a single frame is not an answer: mapping ``present: false`` onto
        an option (``None`` / ``No, there is none``) scored 0/7 in the trace audit
        and overrode correct four-image answers on q28 and q39.
        """
        confirmed = self._confirmed_vlm_answer_evidence()
        if confirmed is not None:
            return confirmed.letter
        if self.decision_policy == "grounded_v2":
            evidence = self._best_vlm_answer_evidence()
            return evidence.letter if evidence is not None else ""
        assess = self._last_vlm_assess or {}
        if assess.get("present"):
            letter = self._mcq_letter_from_text(str(assess.get("suggested_answer") or ""))
            if letter:
                return letter
        return self._last_positive_letter

    def _eqa_self_answer_text(self) -> str:
        """Semantic answer from the four-image EQA before post-model overrides."""
        gm = self.graph_memory
        if self.decision_policy == "grounded_v2" and gm is not None:
            parsed = getattr(gm, "last_eqa_model_parsed", None)
            model_raw_value = getattr(gm, "last_eqa_model_raw", "")
            model_raw = str(model_raw_value or "") if isinstance(model_raw_value, str) else ""
            if isinstance(parsed, tuple) and len(parsed) >= 2:
                field = str(parsed[1] or "").strip()
                if field and not should_abstain_location_mcq(
                    model_raw or field, parse_mcq_choices_from_question(self.question)
                ):
                    return field
            # Older graph memories and debug fixtures may only expose raw JSON.
            raw_candidate = model_raw or str(getattr(gm, "last_eqa_raw", "") or "")
            if raw_candidate.strip():
                try:
                    from emet.utils.json_parse import first_json_dict_lenient

                    data = first_json_dict_lenient(raw_candidate)
                except (ImportError, TypeError, ValueError):
                    data = None
                if isinstance(data, dict):
                    field = str(data.get("answer") or "").strip()
                    if field and not should_abstain_location_mcq(
                        raw_candidate, parse_mcq_choices_from_question(self.question)
                    ):
                        return field
        raw = str(getattr(gm, "last_eqa_raw", "") or "") if gm is not None else ""
        if not raw.strip():
            return ""
        head = re.split(r"\n\s*\[(?:salvage|memory-location|agentic_submit)\]", raw, maxsplit=1)[0]
        choices = parse_mcq_choices_from_question(self.question)
        m = re.search(r"(?:^|\n)\s*answer\s*:\s*([^\n]*)", head, flags=re.IGNORECASE)
        if not m:
            # Terse ``A}`` / ``A) <choice text>`` reply: no labeled fields at all.
            letter = extract_mcq_letter(head, choices or None)
            return letter or ""
        field = m.group(1).strip()
        if not field:
            return ""
        # The EQA explicitly declining ("answer: No, I did not see it") is a real
        # signal — let the ladder move on rather than forcing this text to a letter.
        if should_abstain_location_mcq(head, choices or None):
            return ""
        return field

    def _eqa_self_answer_letter(self) -> str:
        """Benchmark choice encoding derived from the EQA's semantic answer."""
        from emet.habitat.metrics import parse_mcq_choices_from_question

        field = self._eqa_self_answer_text()
        if not field:
            return ""
        letter = self._mcq_letter_from_text(field)
        if letter:
            return letter
        choices = parse_mcq_choices_from_question(self.question)
        if choices:
            idx = match_freeform_to_choice(field, choices)
            if idx is not None and 0 <= idx < 5:
                return chr(ord("A") + idx)
        return ""

    def _grounded_submit_decision(
        self,
        *,
        prefer_answer: str,
        query_answer: str,
    ) -> FinalAnswerDecision:
        """Resolve one scored answer and keep its supporting view attached.

        Image evidence outranks graph-steered multi-image EQA. The view that
        opened the ANSWER gate (or any present+answerable assess) is the scored
        channel when it exists; EQA is only a fallback when no such view exists.
        """
        from emet.habitat.metrics import parse_mcq_choices_from_question

        qa = str(query_answer or "").strip()

        confirmed = self._confirmed_vlm_answer_evidence()
        if confirmed is not None:
            prefer_letter = self._mcq_letter_from_text(prefer_answer)
            if prefer_letter and prefer_letter != confirmed.letter:
                self._append_trace(
                    {
                        "event": "answer_proposal_rejected",
                        "source": "prefer",
                        "answer": self._semantic_answer_text(prefer_answer, prefer_letter),
                        "choice_index": ord(prefer_letter) - ord("A"),
                        "reason": "conflicts with confirmed VLM answer evidence",
                        "confirmed_answer": confirmed.answer_text,
                        "confirmed_choice_index": ord(confirmed.letter) - ord("A"),
                        "confirmed_obs_id": confirmed.obs_id,
                    }
                )
            return self._decision_for_letter(
                confirmed.letter,
                "vlm_suggested",
                evidence=confirmed,
            )

        evidence = self._best_vlm_answer_evidence()
        if evidence is not None:
            return self._decision_for_letter(
                evidence.letter,
                "vlm_suggested",
                evidence=evidence,
            )

        prefer_letter = self._mcq_letter_from_text(prefer_answer)
        if prefer_letter:
            aligned = self._best_vlm_answer_evidence(letter=prefer_letter)
            if aligned is not None:
                return self._decision_for_letter(
                    prefer_letter,
                    "prefer",
                    evidence=aligned,
                )
            self._append_trace(
                {
                    "event": "answer_proposal_rejected",
                    "source": "prefer",
                    "answer": self._semantic_answer_text(prefer_answer, prefer_letter),
                    "choice_index": ord(prefer_letter) - ord("A"),
                    "reason": "no aligned present+answerable view",
                }
            )

        # No image-backed answer: fall back to the multi-image EQA native parse.
        eqa_answer_text = self._eqa_self_answer_text()
        eqa_letter = self._eqa_self_answer_letter()
        if eqa_letter:
            return self._decision_for_letter(
                eqa_letter,
                "eqa_answer",
                answer_text=eqa_answer_text or self._choice_text_for_letter(eqa_letter),
            )

        qa_letter = self._mcq_letter_from_text(qa)
        if qa_letter and not self._looks_like_coordinate_dump(qa):
            return self._decision_for_letter(
                qa_letter,
                "query",
                answer_text=qa,
            )
        if qa and not self._looks_like_coordinate_dump(qa):
            return FinalAnswerDecision(
                answer=qa,
                source="query",
                confidence=self._confidence_for_provenance("query"),
                answer_text=qa,
            )
        if not parse_mcq_choices_from_question(self.question) and prefer_answer.strip():
            return FinalAnswerDecision(
                answer=prefer_answer.strip(),
                source="prefer",
                confidence=self._confidence_for_provenance("prefer"),
                answer_text=prefer_answer.strip(),
            )
        return FinalAnswerDecision(
            answer="Unknown",
            source="query",
            confidence=0.0,
        )

    def _resolve_submit_answer_text(
        self,
        *,
        prefer_answer: str,
        query_answer: str,
    ) -> tuple[str, str]:
        """Pick the scored answer text and record which channel produced it.

        Precedence:

        1. ``vlm_suggested`` — the view that opened the confirmed ANSWER gate.
        2. ``prefer`` — explicit semantic option text when no confirmed view exists.
        3. ``eqa_answer`` — the four-image EQA's own ``Answer:`` block.
        4. An unconfirmed ``vlm_suggested`` view that still saw the target.
        5. ``query`` — ``query_answer`` prose, unless it is a nearest-furniture XYZ
           dump, which is about whatever object happened to be closest and is
           therefore not an answer at all.
        """
        from emet.habitat.metrics import parse_mcq_choices_from_question

        prefer = (prefer_answer or "").strip()
        qa = (query_answer or "").strip()

        if self.decision_policy == "grounded_v2":
            decision = self._grounded_submit_decision(
                prefer_answer=prefer,
                query_answer=qa,
            )
            self._final_answer_decision = decision
            return decision.answer, decision.source

        prefer_letter = self._mcq_letter_from_text(prefer)
        confirmed = self._confirmed_vlm_answer_evidence()
        if confirmed is not None:
            if prefer_letter and prefer_letter != confirmed.letter:
                self._append_trace(
                    {
                        "event": "answer_proposal_rejected",
                        "source": "prefer",
                        "answer": self._semantic_answer_text(prefer, prefer_letter),
                        "choice_index": ord(prefer_letter) - ord("A"),
                        "reason": "conflicts with confirmed VLM answer evidence",
                        "confirmed_answer": confirmed.answer_text,
                        "confirmed_choice_index": ord(confirmed.letter) - ord("A"),
                        "confirmed_obs_id": confirmed.obs_id,
                    }
                )
            self._final_answer_decision = self._decision_for_letter(
                confirmed.letter,
                "vlm_suggested",
                evidence=confirmed,
            )
            return self._final_answer_decision.answer, "vlm_suggested"

        if prefer_letter:
            self._final_answer_decision = self._decision_for_letter(
                prefer_letter,
                "prefer",
                answer_text=self._semantic_answer_text(prefer, prefer_letter),
            )
            return self._final_answer_decision.answer, "prefer"

        eqa_letter = self._eqa_self_answer_letter()
        if eqa_letter:
            self._final_answer_decision = self._decision_for_letter(
                eqa_letter,
                "eqa_answer",
                answer_text=self._semantic_answer_text(self._eqa_self_answer_text(), eqa_letter),
            )
            return self._final_answer_decision.answer, "eqa_answer"

        suggested_letter = self._trusted_vlm_letter()
        if suggested_letter:
            suggested_evidence = self._best_vlm_answer_evidence(letter=suggested_letter)
            self._final_answer_decision = self._decision_for_letter(
                suggested_letter,
                "vlm_suggested",
                evidence=suggested_evidence,
                answer_text=(
                    suggested_evidence.answer_text
                    if suggested_evidence is not None
                    else self._choice_text_for_letter(suggested_letter)
                ),
            )
            return self._final_answer_decision.answer, "vlm_suggested"

        # A coordinate dump is not an answer; fall through rather than let it win.
        qa_letter = self._mcq_letter_from_text(qa)
        if qa_letter and not self._looks_like_coordinate_dump(qa):
            self._final_answer_decision = self._decision_for_letter(
                qa_letter,
                "query",
                answer_text=qa,
            )
            return self._final_answer_decision.answer, "query"
        if qa and not self._looks_like_coordinate_dump(qa):
            self._final_answer_decision = FinalAnswerDecision(
                qa,
                "query",
                self._confidence_for_provenance("query"),
                answer_text=qa,
                evidence_event_ids=(self._verified_evidence_event_ids if self._verified else ()),
            )
            return qa, "query"
        if prefer:
            self._final_answer_decision = FinalAnswerDecision(
                prefer,
                "prefer",
                self._confidence_for_provenance("prefer"),
                answer_text=prefer,
                evidence_event_ids=(self._verified_evidence_event_ids if self._verified else ()),
            )
            return prefer, "prefer"
        if qa and not parse_mcq_choices_from_question(self.question):
            # Non-MCQ: the prose is all we have, so return it verbatim.
            self._final_answer_decision = FinalAnswerDecision(
                qa,
                "query",
                self._confidence_for_provenance("query"),
                answer_text=qa,
                evidence_event_ids=(self._verified_evidence_event_ids if self._verified else ()),
            )
            return qa, "query"
        self._final_answer_decision = FinalAnswerDecision("Unknown", "query", 0.0)
        return "Unknown", "query"

    def _best_evidence_obs_id(self) -> int | None:
        """Highest-signal VLM-assessed view for the final EQA Image 1.

        Rank: answerable+present (no more views needed) > present or answerable.
        Views where the VLM saw nothing are never used.
        """
        if self.decision_policy == "grounded_v2":
            grounded = self._best_vlm_answer_evidence()
            if grounded is not None:
                return grounded.obs_id
        if not self._assess_history:
            return None
        best_oid: int | None = None
        best_rank = (-1, -1)
        for oid, h in self._assess_history.items():
            present = bool(h.get("present"))
            answerable = bool(h.get("answerable"))
            need_more = bool(h.get("need_more_views"))
            rank = (int(present and answerable and not need_more), int(present or answerable))
            if rank > best_rank:
                best_rank, best_oid = rank, int(oid)
        if best_rank == (0, 0):
            return None
        return best_oid

    def _count_find_obs_ids(self) -> list[int]:
        """Graph views to attach for a count MCQ (Action / FIND nodes, not the bathroom)."""
        gm = self.graph_memory
        if gm is None:
            return []
        if not choices_are_count_mcq(parse_mcq_choices_from_question(self.question)):
            return []
        out: list[int] = []
        seen: set[int] = set()

        def _add(raw: Any) -> None:
            try:
                oid = int(raw)
            except (TypeError, ValueError):
                return
            if oid <= 0 or oid in seen:
                return
            usable = getattr(gm, "_obs_usable_for_eqa_image", None)
            if callable(usable):
                try:
                    if not usable(oid):
                        return
                except Exception:
                    pass
            seen.add(oid)
            out.append(oid)

        _add(getattr(gm, "last_eqa_look_obs_id", None))
        _add(getattr(gm, "last_eqa_action_obs_id", None))
        fn = getattr(gm, "_count_candidate_nodes", None)
        if callable(fn):
            try:
                found = fn(self.question)
            except Exception:
                found = None
            nodes = found[0] if isinstance(found, tuple) and found else ()
            if not isinstance(nodes, (list, tuple)):
                nodes = ()
            for node in nodes:
                _add(getattr(node, "obs_id", None))
        return out

    def _count_find_unattached_obs_ids(self) -> list[int]:
        """FIND views that were not in the last attached Image 1..K set."""
        attached = {int(oid) for oid in (getattr(self.graph_memory, "last_eqa_obs_ids", None) or [])}
        return [oid for oid in self._count_find_obs_ids() if oid not in attached]

    def _downgrade_unattached_count_none(self, answer: str, confidence: bool) -> bool:
        """Do not score confident None while stool/lamp FIND RGB was never attached."""
        choices = parse_mcq_choices_from_question(self.question)
        if not choices_are_count_mcq(choices):
            return confidence
        if not count_answer_is_none_or_zero(str(answer or ""), choices):
            return confidence
        missing = self._count_find_unattached_obs_ids()
        if not missing:
            return confidence
        self._pin_eqa_look_obs(missing[0])
        gm = self.graph_memory
        if gm is not None and getattr(gm, "last_eqa_action_obs_id", None) is None:
            gm.last_eqa_action_obs_id = missing[0]
        return False

    def _do_submit_answer(self, prefer_answer: str = "") -> dict[str, Any]:
        from emet.eval.dynagraph_vram import release_siglip_for_vlm

        agent = self.agent
        gm = self.graph_memory
        release_siglip_for_vlm(agent)
        discord_text = ""
        confidence = False
        relevant_images: list[Any] = []
        prefer = (prefer_answer or "").strip()
        query_ans = ""
        answer_source = "prefer"
        force_obs_ids: list[int] | None = None
        if gm is not None and hasattr(gm, "query_answer"):
            # Prefer verified observation as Image 1 (query_answer must honor force_obs_ids;
            # setting last_eqa_obs_ids alone was overwritten by diversified selection).
            confirmed_evidence = self._confirmed_vlm_answer_evidence()
            confirmed_obs_id = confirmed_evidence.obs_id if confirmed_evidence is not None else self._verified_obs_id
            if confirmed_obs_id is not None and hasattr(gm, "select_obs_ids_for_verified_answer"):
                force_obs_ids = gm.select_obs_ids_for_verified_answer(confirmed_obs_id, max_images=1)
                gm.last_eqa_obs_ids = list(force_obs_ids)
            elif self._evidence_image and hasattr(gm, "select_obs_ids_for_verified_answer"):
                # Unverified: pin the best VLM-assessed view instead of a pure
                # diversified pick — the assess already said where the evidence is.
                evidence_obs_id = self._best_evidence_obs_id()
                if evidence_obs_id is not None:
                    force_obs_ids = gm.select_obs_ids_for_verified_answer(evidence_obs_id, max_images=1)
                    gm.last_eqa_obs_ids = list(force_obs_ids)
            find_ids = self._count_find_obs_ids()
            if find_ids:
                rest = [int(oid) for oid in (force_obs_ids or []) if int(oid) not in set(find_ids)]
                force_obs_ids = find_ids + rest
                gm.last_eqa_obs_ids = list(force_obs_ids)
            # Do not clamp EMET_EQA_ANSWER_MAX_NEW_TOKENS here. A prior setdefault("64")
            # truncated Reasoning mid-stream and forced [salvage] on every bal-32 agentic
            # answer; the budget belongs to eqa_vl/answer_max_new_tokens so it can be tuned
            # per VLM.
            xyt = self._robot_xyt_world()
            planner = getattr(agent, "planner", None)
            try:
                (
                    _reasoning,
                    ans,
                    confidence,
                    _cr,
                    _tp,
                    relevant_images,
                ) = gm.query_answer(
                    self.question,
                    xyt,
                    planner,
                    force_obs_ids=force_obs_ids,
                )
                query_ans = (ans or "").strip()
            except Exception as e:
                discord_text = f"Answer:Unknown\nEQA failed: {e}"
                query_ans = "Unknown"
                confidence = False
            answer, answer_source = self._resolve_submit_answer_text(
                prefer_answer=prefer,
                query_answer=query_ans,
            )
            # Semantic answer from VLM assess / tool arg is the decision we want to score; do not
            # inherit False confidence from a coordinate-dump query_answer path.
            if answer_source in ("prefer", "vlm_suggested") and self._mcq_letter_from_text(answer):
                confidence = bool(self._verified) or bool(confidence)
            confidence = self._downgrade_unattached_count_none(answer, bool(confidence))
            discord_text = f"Answer:{answer}\nConfidence:{confidence}\n[submit_source:{answer_source}]"
            if query_ans and query_ans != answer:
                discord_text += f"\n[query_answer:{query_ans[:160]}]"
        elif prefer:
            answer, answer_source = self._resolve_submit_answer_text(
                prefer_answer=prefer,
                query_answer="",
            )
            discord_text = f"Answer:{answer}\n[submit_source:{answer_source}]"
            confidence = bool(self._verified)
        else:
            discord_text = "Answer:Unknown\nNo graph memory"
            answer = "Unknown"
            answer_source = "query"
        confidence = self._downgrade_unattached_count_none(str(answer or ""), bool(confidence))
        self._append_trace(
            {
                "tool": "submit_answer",
                "final_answer": answer,
                "confidence": bool(confidence),
                "verified": self._verified,
                "verified_obs_id": self._verified_obs_id,
                "answer_source": answer_source,
                "query_answer": query_ans or None,
                "prefer_answer": prefer or None,
                "vlm_suggested": (
                    None if self._last_vlm_assess is None else self._last_vlm_assess.get("suggested_answer")
                ),
                "answerable": self._evidence_policy.state == AgenticState.ANSWER,
                "answerability_probability": (
                    self._evidence_policy.beliefs[self._evidence_policy.active_hypothesis_id].answerability_probability
                    if self._evidence_policy.active_hypothesis_id in self._evidence_policy.beliefs
                    else None
                ),
                "force_obs_ids": list(force_obs_ids) if force_obs_ids else None,
                "last_eqa_obs_ids": list(getattr(gm, "last_eqa_obs_ids", []) or []) if gm is not None else None,
                "spatial_rag": getattr(gm, "last_eqa_spatial_rag", None) if gm is not None else None,
                "final_decision": (
                    self._final_answer_decision.to_dict() if self._final_answer_decision is not None else None
                ),
            }
        )
        return {
            "ok": True,
            "answer": answer,
            "answer_source": answer_source,
            "discord_text": discord_text,
            "confidence": bool(confidence),
            "relevant_images": relevant_images,
            "final_decision": (
                self._final_answer_decision.to_dict() if self._final_answer_decision is not None else None
            ),
        }

    def _answer_unknownish(self, answer: Any) -> bool:
        from emet.habitat.metrics import parse_mcq_choices_from_question

        return answer_is_unknownish(
            str(answer or ""),
            parse_mcq_choices_from_question(self.question),
        )

    def _finalize_unknown_location_letter(self, submit_out: dict[str, Any]) -> dict[str, Any]:
        """Keep scored Unknown; optionally log a salvage counterfactual letter.

        Scored answer stays honest Unknown/empty (no ``final_location_salvage``
        lottery). When the question is a location MCQ and images are available,
        still call ``_salvage_location_mcq_letter`` once and record
        ``final_location_salvage_counterfactual`` so summaries can report both
        no-salvage and with-salvage accuracies.
        """
        self._salvage_counterfactual_letter = ""
        if self.graph_memory is not None:
            self.graph_memory.last_salvage_counterfactual_letter = ""
        if self.mode != "answer" or not self._answer_unknownish(submit_out.get("answer")):
            return submit_out

        prior = submit_out.get("answer")
        self._append_trace(
            {
                "event": "final_location_salvage_skipped",
                "reason": "agentic_no_salvage",
                "prior_answer": prior,
            }
        )

        gm = self.graph_memory
        if gm is None or not hasattr(gm, "_salvage_location_mcq_letter"):
            return submit_out

        from emet.habitat.metrics import (
            choices_are_location_mcq,
            parse_mcq_choices_from_question,
            question_is_attribute_state,
        )

        choices = parse_mcq_choices_from_question(self.question)
        if not choices or question_is_attribute_state(self.question) or not choices_are_location_mcq(choices):
            return submit_out

        images = list(submit_out.get("relevant_images") or [])
        if not images:
            for attr in ("last_eqa_images", "last_relevant_images"):
                cand = getattr(gm, attr, None)
                if cand:
                    images = list(cand)
                    break
        if not images:
            return submit_out

        letter = str(gm._salvage_location_mcq_letter(self.question, choices, images) or "").strip()
        if not letter:
            return submit_out

        self._salvage_counterfactual_letter = letter
        if gm is not None:
            gm.last_salvage_counterfactual_letter = letter
        self._append_trace(
            {
                "event": "final_location_salvage_counterfactual",
                "letter": letter,
                "prior_answer": prior,
                "n_unknown_explore": self._n_unknown_explore,
                "n_images": len(images),
                "applied": False,
            }
        )
        return submit_out

    def _maybe_follow_eqa_explore_action(self, submit_out: dict[str, Any]) -> bool:
        """Navigate to EQA ``Action: N`` when submit returned an ungrounded guess.

        Location MCQs often answer Unknown with an image index to explore. Inventing
        a salvage letter (holdout q104/q105) is worse than following that action.
        Count MCQs often guess ``One`` from the wrong RGB while GRAPH_COUNT points
        at a different obs id — follow that Action even when the text is a number.
        Allows one soft-over-budget nav so Action:N still runs after explore used
        the nominal ``max_nav_steps``.

        When Action:N is missing or out of range for the prompt image list (q105:
        ``Action:2`` with only one image), or the action target was already followed
        and the model is still Unknown, fall back to ``explore_frontier`` a few times
        instead of locking an empty letter.
        """
        if self.mode != "answer" or self.decision_policy == "grounded_v2":
            return False
        gm = self.graph_memory
        if gm is None:
            return False
        conf = bool(submit_out.get("confidence"))
        unknownish = self._answer_unknownish(submit_out.get("answer"))
        count_mcq = choices_are_count_mcq(parse_mcq_choices_from_question(self.question))
        missing_find = self._count_find_unattached_obs_ids() if count_mcq else []
        none_without_find = bool(
            missing_find
            and count_answer_is_none_or_zero(
                str(submit_out.get("answer") or ""),
                parse_mcq_choices_from_question(self.question),
            )
        )
        if none_without_find:
            conf = False
            submit_out["confidence"] = False
            if getattr(gm, "last_eqa_action_obs_id", None) is None:
                gm.last_eqa_action_obs_id = missing_find[0]
            self._pin_eqa_look_obs(missing_find[0])
        if conf and not unknownish:
            return False
        # Unconfident count + Action:N: the integer is from the attached (wrong) RGB.
        if not unknownish and not (count_mcq and not conf):
            return False
        obs_id = getattr(gm, "last_eqa_action_obs_id", None)
        if obs_id is not None:
            oid = int(obs_id)
            spent_fn = getattr(gm, "eqa_obs_look_spent", None)
            spent_look = bool(callable(spent_fn) and spent_fn(oid) is True)
            if spent_look:
                nxt_fn = getattr(gm, "next_unspent_eqa_obs_id", None)
                alt = nxt_fn(missing_find, skip={oid}) if count_mcq and missing_find and callable(nxt_fn) else None
                self._append_trace(
                    {
                        "event": "skip_spent_eqa_action",
                        "obs_id": oid,
                        "alt_obs_id": alt,
                        "prior_answer": submit_out.get("answer"),
                    }
                )
                if alt is not None and int(alt) not in self._followed_eqa_actions:
                    oid = int(alt)
                    spent_look = False
                else:
                    gm.last_eqa_action_obs_id = None
            if not spent_look and oid not in self._followed_eqa_actions:
                # Soft +1 budget so Action:N is not starved by prior explore_frontier calls.
                if self._n_nav + self._n_explore >= self.max_nav_steps + 1:
                    return False
                self._followed_eqa_actions.add(oid)
                gm.last_eqa_action_obs_id = None
                # Next query_answer must attach this RGB even if verify pins another view.
                self._pin_eqa_look_obs(oid)
                # Force re-verify at the action target before the next submit.
                self._verified = False
                self._verified_obs_id = None
                self._last_verify = None
                # Temporarily raise budget so navigate_to_obs accepts the Action follow.
                old_budget = self.max_nav_steps
                self.max_nav_steps = max(old_budget, self._n_nav + self._n_explore + 1)
                try:
                    nav = self.handle_tool("navigate_to_obs", {"obs_id": oid})
                    # Verify the post-nav capture, not the historical hyp obs_id.
                    self.handle_tool("verify_siglip", {})
                finally:
                    self.max_nav_steps = old_budget
                self._append_trace(
                    {
                        "event": "follow_eqa_action",
                        "obs_id": oid,
                        "nav_ok": bool(nav.get("ok")),
                        "prior_answer": submit_out.get("answer"),
                    }
                )
                return True
        # No resolvable Action:N, or already followed that obs and still Unknown.
        # Do not soft-explore on an unconfident count number — that burns budget
        # after a One guess. Unknown location still explores.
        # Cap soft explores so we do not loop forever on location MCQs.
        # Soft +2 beyond max_nav_steps: Action follow may already have used +1.
        if not unknownish:
            return False
        if self._n_unknown_explore >= 2:
            return False
        if self._n_nav + self._n_explore >= self.max_nav_steps + 2:
            return False
        self._n_unknown_explore += 1
        gm.last_eqa_action_obs_id = None
        self._verified = False
        self._verified_obs_id = None
        self._last_verify = None
        old_budget = self.max_nav_steps
        self.max_nav_steps = max(old_budget, self._n_nav + self._n_explore + 1)
        try:
            nav = self.handle_tool("explore_frontier", {})
            self.handle_tool("verify_siglip", {})
        finally:
            self.max_nav_steps = old_budget
        self._append_trace(
            {
                "event": "follow_unknown_explore",
                "nav_ok": bool(nav.get("ok")),
                "prior_answer": submit_out.get("answer"),
                "n_unknown_explore": self._n_unknown_explore,
            }
        )
        return True

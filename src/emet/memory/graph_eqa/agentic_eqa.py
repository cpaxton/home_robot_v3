# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unified agentic GraphEQA loop: explore / navigate / verify / answer with tools."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from emet.agent.prompt import parse_tool_calls_response
from emet.habitat.metrics import (
    choices_are_count_mcq,
    extract_mcq_letter,
    parse_mcq_choices_from_question,
    should_abstain_location_mcq,
)
from emet.memory.graph_eqa.action_history import (
    ActionHistoryEntry,
    ActionSignature,
    ActionTarget,
    GateDecision,
    ProgressToken,
    decide_candidate,
    quantized_xy,
    render_history_entry,
    resolve_action_progress_mode,
    stable_digest,
    status_outcome_class,
)
from emet.memory.graph_eqa.agentic_policy import (
    AgenticState,
    EvidencePolicy,
    EvidenceRecord,
)
from emet.memory.graph_eqa.agentic_tools import (
    build_agentic_eqa_tools,
    build_graph_eqa_system_prompt,
    build_state_message,
    coerce_room_label,
    normalize_current_room,
)
from emet.memory.graph_eqa.graph_memory import (
    _QUESTION_VERB_FILLERS,
    SIGLIP_PRESENT_THRESHOLD,
    NavHypothesis,
    VerifyResult,
    label_matches_relevant_object,
    question_stem_for_keywords,
)
from emet.memory.graph_eqa.mcq_debias import (
    answer_is_unknownish,
    match_freeform_to_choice,
    valid_choice_indices,
)
from emet.memory.graph_eqa.room_clusters import (
    merge_room_estimates,
    question_target_rooms,
    resolve_investigate_room_stamp,
    resolve_room_policy,
    room_leave_needed,
    room_mismatches_question,
)
from emet.utils.logger import Logger

_logger = Logger(__name__)

# Image-space SigLIP is a **high-recall / high-FP** proposal (see agentic_scale.md).
# Three bands on Habitat RGB (offline calib: real hits cluster ~0.10–0.14):
#   >= PRESENT (0.12)  → PRESENT
#   >= ABSENT  (0.10)  → CANDIDATE
#   <  ABSENT  (0.10)  → ABSENT   (true-negative for *this* view — move on)
# Do not treat ABSENT as proof the object is gone from the scene.
# SigLIP ranks WHERE to navigate next to grow the graph; Qwen vlm_assess on
# pixels decides answerability (detector scores are not fed into that prompt).
SIGLIP_IMAGE_PRESENT_THRESHOLD = 0.12
SIGLIP_IMAGE_ABSENT_THRESHOLD = 0.10

# query_answer sometimes echoes graph XYZ ("The fan is at approximately (x,y,z) m")
# instead of an MCQ letter — keep Qwen's letter when that happens.
_COORD_DUMP_RE = re.compile(
    r"approximately\s*\([^)]+\)\s*m|\bat approximately\b.*\bm\b",
    re.IGNORECASE | re.DOTALL,
)

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})
_DECISION_POLICIES = frozenset({"legacy", "grounded_v2"})

# Region escape: after this many consecutive "target not visible" view assessments,
# require the next frontier to be at least ESCAPE_MIN_TRAVEL_M away so the robot
# leaves the area instead of re-scanning it (holdout q104/q105 circled their spawn).
NOT_PRESENT_ESCAPE_STREAK = 2
ESCAPE_MIN_TRAVEL_M = 3.0

# Confidence prior per answer channel, ordered by reliability in the 2026-07 trace
# audit (docs/experiments/agentic_eqa_trace_audit.md). Priors, not fitted values.
_PROVENANCE_CONFIDENCE = {
    "prefer": 0.70,
    "eqa_answer": 0.65,
    "vlm_suggested": 0.50,
    "query": 0.45,
    "pending_letter": 0.35,
    "uniform_prior": 0.25,
}

# Same hyp obs_id navigated this many times without a fresh graph obs → stall / break loop.
# Multi-view investigate samples up to PLACE_APPROACH_SAMPLES distinct bearings first.
NAV_SAME_OBS_LOOP_LIMIT = 2

# Distinct planar approach samples around a place card (re-investigate = next bearing).
PLACE_APPROACH_SAMPLES = 4

# Consecutive planner misses on investigate navigation → treat every remaining
# candidate as unreachable and stop retrying them (forces candidate switch /
# answer-from-graph instead of a sample_nav_failed loop).
NAV_CONSECUTIVE_FAIL_LIMIT = 2

# After this many successful explore_frontier calls in a row, prefer an untried
# investigate hyp over another frontier (stops leave/ABSENT explore-only loops).
EXPLORE_STREAK_FORCE_INVESTIGATE = 2
# OVMM find_recep: prefer investigate when a place card is this close (teleport sim).
OVMM_NEAR_INVESTIGATE_M = 3.5

# Question cues that force a close-look preference (time/state/count/detail).
_CLOSE_LOOK_CUES = (
    "what time",
    "time is it",
    "o'clock",
    "time of day",
    "what hour",
    "clock",
    "how many",
    "how much",
    "number of",
    " on or off",
    " open or closed",
    "is the ",
    "turned on",
    "turned off",
    "set to",
    "what color",
    "what colour",
    "what brand",
    "what does it say",
    "what is written",
    "read the ",
)


def question_requires_close_look_keywords(question: str) -> bool:
    """Cheap heuristic: does answering need a close look (clock/count/state/detail)?

    Used as the pre-VLM shortcut and the no-VLM fallback for the close-look flag;
    the VLM classifier (``extract_target_from_question``) is the authoritative path.
    """
    q = str(question or "").strip().lower()
    if not q:
        return False
    return any(cue in q for cue in _CLOSE_LOOK_CUES)


# Hyp recall: how many evidence cards to show the router / walk in fallback.
DEFAULT_HYP_RECALL_K = 6

# Investigate vs explore: place-card sources worth a closer look.
INVESTIGATE_SOURCES = frozenset({"graph", "confirmed", "siglip"})
PLACE_INSPECT_RECENT_K = 3
# Compact tool outcomes shown to the VLM router (avoid re-picking stuck loops).
RECENT_ACTIONS_K = 6

# Routing turns are text-only JSON; a two-call reply with arguments needs more than 64 tokens.
ROUTER_MAX_NEW_TOKENS = 128


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
        return f"investigated={self.investigate_count} closest={closest} {close} {ap} {cov}{more} recent: {recent}"


def env_eqa_agentic_verify() -> bool | None:
    v = os.environ.get("EMET_EQA_AGENTIC_VERIFY", "").strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None


def env_eqa_agentic_router() -> bool | None:
    v = os.environ.get("EMET_EQA_AGENTIC_ROUTER", "").strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None


def env_eqa_room_policy() -> str | None:
    """Override ``eqa.room_policy``: ``canonical`` or ``llm``."""
    v = os.environ.get("EMET_EQA_ROOM_POLICY", "").strip().lower()
    if not v:
        return None
    return resolve_room_policy(v)


def resolve_agentic_decision_policy(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    return value if value in _DECISION_POLICIES else "legacy"


def env_eqa_agentic_decision_policy() -> str | None:
    value = os.environ.get("EMET_EQA_AGENTIC_DECISION_POLICY", "").strip().lower()
    if not value:
        return None
    return resolve_agentic_decision_policy(value)


def env_eqa_router_room_images() -> int:
    """Nearby object images for multimodal router (0 = text-only; default 3)."""
    raw = os.environ.get("EMET_EQA_ROUTER_ROOM_IMAGES", "").strip()
    if not raw:
        return 3
    try:
        return max(0, int(raw))
    except ValueError:
        return 3


def env_eqa_hyp_recall_k() -> int:
    """Top-K evidence cards for agentic hyp recall (default 6)."""
    raw = os.environ.get("EMET_EQA_HYP_RECALL_K", "").strip()
    if not raw:
        return DEFAULT_HYP_RECALL_K
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_HYP_RECALL_K


def _env_positive_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value}")
    return value


def env_eqa_collect_trace() -> bool:
    v = os.environ.get("EMET_EQA_TRACE", "").strip().lower()
    return v in _TRUE


def env_eqa_force_answer() -> bool:
    """Emit a best-guess letter instead of ``Unknown`` at exhaustion (default on).

    ``EMET_EQA_FORCE_ANSWER=0`` restores the legacy abstain for A/B comparison.
    Forced answers still carry ``answer_provenance`` and a calibrated confidence,
    so downstream consumers can present "I think X, but I'm unsure".
    """
    v = os.environ.get("EMET_EQA_FORCE_ANSWER", "").strip().lower()
    if v in _FALSE:
        return False
    return True


def env_eqa_agentic_require_verified() -> bool | None:
    """When True, refuse unverified ``submit_answer`` (incl. fallback / budget exhaust).

    Env: ``EMET_EQA_AGENTIC_REQUIRE_VERIFIED=1``. Unverified exhaust falls to the
    forced-answer ladder (or, with ``EMET_EQA_FORCE_ANSWER=0``, to ``Unknown``).
    """
    v = os.environ.get("EMET_EQA_AGENTIC_REQUIRE_VERIFIED", "").strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None


def env_eqa_room_stamp_investigate() -> bool:
    """Stamp room clusters after investigate (default **off**).

    ``EMET_EQA_ROOM_STAMP_INVESTIGATE=1`` enables close-look stamps. Default off:
    stamps regressed HM-EQA accuracy vs explore-streak-only; keep for A/B.
    Also ``eqa.room_stamp_investigate``.
    """
    v = os.environ.get("EMET_EQA_ROOM_STAMP_INVESTIGATE", "").strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return False


def env_eqa_answerable_confirm() -> bool:
    """Hybrid confirm before unlock (default on). ``EMET_EQA_ANSWERABLE_CONFIRM=0`` disables."""
    v = os.environ.get("EMET_EQA_ANSWERABLE_CONFIRM", "").strip().lower()
    if v in _FALSE:
        return False
    if v in _TRUE:
        return True
    return True


def env_eqa_agentic_mcq_debias() -> bool | None:
    """Debias unverified forced answers (default on).

    ``EMET_EQA_AGENTIC_MCQ_DEBIAS=0`` restores the raw EQA letter for forced
    budget-exhaustion answers (letter-position bias returns).
    """
    v = os.environ.get("EMET_EQA_AGENTIC_MCQ_DEBIAS", "").strip().lower()
    if v in _FALSE:
        return False
    if v in _TRUE:
        return True
    return None


def env_eqa_agentic_close_look() -> bool | None:
    """Ask whether the question needs a close look before the loop (default on).

    ``EMET_EQA_AGENTIC_CLOSE_LOOK=0`` disables the keyword/VLM classifier and the
    close-look redirect (time/state/count questions may explore forever again).
    """
    v = os.environ.get("EMET_EQA_AGENTIC_CLOSE_LOOK", "").strip().lower()
    if v in _FALSE:
        return False
    if v in _TRUE:
        return True
    return None


def env_eqa_agentic_no_early_unverified() -> bool | None:
    """Hold unverified auto-submits while budget remains (default on).

    ``EMET_EQA_AGENTIC_NO_EARLY_UNVERIFIED=0`` restores early auto-submit on a
    bare ``answerable`` state even when the evidence was never corroborated.
    """
    v = os.environ.get("EMET_EQA_AGENTIC_NO_EARLY_UNVERIFIED", "").strip().lower()
    if v in _FALSE:
        return False
    if v in _TRUE:
        return True
    return None


def env_eqa_agentic_single_view_confirm() -> bool | None:
    """Confirm on one present+answerable view (default on).

    ``EMET_EQA_AGENTIC_SINGLE_VIEW_CONFIRM=0`` restores the phrase-token / two-view
    corroboration gate (verification rate drops, so more forced guesses).
    """
    v = os.environ.get("EMET_EQA_AGENTIC_SINGLE_VIEW_CONFIRM", "").strip().lower()
    if v in _FALSE:
        return False
    if v in _TRUE:
        return True
    return None


def env_eqa_agentic_evidence_image() -> bool | None:
    """Pin the best VLM-assessed view as EQA Image 1 (default on).

    ``EMET_EQA_AGENTIC_EVIDENCE_IMAGE=0`` restores pure diversified image
    selection for unverified final answers.
    """
    v = os.environ.get("EMET_EQA_AGENTIC_EVIDENCE_IMAGE", "").strip().lower()
    if v in _FALSE:
        return False
    if v in _TRUE:
        return True
    return None


def _eqa_cfg(agent: Any) -> dict[str, Any]:
    params = getattr(agent, "parameters", None) or {}
    if hasattr(params, "get"):
        raw = params.get("eqa", {}) or {}
    else:
        raw = {}
    return dict(raw) if isinstance(raw, dict) else {}


def agentic_verify_enabled(agent: Any) -> bool:
    env = env_eqa_agentic_verify()
    if env is not None:
        return env
    return bool(_eqa_cfg(agent).get("agentic_verify", False))


def _feat_list(arr: np.ndarray | None) -> list[float] | None:
    if arr is None:
        return None
    return [float(x) for x in np.asarray(arr, dtype=np.float32).reshape(-1).tolist()]


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


class AgenticEQAExecutor:
    """Bounded tool loop for post-explore / world-change EQA."""

    def __init__(
        self,
        agent: Any,
        question: str | None,
        *,
        goal: str = "",
        max_rounds: int = 8,
        max_nav_steps: int = 8,
        verify_min_sim: float = SIGLIP_IMAGE_PRESENT_THRESHOLD,
        trace_path: Path | str | None = None,
        trace_meta: dict[str, Any] | None = None,
        collect_trace: bool | None = None,
        router: bool | None = None,
        require_verified: bool | None = None,
        mcq_debias: bool | None = None,
        close_look: bool | None = None,
        no_early_unverified: bool | None = None,
        single_view_confirm: bool | None = None,
        evidence_image: bool | None = None,
    ):
        self.agent = agent
        self.mode = "answer" if question else "explore"
        self.question = question or ""
        self.goal = goal or "explore the environment and update the map"
        self.max_rounds = max(1, int(max_rounds))
        self.max_nav_steps = max(0, int(max_nav_steps))
        self.verify_min_sim = float(verify_min_sim)
        self._verified = False
        self._verified_obs_id: int | None = None
        self._last_verify: VerifyResult | None = None
        self._hypotheses: list[NavHypothesis] = []
        self._hyp_i = 0
        self._n_nav = 0
        self._n_explore = 0
        # Consecutive investigate nav misses; >= NAV_CONSECUTIVE_FAIL_LIMIT blocks
        # every remaining candidate (prevents sample_nav_failed loops).
        self._consecutive_nav_fail = 0
        self._unreachable_obs_ids: set[int] = set()
        self._tool_log: list[str] = []
        # Compatibility rendering cache; structured entries are the source of truth.
        self._recent_actions: list[str] = []
        self._action_history: list[ActionHistoryEntry] = []
        self._tool_dispatch_depth = 0
        self._action_selected_by = "internal"
        self._last_action_gate_decisions: list[Any] = []
        self._trace_rows: list[dict[str, Any]] = []
        # obs_id → successful navigate/investigate attempts (loop detection for the router).
        self._nav_to_obs_counts: dict[int, int] = {}
        self._nav_loop_flags: list[dict[str, Any]] = []
        # Per place-card investigate history for this question (count / closest / recent).
        self._place_inspect: dict[int, PlaceInspectRecord] = {}
        # Capture stations from investigate — not place cards (avoids patio station-chase).
        self._station_obs_ids: set[int] = set()
        # After VLM assess_present=False, prefer one explore before the next investigate.
        self._prefer_explore: bool = False
        self._prefer_explore_reason: str = ""
        # Successful explore_frontier streak; reset on investigate (see EXPLORE_STREAK_*).
        self._n_consecutive_explore: int = 0
        # Soft answerable waiting for phrase hit / second agreeing view.
        self._pending_answerable: dict[str, Any] | None = None
        self._answerable_confirm = env_eqa_answerable_confirm()
        # Last view assess that actually saw the target. A ``present: false`` assess
        # reports what is missing from one frame and must never supply the scored
        # letter (see docs/experiments/agentic_eqa_trace_audit.md).
        self._last_positive_letter: str = ""
        self._last_positive_obs_id: int | None = None
        self._answer_evidence: list[AnswerEvidenceRecord] = []
        # Episode-level answer evidence survives a later coverage motion. The
        # current EvidencePolicy hypothesis may reset, but a confirmed view must
        # not be silently replaced by an unverified budget guess.
        self._confirmed_answer_evidence: AnswerEvidenceRecord | None = None
        self._final_answer_decision: FinalAnswerDecision | None = None
        self._verified_evidence_event_ids: tuple[str, ...] = ()
        # How the final letter was obtained when the episode could not verify.
        self._answer_provenance: str = ""
        self._force_answer = env_eqa_force_answer()
        self._last_room_estimate: str = "unknown"
        self._last_router_room_estimate: str = "unknown"
        self._graph_room_estimate: str = "unknown"
        self._current_room_source: str = "unknown"
        self._room_estimate_stale: bool = True
        self._graph_room_stale: bool = True
        self._router_room_stale: bool = True
        self._room_pose_round: int | None = None
        self._room_world_step: int | None = None
        self._room_estimates: list[str] = []
        self._last_router_n_images: int = 0
        self._last_router_ms: float | None = None
        self._last_capture_status: str | None = None
        env_policy = env_eqa_room_policy()
        cfg_policy = _eqa_cfg(agent).get("room_policy", "canonical")
        self.room_policy = resolve_room_policy(env_policy if env_policy is not None else cfg_policy)
        env_decision = env_eqa_agentic_decision_policy()
        cfg_decision = _eqa_cfg(agent).get("agentic_decision_policy", "legacy")
        self.decision_policy = resolve_agentic_decision_policy(
            env_decision if env_decision is not None else cfg_decision
        )
        eqa_cfg = _eqa_cfg(agent)
        graph_mode = os.environ.get("EMET_EQA_GRAPH_EVIDENCE_MODE", "") or eqa_cfg.get("graph_evidence_mode", "off")
        history_mode = os.environ.get("EMET_EQA_ROOM_HISTORY_MODE", "") or eqa_cfg.get("room_history_mode", "off")
        attempt_mode = os.environ.get("EMET_EQA_ATTEMPT_LEDGER_MODE", "") or eqa_cfg.get("attempt_ledger_mode", "off")
        action_progress_mode = os.environ.get("EMET_EQA_ACTION_PROGRESS_MODE", "") or eqa_cfg.get(
            "action_progress_mode", "off"
        )
        self.graph_evidence_mode = str(graph_mode).strip().lower()
        self.room_history_mode = str(history_mode).strip().lower()
        self.attempt_ledger_mode = str(attempt_mode).strip().lower()
        self.action_progress_mode = resolve_action_progress_mode(action_progress_mode)
        if self.graph_evidence_mode not in {"off", "shadow", "agent"}:
            raise ValueError(f"invalid graph_evidence_mode: {graph_mode!r}")
        if self.room_history_mode not in {"off", "shadow", "agent"}:
            raise ValueError(f"invalid room_history_mode: {history_mode!r}")
        if self.attempt_ledger_mode not in {"off", "shadow", "agent"}:
            raise ValueError(f"invalid attempt_ledger_mode: {attempt_mode!r}")
        if str(action_progress_mode).strip().lower() not in {"off", "shadow", "enforce"}:
            raise ValueError(f"invalid action_progress_mode: {action_progress_mode!r}")
        if self.action_progress_mode != "off" and self.decision_policy != "grounded_v2":
            raise ValueError("action_progress_mode shadow/enforce requires agentic_decision_policy=grounded_v2")
        hints_env = os.environ.get("EMET_EQA_ROOM_TARGET_HINTS", "").strip().lower()
        if hints_env in _TRUE:
            self.room_target_hints = True
        elif hints_env in _FALSE:
            self.room_target_hints = False
        else:
            self.room_target_hints = bool(eqa_cfg.get("room_target_hints", True))
        self.agent_state_max_chars = max(1000, int(eqa_cfg.get("agent_state_max_chars", 6000)))
        self._last_agent_state_snapshot: Any | None = None
        self._router_call_seq = 0
        self._last_rendered_action_allowlist: dict[str, tuple[Any, ...]] = {
            "place_ids": (),
            "place_obs_ids": (),
            "frontier_ids": (),
            "event_ids": (),
        }
        self._router_action_allowlists: dict[str, dict[str, tuple[Any, ...]]] = {}
        self._router_path_world: list[list[float]] = []
        self._router_path_m = 0.0
        cfg_stamp = _eqa_cfg(agent).get("room_stamp_investigate", None)
        if os.environ.get("EMET_EQA_ROOM_STAMP_INVESTIGATE", "").strip():
            self._room_stamp_investigate = env_eqa_room_stamp_investigate()
        elif cfg_stamp is not None:
            self._room_stamp_investigate = bool(cfg_stamp)
        else:
            self._room_stamp_investigate = False
        self._in_target_area: bool | None = None
        self._collect_trace = (
            bool(collect_trace)
            if collect_trace is not None
            else (env_eqa_collect_trace() or bool(_eqa_cfg(agent).get("collect_agentic_trace", False)))
        )
        self._trace_path = Path(trace_path) if trace_path else None
        self._trace_meta = dict(trace_meta or {})
        raw_question_id = self._trace_meta.get("question_id")
        if raw_question_id is None:
            raw_question_id = self._trace_meta.get("qid")
        self._question_id = str(
            raw_question_id
            if raw_question_id is not None
            else hashlib.sha1(self.question.encode("utf-8")).hexdigest()[:12]
        )
        raw_session_id = self._trace_meta.get("session_id")
        if raw_session_id is None:
            raw_session_id = self._trace_meta.get("episode_id")
        if raw_session_id is None:
            raw_session_id = self._trace_meta.get("run_id")
        if raw_session_id is None:
            raw_session_id = getattr(agent, "_episode_debug_dir", None)
        self._session_id = str(raw_session_id or f"session:{self._question_id}")
        self._trace_meta.setdefault("question_id", self._question_id)
        self._trace_meta.setdefault("session_id", self._session_id)
        gm_context = self.graph_memory
        bind_context = getattr(gm_context, "bind_episode_context", None) if gm_context is not None else None
        if callable(bind_context):
            bind_context(question_id=self._question_id, session_id=self._session_id)
        self._gt_placements: dict[str, Any] | None = None
        self._round = 0
        self._tried: dict[int, str] = {}  # obs_id → last verify summary (never re-verify)
        self._followed_eqa_actions: set[int] = set()
        # Soft explores after Unknown when Action:N is missing/OOB or already followed.
        self._n_unknown_explore = 0
        # Counterfactual salvage letter (logged, never applied to scored answer).
        self._salvage_counterfactual_letter = ""
        # Obs ids freshly produced by capture_and_update this turn (eligible for one verify).
        self._fresh_obs_ids: set[int] = set()
        self._vlm_assessed_obs_ids: set[int] = set()
        self._target_phrase: str = ""
        self._question_type: str = "other"
        self._last_vlm_assess: dict[str, Any] | None = None
        self._not_present_streak = 0
        self._frontier_pick_waypoints: list[tuple[float, float]] = []
        self._frontier_pick_dir: Path | None = None
        self._evidence_policy = EvidencePolicy()
        self._presence_detector: Any | None = None
        self._presence_detector_initialized = False
        env_router = env_eqa_agentic_router()
        cfg_router = _eqa_cfg(agent).get("agentic_vlm_router", True)
        self._router_enabled = bool(
            router if router is not None else (env_router if env_router is not None else cfg_router)
        )
        env_req = env_eqa_agentic_require_verified()
        cfg_req = _eqa_cfg(agent).get("agentic_require_verified", False)
        self._require_verified = bool(
            require_verified if require_verified is not None else (env_req if env_req is not None else cfg_req)
        )
        env_debias = env_eqa_agentic_mcq_debias()
        cfg_debias = _eqa_cfg(agent).get("agentic_mcq_debias", True)
        self._mcq_debias = bool(
            mcq_debias if mcq_debias is not None else (env_debias if env_debias is not None else cfg_debias)
        )
        env_close_look = env_eqa_agentic_close_look()
        cfg_close_look = _eqa_cfg(agent).get("agentic_close_look", True)
        self._close_look = bool(
            close_look if close_look is not None else (env_close_look if env_close_look is not None else cfg_close_look)
        )
        env_no_early = env_eqa_agentic_no_early_unverified()
        cfg_no_early = _eqa_cfg(agent).get("agentic_no_early_unverified", True)
        self._no_early_unverified = bool(
            no_early_unverified
            if no_early_unverified is not None
            else (env_no_early if env_no_early is not None else cfg_no_early)
        )
        env_svc = env_eqa_agentic_single_view_confirm()
        cfg_svc = _eqa_cfg(agent).get("agentic_single_view_confirm", True)
        self._single_view_confirm = bool(
            single_view_confirm if single_view_confirm is not None else (env_svc if env_svc is not None else cfg_svc)
        )
        env_ev = env_eqa_agentic_evidence_image()
        cfg_ev = _eqa_cfg(agent).get("agentic_evidence_image", True)
        self._evidence_image = bool(
            evidence_image if evidence_image is not None else (env_ev if env_ev is not None else cfg_ev)
        )
        self._assess_history: dict[int, dict[str, Any]] = {}
        self._close_look_required = False
        self._close_look_source = "disabled"
        self._tools: list[Any] | None = None
        self._tool_names: set[str] = set()
        self._system_prompt: str = ""

    @property
    def graph_memory(self) -> Any:
        return getattr(self.agent, "graph_memory", None)

    @property
    def query_text(self) -> str:
        """Phrase used to bias graph inspection / frontier picks (question or explore goal)."""
        return self.question or self.goal

    def _robot_xyt(self) -> np.ndarray | None:
        robot = getattr(self.agent, "robot", None)
        if robot is None or not hasattr(robot, "get_base_pose"):
            return None
        try:
            pose = np.asarray(robot.get_base_pose(), dtype=float).reshape(-1)
        except Exception:
            return None
        if pose.size < 2 or not np.isfinite(pose[:2]).all():
            return None
        return pose

    def _robot_xyt_world(self) -> np.ndarray | None:
        """Robot base ``(x, y, θ)`` in the voxel-map / world frame for A* planning.

        ``get_base_pose`` is episode-relative (ZMQ gps/compass), but the voxel map
        and ``navigate_to_target_pose`` plan in the world frame anchored at
        ``navigation_origin_xyt``. For sims whose spawn is not at world (0,0)
        (robocasa origin ≈ (2.9,-1.7)) planning from the raw episode pose puts the
        A* start at grid center / an unexplored cell → "non navigable point".
        """
        local = self._robot_xyt()
        if local is None:
            return None
        agent = self.agent
        convert = getattr(agent, "_planning_base_xyt", None)
        if callable(convert):
            try:
                world = np.asarray(convert(local), dtype=float).reshape(-1)
                if world.size >= 2 and np.isfinite(world[:2]).all():
                    if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
                        print(
                            f"[navstart] local={local.round(3).tolist()} world={world.round(3).tolist()}",
                            flush=True,
                        )
                    return world
                if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
                    print(
                        f"[navstart] invalid world pose={world.tolist()}; using local={local.round(3).tolist()}",
                        flush=True,
                    )
            except Exception:
                pass
        return local

    def _graph_world_step(self) -> int:
        gm = self.graph_memory
        getter = getattr(gm, "_effective_timestep", None) if gm is not None else None
        if callable(getter):
            try:
                return int(getter())
            except (TypeError, ValueError):
                pass
        return int(getattr(gm, "_graph_timestep", 0) or 0) if gm is not None else 0

    def _refresh_graph_room_estimate(self, *, after_motion: bool = False) -> str:
        """Recompute current-pose canonical room and room-target status."""
        gm = self.graph_memory
        room_fn = getattr(gm, "graph_room_at_robot", None) if gm is not None else None
        if after_motion:
            self._graph_room_stale = True
            self._router_room_stale = True
        if not callable(room_fn):
            if after_motion:
                self._room_estimate_stale = True
                self._current_room_source = "stale_router"
                self._in_target_area = None
            return self._graph_room_estimate
        try:
            graph_room = coerce_room_label(
                room_fn(self._robot_xyt_world()),
                room_policy=self.room_policy,
            )
        except Exception as e:
            _logger.warning(f"graph room refresh before router state failed: {e}")
            if after_motion:
                self._room_estimate_stale = True
                self._current_room_source = "stale_router"
                self._in_target_area = None
            return self._graph_room_estimate
        self._graph_room_estimate = graph_room
        if graph_room != "unknown":
            self._last_room_estimate = graph_room
            self._current_room_source = "graph_current_pose"
            self._room_estimate_stale = False
            self._graph_room_stale = False
            self._room_pose_round = int(self._round) + 1
            self._room_world_step = self._graph_world_step()
        elif after_motion:
            self._room_estimate_stale = True
            self._current_room_source = "stale_router"
        targets = question_target_rooms(self.question)
        if self._last_room_estimate != "unknown" and targets and not self._room_estimate_stale:
            self._in_target_area = not room_mismatches_question(
                self._last_room_estimate,
                self.question,
            )
        elif after_motion:
            self._in_target_area = None
        return graph_room

    def _refresh_room_after_motion(self) -> str:
        """Invalidate prior-pose router room and establish a current-pose room."""
        return self._refresh_graph_room_estimate(after_motion=True)

    def _observation_room(self, obs_id: int | None) -> str:
        """Resolve evidence room from the immutable observation view/place."""
        if obs_id is None:
            return ""
        gm = self.graph_memory
        room_fn = getattr(gm, "observation_room", None) if gm is not None else None
        if callable(room_fn):
            try:
                _room_id, room_name = room_fn(int(obs_id))
                normalized = normalize_current_room(room_name)
                if normalized != "unknown":
                    return normalized
            except (TypeError, ValueError):
                pass
        return ""

    def _append_trace(self, row: dict[str, Any]) -> None:
        if not self._collect_trace:
            return
        payload = {
            **self._trace_meta,
            "trace_schema_version": 2,
            "question": self.question,
            "mode": self.mode,
            "round": self._round,
            **row,
        }
        self._trace_rows.append(payload)
        if self._trace_path is not None:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._trace_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")

    def _attach_gt(self, row: dict[str, Any], xyz: np.ndarray | None) -> None:
        placements = self._gt_placements
        if placements is None:
            try:
                from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

                robot = getattr(self.agent, "robot", None)
                session = robot.get_emet_session() if robot is not None and hasattr(robot, "get_emet_session") else None
                placements = read_sim_object_placements(session) or {}
                self._gt_placements = placements
            except Exception:
                placements = {}
                self._gt_placements = {}
        gt_key = self._trace_meta.get("gt_body_key") or ""
        if not gt_key or gt_key not in placements:
            return
        info = placements[gt_key]
        gt_xyz = np.asarray(info.get("pos"), dtype=float).reshape(-1)[:3]
        row["gt_body_key"] = gt_key
        row["gt_xyz"] = [float(x) for x in gt_xyz.tolist()]
        if xyz is not None:
            d = float(np.linalg.norm(np.asarray(xyz, dtype=float).reshape(-1)[:2] - gt_xyz[:2]))
            row["gt_dist_m"] = d
            row["gt_present"] = bool(d <= 1.5)

    def _graph_node_for_obs(self, obs_id: int) -> Any | None:
        gm = self.graph_memory
        for node in list(getattr(gm, "_nodes", None) or ()):
            if int(getattr(node, "obs_id", -1)) != int(obs_id):
                continue
            if bool(getattr(node, "is_frontier", False)) or bool(getattr(node, "is_viewpoint", False)):
                continue
            return node
        return None

    def _action_target_for_obs(self, obs_id: int) -> ActionTarget:
        """Resolve a mutable adapter ID to stable place/view semantics."""
        oid = int(obs_id)
        gm = self.graph_memory
        world = getattr(gm, "world_evidence", None) if gm is not None else None
        node = self._graph_node_for_obs(oid)
        view = world.view_for_obs(oid) if world is not None else None
        entity = None
        if world is not None and node is not None:
            entity = world.entity_for_node(int(node.node_id))
        place_id = str(getattr(view, "place_id", "") or getattr(entity, "place_id", "") or f"obs:{oid}")
        labels = tuple(str(item) for item in (getattr(node, "labels", None) or ()))
        if not labels:
            labels = tuple(str(item) for item in (getattr(view, "labels", None) or ()))
        if not labels:
            hypothesis = next(
                (item for item in self._hypotheses if int(item.obs_id) == oid),
                None,
            )
            phrase = str(getattr(hypothesis, "phrase", "") or "").strip()
            labels = (phrase,) if phrase else ()
        room = self._observation_room(oid) or "unknown"
        if room == "unknown" and world is not None:
            place = world.places.get(place_id)
            room_record = world.rooms.get(place.room_id) if place is not None and place.room_id else None
            room = str(getattr(room_record, "room_name", "") or "unknown")
        xyz_value = getattr(node, "xyz", None) if node is not None else getattr(view, "object_xyz", None)
        xyz = None
        if xyz_value is not None:
            values = np.asarray(xyz_value, dtype=float).reshape(-1)
            if values.size >= 2:
                xyz = (
                    float(values[0]),
                    float(values[1]),
                    float(values[2]) if values.size >= 3 else 0.0,
                )
        return ActionTarget(
            kind="place",
            stable_id=place_id,
            labels=labels,
            room=room,
            adapter_id=oid,
            view_id=str(getattr(view, "view_id", "") or "") or None,
            revision=(int(view.revision) if view is not None else None),
            xyz=xyz,
        )

    def _action_target_for_frontier(self, frontier_id: str) -> ActionTarget:
        fid = str(frontier_id or "").strip()
        gm = self.graph_memory
        world = getattr(gm, "world_evidence", None) if gm is not None else None
        record = world.frontiers.get(fid) if world is not None and fid else None
        labels: list[str] = []
        room = "unknown"
        if record is not None and world is not None:
            for place_id in tuple(record.attachment_ids)[:3]:
                place = world.places.get(str(place_id))
                entity = world.entities.get(place.entity_id) if place is not None else None
                labels.extend(str(item) for item in (getattr(entity, "labels", None) or ()))
                room_record = world.rooms.get(place.room_id) if place is not None and place.room_id else None
                if room == "unknown" and room_record is not None:
                    room = str(room_record.room_name or "unknown")
        xyz = tuple(record.centroid_xyz) if record is not None else None
        return ActionTarget(
            kind="frontier",
            stable_id=fid or "unresolved",
            labels=tuple(dict.fromkeys(labels))[:3],
            room=room,
            adapter_id=(int(record.obs_id) if record is not None and record.obs_id is not None else None),
            revision=(int(record.revision) if record is not None else None),
            xyz=xyz,
        )

    def _frontier_geometry_id(self, frontier_id: str) -> str:
        gm = self.graph_memory
        world = getattr(gm, "world_evidence", None) if gm is not None else None
        record = world.frontiers.get(str(frontier_id)) if world is not None else None
        if record is None:
            return "unknown"
        centroid = quantized_xy(record.centroid_xyz, cell_m=0.5)
        return stable_digest(
            "frontier-material",
            {
                "status": str(record.status),
                "centroid_cell_0p5m": centroid,
                "cells": tuple(sorted(record.cells)),
                "parents": tuple(sorted(record.parent_ids)),
            },
        )

    def _relevant_evidence_digest(self, target: ActionTarget) -> str:
        """Hash target-local non-attempt evidence only."""
        gm = self.graph_memory
        world = getattr(gm, "world_evidence", None) if gm is not None else None
        if world is None:
            return "none"
        rows: list[tuple[Any, ...]] = []
        for event in list(getattr(world, "events", None) or ()):
            payload = dict(getattr(event, "payload", None) or {})
            if "outcome" in payload and "status_code" in payload:
                continue
            relevant = str(getattr(event, "subject_id", "")) == target.stable_id
            relevant = relevant or str(getattr(event, "place_id", "") or "") == target.stable_id
            relevant = relevant or str(getattr(event, "frontier_id", "") or "") == target.stable_id
            relevant = relevant or bool(target.view_id and getattr(event, "view_id", None) == target.view_id)
            if not relevant:
                continue
            rows.append(
                (
                    str(getattr(event, "event_id", "")),
                    str(getattr(event, "predicate", "")),
                    str(getattr(event, "polarity", "")),
                    str(getattr(event, "view_id", "") or ""),
                )
            )
        return stable_digest("target-evidence", rows)

    def _action_signature(
        self,
        name: str,
        args: dict[str, Any],
        *,
        out: dict[str, Any] | None = None,
    ) -> ActionSignature:
        tool = str(name or "").strip().lower()
        result = dict(out or {})
        intent = str(
            args.get("phrase")
            or args.get("toward")
            or getattr(self, "_target_phrase", "")
            or self.query_text
            or self.question
        )
        if tool in {"investigate", "navigate_to_obs"}:
            raw_obs = args.get("obs_id", result.get("obs_id", -1))
            try:
                obs_id = int(raw_obs)
            except (TypeError, ValueError):
                obs_id = -1
            target = self._action_target_for_obs(obs_id)
            raw_approach = result.get("approach_index")
            if raw_approach is None and obs_id >= 0:
                requested = args.get("approach_index", args.get("approach"))
                try:
                    preferred = int(requested) if requested is not None else None
                except (TypeError, ValueError):
                    preferred = None
                if preferred is not None and self.action_progress_mode == "enforce":
                    raw_approach = preferred % PLACE_APPROACH_SAMPLES
                else:
                    raw_approach = self._next_approach_index(obs_id, prefer=preferred)
            return ActionSignature.build(
                tool_name=tool,
                family="inspect_place",
                intent=intent,
                target=target,
                variant={"approach_index": raw_approach},
            )
        if tool == "verify_siglip":
            raw_obs = args.get("obs_id", result.get("obs_id"))
            if raw_obs is None:
                raw_obs = self._latest_obs_id()
            target = self._action_target_for_obs(int(raw_obs) if raw_obs is not None else -1)
            phrase = str(args.get("phrase") or getattr(self, "_target_phrase", "") or self.query_text)
            return ActionSignature.build(
                tool_name=tool,
                family="verify_view",
                intent=phrase,
                target=target,
                variant={
                    "view_id": target.view_id or f"obs:{target.adapter_id}",
                    "verifier_profile": "siglip+vlm",
                },
            )
        if tool == "explore_frontier":
            frontier_id = str(args.get("frontier_id") or result.get("frontier_id") or "")
            target = self._action_target_for_frontier(frontier_id)
            goal_xyz = result.get("frontier_xyz") or target.xyz
            frontier_intent = str(getattr(self, "_target_phrase", "") or self.query_text or self.question)
            return ActionSignature.build(
                tool_name=tool,
                family="explore_frontier",
                intent=intent,
                target=target,
                # ``toward`` is useful display context but only a weak navigation
                # hint: rewording it must not bypass the same static frontier work.
                work_intent=frontier_intent,
                variant={
                    "frontier_geometry_id": self._frontier_geometry_id(frontier_id),
                    "goal_cell": quantized_xy(goal_xyz),
                },
            )
        pose = self._robot_xyt_world()
        pose_cell = quantized_xy(pose)
        if tool == "look_around":
            target = ActionTarget(kind="pose", stable_id=f"pose:{pose_cell}", xyz=None)
            return ActionSignature.build(
                tool_name=tool,
                family="scan_view",
                intent=intent,
                target=target,
                variant={"pose_cell": pose_cell, "sensor_profile": "head-rgb"},
            )
        if tool == "submit_answer":
            target = ActionTarget(kind="question", stable_id=self._question_id)
            return ActionSignature.build(
                tool_name=tool,
                family="submit_answer",
                intent=self.question,
                target=target,
                variant={"answer": str(args.get("answer") or result.get("final_answer") or "")},
            )
        target = ActionTarget(kind="episode", stable_id=self._question_id or self._session_id)
        return ActionSignature.build(
            tool_name=tool,
            family="finish" if tool == "finish" else tool,
            intent=intent,
            target=target,
            variant={},
        )

    def _action_progress_token(self, signature: ActionSignature) -> ProgressToken:
        target = signature.target
        components: dict[str, Any] = {
            "family": signature.family,
            "robot_pose_cell": quantized_xy(self._robot_xyt_world()),
        }
        if signature.family in {"inspect_place", "verify_view"}:
            current = self._action_target_for_obs(target.adapter_id) if target.adapter_id is not None else target
            components.update(
                {
                    "target_id": current.stable_id,
                    "view_id": current.view_id,
                    "view_revision": current.revision,
                    "relevant_evidence_digest": self._relevant_evidence_digest(current),
                }
            )
            rec = self._place_inspect.get(int(target.adapter_id)) if target.adapter_id is not None else None
            if rec is not None:
                components["coverage"] = str(rec.coverage)
                components["local_frontier_cells"] = int(rec.local_frontier_cells)
        elif signature.family == "explore_frontier":
            components.update(
                {
                    "target_id": target.stable_id,
                    "frontier_geometry_id": self._frontier_geometry_id(target.stable_id),
                    "relevant_evidence_digest": self._relevant_evidence_digest(target),
                }
            )
        return ProgressToken.build(components)

    @staticmethod
    def _action_progress_reasons(
        signature: ActionSignature,
        before: ProgressToken,
        after: ProgressToken,
        out: dict[str, Any],
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        for key, reason in (
            ("view_id", "new_view"),
            ("view_revision", "view_revision"),
            ("relevant_evidence_digest", "target_evidence"),
            ("coverage", "coverage"),
            ("local_frontier_cells", "local_geometry"),
            ("frontier_geometry_id", "frontier_geometry"),
        ):
            if before.value(key) != after.value(key):
                reasons.append(reason)
        pose_changed = before.value("robot_pose_cell") != after.value("robot_pose_cell")
        partial_nav = bool(out.get("nav_progress")) and not bool(out.get("nav_finished"))
        if pose_changed and (signature.family == "explore_frontier" or partial_nav):
            reasons.append("motion")
        capture = out.get("capture")
        capture_status = str(capture.get("status") or "") if isinstance(capture, dict) else ""
        if capture_status in {"NEW_OBS", "CONTENT_REFRESHED"} and "new_view" not in reasons:
            reasons.append("new_view")
        return tuple(dict.fromkeys(reasons))

    def _record_action_history(
        self,
        name: str,
        args: dict[str, Any],
        out: dict[str, Any],
        *,
        signature: ActionSignature,
        progress_before: ProgressToken,
    ) -> None:
        if name in {"inspect_graph", "capture_and_update"}:
            return
        if signature.family == "inspect_place" and out.get("ok"):
            self._n_consecutive_explore = 0
        progress_after = self._action_progress_token(signature)
        progress_reasons = self._action_progress_reasons(
            signature,
            progress_before,
            progress_after,
            out,
        )
        verify = out.get("verify")
        verify_status = ""
        if isinstance(verify, dict):
            verify_status = str(verify.get("status") or verify.get("decision") or "")
        elif name == "verify_siglip":
            verify_status = str(out.get("status") or out.get("decision") or "")
        capture = out.get("capture")
        capture_status = str(capture.get("status") or "") if isinstance(capture, dict) else ""
        status = str(
            out.get("status")
            or out.get("nav_status_code")
            or out.get("error")
            or verify_status
            or ("ok" if out.get("ok") else "failed")
        )
        closest = None
        adapter = signature.target.adapter_id
        rec = self._place_inspect.get(int(adapter)) if adapter is not None else None
        if rec is not None and rec.closest_m is not None:
            closest = float(rec.closest_m)
        entry = ActionHistoryEntry(
            schema_version=1,
            round_index=int(self._round) + 1,
            selected_by=str(self._action_selected_by or "internal"),
            signature=signature,
            progress_before=progress_before,
            progress_after=progress_after,
            outcome_class=status_outcome_class(
                family=signature.family,
                ok=bool(out.get("ok")),
                status=verify_status or status,
                progress_reasons=progress_reasons,
            ),
            status=status[:120],
            ok=bool(out.get("ok")),
            progress_reasons=progress_reasons,
            closest_m=closest,
            capture_status=capture_status,
            verify_status=verify_status,
            nav_outcome=str(out.get("nav_outcome") or "")[:80],
        )
        self._action_history.append(entry)
        self._action_history = self._action_history[-32:]
        line = render_history_entry(entry)
        self._recent_actions.append(line)
        self._recent_actions = self._recent_actions[-RECENT_ACTIONS_K:]
        self._append_trace({"event": "action_history", "entry": entry.to_dict()})

    def _inspect_action_gate_decision(
        self,
        name: str,
        args: dict[str, Any],
    ) -> GateDecision:
        try:
            obs_id = int(args.get("obs_id"))
        except (TypeError, ValueError):
            signature = self._action_signature(name, args)
            progress = self._action_progress_token(signature)
            return decide_candidate(self._action_history, signature, progress)

        requested = args.get("approach_index", args.get("approach"))
        try:
            preferred = int(requested) % PLACE_APPROACH_SAMPLES if requested is not None else None
        except (TypeError, ValueError):
            preferred = None

        rec = self._place_inspect.get(obs_id)
        tried = [int(item) % PLACE_APPROACH_SAMPLES for item in (rec.tried_approaches if rec is not None else ())]
        probe = self._action_signature(name, args, out={"approach_index": 0})
        current_progress = self._action_progress_token(probe)
        continuation: list[int] = []
        if preferred is None:
            for entry in reversed(self._action_history):
                if (
                    entry.signature.work_key == probe.work_key
                    and entry.outcome_class == "progress"
                    and "motion" in entry.progress_reasons
                    and entry.progress_after.digest == current_progress.digest
                ):
                    approach = entry.signature.variant_value("approach_index")
                    if approach is not None:
                        continuation.append(int(approach) % PLACE_APPROACH_SAMPLES)
                    break

        if preferred is not None:
            order = [preferred]
        else:
            order = [
                *continuation,
                *(index for index in range(PLACE_APPROACH_SAMPLES) if index not in tried),
                *tried,
            ]
        order = list(dict.fromkeys(order))
        decisions: list[GateDecision] = []
        for approach in order:
            signature = self._action_signature(name, args, out={"approach_index": approach})
            progress = self._action_progress_token(signature)
            decision = decide_candidate(self._action_history, signature, progress)
            decisions.append(decision)
            if decision.allowed:
                return decision

        if preferred is not None and decisions:
            return decisions[0]
        prior_rounds = tuple(
            dict.fromkeys(round_index for decision in decisions for round_index in decision.prior_rounds)
        )
        base = (
            decisions[0]
            if decisions
            else decide_candidate(
                self._action_history,
                probe,
                current_progress,
            )
        )
        return GateDecision(
            allowed=False,
            disposition="would_suppress_saturated",
            reason="all finite approach variants are temporarily ineligible for the unchanged place state",
            signature=base.signature,
            progress=base.progress,
            prior_rounds=prior_rounds[-4:],
        )

    def _action_gate_decision(self, name: str, args: dict[str, Any]) -> GateDecision:
        tool = str(name or "").strip().lower()
        if tool in {"investigate", "navigate_to_obs"}:
            return self._inspect_action_gate_decision(tool, args)
        signature = self._action_signature(tool, args)
        progress = self._action_progress_token(signature)
        return decide_candidate(self._action_history, signature, progress)

    def _prepare_action_progress_dispatch(
        self,
        name: str,
        args: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if self.action_progress_mode not in {"shadow", "enforce"} or name not in {
            "investigate",
            "navigate_to_obs",
            "verify_siglip",
            "explore_frontier",
        }:
            return args, None

        prepared = dict(args)
        if (
            self.action_progress_mode == "enforce"
            and name == "explore_frontier"
            and not str(prepared.get("frontier_id") or "").strip()
        ):
            frontier_ids = tuple(self._rendered_action_allowlist().get("frontier_ids", ()))
            if not frontier_ids:
                self._append_trace(
                    {
                        "event": "action_gate_dispatch",
                        "mode": self.action_progress_mode,
                        "tool": name,
                        "allowed": False,
                        "disposition": "no_eligible_frontier",
                    }
                )
                return prepared, {
                    "ok": False,
                    "status": "NO_ELIGIBLE_ACTION",
                    "error": "no eligible rendered frontier remains while static progress gating is enforced",
                }
            prepared["frontier_id"] = str(frontier_ids[0])

        decision = self._action_gate_decision(name, prepared)
        self._append_trace(
            {
                "event": "action_gate_dispatch",
                "mode": self.action_progress_mode,
                "tool": name,
                "decision": decision.to_dict(),
            }
        )
        if self.action_progress_mode == "shadow":
            return args, None
        if not decision.allowed:
            return prepared, {
                "ok": False,
                "status": "ACTION_PROGRESS_SUPPRESSED",
                "disposition": decision.disposition,
                "target_id": decision.signature.target.stable_id,
                "error": decision.reason,
            }
        if decision.signature.family == "inspect_place":
            approach = decision.signature.variant_value("approach_index")
            if approach is not None:
                prepared["approach_index"] = int(approach)
        return prepared, None

    def handle_tool(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(args or {})
        name = (name or "").strip().lower()
        top_level = self._tool_dispatch_depth == 0
        if top_level:
            args, gate_error = self._prepare_action_progress_dispatch(name, args)
            if gate_error is not None:
                return gate_error
        signature = self._action_signature(name, args) if top_level else None
        progress_before = self._action_progress_token(signature) if signature is not None else None
        self._tool_dispatch_depth += 1
        self._tool_log.append(name)
        if name == "inspect_graph":
            out = self._tool_inspect_graph()
        elif name == "explore_frontier":
            out = self._tool_explore_frontier(
                str(args.get("toward") or ""),
                frontier_id=str(args.get("frontier_id") or ""),
            )
        elif name in ("investigate", "navigate_to_obs"):
            raw_ap = args.get("approach_index", args.get("approach"))
            approach_index = None
            if raw_ap is not None and str(raw_ap).strip() != "":
                try:
                    approach_index = int(raw_ap)
                except (TypeError, ValueError):
                    approach_index = None
            raw_obs_id = args.get("obs_id", -1)
            try:
                obs_id = int(raw_obs_id)
            except (TypeError, ValueError):
                listed = sorted({int(h.obs_id) for h in self._hypotheses})
                out = {
                    "ok": False,
                    "status": "OBS_NOT_IN_EVIDENCE",
                    "obs_id": raw_obs_id,
                    "listed_obs_ids": listed,
                    "error": "obs_id must be a numeric evidence-card id",
                }
            else:
                out = self._tool_investigate(
                    obs_id,
                    tool_name=name,
                    approach_index=approach_index,
                )
        elif name == "look_around":
            out = self._tool_look_around()
        elif name == "capture_and_update":
            out = self._tool_capture_and_update()
        elif name == "verify_siglip":
            raw_obs_id = args.get("obs_id")
            try:
                obs_id = int(raw_obs_id) if raw_obs_id is not None else None
            except (TypeError, ValueError):
                out = {
                    "ok": False,
                    "status": "OBS_NOT_IN_EVIDENCE",
                    "obs_id": raw_obs_id,
                    "error": "obs_id must be a numeric evidence-card id",
                }
            else:
                out = self._tool_verify_siglip(
                    str(args.get("phrase") or ""),
                    obs_id,
                )
        elif name == "submit_answer":
            out = self._tool_submit_answer(str(args.get("answer") or ""))
        elif name == "finish":
            out = self._tool_finish(str(args.get("summary") or ""))
        else:
            out = {"ok": False, "error": f"unknown tool {name!r}"}
        self._tool_dispatch_depth = max(0, self._tool_dispatch_depth - 1)
        if not isinstance(out, dict):
            out = {"ok": False, "error": f"non-dict tool result for {name!r}"}
        completed_signature = out.pop("_action_history_signature", None) or signature
        completed_progress_before = out.pop("_action_history_progress_before", None) or progress_before
        if top_level and signature is not None and progress_before is not None:
            self._record_action_history(
                name,
                args,
                out,
                signature=completed_signature,
                progress_before=completed_progress_before,
            )
        # Shared ToolOutcome → attempt ledger (no-op when ledger off / tool not mapped).
        try:
            from emet.agent.tool_outcome import ToolOutcome, maybe_record_tool_attempt

            # navigate/investigate already dual-write via record_nav_attempt; skip
            # duplicate navigate rows. Still record verify / explore / closer_look.
            if name not in ("investigate", "navigate_to_obs"):
                maybe_record_tool_attempt(
                    self.graph_memory,
                    ToolOutcome.from_eqa_dict(name, out),
                    source="eqa",
                )
        except Exception:
            pass
        return out

    def _record_recent_action(
        self,
        name: str,
        args: dict[str, Any],
        out: dict[str, Any],
    ) -> None:
        """Compatibility helper for tests and callers with a completed result."""
        signature = self._action_signature(name, dict(args or {}), out=out)
        self._record_action_history(
            name,
            dict(args or {}),
            dict(out or {}),
            signature=signature,
            progress_before=self._action_progress_token(signature),
        )

    def _ovmm_phase(self) -> str:
        return str(self._trace_meta.get("ovmm_phase") or "").strip()

    def _apply_ovmm_trace_target(self) -> None:
        """Episode metadata from OVMM find overrides heuristic/VLM target extraction."""
        phase = self._ovmm_phase()
        if phase == "find_recep":
            recep = str(self._trace_meta.get("goal_recep") or "").strip()
            if recep:
                self._target_phrase = recep
        elif phase == "find_object":
            obj = str(self._trace_meta.get("object") or "").strip()
            if obj:
                self._target_phrase = obj

    def _ovmm_boost_phrases(self) -> list[str]:
        phase = self._ovmm_phase()
        if not phase:
            return []
        ordered: list[str] = []
        if phase == "find_recep":
            primary = str(self._trace_meta.get("goal_recep") or "").strip()
            if primary:
                ordered.append(primary)
        elif phase == "find_object":
            primary = str(self._trace_meta.get("object") or "").strip()
            if primary:
                ordered.append(primary)
            start = str(self._trace_meta.get("start_recep") or "").strip()
            if start and start not in ordered:
                ordered.append(start)
        for key in ("object", "goal_recep", "start_recep"):
            val = str(self._trace_meta.get(key) or "").strip()
            if val and val not in ordered:
                ordered.append(val)
        return ordered

    def _ovmm_sim_placement_hypotheses(self) -> list[NavHypothesis]:
        """GT placement seeds for OVMM find (recep/obj near spawn)."""
        phase = self._ovmm_phase()
        if not phase:
            return []
        robot = getattr(self.agent, "robot", None)
        if robot is None or not hasattr(robot, "get_emet_session"):
            return []
        from emet.eval.ovmm_find_phase import bodies_matching_category
        from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

        placements = read_sim_object_placements(robot.get_emet_session()) or {}
        if not placements:
            return []
        out: list[NavHypothesis] = []
        if phase == "find_recep":
            query = str(self._trace_meta.get("goal_recep") or self._target_phrase or "").strip()
            bodies = bodies_matching_category(placements, query) if query else []
            for i, body in enumerate(bodies[:3]):
                pos = np.asarray(placements[body]["pos"], dtype=float).reshape(-1)[:3]
                out.append(
                    NavHypothesis(
                        phrase=query or str(placements[body].get("cat") or body),
                        obs_id=-3_000_000 - i,
                        xyz=pos.copy(),
                        score=0.0,
                        source="graph",
                    )
                )
        elif phase == "find_object":
            body = str(self._trace_meta.get("object_gt_body") or "").strip()
            if body and body in placements:
                bodies = [body]
            else:
                query = str(self._trace_meta.get("object") or self._target_phrase or "").strip()
                bodies = bodies_matching_category(placements, query) if query else []
            for i, bname in enumerate(bodies[:3]):
                pos = np.asarray(placements[bname]["pos"], dtype=float).reshape(-1)[:3]
                phrase = str(self._trace_meta.get("object") or placements[bname].get("cat") or bname)
                out.append(
                    NavHypothesis(
                        phrase=phrase,
                        obs_id=-3_100_000 - i,
                        xyz=pos.copy(),
                        score=0.0,
                        source="graph",
                    )
                )
        return out

    def _recall_nav_hypotheses(self) -> list[NavHypothesis]:
        gm = self.graph_memory
        if gm is None:
            return []
        boost = self._ovmm_boost_phrases()
        try:
            hypotheses = gm.hypothesize_nav_targets(
                self.query_text,
                max_k=env_eqa_hyp_recall_k(),
                robot_xyt=self._robot_xyt_world(),
                boost_phrases=boost or None,
            )
        except TypeError:
            hypotheses = gm.hypothesize_nav_targets(self.query_text, max_k=env_eqa_hyp_recall_k())
        sim_hyps = self._ovmm_sim_placement_hypotheses()
        if sim_hyps:
            seen = {int(h.obs_id) for h in hypotheses}
            prefixed = [h for h in sim_hyps if int(h.obs_id) not in seen]
            if prefixed:
                hypotheses = prefixed + list(hypotheses)
        if not any(str(h.source) in INVESTIGATE_SOURCES for h in hypotheses):
            adjacent = self._receptacle_adjacent_hypotheses(gm)
            if adjacent:
                hypotheses = adjacent + list(hypotheses)
        return hypotheses

    def _ovmm_prefers_nearby_investigate(self) -> bool:
        """OVMM find phases and close-look HM-EQA should bias toward nearby place cards."""
        phase = self._ovmm_phase()
        if phase in ("find_recep", "find_object"):
            return True
        return bool(self._close_look_required)

    def _investigate_matches_ovmm_target(self, hyp: NavHypothesis | None, obs_id: int) -> bool:
        """For OVMM find phases, only absent-at-close on the target should nudge explore."""
        phase = self._ovmm_phase()
        if phase not in ("find_recep", "find_object"):
            return True
        if phase == "find_recep":
            target = (self._target_phrase or str(self._trace_meta.get("goal_recep") or "")).strip().lower()
        else:
            target = (self._target_phrase or str(self._trace_meta.get("object") or "")).strip().lower()
        if not target:
            return True
        if hyp is not None:
            phrase = str(hyp.phrase or "").lower()
            if target in phrase or label_matches_relevant_object(target, phrase):
                return True
            for tok in self._FIXTURE_LABEL_TOKENS:
                if tok in target and tok in phrase:
                    return True
        gm = self.graph_memory
        if gm is not None and hasattr(gm, "_observation_by_id"):
            obs = gm._observation_by_id(int(obs_id))
            if obs is not None:
                for lab in getattr(obs, "labels", None) or []:
                    if label_matches_relevant_object(target, str(lab)):
                        return True
        return False

    def _nearby_untried_investigate_hyp(self, max_dist_m: float = OVMM_NEAR_INVESTIGATE_M) -> NavHypothesis | None:
        best: NavHypothesis | None = None
        best_d = float("inf")
        for h in self._investigate_hypotheses():
            oid = int(h.obs_id)
            if self._obs_already_verified(oid) or self._hypothesis_nav_blocked(oid):
                continue
            if self._place_approaches_exhausted(oid):
                continue
            d = self._dist_to_anchor_m(oid, h)
            if d is None or d > float(max_dist_m):
                continue
            if d < best_d:
                best_d = d
                best = h
        return best

    def _tool_inspect_graph(self) -> dict[str, Any]:
        gm = self.graph_memory
        if gm is None:
            return {"ok": False, "error": "no graph_memory"}
        if hasattr(gm, "extract_relevant_objects") and getattr(gm, "image_description_client", None) is not None:
            gm.extract_relevant_objects(self.query_text)
        if getattr(gm, "memory_summary_enabled", False) and hasattr(gm, "refresh_siglip_confirmed_memory"):
            gm.refresh_siglip_confirmed_memory()
        hypotheses = self._recall_nav_hypotheses()
        self._set_hypotheses(hypotheses)
        out = {
            "ok": True,
            "n_hypotheses": len(self._hypotheses),
            "hypotheses": [
                {
                    "phrase": h.phrase,
                    "obs_id": int(h.obs_id),
                    "xyz": [float(x) for x in np.asarray(h.xyz).reshape(-1)[:3]],
                    "source": h.source,
                    "siglip_sim": (float(h.siglip_sim) if getattr(h, "siglip_sim", None) is not None else None),
                }
                for h in self._hypotheses
            ],
        }
        self._append_trace(
            {
                "tool": "inspect_graph",
                "picked_by": "loop",
                "policy_state": self._evidence_policy.state,
                "n_hypotheses": out["n_hypotheses"],
                "hypotheses": out["hypotheses"],
            }
        )
        return out

    def _tool_explore_frontier(self, toward: str = "", *, frontier_id: str = "") -> dict[str, Any]:
        if self._n_nav + self._n_explore >= self.max_nav_steps:
            return {"ok": False, "error": "nav budget exhausted"}
        toward = (toward or "").strip()
        requested_frontier_id = str(frontier_id or "").strip()
        grounded = self.decision_policy == "grounded_v2"
        if grounded and self.action_progress_mode == "enforce" and not requested_frontier_id:
            eligible = tuple(self._rendered_action_allowlist().get("frontier_ids", ()))
            if not eligible:
                return {
                    "ok": False,
                    "status": "NO_ELIGIBLE_ACTION",
                    "error": "no eligible rendered frontier remains while static progress gating is enforced",
                }
            requested_frontier_id = str(eligible[0])
        leave = False
        if not grounded:
            leave = room_leave_needed(
                room_policy=self.room_policy,
                current_room=self._last_room_estimate,
                question=self.question,
                in_target_area=self._in_target_area,
            )
        # Do not invent toward= from MCQ options / room enums — pass the Question
        # to the frontier VLM and let it pick among graph-contexted views.
        bias = toward or self.query_text
        agent = self.agent
        gm = self.graph_memory
        frontier_xyz = None
        frontier_id = requested_frontier_id
        pick_source = "pick_uncovered"
        frontier_room = "unknown"
        history_signature = None
        history_progress_before = None
        try:
            from emet.controller.habitat_nav import pick_uncovered_explore_target

            escape_m = 0.0 if grounded else self._escape_min_travel_m()
            candidates: list[np.ndarray | None] = []
            if requested_frontier_id:
                world = getattr(gm, "world_evidence", None) if gm is not None else None
                record = world.frontiers.get(requested_frontier_id) if world is not None else None
                if record is None or record.status != "active":
                    return {
                        "ok": False,
                        "error": f"stale or unknown frontier_id: {requested_frontier_id}",
                        "status": "STALE_FRONTIER_ID",
                        "frontier_id": requested_frontier_id,
                    }
                frontier_xyz = np.asarray(record.centroid_xyz, dtype=float)
                pick_source = "router_frontier_id"
            # GraphEQA-style: VLM ranks a small pool of reachable frontier RGBs.
            # Agentic always tries this; classic coverage path still uses EMET_VLM_FRONTIER_SCORING.
            if frontier_xyz is None and hasattr(agent, "_vlm_frontier_choice"):
                try:
                    vlm_pt = agent._vlm_frontier_choice(
                        bias,
                        current_room=self._last_room_estimate,
                        room_policy=self.room_policy,
                        leave_hint=leave,
                    )
                except TypeError:
                    # Older controllers without room kwargs.
                    try:
                        vlm_pt = agent._vlm_frontier_choice(bias)
                    except Exception as e:
                        _logger.warning(f"vlm_frontier_choice failed: {e}")
                        vlm_pt = None
                except Exception as e:
                    _logger.warning(f"vlm_frontier_choice failed: {e}")
                    vlm_pt = None
                if vlm_pt is not None:
                    candidates.append(vlm_pt)
                    pick_source = "vlm_frontier"
            # SigLIP guidance aims at the frontier nearest the best-matching *already
            # observed* point, so while escaping it just pulls us back into the area we
            # already rejected. Let region utility choose instead.
            if frontier_xyz is None and escape_m <= 0.0 and hasattr(agent, "_siglip_guided_frontier"):
                candidates.append(agent._siglip_guided_frontier(bias))
            if frontier_xyz is None and hasattr(agent, "_best_frontier_point_from_graph"):
                candidates.append(agent._best_frontier_point_from_graph(bias))
            if frontier_xyz is None:
                frontier_xyz = pick_uncovered_explore_target(
                    agent,
                    question=bias,
                    candidates=candidates,
                    blocked=getattr(agent, "_habitat_blocked_goals", None),
                    recent_goals=getattr(agent, "_habitat_recent_goals", None),
                    min_travel_m=escape_m,
                )
            if frontier_xyz is not None and pick_source == "vlm_frontier":
                # Confirm the accepted goal is still the VLM pick (not a later fallback).
                vlm0 = candidates[0] if candidates else None
                if (
                    vlm0 is None
                    or float(
                        np.linalg.norm(
                            np.asarray(frontier_xyz, dtype=float).reshape(-1)[:2]
                            - np.asarray(vlm0, dtype=float).reshape(-1)[:2]
                        )
                    )
                    > 0.35
                ):
                    pick_source = "pick_uncovered"
        except Exception as e:
            _logger.warning(f"explore_frontier pick failed: {e}")
            pick_source = "pick_uncovered"
        if frontier_xyz is not None:
            room_fn = getattr(gm, "graph_room_at_robot", None) if gm is not None else None
            if callable(room_fn):
                try:
                    frontier_room = coerce_room_label(
                        room_fn(
                            (
                                float(np.asarray(frontier_xyz).reshape(-1)[0]),
                                float(np.asarray(frontier_xyz).reshape(-1)[1]),
                            )
                        ),
                        room_policy=self.room_policy,
                    )
                except Exception:
                    frontier_room = "unknown"
            frontier_id_fn = getattr(gm, "frontier_id_near_xy", None) if gm is not None else None
            if not frontier_id and callable(frontier_id_fn):
                try:
                    frontier_id = str(frontier_id_fn(frontier_xyz) or "")
                except (TypeError, ValueError):
                    frontier_id = ""
        if frontier_id:
            history_signature = self._action_signature(
                "explore_frontier",
                {"frontier_id": frontier_id, "toward": toward},
            )
            history_progress_before = self._action_progress_token(history_signature)
        frontier_key = -1_000_000 - self._n_explore
        hypothesis_id = self._begin_policy_approach(
            "frontier",
            frontier_key,
            bias,
        )
        ok = False
        motion_progress = False
        reached = False
        nav_outcome_str = ""
        used_nav_target = False
        start = self._robot_xyt_world()
        if start is None:
            start = np.array([0.0, 0.0, 0.0])
        if frontier_xyz is not None and hasattr(agent, "navigate_to_target_pose"):
            used_nav_target = True
            try:
                nav_outcome = agent.navigate_to_target_pose(frontier_xyz, start, None)
            except TypeError:
                nav_outcome = agent.navigate_to_target_pose(frontier_xyz, start)
            nav_outcome_str = str(nav_outcome)
            ok = bool(nav_outcome)
            nav_outcome_str = str(nav_outcome)
            motion_progress = ok
            self._n_explore += 1
            if ok:
                self._consecutive_nav_fail = 0
                reached = bool(getattr(nav_outcome, "finished", ok))
                if self.decision_policy != "grounded_v2" or reached:
                    self._retire_visited_frontier(frontier_xyz=frontier_xyz)
            elif self._explore_nav_progressed():
                # Chunked path: robot moved toward the frontier even if not finished.
                self._consecutive_nav_fail = 0
                motion_progress = True
        elif hasattr(agent, "run_exploration"):
            ok = bool(agent.run_exploration())
            motion_progress = ok
            self._n_explore += 1
            if ok:
                self._consecutive_nav_fail = 0
            pick_source = "run_exploration_fallback"
            # Recover a goal for viz/trace when the uncovered picker returned None.
            if frontier_xyz is None:
                recent = list(getattr(agent, "_habitat_recent_goals", None) or [])
                if recent:
                    frontier_xyz = np.array([float(recent[-1][0]), float(recent[-1][1]), 1.0])
                else:
                    after = self._robot_xyt_world()
                    if after is not None:
                        frontier_xyz = np.asarray(after, dtype=float).reshape(-1)[:3]
            if ok and frontier_xyz is not None:
                self._retire_visited_frontier(frontier_xyz=frontier_xyz)
                reached = True
        nav_result = getattr(agent, "_last_nav_attempt", None) if used_nav_target else None
        if nav_result is not None:
            from emet.controller.nav_attempt import nav_status_code

            nav_status = nav_status_code(nav_result)
        else:
            nav_status = nav_outcome_str or ("ok" if ok else "failed")
        if motion_progress:
            self._refresh_room_after_motion()
        cap = self._tool_capture_and_update()
        look_retry = False
        # After a successful explore nav, a mid-floor / already-mapped goal often yields
        # NO_NEW_OBS. Spin in place so we still peel new coverage from this pose.
        if ok and not cap.get("ok") and str(cap.get("status") or "") == "NO_NEW_OBS":
            look_retry = True
            look = self._tool_look_around(verify=False)
            look_cap = look.get("capture") if isinstance(look, dict) else None
            if isinstance(look_cap, dict) and look_cap.get("ok"):
                cap = look_cap
        verify_out = None
        if cap.get("ok") and cap.get("obs_id") is not None:
            self._policy_approached(hypothesis_id, int(cap["obs_id"]))
            if self.mode == "answer":
                verify_out = self._verify_after_motion(phrase=self.query_text)
        panel_path = self._save_frontier_pick_panel(
            frontier_xyz,
            robot_xyt_before=start,
        )
        room_aligned: bool | None = None
        if self.room_policy == "canonical":
            targets = question_target_rooms(self.question)
            fr = normalize_current_room(frontier_room)
            if fr != "unknown" and targets:
                room_aligned = fr in targets
        elif self._in_target_area is False:
            # Soft leave explore: alignment unknown until next router turn.
            room_aligned = None
        row = {
            "tool": "explore_frontier",
            "ok": ok,
            "frontier_xyz": [float(x) for x in np.asarray(frontier_xyz).reshape(-1)[:3]]
            if frontier_xyz is not None
            else None,
            "frontier_id": frontier_id or None,
            "source": pick_source,
            "pick_panel": str(panel_path) if panel_path else None,
            "look_around_on_no_new_obs": look_retry,
            "room_policy": self.room_policy,
            "current_room": self._last_room_estimate,
            "frontier_room": frontier_room,
            "room_leave_hint": leave,
            "toward": toward or None,
            "room_aligned": room_aligned,
            "in_target_area": self._in_target_area,
            "nav_outcome": nav_outcome_str or None,
            "nav_status_code": nav_status,
        }
        self._attach_gt(row, frontier_xyz)
        self._append_trace(row)
        if ok:
            self._prefer_explore = False
            self._prefer_explore_reason = ""
            self._n_consecutive_explore = int(getattr(self, "_n_consecutive_explore", 0) or 0) + 1
        return {
            "ok": ok,
            "capture": cap,
            "frontier_xyz": row["frontier_xyz"],
            "xyz": row["frontier_xyz"],
            "frontier_id": frontier_id or None,
            "target_kind": "frontier" if frontier_id else "",
            "target_id": frontier_id or "",
            "phrase": toward or self.query_text,
            "room": frontier_room,
            "motion_progress": bool(motion_progress),
            "nav_outcome": nav_outcome_str or None,
            "nav_progress": bool(motion_progress),
            "nav_finished": bool(reached),
            "nav_status_code": nav_status,
            "status": nav_status,
            "verify": verify_out,
            "_action_history_signature": history_signature,
            "_action_history_progress_before": history_progress_before,
        }

    def _investigate_hypotheses(self) -> list[NavHypothesis]:
        return [h for h in self._hypotheses if str(h.source) in INVESTIGATE_SOURCES]

    def _explore_hypotheses(self) -> list[NavHypothesis]:
        return [h for h in self._hypotheses if str(h.source) not in INVESTIGATE_SOURCES]

    def _place_anchor_xy(self, obs_id: int, hyp: NavHypothesis | None) -> tuple[float, float] | None:
        if hyp is not None:
            xyz = np.asarray(hyp.xyz, dtype=float).reshape(-1)
            if xyz.size >= 2:
                return float(xyz[0]), float(xyz[1])
        gm = self.graph_memory
        if gm is not None and hasattr(gm, "_observation_by_id"):
            obs = gm._observation_by_id(int(obs_id))
            if obs is not None:
                xyz = np.asarray(obs.xyz, dtype=float).reshape(-1)
                if xyz.size >= 2:
                    return float(xyz[0]), float(xyz[1])
        return None

    def _dist_to_anchor_m(self, obs_id: int, hyp: NavHypothesis | None) -> float | None:
        anchor = self._place_anchor_xy(obs_id, hyp)
        robot = self._robot_xyt_world()
        if anchor is None or robot is None:
            return None
        return float(np.hypot(float(robot[0]) - anchor[0], float(robot[1]) - anchor[1]))

    def _record_place_inspect(
        self,
        obs_id: int,
        *,
        closest_m: float | None,
        verify_out: dict[str, Any] | None,
        approach_index: int | None = None,
    ) -> PlaceInspectRecord:
        oid = int(obs_id)
        rec = self._place_inspect.get(oid) or PlaceInspectRecord()
        dist = (
            float(closest_m) if closest_m is not None else (float(rec.closest_m) if rec.closest_m is not None else 99.0)
        )
        if rec.closest_m is None or dist < float(rec.closest_m):
            rec.closest_m = dist
        verify_status = ""
        if isinstance(verify_out, dict):
            verify_status = str(verify_out.get("status") or verify_out.get("decision") or "")
        assess = self._last_vlm_assess if isinstance(self._last_vlm_assess, dict) else {}
        present = assess.get("present") if assess else None
        answerable = assess.get("answerable") if assess else None
        suggested = str(assess.get("suggested_answer") or "") if assess else ""
        if present is None and isinstance(verify_out, dict):
            present = verify_out.get("present")
            answerable = verify_out.get("answerable")
        if approach_index is not None:
            ap = int(approach_index) % PLACE_APPROACH_SAMPLES
            if ap not in rec.tried_approaches:
                rec.tried_approaches.append(ap)
        visit = PlaceInspectVisit(
            round=int(self._round),
            closest_m=dist,
            verify=verify_status,
            assess_present=bool(present) if present is not None else None,
            assess_answerable=bool(answerable) if answerable is not None else None,
            suggested=suggested,
            approach_index=int(approach_index) if approach_index is not None else None,
        )
        rec.investigate_count += 1
        rec.recent.append(visit)
        if len(rec.recent) > PLACE_INSPECT_RECENT_K:
            rec.recent = rec.recent[-PLACE_INSPECT_RECENT_K:]
        rec.last_verify = verify_status
        rec.last_assess_present = visit.assess_present
        rec.last_assess_answerable = visit.assess_answerable
        rec.last_suggested = suggested
        self._place_inspect[oid] = rec
        # Close look: only VLM assess_present=False nudges explore (not SigLIP ABSENT alone).
        # OVMM find_recep: absent at a non-target fixture must not pull explore away.
        hyp = next((h for h in self._investigate_hypotheses() if int(h.obs_id) == oid), None)
        if (
            self.decision_policy != "grounded_v2"
            and dist <= 1.0
            and visit.assess_present is False
            and self._investigate_matches_ovmm_target(hyp, oid)
        ):
            self._prefer_explore = True
            self._prefer_explore_reason = "absent"
        return rec

    def _place_approaches_exhausted(self, obs_id: int) -> bool:
        """True when the fixed approach sample budget is spent."""
        rec = self._place_inspect.get(int(obs_id))
        if rec is None:
            return False
        return int(rec.approaches_left) <= 0

    def _next_approach_index(self, obs_id: int, *, prefer: int | None = None) -> int | None:
        """Next unused approach sample index, or None if count-exhausted."""
        rec = self._place_inspect.get(int(obs_id))
        tried = {int(i) for i in (rec.tried_approaches if rec is not None else [])}
        if prefer is not None:
            p = int(prefer) % PLACE_APPROACH_SAMPLES
            if p not in tried:
                return p
        for i in range(PLACE_APPROACH_SAMPLES):
            if i not in tried:
                return i
        return None

    def _place_close_and_absent(self, obs_id: int) -> bool:
        rec = self._place_inspect.get(int(obs_id))
        if rec is None or not rec.approached_close or rec.investigate_count <= 0:
            return False
        if rec.last_assess_answerable:
            return False
        if rec.last_assess_present is False:
            return True
        return str(rec.last_verify).upper() in {"ABSENT", "SKIPPED_SAME_VIEW"}

    def _labels_for_room_stamp(self, obs_id: int, hyp: NavHypothesis | None = None) -> list[str]:
        """Local labels for room stamping: hyp node labels + this obs only.

        Deliberately omits ``hyp.phrase`` (often question/MCQ text like
        ``wall clock kitchen``) and ``labels_near_obs`` / station merges, which
        leak nonlocal kitchen/bathroom words in open-plan scenes.
        """
        labels: list[str] = []
        if hyp is not None:
            for lab in list(getattr(hyp, "labels", None) or []):
                s = str(lab).strip()
                if s and s not in labels:
                    labels.append(s)
        gm = self.graph_memory
        if gm is not None:
            for obs in list(getattr(gm, "_observations", None) or []):
                if int(getattr(obs, "obs_id", -1)) != int(obs_id):
                    continue
                for lab in list(getattr(obs, "labels", None) or []):
                    s = str(lab).strip()
                    if s and s not in labels:
                        labels.append(s)
                break
        return labels[:48]

    def _stamp_room_after_investigate(
        self,
        obs_id: int,
        *,
        hyp: NavHypothesis | None,
        station_oid: int | None,
    ) -> dict[str, Any]:
        """Refresh graph room cluster from close-look evidence (deferred room-stamp)."""
        if not bool(getattr(self, "_room_stamp_investigate", False)):
            return {"ok": False, "reason": "disabled"}
        gm = self.graph_memory
        if gm is None or not hasattr(gm, "stamp_vlm_room_at_robot"):
            return {"ok": False, "reason": "no_graph"}
        # Station labels stay out of the bag (trace-only); they bleed open-plan kitchens.
        labels = self._labels_for_room_stamp(int(obs_id), hyp)
        label_source = "obs_and_hyp_labels"
        proposed = resolve_investigate_room_stamp(
            labels=labels,
            current_room=self._last_room_estimate,
            room_policy=self.room_policy,
        )
        if proposed == "unknown":
            return {
                "ok": False,
                "reason": "no_room",
                "labels": labels[:12],
                "label_source": label_source,
                "station_obs_id": int(station_oid) if station_oid is not None else None,
            }
        stamp_xy = None
        if hyp is not None and getattr(hyp, "xyz", None) is not None:
            try:
                xyz = np.asarray(hyp.xyz, dtype=float).reshape(-1)
                stamp_xy = (float(xyz[0]), float(xyz[1]))
            except Exception:
                stamp_xy = None
        if stamp_xy is None:
            xyt = self._robot_xyt_world()
            if xyt is not None:
                stamp_xy = (float(xyt[0]), float(xyt[1]))
        if stamp_xy is None:
            return {"ok": False, "reason": "no_xy", "proposed": proposed}
        prev = "unknown"
        if hasattr(gm, "graph_room_at_robot"):
            try:
                prev = coerce_room_label(gm.graph_room_at_robot(stamp_xy), room_policy=self.room_policy)
            except Exception as e:
                _logger.warning(f"graph_room_at_robot before investigate stamp failed: {e}")
        try:
            stamp_kwargs = {
                "protect_indoor_from_outdoor": True,
                "corroborating_labels": labels,
                "source": "investigate_vlm",
                "source_view_id": (
                    gm.view_id_for_obs(int(station_oid))
                    if station_oid is not None and hasattr(gm, "view_id_for_obs")
                    else None
                ),
            }
            stamped = gm.stamp_vlm_room_at_robot(
                stamp_xy,
                proposed,
                **stamp_kwargs,
                agent_round=int(self._round) + 1,
                pose_round=int(self._round) + 1,
            )
        except Exception as e:
            _logger.warning(f"stamp_vlm_room_at_robot after investigate failed: {e}")
            return {"ok": False, "reason": "stamp_failed", "error": str(e), "proposed": proposed}
        stamped_s = coerce_room_label(stamped, room_policy=self.room_policy)
        if stamped_s == "unknown":
            payload = {
                "ok": False,
                "reason": "blocked_or_noop",
                "proposed": proposed,
                "prev": prev,
                "labels": labels[:12],
                "label_source": label_source,
                "station_obs_id": int(station_oid) if station_oid is not None else None,
            }
            self._append_trace({"event": "room_stamp_investigate", **payload})
            return payload
        graph_room = stamped_s
        if hasattr(gm, "graph_room_at_robot"):
            try:
                graph_room = coerce_room_label(gm.graph_room_at_robot(stamp_xy), room_policy=self.room_policy)
            except Exception:
                graph_room = stamped_s
        self._graph_room_estimate = graph_room
        self._graph_room_stale = graph_room == "unknown"
        merged = merge_room_estimates(proposed, graph_room, room_policy=self.room_policy)
        # Prefer the stamp we just applied when merge would keep a stale VLM outdoor.
        if normalize_current_room(merged) == "outdoor" and not (normalize_current_room(proposed) == "outdoor"):
            merged = proposed
        self._last_room_estimate = merged
        self._last_router_room_estimate = proposed
        self._current_room_source = "investigate_vlm+graph"
        self._room_estimate_stale = False
        self._router_room_stale = False
        self._room_pose_round = int(self._round) + 1
        self._room_world_step = self._graph_world_step()
        self._room_estimates.append(merged)
        if len(self._room_estimates) > 8:
            self._room_estimates = self._room_estimates[-8:]
        payload = {
            "ok": True,
            "obs_id": int(obs_id),
            "station_obs_id": int(station_oid) if station_oid is not None else None,
            "proposed": proposed,
            "stamped": stamped_s,
            "prev": prev,
            "current_room": merged,
            "labels": labels[:12],
            "label_source": label_source,
            "xy": [float(stamp_xy[0]), float(stamp_xy[1])],
        }
        self._append_trace({"event": "room_stamp_investigate", **payload})
        self._record_room_timeline(
            kind="stamp",
            room=merged,
            obs_id=int(obs_id),
            note=f"investigate stamp prev={prev}",
        )
        if gm is not None and hasattr(gm, "record_attempt"):
            gm.record_attempt(
                action_kind="investigate",
                outcome="ok",
                status_code="room_stamp",
                note=f"stamp room={merged} at obs {int(obs_id)}",
                step=self._graph_world_step(),
                obs_id=int(obs_id),
                phrase="",
                source="eqa",
                room=normalize_current_room(merged),
            )
        return payload

    def _voxel_planner(self) -> tuple[Any | None, Any | None]:
        agent = self.agent
        voxel_map = getattr(agent, "voxel_map", None)
        planner = getattr(agent, "planner", None) or getattr(agent, "_planner", None)
        return voxel_map, planner

    def _known_room_for_event(self) -> str:
        """Canonical room label for timeline writes; empty when unknown (never invent)."""
        if self._room_estimate_stale:
            return ""
        for raw in (self._last_room_estimate, self._graph_room_estimate):
            room = normalize_current_room(raw)
            if room != "unknown":
                return room
        return ""

    def _record_room_timeline(
        self,
        *,
        kind: str,
        room: str | None = None,
        phrase: str = "",
        obs_id: int | None = None,
        note: str = "",
    ) -> dict[str, Any] | None:
        gm = self.graph_memory
        if gm is None or not hasattr(gm, "record_room_event"):
            return None
        observed_room = self._observation_room(obs_id)
        label = observed_room or (normalize_current_room(room) if room else self._known_room_for_event())
        if label == "unknown" or not label:
            label = self._known_room_for_event()
        if not label:
            return None
        try:
            return gm.record_room_event(
                room=label,
                kind=kind,
                step=self._graph_world_step(),
                phrase=phrase,
                obs_id=obs_id,
                note=note,
                agent_round=int(self._round) + 1,
            )
        except Exception as e:
            _logger.warning(f"record_room_event failed: {e}")
            return None

    def _refresh_place_coverage(self, obs_id: int) -> PlaceInspectRecord:
        """Update Investigate-card coverage= from footprint ∩ unexplored frontier."""
        oid = int(obs_id)
        rec = self._place_inspect.get(oid) or PlaceInspectRecord()
        prev_cov = str(rec.coverage or "unknown")
        gm = self.graph_memory
        voxel_map, planner = self._voxel_planner()
        cov = None
        if gm is not None and hasattr(gm, "place_coverage_for_obs"):
            try:
                cov = gm.place_coverage_for_obs(
                    oid,
                    voxel_map=voxel_map,
                    planner=planner,
                    robot_xyt=self._robot_xyt_world(),
                )
            except Exception as e:
                _logger.warning(f"place coverage refresh failed: {e}")
                cov = None
        if cov is not None:
            rec.coverage = str(getattr(cov, "status", "unknown") or "unknown")
            rec.local_frontier_cells = int(getattr(cov, "local_frontier_cells", 0) or 0)
        self._place_inspect[oid] = rec
        if prev_cov != "closed" and rec.coverage == "closed":
            self._record_room_timeline(
                kind="coverage_closed",
                obs_id=oid,
                note=f"obs {oid} local frontier closed",
            )
        return rec

    def _mark_approach_tried(
        self,
        obs_id: int,
        approach_index: int,
        *,
        target_xy: tuple[float, float] | None = None,
    ) -> None:
        oid = int(obs_id)
        rec = self._place_inspect.get(oid) or PlaceInspectRecord()
        ap = int(approach_index) % PLACE_APPROACH_SAMPLES
        if ap not in rec.tried_approaches:
            rec.tried_approaches.append(ap)
        if target_xy is not None:
            xy = (float(target_xy[0]), float(target_xy[1]))
            if all(math.hypot(xy[0] - p[0], xy[1] - p[1]) > 0.25 for p in rec.tried_xy):
                rec.tried_xy.append(xy)
        self._place_inspect[oid] = rec

    def _investigate_target_xyz(self, obs_id: int, approach_index: int) -> np.ndarray | None:
        gm = self.graph_memory
        if gm is None:
            return None
        xyt = self._robot_xyt_world()
        voxel_map, planner = self._voxel_planner()
        rec = self._place_inspect.get(int(obs_id))
        avoid = list(rec.tried_xy) if rec is not None else None

        def _as_xy(raw: Any) -> np.ndarray | None:
            if raw is None:
                return None
            try:
                arr = np.asarray(raw, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                return None
            if arr.size < 2 or not np.isfinite(arr[:2]).all():
                return None
            return np.array([float(arr[0]), float(arr[1]), 1.0], dtype=float)

        # Habitat: prefer navmesh-reachable approaches (can sit through doorways).
        try:
            from emet.controller.habitat_nav import (
                habitat_perfect_nav_enabled,
                is_habitat_robot_client,
                sample_habitat_navmesh_approach_xy,
            )

            agent = self.agent
            params = getattr(agent, "parameters", None)
            robot = getattr(agent, "robot", None)
            if (
                params is not None
                and robot is not None
                and habitat_perfect_nav_enabled(params)
                and is_habitat_robot_client(robot)
            ):
                sim = getattr(robot, "_sim", None)
                anchor = None
                if hasattr(gm, "_obs_nav_anchor"):
                    anchor = gm._obs_nav_anchor(int(obs_id))
                if sim is not None and anchor is not None:
                    robot_xy = None
                    if xyt is not None:
                        robot_xy = (float(xyt[0]), float(xyt[1]))
                    r_in = max(
                        0.35,
                        float(getattr(gm, "image_nav_min_approach_m", 0.35) or 0.35),
                    )
                    got = sample_habitat_navmesh_approach_xy(
                        sim,
                        anchor_xy=(float(anchor[0]), float(anchor[1])),
                        robot_xy=robot_xy,
                        approach_index=int(approach_index),
                        radius_inner_m=r_in,
                        avoid_xy=avoid,
                    )
                    if got is not None:
                        return np.array([float(got[0]), float(got[1]), 1.0], dtype=float)
        except Exception as e:
            _logger.debug(f"habitat navmesh approach unavailable: {e}")

        fn = getattr(gm, "_navigation_approach_waypoint_for_obs", None)
        if callable(fn):
            try:
                got = _as_xy(
                    fn(
                        int(obs_id),
                        xyt,
                        approach_index=int(approach_index),
                        n_approaches=PLACE_APPROACH_SAMPLES,
                        avoid_xy=avoid,
                        voxel_map=voxel_map,
                        planner=planner,
                    )
                )
            except TypeError:
                try:
                    got = _as_xy(fn(int(obs_id), xyt, approach_index=int(approach_index)))
                except TypeError:
                    got = None
            if got is not None:
                return got
        if hasattr(gm, "_navigation_waypoint_for_obs"):
            return _as_xy(gm._navigation_waypoint_for_obs(int(obs_id), xyt))
        # Synthetic cards (SigLIP soft-seeded search targets) carry their own xyz; the
        # graph has no node for these obs_ids, so use the hypothesis position directly.
        for h in self._hypotheses:
            if int(h.obs_id) == int(obs_id):
                return _as_xy(h.xyz)
        return None

    def _maybe_retract_claim_after_station(
        self,
        obs_id: int,
        *,
        closest_m: float | None,
        verify_out: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """If a close look says ABSENT for phrase P, stop advertising P at that obs."""
        gm = self.graph_memory
        if gm is None or not hasattr(gm, "retract_phrase_claim_at_obs"):
            return None
        if closest_m is None or float(closest_m) > 1.0:
            return None
        if not isinstance(verify_out, dict):
            return None
        status = str(verify_out.get("status") or verify_out.get("decision") or "").upper()
        if status != "ABSENT":
            return None
        evidence_obs_id = int(verify_out.get("obs_id") or obs_id)
        evidence_source = "siglip"
        if self.decision_policy == "grounded_v2":
            vlm = verify_out.get("vlm_assess")
            if not isinstance(vlm, dict) or not vlm.get("ok"):
                return None
            # Cheap ABSENT is proposal-only. Only an explicit VLM miss on this fresh
            # station view may become additive negative evidence.
            if vlm.get("present") is not False or bool(vlm.get("answerable")):
                return None
            evidence_obs_id = int(vlm.get("obs_id") or evidence_obs_id)
            evidence_source = "vlm"
        phrase = str(verify_out.get("phrase") or self._target_phrase or "").strip()
        if not phrase:
            return None
        out = gm.retract_phrase_claim_at_obs(
            int(obs_id),
            phrase,
            room=self._observation_room(evidence_obs_id) or self._known_room_for_event() or None,
            step=self._graph_world_step(),
            strip_matching_labels=self.decision_policy != "grounded_v2",
            apply_blacklist=self.decision_policy != "grounded_v2",
            evidence_obs_id=evidence_obs_id,
            evidence_source=evidence_source,
        )
        self._append_trace(
            {
                "event": "retract_claim",
                "obs_id": int(obs_id),
                "claim_obs_id": int(obs_id),
                "evidence_obs_id": evidence_obs_id,
                "evidence_source": evidence_source,
                "phrase": str(out.get("phrase") or phrase),
                "closest_m": float(closest_m),
                "room": out.get("room"),
                **{k: out.get(k) for k in ("stripped_obs", "stripped_nodes", "ok")},
            }
        )
        return out

    def _tool_investigate(
        self,
        obs_id: int,
        *,
        tool_name: str = "investigate",
        approach_index: int | None = None,
    ) -> dict[str, Any]:
        """Closer look at a place card: sample an approach, look around, verify/assess."""
        if self._n_nav + self._n_explore >= self.max_nav_steps:
            return {"ok": False, "error": "nav budget exhausted"}
        gm = self.graph_memory
        agent = self.agent
        if gm is None or not hasattr(agent, "navigate_to_target_pose"):
            return {"ok": False, "error": "nav unavailable"}
        oid = int(obs_id)
        trace_tool = "investigate" if tool_name == "investigate" else "navigate_to_obs"
        if oid in self._station_obs_ids:
            self._append_trace(
                {
                    "tool": trace_tool,
                    "ok": False,
                    "obs_id": oid,
                    "status": "STATION_OBS_NOT_PLACE",
                }
            )
            return {
                "ok": False,
                "error": (
                    f"obs_id={oid} is a capture station from a prior investigate — "
                    "not a place card; use explore_frontier or investigate a listed place"
                ),
                "status": "STATION_OBS_NOT_PLACE",
                "obs_id": oid,
            }
        prior_visits = int(self._nav_to_obs_counts.get(oid, 0))
        if self.action_progress_mode == "enforce" and approach_index is not None:
            next_ap = int(approach_index) % PLACE_APPROACH_SAMPLES
        else:
            next_ap = self._next_approach_index(oid, prefer=approach_index)
        # Exhausted orbit samples / stall — not bare "nav failed". Close+ABSENT alone
        # no longer blocks while unused approach bearings remain.
        if next_ap is None or self._hypothesis_nav_blocked(oid):
            flag = {
                "obs_id": oid,
                "visits": prior_visits,
                "status": "NAV_LOOP_BLOCKED",
                "prior": self._tried.get(oid),
                "place_inspect": (self._place_inspect[oid].card_bits() if oid in self._place_inspect else None),
                "approaches_exhausted": True,
            }
            self._nav_loop_flags.append(flag)
            self._append_trace({"tool": trace_tool, "ok": False, **flag})
            return {
                "ok": False,
                "error": (
                    f"investigate blocked on obs_id={oid} (visits={prior_visits}, "
                    f"approaches exhausted); pick another investigate card or explore_frontier"
                ),
                "status": "NAV_LOOP_BLOCKED",
                "obs_id": oid,
            }
        inv = self._investigate_hypotheses()
        hyp = next((h for h in inv if int(h.obs_id) == oid), None)
        if hyp is None:
            hyp = next((h for h in self._hypotheses if int(h.obs_id) == oid), None)
            if hyp is not None and str(hyp.source) not in INVESTIGATE_SOURCES:
                listed = sorted({int(h.obs_id) for h in inv})
                self._append_trace(
                    {
                        "tool": trace_tool,
                        "ok": False,
                        "obs_id": oid,
                        "status": "NOT_INVESTIGATE_CARD",
                        "listed_obs_ids": listed,
                    }
                )
                return {
                    "ok": False,
                    "error": (
                        f"obs_id={oid} is an explore frontier, not an investigate place; "
                        f"use investigate on {listed} or explore_frontier"
                    ),
                    "status": "NOT_INVESTIGATE_CARD",
                    "obs_id": oid,
                    "listed_obs_ids": listed,
                }
        if hyp is None and self._hypotheses:
            listed = sorted({int(h.obs_id) for h in inv} or {int(h.obs_id) for h in self._hypotheses})
            self._append_trace(
                {
                    "tool": trace_tool,
                    "ok": False,
                    "obs_id": oid,
                    "status": "OBS_NOT_IN_EVIDENCE",
                    "listed_obs_ids": listed,
                }
            )
            return {
                "ok": False,
                "error": (
                    f"obs_id={oid} is not in the Investigate list {listed}; "
                    "investigate a listed place or explore_frontier"
                ),
                "status": "OBS_NOT_IN_EVIDENCE",
                "obs_id": oid,
                "listed_obs_ids": listed,
            }
        phrase = self._target_phrase or self.query_text
        source = hyp.source if hyp is not None else "graph"
        hypothesis_id = self._begin_policy_approach(source, oid, phrase)
        xyt = self._robot_xyt()
        target = self._investigate_target_xyz(oid, next_ap)
        if target is None:
            return {"ok": False, "error": f"no waypoint for obs_id={obs_id}"}
        start = self._robot_xyt_world() if xyt is not None else np.array([0.0, 0.0, 0.0])
        # Face the OBJECT on arrival, not the approach waypoint. navigate_to_target_pose
        # with target_theta=None leaves the final yaw arbitrary (often a wall), so the
        # arrival capture sees a brick wall and the VLM assess reports present=False.
        # theta toward the object anchor from the standing waypoint makes the head look
        # at the target itself.
        try:
            t_arr = np.asarray(target, dtype=float).reshape(-1)
            look_at = t_arr[:2]
            gm = self.graph_memory
            if gm is not None and hasattr(gm, "_obs_nav_anchor"):
                anchor = gm._obs_nav_anchor(int(oid))
                if anchor is not None:
                    a_arr = np.asarray(anchor, dtype=float).reshape(-1)
                    if a_arr.size >= 2 and np.isfinite(a_arr[:2]).all():
                        look_at = a_arr[:2]
            target_theta = float(np.arctan2(look_at[1] - t_arr[1], look_at[0] - t_arr[0]))
        except Exception:
            target_theta = None
        try:
            nav_outcome = agent.navigate_to_target_pose(target, start, target_theta, target_obs_id=oid)
        except TypeError:
            nav_outcome = agent.navigate_to_target_pose(target, start, target_theta)
        finished = bool(nav_outcome.finished)
        nav_outcome_str = str(nav_outcome)
        self._n_nav += 1
        self._nav_to_obs_counts[oid] = prior_visits + 1
        # Consume this sample even on planner miss so the next call draws a new XY.
        tgt_xy = (
            float(np.asarray(target).reshape(-1)[0]),
            float(np.asarray(target).reshape(-1)[1]),
        )
        self._mark_approach_tried(oid, next_ap, target_xy=tgt_xy)
        nav_res = getattr(agent, "_last_nav_attempt", None)
        dist_m = float(getattr(nav_res, "dist_m", 0.0) or 0.0) if nav_res else 0.0
        note = str(getattr(nav_res, "note", "") or "") if nav_res else ""
        # ``finished`` is False for chunked (path >8 waypoints) plans even when the
        # robot made real progress toward the obs — in teleport mode that is the
        # common case and must not be treated as a failure. Use the NavOutcome
        # (reached/progress) as the "reached / progressing" signal.
        nav_progress = bool(nav_outcome.ok)
        from emet.controller.nav_attempt import nav_status_code

        # Ledger dual-write is owned by DynamemController._log_nav_attempt
        # (sync_nav_attempt_to_ledger). Fallback only when no result was published.
        if nav_res is None and hasattr(gm, "record_nav_attempt"):
            gm.record_nav_attempt(oid, success=nav_progress, note=note or "agentic", dist_m=dist_m)
            status = "ok" if nav_progress else "failed"
        else:
            status = nav_status_code(nav_res) if nav_res is not None else ("ok" if nav_progress else "failed")
        if not nav_progress:
            self._consecutive_nav_fail += 1
            self._tried.setdefault(oid, f"nav failed ({status})")
            if self._consecutive_nav_fail >= NAV_CONSECUTIVE_FAIL_LIMIT:
                # Stop retrying unreachable candidates: block every remaining
                # investigate obs so the router must switch or fall back to the graph.
                block = sorted({int(h.obs_id) for h in inv if not self._hypothesis_nav_blocked(int(h.obs_id))})
                self._unreachable_obs_ids.update(block)
                self._append_trace(
                    {
                        "event": "nav_fallback",
                        "reason": f"{self._consecutive_nav_fail} consecutive nav failures",
                        "blocked_obs_ids": block,
                        "obs_id": oid,
                    }
                )
                _logger.info(
                    "agentic: %d consecutive nav failures — blocking unreachable obs %s",
                    self._consecutive_nav_fail,
                    block,
                )
        else:
            self._consecutive_nav_fail = 0
        row = {
            "tool": trace_tool,
            "obs_id": oid,
            "approach_index": int(next_ap),
            "target_xyz": [float(x) for x in np.asarray(target).reshape(-1)[:3]],
            "nav_outcome": nav_outcome_str,
            "nav_success": bool(finished),
            "nav_progress": bool(nav_progress),
            "nav_dist_m": dist_m,
            "nav_note": note,
            "nav_status_code": status,
            "nav_visit_n": self._nav_to_obs_counts[oid],
        }
        self._attach_gt(row, target)
        self._append_trace(row)
        if not nav_progress:
            return {
                "ok": False,
                "status": status,
                "obs_id": oid,
                "target_xyz": row["target_xyz"],
                "approach_index": int(next_ap),
                "nav_outcome": nav_outcome_str,
                "nav_progress": False,
                "nav_finished": bool(finished),
                "nav_status_code": status,
                "nav_note": note,
                "capture": None,
                "verify": None,
            }

        # Historical FIND RGB (obs 163) must be Image 1 on the next answer even if
        # the arrival capture is a wall and verify pins that station instead.
        self._pin_eqa_look_obs(oid)
        self._refresh_room_after_motion()
        cap = self._tool_capture_and_update()
        cap_adv = isinstance(cap, dict) and cap.get("ok") and cap.get("obs_id") is not None
        station_oid = None
        # Only a successful capture advance counts as a new station view.
        if cap_adv:
            station_oid = int(cap["obs_id"])
            self._station_obs_ids.add(station_oid)
            self._fresh_obs_ids.add(station_oid)
            self._tried.pop(station_oid, None)
            scored = getattr(self._evidence_policy, "_globally_scored_obs_ids", None)
            if isinstance(scored, set):
                scored.discard(station_oid)
            self._vlm_assessed_obs_ids.discard(station_oid)
            self._policy_approached(hypothesis_id, station_oid)

        verify_out = None
        if self.mode == "answer":
            if station_oid is not None:
                # The arrival view faces the object (investigate passes target_theta
                # toward it) — verify THIS view, before any map-coverage sweep turns
                # the head away. Scoring the sweep's last pan made the assess look at
                # a wall instead of the object.
                verify_out = self.handle_tool(
                    "verify_siglip",
                    {"phrase": self._siglip_phrase(phrase), "obs_id": station_oid},
                )
            else:
                # Arrived but capture did not advance — score the live arrival view
                # (still facing the object) once, then block re-nav.
                verify_out = self._verify_stalled_nav_view(oid, phrase=phrase)
                flag = {
                    "obs_id": oid,
                    "visits": self._nav_to_obs_counts[oid],
                    "status": "STALLED_NAV_LOOP",
                    "look_around_on_no_new_obs": True,
                    "verify_status": (verify_out or {}).get("status") if isinstance(verify_out, dict) else None,
                    "approach_index": int(next_ap),
                }
                self._nav_loop_flags.append(flag)
                self._tried[oid] = f"STALLED_NAV_LOOP verify={flag.get('verify_status') or 'none'}"
                self._append_trace({"event": "nav_loop", **flag})
                _logger.warning(
                    f"agentic nav loop: obs_id={oid} visits={flag['visits']} verify={flag.get('verify_status')}"
                )

        # Sweep only for map coverage when the arrive-capture did not advance a fresh
        # view. Never overwrite the verified arrival capture with a sweep pan.
        if not cap_adv:
            self._tool_look_around(verify=False)

        closest = self._dist_to_anchor_m(oid, hyp)
        rec = self._record_place_inspect(
            oid,
            closest_m=closest,
            verify_out=verify_out,
            approach_index=next_ap,
        )
        rec = self._refresh_place_coverage(oid)
        self._maybe_retract_claim_after_station(
            oid,
            closest_m=closest,
            verify_out=verify_out if isinstance(verify_out, dict) else None,
        )
        self._append_trace(
            {
                "event": "station_inspect",
                "tool": trace_tool,
                "obs_id": oid,
                "station_obs_id": station_oid,
                "closest_m": closest,
                "approach_index": int(next_ap),
                "coverage": rec.coverage,
                "local_frontier_cells": rec.local_frontier_cells,
                "place_inspect": rec.card_bits(),
                "verify_status": (
                    (verify_out or {}).get("status") or (verify_out or {}).get("decision")
                    if isinstance(verify_out, dict)
                    else None
                ),
            }
        )
        room_stamp = self._stamp_room_after_investigate(oid, hyp=hyp, station_oid=station_oid)
        return {
            "ok": True,
            "obs_id": oid,
            "target_xyz": row["target_xyz"],
            "approach_index": int(next_ap),
            "nav_outcome": nav_outcome_str,
            "nav_progress": bool(nav_progress),
            "nav_finished": bool(finished),
            "nav_status_code": status,
            "nav_note": note,
            "capture": cap,
            "verify": verify_out,
            "look_around_on_no_new_obs": True,
            "place_inspect": rec.card_bits(),
            "coverage": rec.coverage,
            "station_obs_id": station_oid,
            "room_stamp": room_stamp,
        }

    def _tool_navigate_to_obs(self, obs_id: int) -> dict[str, Any]:
        """Compat alias — ``navigate_to_obs`` shares the investigate approach path."""
        return self._tool_investigate(int(obs_id), tool_name="navigate_to_obs")

    def _pin_eqa_look_obs(self, obs_id: int | None) -> None:
        """Next ``query_answer`` attaches this graph RGB as Image 1."""
        gm = self.graph_memory
        if gm is None or obs_id is None:
            return
        try:
            oid = int(obs_id)
        except (TypeError, ValueError):
            return
        if oid <= 0:
            return
        gm.last_eqa_look_obs_id = oid

    def _tool_look_around(self, *, verify: bool = True) -> dict[str, Any]:
        agent = self.agent
        hypothesis_id = None
        if verify:
            hypothesis_id = self._begin_policy_approach(
                "look",
                -2_000_000 - self._n_nav - self._n_explore,
                self.query_text,
            )
        ok = False
        if hasattr(agent, "look_around"):
            try:
                agent.look_around()
                ok = True
            except Exception as e:
                _logger.warning(f"look_around failed: {e}")
        if ok:
            self._refresh_room_after_motion()
        cap = self._tool_capture_and_update()
        verify_out = None
        if hypothesis_id is not None and cap.get("ok") and cap.get("obs_id") is not None:
            self._policy_approached(hypothesis_id, int(cap["obs_id"]))
            if verify and self.mode == "answer":
                verify_out = self._verify_after_motion(phrase=self.query_text)
        self._append_trace({"tool": "look_around", "ok": ok})
        return {"ok": ok, "capture": cap, "verify": verify_out}

    def _siglip_phrase(self, phrase: str = "") -> str:
        """Short object phrase for SigLIP — never feed the full MCQ question text."""
        text = (phrase or "").strip()
        q = (self.question or "").strip()
        # Callers sometimes pass query_text (== full question); prefer extracted target.
        if (not text) or (q and text == q) or ("?" in text and len(text.split()) > 6):
            text = (self._target_phrase or "").strip()
        if not text:
            # Last resort: stem without choices / trailing "Answer:"
            stem = question_stem_for_keywords(self.question or "")
            text = (stem or self.query_text or "").strip()
        return text

    def _verify_after_motion(self, *, phrase: str = "") -> dict[str, Any]:
        """Run verify on the newest captured view (router and fallback both need this)."""
        return self.handle_tool(
            "verify_siglip",
            {"phrase": self._siglip_phrase(phrase)},
        )

    def _verify_stalled_nav_view(self, obs_id: int, *, phrase: str = "") -> dict[str, Any]:
        """When capture does not advance, still score the current view once for the planner."""
        oid = int(obs_id)
        # Allow verify despite REQUIRES_FRESH_VIEW — we intentionally revisit this station.
        self._fresh_obs_ids.add(oid)
        # Clear prior same-view skip so this stall path can record ABSENT/CANDIDATE.
        self._tried.pop(oid, None)
        scored = getattr(self._evidence_policy, "_globally_scored_obs_ids", None)
        if isinstance(scored, set):
            scored.discard(oid)
        return self.handle_tool(
            "verify_siglip",
            {
                "phrase": self._siglip_phrase(phrase),
                "obs_id": oid,
            },
        )

    def _obs_revision_snapshot(self, gm: Any) -> dict[int, int]:
        """Safe obs_id→revision map (ignores MagicMock / non-int backends)."""
        out: dict[int, int] = {}
        if gm is None:
            return out
        fn = getattr(gm, "obs_revision", None)
        if not callable(fn):
            return out
        tracked: set[int] = set()
        before = self._latest_obs_id()
        if before is not None:
            tracked.add(int(before))
        for h in self._hypotheses[:5]:
            tracked.add(int(h.obs_id))
        for oid in tracked:
            try:
                out[int(oid)] = int(fn(int(oid)))
            except (TypeError, ValueError):
                continue
        return out

    def _obs_revisions_advanced(self, gm: Any, before_revs: dict[int, int]) -> list[int]:
        if gm is None or not before_revs:
            return []
        fn = getattr(gm, "obs_revision", None)
        if not callable(fn):
            return []
        advanced: list[int] = []
        check_ids = set(before_revs)
        last_u = getattr(gm, "_last_obs_content_update_id", None)
        if isinstance(last_u, int):
            check_ids.add(int(last_u))
        for oid in check_ids:
            try:
                cur = int(fn(int(oid)))
            except (TypeError, ValueError):
                continue
            if cur > int(before_revs.get(int(oid), 0)):
                advanced.append(int(oid))
        return advanced

    def _tool_capture_and_update(self) -> dict[str, Any]:
        before = self._latest_obs_id()
        gm = self.graph_memory
        before_revs = self._obs_revision_snapshot(gm)
        agent = self.agent
        if hasattr(agent, "update"):
            try:
                agent.update()
            except Exception as e:
                _logger.warning(f"capture_and_update agent.update failed: {e}")
        # Always refresh graph-side confirmed memory after a voxel update when enabled.
        if gm is not None and getattr(gm, "memory_summary_enabled", False):
            if hasattr(gm, "refresh_siglip_confirmed_memory"):
                gm.refresh_siglip_confirmed_memory()
        fresh = self._latest_obs_id()
        refreshed_ids = self._obs_revisions_advanced(gm, before_revs)

        # New observation id — full advance.
        if fresh is not None and (before is None or int(fresh) != int(before)):
            self._fresh_obs_ids.add(int(fresh))
            if self.mode == "answer" and before is not None:
                try:
                    self._refresh_hypotheses_from_graph()
                except Exception as exc:
                    _logger.warning(f"hypothesis refresh after capture failed: {exc}")
            self._last_capture_status = "OK"
            self._append_trace({"tool": "capture_and_update", "ok": True, "obs_id": fresh})
            return {"ok": True, "obs_id": fresh, "status": "NEW_OBS"}

        # Same obs_id but candidate RGB/evidence refreshed via spatial merge.
        if refreshed_ids:
            use_id = int(refreshed_ids[0])
            if fresh is not None and int(fresh) in refreshed_ids:
                use_id = int(fresh)
            refreshed_set = {int(item) for item in refreshed_ids}
            self._fresh_obs_ids.update(refreshed_set)
            # Allow re-verify: old evidence on these stable ids is stale once RGB changed.
            for refreshed_id in refreshed_set:
                self._tried.pop(refreshed_id, None)
                self._vlm_assessed_obs_ids.discard(refreshed_id)
                self._assess_history.pop(refreshed_id, None)
            self._answer_evidence = [item for item in self._answer_evidence if item.obs_id not in refreshed_set]
            if self._confirmed_answer_evidence is not None and self._confirmed_answer_evidence.obs_id in refreshed_set:
                self._confirmed_answer_evidence = None
            if self._pending_answerable is not None and self._pending_answerable.get("obs_id") in refreshed_set:
                self._pending_answerable = None
            if self._verified_obs_id in refreshed_set:
                self._verified = False
                self._verified_obs_id = None
                self._verified_evidence_event_ids = ()
                self._final_answer_decision = None
            decision_evidence = (
                self._final_answer_decision.evidence if self._final_answer_decision is not None else None
            )
            if decision_evidence is not None and decision_evidence.obs_id in refreshed_set:
                self._final_answer_decision = None
            if self._last_positive_obs_id in refreshed_set:
                self._last_positive_obs_id = None
                self._last_positive_letter = ""
            scored = getattr(self._evidence_policy, "_globally_scored_obs_ids", None)
            if isinstance(scored, set):
                scored.difference_update(refreshed_set)
            policy_invalidated = False
            for belief in self._evidence_policy.beliefs.values():
                prior_evidence = list(belief.evidence)
                belief.evidence = [item for item in prior_evidence if int(item.obs_id) not in refreshed_set]
                belief.attempted_obs_ids.difference_update(refreshed_set)
                policy_invalidated = policy_invalidated or len(belief.evidence) != len(prior_evidence)
            if policy_invalidated:
                self._evidence_policy.reset_for_new_approach()
            if self.mode == "answer":
                try:
                    self._refresh_hypotheses_from_graph()
                except Exception as exc:
                    _logger.warning(f"hypothesis refresh after content refresh failed: {exc}")
            self._last_capture_status = "CONTENT_REFRESHED"
            self._append_trace(
                {
                    "tool": "capture_and_update",
                    "ok": True,
                    "obs_id": use_id,
                    "status": "CONTENT_REFRESHED",
                    "refreshed_obs_ids": refreshed_ids,
                }
            )
            return {
                "ok": True,
                "obs_id": use_id,
                "status": "CONTENT_REFRESHED",
                "refreshed_obs_ids": refreshed_ids,
            }

        # Reject non-advancing captures (same obs_id, no candidate refresh).
        if fresh is not None and before is not None and int(fresh) == int(before):
            self._last_capture_status = "NO_NEW_OBS"
            self._append_trace(
                {
                    "tool": "capture_and_update",
                    "ok": False,
                    "obs_id": fresh,
                    "prior_obs_id": before,
                    "status": "NO_NEW_OBS",
                }
            )
            return {
                "ok": False,
                "error": "capture did not advance observation — move before re-capturing",
                "obs_id": fresh,
                "status": "NO_NEW_OBS",
            }
        self._last_capture_status = "NO_OBS"
        self._append_trace({"tool": "capture_and_update", "ok": True, "obs_id": fresh})
        return {"ok": True, "obs_id": fresh}

    def _refresh_hypotheses_from_graph(self) -> None:
        """Re-retrieve nav evidence cards after voxel/graph grew — no VLM extract."""
        gm = self.graph_memory
        if gm is None or not hasattr(gm, "hypothesize_nav_targets"):
            return
        hypotheses = self._recall_nav_hypotheses()
        self._set_hypotheses(hypotheses)

    _FIXTURE_LABEL_TOKENS = frozenset(
        {
            "cabinet",
            "counter",
            "shelf",
            "table",
            "desk",
            "dresser",
            "chest",
            "drawer",
            "stove",
            "oven",
            "refrigerator",
            "fridge",
            "microwave",
            "countertop",
        }
    )

    def _receptacle_adjacent_hypotheses(self, gm: Any) -> list[NavHypothesis]:
        """Container/fixture nodes to look at when a receptacle phrase has no direct
        place card (microwave/table/cab often sit on/under these).

        Two soft-search sources, both label-free:
          (1) SigLIP text grounding of the target phrase against the voxel semantic
              memory — the top-similarity world point is where the object is likely
              to be even if YoloE never made a labeled node for it.
          (2) container/fixture node labels (cabinet/counter/shelf/table/...) as a
              geometric fallback when SigLIP has nothing above threshold.
        """
        if gm is None:
            return []
        out: list[NavHypothesis] = []
        seen: set[int] = set()

        # (1) SigLIP soft ground: top-similarity voxel point for the target phrase.
        # A soft *explore* seed only needs to point at the most likely spot, not a
        # PRESENT-level confirmation, so use a low bar — if the semantic memory has
        # any microwave-like features we want to go look there.
        voxel_map, _ = self._voxel_planner()
        target = self._target_phrase or self._siglip_phrase()
        if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
            _logger.info(
                "[siglip-seed] target=%r voxel=%s target_phrase=%r",
                target,
                bool(voxel_map is not None),
                self._target_phrase,
            )
        if voxel_map is not None and target:
            try:
                sim = voxel_map.find_alignment_over_model(target)
                points, _, _, _ = voxel_map.semantic_memory.get_pointcloud()
                if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
                    _logger.info(
                        "[siglip-seed] sim=%s n_points=%s",
                        "None" if sim is None else f"{sim.numel()}",
                        "None" if points is None else str(tuple(points.shape)),
                    )
                if sim is not None and points is not None and sim.numel() > 0:
                    best = int(sim.cpu().argmax(dim=-1))
                    best_sim = float(sim.cpu().max(dim=-1)[0].item())
                    if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
                        _logger.info(
                            "[siglip-seed] target=%r top_sim=%.3f n_points=%d",
                            target,
                            best_sim,
                            int(points.shape[0]),
                        )
                    if best_sim > 0.12:
                        xyz = np.asarray(points[best].detach().cpu().numpy(), dtype=float).reshape(-1)[:3]
                        if xyz.size >= 3 and np.isfinite(xyz).all():
                            out.append(
                                NavHypothesis(
                                    phrase=f"siglip {self._target_phrase or target}",
                                    obs_id=-2_000_000 - len(out),
                                    xyz=xyz,
                                    score=0.0,
                                    source="siglip",
                                )
                            )
            except Exception as e:
                if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
                    _logger.warning(f"siglip receptacle seed failed for {target!r}: {e}")

        # (2) container/fixture node labels as a geometric fallback.
        if not out and hasattr(gm, "get_nodes"):
            for node in gm.get_nodes():
                if getattr(node, "is_frontier", False) or getattr(node, "is_viewpoint", False):
                    continue
                oid = int(getattr(node, "obs_id", -1))
                if oid < 0 or oid in seen:
                    continue
                labels = [str(lab).lower() for lab in (getattr(node, "labels", None) or [])]
                if not any(tok in lab for lab in labels for tok in self._FIXTURE_LABEL_TOKENS):
                    continue
                seen.add(oid)
                xyz = np.asarray(node.xyz, dtype=float).reshape(-1)
                out.append(
                    NavHypothesis(
                        phrase="nearby fixture",
                        obs_id=oid,
                        xyz=xyz[:3],
                        score=0.0,
                        source="graph",
                    )
                )
                if len(out) >= 4:
                    break
        return out

    def _set_hypotheses(self, hypotheses: list[NavHypothesis]) -> None:
        """Install recalled hyps: drop visited frontiers; prefer untried in order."""
        if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
            inv = [h for h in hypotheses if str(h.source) in INVESTIGATE_SOURCES]
            exp = [h for h in hypotheses if str(h.source) not in INVESTIGATE_SOURCES]
            _logger.info(
                "[hyps] q=%r investigate=%d (%s) explore=%d (%s)",
                self.query_text[:50],
                len(inv),
                [f"{int(h.obs_id)}:{h.phrase}" for h in inv][:8],
                len(exp),
                [f"{int(h.obs_id)}:{h.phrase}" for h in exp][:8],
            )
        filtered: list[NavHypothesis] = []
        for h in hypotheses:
            oid = int(h.obs_id)
            if str(h.source) == "frontier" and (
                int(self._nav_to_obs_counts.get(oid, 0)) >= 1 or self._hypothesis_nav_blocked(oid)
            ):
                continue
            # Capture stations are verify views, not places to orbit next.
            if str(h.source) in INVESTIGATE_SOURCES and oid in self._station_obs_ids:
                continue
            filtered.append(h)
        # Anti-echo: untried / low visits first, then tried graph/siglip for context.
        untried: list[NavHypothesis] = []
        tried: list[NavHypothesis] = []
        for h in filtered:
            oid = int(h.obs_id)
            if self._hypothesis_nav_blocked(oid) or int(self._nav_to_obs_counts.get(oid, 0)) >= 1:
                if str(h.source) != "frontier":
                    tried.append(h)
            else:
                untried.append(h)
        packed = untried + tried
        self._hypotheses = packed
        self._hyp_i = 0
        _SOURCE_PRIOR = {
            "graph": 0.55,
            "confirmed": 0.5,
            "siglip": 0.4,
            "frontier": 0.2,
        }
        for h in self._hypotheses:
            self._evidence_policy.register_hypothesis(
                f"{h.source}:{int(h.obs_id)}",
                h.phrase,
                prior_probability=_SOURCE_PRIOR.get(str(h.source), 0.3),
            )

    def _latest_obs_id(self) -> int | None:
        """Newest non-frontier observation id (the frame just captured), if any."""
        gm = self.graph_memory
        observations = list(getattr(gm, "_observations", None) or [])
        for obs in reversed(observations):
            oid = int(obs.obs_id)
            usable = getattr(gm, "_obs_usable_for_eqa_image", None)
            if usable is not None and not usable(oid):
                continue
            return oid
        return None

    def _obs_already_verified(self, obs_id: int) -> bool:
        """True when this obs was already SigLIP-scored — do not score it again.

        Bare ``\"nav failed\"`` in ``_tried`` is a transient planner miss, not a
        verify score; callers may still navigate/verify that obs_id.
        """
        oid = int(obs_id)
        if oid in self._evidence_policy.scored_obs_ids:
            return True
        tried = str(self._tried.get(oid) or "")
        if not tried or tried == "nav failed":
            return False
        if tried.startswith("STALLED_NAV_LOOP") or tried.startswith("verify "):
            return True
        # Legacy / unknown tried markers — preserve no-reverify.
        return True

    def _begin_policy_approach(self, source: str, obs_id: int, phrase: str) -> str:
        # A prior verify may have left the policy in ANSWER (a different hypothesis
        # was confirmed). Starting a new investigate must reset to a fresh
        # SEARCH→APPROACH so the next capture+assess can confirm again — otherwise
        # apply_vlm_assessment raises 'invalid in state ANSWER' and _verified never
        # updates even when the VLM keeps reporting present=True. Only reset when
        # switching to a new hypothesis (not re-verifying the same confirmed view).
        if (
            self._evidence_policy.state in (AgenticState.REPLAN, AgenticState.ANSWER)
            and self._evidence_policy.active_hypothesis_id
            and self._evidence_policy.active_hypothesis_id != f"{source}:{int(obs_id)}"
        ):
            self._evidence_policy.reset_for_new_approach()
            self._verified = False
            self._verified_obs_id = None
        elif self._evidence_policy.state == AgenticState.REPLAN:
            self._evidence_policy.replan()
            self._verified = False
            self._verified_obs_id = None
        hypothesis_id = f"{source}:{int(obs_id)}"
        self._evidence_policy.register_hypothesis(hypothesis_id, phrase)
        if self._evidence_policy.state == AgenticState.SEARCH:
            self._evidence_policy.choose(hypothesis_id)
        return hypothesis_id

    def _policy_approached(self, hypothesis_id: str, fresh_obs_id: int) -> None:
        if self._evidence_policy.active_hypothesis_id != hypothesis_id:
            return
        try:
            self._evidence_policy.approached(int(fresh_obs_id))
        except (RuntimeError, ValueError) as exc:
            _logger.warning(f"evidence-policy approach rejected: {exc}")

    def _next_untried_hypothesis(self) -> NavHypothesis | None:
        """Prefer Investigate cards with unused approach samples left."""
        for h in self._investigate_hypotheses():
            oid = int(h.obs_id)
            if self._obs_already_verified(oid):
                continue
            if self._place_approaches_exhausted(oid) or self._hypothesis_nav_blocked(oid):
                continue
            rec = self._place_inspect.get(oid)
            if rec is None or not rec.approached_close or rec.approaches_left > 0:
                return h
        for h in self._investigate_hypotheses():
            oid = int(h.obs_id)
            if self._obs_already_verified(oid):
                continue
            if self._place_approaches_exhausted(oid) or self._hypothesis_nav_blocked(oid):
                continue
            return h
        return None

    def _hypothesis_nav_blocked(self, obs_id: int) -> bool:
        """True if investigate must refuse this id (approaches/coverage exhausted / stall)."""
        oid = int(obs_id)
        if self.action_progress_mode == "enforce":
            # Semantic dispatch already chose an eligible concrete approach.
            return False
        if self._place_approaches_exhausted(oid):
            return True
        if self._next_approach_index(oid) is None:
            return True
        tried = str(self._tried.get(oid) or "")
        if tried.startswith("STALLED_NAV_LOOP"):
            return True
        # Consecutive planner misses marked this candidate unreachable.
        if oid in self._unreachable_obs_ids:
            return True
        # Hard cap so planner thrashing cannot consume the whole nav budget.
        max_attempts = PLACE_APPROACH_SAMPLES + NAV_SAME_OBS_LOOP_LIMIT
        if int(self._nav_to_obs_counts.get(oid, 0)) >= max_attempts:
            return True
        return False

    def _explore_nav_progressed(self) -> bool:
        """True if the last nav attempt made real progress (chunked path is not a miss)."""
        agent = self.agent
        if agent is None:
            return False
        nav_res = getattr(agent, "_last_nav_attempt", None)
        if nav_res is None:
            return False
        return bool(getattr(nav_res, "success", False))

    def _retire_visited_frontier(
        self,
        *,
        frontier_obs_id: int | None = None,
        frontier_xyz: Any = None,
    ) -> None:
        """Visited frontiers are not frontiers — drop them from the graph."""
        gm = self.graph_memory
        if gm is None:
            return
        if frontier_obs_id is not None and hasattr(gm, "retire_frontier_obs"):
            try:
                gm.retire_frontier_obs(int(frontier_obs_id))
            except Exception as e:
                _logger.warning(f"retire_frontier_obs({frontier_obs_id}) failed: {e}")
        if frontier_xyz is not None and hasattr(gm, "retire_frontier_near_xy"):
            try:
                gm.retire_frontier_near_xy(frontier_xyz, radius_m=1.25)
            except Exception as e:
                _logger.warning(f"retire_frontier_near_xy failed: {e}")
        # Mirror voxel mask → graph so remaining clusters stay accurate.
        agent = self.agent
        vm = getattr(agent, "voxel_map", None)
        planner = getattr(agent, "planner", None) or getattr(agent, "_planner", None)
        xyt = self._robot_xyt_world()
        if vm is not None and planner is not None and xyt is not None:
            try:
                from emet.memory.graph_eqa.dynamem_graph_hooks import sync_graph_frontier_nodes

                sync_graph_frontier_nodes(
                    graph_memory=gm,
                    voxel_map=vm,
                    planner=planner,
                    base_xyt=xyt,
                    question=self.query_text,
                )
            except Exception as e:
                _logger.warning(f"sync_graph_frontier_nodes after visit failed: {e}")

    def _dense_max_sim_for_rgb(self, phrase: str, rgb: np.ndarray | None) -> float | None:
        """Max patch-token SigLIP cosine for *phrase* vs *rgb* (MaskSigLIP space).

        Full-frame pool cosines rarely exceed ~0.12; dense max is closer to DynaMem's
        per-point features and can reach the 0.21 PRESENT bar when the object is in view.
        """
        text = (phrase or "").strip()
        if not text or rgb is None:
            return None
        enc = None
        gm = self.graph_memory
        if gm is not None:
            enc = getattr(gm, "_confirmed_memory_siglip_encoder", None)
        if enc is None:
            try:
                from emet.perception.encoders.siglip_encoder import get_shared_mask_siglip_encoder

                enc = get_shared_mask_siglip_encoder()
            except Exception:
                return None
        try:
            import torch
            import torch.nn.functional as F

            text_t = enc.encode_text(text).detach().float().reshape(-1)
            text_t = text_t / (text_t.norm() + 1e-12)
            inputs = enc._to_model_inputs(enc.processor(images=np.asarray(rgb, dtype=np.uint8), return_tensors="pt"))
            with torch.no_grad():
                out = enc.model.vision_model(inputs["pixel_values"], output_hidden_states=True)
                feat = F.normalize(out.last_hidden_state.float(), dim=-1)
                sims = feat @ text_t.to(device=feat.device, dtype=feat.dtype).reshape(-1, 1)
                return float(sims.max().item())
        except Exception as e:
            _logger.warning(f"dense_max_sim_for_rgb failed: {e}")
            return None

    def _voxel_max_sim_for_obs(self, phrase: str, obs_id: int) -> tuple[float, str] | None:
        """Max per-point SigLIP cosine for *phrase* on voxel features in *obs_id*.

        Full-frame ``encode_image`` cosines top out ~0.10–0.16 in Habitat, while DynaMem's
        0.21 / 0.28 thresholds were calibrated on dense voxel point features. Prefer those
        when the voxel map is available so PRESENT can actually fire.
        """
        text = (phrase or "").strip()
        if not text:
            return None
        vm = getattr(self.agent, "voxel_map", None)
        if vm is None or not hasattr(vm, "find_alignment_over_model"):
            return None
        try:
            alignments = vm.find_alignment_over_model(text)
        except Exception:
            return None
        if alignments is None:
            return None
        try:
            import torch

            if not isinstance(alignments, torch.Tensor):
                return None
        except ImportError:
            return None
        a = alignments.detach().cpu().float().reshape(-1)
        if a.numel() == 0:
            return None
        sm = getattr(vm, "semantic_memory", None)
        counts = getattr(sm, "_obs_counts", None) if sm is not None else None
        channel = "voxel_global"
        if counts is not None:
            try:
                import torch

                c = counts.detach().cpu().long().reshape(-1)
                if c.numel() == a.numel():
                    mask = c == int(obs_id)
                    if bool(mask.any()):
                        a = a[mask]
                        channel = "voxel_obs"
            except Exception:
                pass
        return float(a.max().item()), channel

    def _detector_for_verify(self) -> Any | None:
        if self._presence_detector_initialized:
            return self._presence_detector
        self._presence_detector_initialized = True
        backend = os.environ.get("EMET_EQA_AGENTIC_VERIFIER", "").strip().lower()
        if backend in ("", "none", "siglip"):
            return None
        try:
            from emet.eval.presence_verifiers import (
                OwlV2PresenceDetector,
                YoloEPresenceDetector,
            )

            if backend == "owlv2":
                self._presence_detector = OwlV2PresenceDetector()
            elif backend == "yoloe":
                self._presence_detector = YoloEPresenceDetector()
            else:
                raise ValueError(f"unsupported EMET_EQA_AGENTIC_VERIFIER={backend!r}")
        except Exception as exc:
            _logger.warning(f"presence detector unavailable: {exc}")
            self._presence_detector = None
        return self._presence_detector

    def _graph_label_matches(self, phrase: str, obs_id: int) -> bool:
        gm = self.graph_memory
        stored = gm._observation_by_id(int(obs_id)) if gm is not None and hasattr(gm, "_observation_by_id") else None
        labels = getattr(stored, "labels", None) or []
        return any(label_matches_relevant_object(phrase, str(label)) for label in labels)

    def _relation_sufficient_for_obs(self, obs_id: int) -> bool:
        from emet.habitat.metrics import (
            choices_are_count_mcq,
            choices_are_location_mcq,
            parse_mcq_choices_from_question,
        )

        choices = parse_mcq_choices_from_question(self.question)
        if not choices:
            choices = list(getattr(self, "_mcq_choices", None) or [])
        if not choices:
            return True
        # Cardinality MCQs: fused target presence is enough; the answer VLM counts.
        if choices_are_count_mcq(choices):
            return True
        if not choices_are_location_mcq(choices):
            return True
        gm = self.graph_memory
        stored = gm._observation_by_id(int(obs_id)) if gm is not None and hasattr(gm, "_observation_by_id") else None
        labels = [str(label) for label in (getattr(stored, "labels", None) or [])]
        if any(label_matches_relevant_object(choice, label) for choice in choices for label in labels):
            return True
        # Landmark overlap with nearby graph nodes (room/fixture context).
        if gm is not None and hasattr(gm, "labels_near_obs"):
            try:
                near_labels = [str(x) for x in (gm.labels_near_obs(int(obs_id)) or [])]
            except Exception:
                near_labels = []
            if any(label_matches_relevant_object(choice, label) for choice in choices for label in near_labels):
                return True
        return False

    def _inventory_labels(self, *, limit: int = 12) -> list[str]:
        gm = self.graph_memory
        labels: list[str] = []
        seen: set[str] = set()
        for obs in list(getattr(gm, "_observations", None) or []):
            for lab in list(getattr(obs, "labels", None) or []):
                s = str(lab).strip()
                if not s or s.lower() in seen:
                    continue
                seen.add(s.lower())
                labels.append(s)
                if len(labels) >= limit:
                    return labels
        return labels

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

    def _frontier_pick_out_dir(self) -> Path:
        """Directory for numbered pick panels (episode bundle when available)."""
        if getattr(self, "_frontier_pick_dir", None):
            out = Path(self._frontier_pick_dir)
            out.mkdir(parents=True, exist_ok=True)
            return out
        ep = getattr(self.agent, "_episode_debug_dir", None) or os.environ.get("EMET_EQA_EPISODE_DIR")
        if ep:
            out = Path(str(ep)).expanduser() / "frontier_picks"
        elif self._trace_path is not None:
            out = self._trace_path.parent / "frontier_picks"
        else:
            out = Path.home() / ".cache" / "habitat_eqa" / "frontier_picks"
        out.mkdir(parents=True, exist_ok=True)
        self._frontier_pick_dir = out
        self.agent._frontier_pick_dir = str(out)
        return out

    def _save_frontier_pick_panel(
        self,
        frontier_xyz: Any,
        *,
        robot_xyt_before: np.ndarray | None = None,
    ) -> Path | None:
        """Write a numbered frontier-pick panel into the episode bundle (best-effort)."""
        if frontier_xyz is None:
            return None
        try:
            arr = np.asarray(frontier_xyz, dtype=float).reshape(-1)
            if arr.size < 2:
                return None
            pick = (float(arr[0]), float(arr[1]))
            self._frontier_pick_waypoints.append(pick)

            voxel_map = getattr(self.agent, "voxel_map", None)
            if voxel_map is None or not hasattr(voxel_map, "get_2d_map"):
                return None
            obstacles, explored = voxel_map.get_2d_map()
            go = getattr(voxel_map, "grid_origin", np.array([0.0, 0.0]))
            if hasattr(go, "detach"):
                go = go.detach().cpu().numpy()
            go = np.asarray(go, dtype=np.float64).reshape(-1)[:2]
            res = float(getattr(voxel_map, "grid_resolution", 0.1) or 0.1)

            robot_xy = None
            if robot_xyt_before is not None:
                r = np.asarray(robot_xyt_before, dtype=float).reshape(-1)
                if r.size >= 2:
                    robot_xy = (float(r[0]), float(r[1]))

            from emet.visualization.frontier_pick_viz import (
                frontier_mask_from_explored,
                render_frontier_pick_rgb,
                save_frontier_pick_rgb,
            )

            n = len(self._frontier_pick_waypoints)
            dist_m = 0.0
            if robot_xy is not None:
                dist_m = float(np.hypot(pick[0] - robot_xy[0], pick[1] - robot_xy[1]))
            title = f"iteration {n - 1} — pick {dist_m:.1f} m ahead ({n} waypoints)"
            rgb = render_frontier_pick_rgb(
                obstacles,
                explored,
                frontier=frontier_mask_from_explored(explored, obstacles),
                robot_xy=robot_xy,
                chosen_xy=pick,
                waypoints=list(self._frontier_pick_waypoints),
                grid_origin_xy=go,
                grid_resolution=res,
                title=title,
            )
            out_dir = self._frontier_pick_out_dir()
            path = save_frontier_pick_rgb(rgb, out_dir / f"iter_{n - 1:02d}.png")
            paths = list(getattr(self.agent, "_frontier_pick_panels", []) or [])
            paths.append(str(path))
            self.agent._frontier_pick_panels = paths
            return path
        except Exception as e:
            _logger.warning(f"frontier pick panel failed: {e}")
            return None

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

    def _tool_verify_siglip(self, phrase: str, obs_id: int | None) -> dict[str, Any]:
        gm = self.graph_memory
        if gm is None:
            return {"ok": False, "error": "no graph_memory"}
        text = (phrase or "").strip()
        if not text:
            if self._target_phrase:
                text = self._target_phrase
            else:
                phrases = list(getattr(gm, "_relevant_phrases", None) or []) + list(
                    getattr(gm, "_relevant_objects", None) or []
                )
                # Prefer phrases from the question stem over MCQ-option nouns
                # (``fruit bowl`` > ``kitchen island``), then noun compounds over
                # leading verb fillers (``fruit bowl`` > ``looking``).
                stem = question_stem_for_keywords(self.question).lower()
                ranked = sorted(
                    phrases,
                    key=lambda p: (
                        1 if (p or "").strip().lower() in stem else 0,
                        0 if (p or "").split()[:1] and (p or "").split()[0].lower() in _QUESTION_VERB_FILLERS else 1,
                        len((p or "").split()),
                        len(p or ""),
                    ),
                    reverse=True,
                )
                text = ranked[0] if ranked else self.question
        oid = obs_id
        if oid is None or int(oid) < 0:
            # No obs_id means "verify what the robot is looking at now". Motion tools call
            # this right after capture_and_update, so the newest observation is the frame
            # just taken; falling back to a hypothesis re-verified the same stale obs every
            # round while the robot explored the far side of the scene (q104/q105).
            oid = self._latest_obs_id()
        if oid is None:
            if self._hypotheses:
                oid = int(self._hypotheses[min(self._hyp_i, len(self._hypotheses) - 1)].obs_id)
            elif getattr(gm, "last_eqa_obs_ids", None):
                oid = int(gm.last_eqa_obs_ids[0])
            else:
                return {"ok": False, "error": "no obs_id"}
        oid = int(oid)
        verify_target = self._action_target_for_obs(oid)
        if (
            self._router_enabled
            and self._evidence_policy.state != AgenticState.VERIFY
            and oid not in self._fresh_obs_ids
        ):
            return {
                "ok": False,
                "error": (f"obs_id {oid} is stale; SEARCH must APPROACH/capture a fresh view before VERIFY"),
                "status": "REQUIRES_FRESH_VIEW",
                "obs_id": oid,
                "phrase": text,
                "target_kind": verify_target.kind,
                "target_id": verify_target.stable_id,
                "view_id": verify_target.view_id,
                "room": verify_target.room,
                "verified": self._verified,
            }
        # Interactive rule: one verify per view. Re-checking the same obs burns rounds
        # without new evidence — move first (nav / explore), then verify the fresh frame.
        if self._obs_already_verified(oid):
            self._append_trace(
                {
                    "tool": "verify_siglip",
                    "phrase": text,
                    "obs_id": oid,
                    "decision": "SKIPPED_SAME_VIEW",
                    "sim": 0.0,
                    "prior": self._tried.get(oid),
                }
            )
            return {
                "ok": False,
                "error": f"obs_id {oid} already verified ({self._tried.get(oid)}); navigate or explore for a new view",
                "status": "SKIPPED_SAME_VIEW",
                "obs_id": oid,
                "phrase": text,
                "target_kind": verify_target.kind,
                "target_id": verify_target.stable_id,
                "view_id": verify_target.view_id,
                "room": verify_target.room,
                "verified": self._verified,
            }
        rgb = None
        live_obs = None
        robot = getattr(self.agent, "robot", None)
        if robot is not None and hasattr(robot, "get_observation"):
            try:
                live_obs = robot.get_observation()
                if live_obs is not None and getattr(live_obs, "rgb", None) is not None:
                    rgb = np.asarray(live_obs.rgb)
            except Exception:
                pass
        if rgb is None:
            stored = gm._observation_by_id(int(oid)) if hasattr(gm, "_observation_by_id") else None
            stored_rgb = getattr(stored, "rgb", None) if stored is not None else None
            if isinstance(stored_rgb, np.ndarray) and stored_rgb.ndim == 3:
                rgb = np.asarray(stored_rgb)
        result = gm.verify_phrase_at_obs(text, int(oid), rgb=rgb, min_sim=self.verify_min_sim)
        full_frame_sim = float(result.sim)
        voxel = self._voxel_max_sim_for_obs(text, int(oid))
        voxel_sim = float(voxel[0]) if voxel is not None else None
        voxel_ch = voxel[1] if voxel is not None else None
        dense_sim = self._dense_max_sim_for_rgb(text, rgb)
        detector_evidence = None
        detector = self._detector_for_verify()
        if detector is not None and rgb is not None:
            try:
                from emet.eval.presence_verifiers import detector_crop_evidence

                enc = getattr(gm, "_confirmed_memory_siglip_encoder", None)
                detector_evidence = (
                    detector_crop_evidence(detector, enc, rgb, text) if enc is not None else detector.score(rgb, text)
                )
            except Exception as exc:
                _logger.warning(f"hybrid presence verify failed: {exc}")
        # Best image-space score (full-frame pool vs dense patch).
        best_img = full_frame_sim
        verify_channel = "full_frame"
        if dense_sim is not None and float(dense_sim) > best_img:
            best_img = float(dense_sim)
            verify_channel = "dense_patch"
        # Voxel wins when it clears the DynaMem bar; else use image three-band.
        if voxel_sim is not None and voxel_sim >= SIGLIP_PRESENT_THRESHOLD:
            status, ok, sim_out, verify_channel = "PRESENT", True, float(voxel_sim), str(voxel_ch)
        elif best_img >= SIGLIP_IMAGE_PRESENT_THRESHOLD:
            status, ok, sim_out = "PRESENT", True, float(best_img)
        elif best_img >= SIGLIP_IMAGE_ABSENT_THRESHOLD:
            status, ok, sim_out = "CANDIDATE", False, float(best_img)
        else:
            status, ok, sim_out = "ABSENT", False, float(best_img)
        result = VerifyResult(
            status=status,
            sim=float(sim_out),
            obs_id=int(oid),
            phrase=text,
            ok=ok,
            text_feat=result.text_feat,
            img_feat=result.img_feat,
        )
        self._last_verify = result
        self._tried[int(result.obs_id)] = f"verify {result.status} sim={float(result.sim):.2f}"
        self._fresh_obs_ids.discard(int(result.obs_id))
        # After motion, capture may not mint a new obs_id (unit mocks / no map growth).
        # Complete APPROACH → VERIFY on the view we are scoring; otherwise open a view hyp.
        if self._evidence_policy.state == AgenticState.APPROACH:
            active = self._evidence_policy.active_hypothesis_id
            if active is not None:
                self._policy_approached(active, int(result.obs_id))
        if self._evidence_policy.state != AgenticState.VERIFY:
            hypothesis_id = self._begin_policy_approach("view", int(result.obs_id), text)
            self._policy_approached(hypothesis_id, int(result.obs_id))
        hypothesis_id = self._evidence_policy.active_hypothesis_id or f"view:{int(result.obs_id)}"
        graph_label_match = self._graph_label_matches(text, int(result.obs_id))
        assessment = None
        try:
            evidence = EvidenceRecord(
                hypothesis_id=hypothesis_id,
                obs_id=int(result.obs_id),
                phrase=text,
                full_frame_sim=full_frame_sim,
                dense_sim=dense_sim,
                voxel_sim=voxel_sim,
                detector_score=(float(detector_evidence.score) if detector_evidence is not None else None),
                crop_siglip_sim=(detector_evidence.crop_siglip_sim if detector_evidence is not None else None),
                graph_label_match=graph_label_match,
                detector_backend=(str(detector_evidence.backend) if detector_evidence is not None else None),
                bbox_xyxy=(detector_evidence.bbox_xyxy if detector_evidence is not None else None),
                provenance=tuple(
                    channel
                    for channel, value in (
                        ("full_frame", full_frame_sim),
                        ("dense_patch", dense_sim),
                        (str(voxel_ch or "voxel"), voxel_sim),
                        (
                            str(detector_evidence.backend) if detector_evidence is not None else "detector",
                            detector_evidence.score if detector_evidence is not None else None,
                        ),
                    )
                    if value is not None
                ),
            )
            self._evidence_policy.add_evidence(evidence)
            assessment = self._evidence_policy.assess(
                relation_sufficient=self._relation_sufficient_for_obs(int(result.obs_id))
            )
        except (RuntimeError, ValueError) as exc:
            _logger.warning(f"evidence-policy verify rejected: {exc}")
        # Cheap channels never unlock submit — multimodal VLM assess is the gate.
        proposal = {
            "phrase": text,
            "detector_score": (float(detector_evidence.score) if detector_evidence is not None else None),
            "dense_sim": dense_sim,
            "full_frame_sim": full_frame_sim,
            "decision": result.status,
            "obs_id": int(result.obs_id),
        }
        proposal_threshold = (
            SIGLIP_PRESENT_THRESHOLD if verify_channel.startswith("voxel") else SIGLIP_IMAGE_PRESENT_THRESHOLD
        )
        proposal_event_id = self._persist_agentic_evidence(
            stage="siglip_proposal",
            outcome=result.status.lower(),
            obs_id=int(result.obs_id),
            phrase=text,
            confidence=min(1.0, max(0.0, float(result.sim))),
            source=verify_channel,
            score=float(result.sim),
            threshold=float(proposal_threshold),
            payload={
                "full_frame_sim": full_frame_sim,
                "dense_sim": dense_sim,
                "voxel_sim": voxel_sim,
                "graph_label_match": bool(graph_label_match),
            },
        )
        proposal["evidence_event_id"] = proposal_event_id or None
        vlm_out = self._run_vlm_view_assess(
            rgb=rgb,
            phrase=text,
            obs_id=int(result.obs_id),
            proposal=proposal,
        )
        row = {
            "tool": "verify_siglip",
            "phrase": text,
            "obs_id": int(result.obs_id),
            "target_kind": verify_target.kind,
            "target_id": verify_target.stable_id,
            "view_id": verify_target.view_id,
            "room": verify_target.room,
            "sim": float(result.sim),
            "decision": result.status,
            "verify_channel": verify_channel,
            "full_frame_sim": full_frame_sim,
            "voxel_sim": voxel_sim,
            "dense_sim": dense_sim,
            "detector_backend": (detector_evidence.backend if detector_evidence is not None else None),
            "detector_score": (float(detector_evidence.score) if detector_evidence is not None else None),
            "detector_bbox_xyxy": (
                list(detector_evidence.bbox_xyxy)
                if detector_evidence is not None and detector_evidence.bbox_xyxy is not None
                else None
            ),
            "crop_siglip_sim": (detector_evidence.crop_siglip_sim if detector_evidence is not None else None),
            "graph_label_match": graph_label_match,
            "policy_state": self._evidence_policy.state,
            "presence_probability": (assessment.presence_probability if assessment is not None else None),
            "answerability_probability": (
                self._evidence_policy.beliefs[self._evidence_policy.active_hypothesis_id].answerability_probability
                if self._evidence_policy.active_hypothesis_id
                and self._evidence_policy.active_hypothesis_id in self._evidence_policy.beliefs
                else (assessment.answerability_probability if assessment is not None else None)
            ),
            "positive_channels": (list(assessment.positive_channels) if assessment is not None else []),
            "contradiction_channels": (list(assessment.contradiction_channels) if assessment is not None else []),
            # Submit unlock = VLM answerable (not cheap SigLIP/OWL alone).
            "fused_verified": bool(self._verified),
            "vlm_answerable": bool(self._verified),
            "answerable": bool(self._evidence_policy.state == AgenticState.ANSWER),
            "vlm_assess": vlm_out,
            "present_bar": (
                SIGLIP_PRESENT_THRESHOLD if verify_channel.startswith("voxel") else SIGLIP_IMAGE_PRESENT_THRESHOLD
            ),
            "absent_bar": None if verify_channel.startswith("voxel") else SIGLIP_IMAGE_ABSENT_THRESHOLD,
            "text_feat": _feat_list(result.text_feat),
            "img_feat": _feat_list(result.img_feat),
            "siglip_evidence_event_id": proposal_event_id or None,
        }
        labeler = getattr(robot, "hm3d_semantic_labeler", None) if robot is not None else None
        semantic = getattr(live_obs, "semantic", None) if live_obs is not None else None
        if labeler is not None and semantic is not None and hasattr(labeler, "visibility_for_phrase"):
            row.update(
                labeler.visibility_for_phrase(
                    semantic,
                    text,
                    getattr(live_obs, "depth", None),
                )
            )
        xyt = self._robot_xyt_world()
        if xyt is not None:
            row["xyt"] = [float(x) for x in xyt.reshape(-1)[:3]]
        hyp = next((h for h in self._hypotheses if int(h.obs_id) == int(result.obs_id)), None)
        if hyp is not None:
            row["target_xyz"] = [float(x) for x in np.asarray(hyp.xyz).reshape(-1)[:3]]
            row["source"] = hyp.source
            self._attach_gt(row, hyp.xyz)
        else:
            row["source"] = "current_view"
        self._append_trace(row)
        return {
            "ok": True,
            "status": result.status,
            "sim": float(result.sim),
            "verified": self._verified,
            "vlm_answerable": bool(self._verified),
            "obs_id": int(result.obs_id),
            "phrase": text,
            "target_kind": verify_target.kind,
            "target_id": verify_target.stable_id,
            "view_id": verify_target.view_id,
            "room": verify_target.room,
            "verify_channel": verify_channel,
            "fused_verified": bool(self._verified),
            "answerable": bool(self._evidence_policy.state == AgenticState.ANSWER),
            "presence_probability": (assessment.presence_probability if assessment is not None else None),
            "answerability_probability": (
                self._evidence_policy.beliefs[self._evidence_policy.active_hypothesis_id].answerability_probability
                if self._evidence_policy.active_hypothesis_id
                and self._evidence_policy.active_hypothesis_id in self._evidence_policy.beliefs
                else (assessment.answerability_probability if assessment is not None else None)
            ),
            "siglip_evidence_event_id": proposal_event_id or None,
            "verified_evidence_event_ids": list(self._verified_evidence_event_ids),
            "vlm_assess": vlm_out,
        }

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

    def _frontier_count(self) -> int:
        gm = self.graph_memory
        try:
            nodes = gm.get_nodes() if gm is not None else []
            return sum(1 for n in nodes if getattr(n, "is_frontier", False))
        except Exception:
            return 0

    def _explore_done(self) -> bool:
        if self._n_nav + self._n_explore >= self.max_nav_steps:
            return True
        return (self._n_nav + self._n_explore) > 0 and self._frontier_count() == 0

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
        if conf and not unknownish:
            return False
        count_mcq = choices_are_count_mcq(parse_mcq_choices_from_question(self.question))
        # Unconfident count + Action:N: the integer is from the attached (wrong) RGB.
        if not unknownish and not (count_mcq and not conf):
            return False
        obs_id = getattr(gm, "last_eqa_action_obs_id", None)
        if obs_id is not None:
            oid = int(obs_id)
            if oid not in self._followed_eqa_actions:
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

    def _rendered_action_allowlist(self) -> dict[str, tuple[Any, ...]]:
        snapshot = self._last_agent_state_snapshot
        return {
            "place_ids": tuple(getattr(snapshot, "visible_place_ids", ()) or ()),
            "place_obs_ids": tuple(getattr(snapshot, "visible_place_obs_ids", ()) or ()),
            "frontier_ids": tuple(getattr(snapshot, "visible_frontier_ids", ()) or ()),
            "event_ids": tuple(getattr(snapshot, "visible_event_ids", ()) or ()),
        }

    def _next_rendered_hypothesis(self) -> NavHypothesis | None:
        visible = {int(item) for item in self._rendered_action_allowlist()["place_obs_ids"]}
        if not visible:
            return None
        candidates = [
            item
            for item in self._investigate_hypotheses()
            if int(item.obs_id) in visible
            and not self._hypothesis_nav_blocked(int(item.obs_id))
            and (self.action_progress_mode == "enforce" or int(item.obs_id) not in self._tried)
        ]
        return candidates[0] if candidates else None

    def _rendered_frontier_args(self, *, toward: str = "") -> dict[str, Any]:
        args: dict[str, Any] = {}
        frontiers = self._rendered_action_allowlist()["frontier_ids"]
        if frontiers:
            args["frontier_id"] = str(frontiers[0])
        if toward:
            args["toward"] = toward
        return args

    def _validate_rendered_tool_calls(
        self,
        calls: list[tuple[str, dict[str, Any]]],
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
        """Reject stable IDs absent from the exact state rendered this turn."""
        if self.decision_policy != "grounded_v2":
            return calls, []
        allowlist = self._last_rendered_action_allowlist
        allowed_obs = {int(item) for item in allowlist.get("place_obs_ids", ())}
        frontier_order = tuple(str(item) for item in allowlist.get("frontier_ids", ()) if str(item))
        allowed_frontiers = set(frontier_order)
        accepted: list[tuple[str, dict[str, Any]]] = []
        rejected: list[dict[str, Any]] = []
        for name, args in calls:
            if name in {"investigate", "navigate_to_obs"}:
                try:
                    obs_id = int(args.get("obs_id"))
                except (TypeError, ValueError):
                    rejected.append({"tool": name, "reason": "invalid_obs_id", "value": args.get("obs_id")})
                    continue
                if obs_id not in allowed_obs:
                    rejected.append({"tool": name, "reason": "obs_id_not_rendered", "value": obs_id})
                    continue
            elif name == "explore_frontier":
                frontier_id = str(args.get("frontier_id") or "").strip()
                if not frontier_id and self.action_progress_mode == "enforce":
                    if not frontier_order:
                        rejected.append(
                            {
                                "tool": name,
                                "reason": "no_eligible_frontier",
                                "value": None,
                            }
                        )
                        continue
                    args = {**args, "frontier_id": frontier_order[0]}
                    frontier_id = frontier_order[0]
                if frontier_id and frontier_id not in allowed_frontiers:
                    rejected.append(
                        {
                            "tool": name,
                            "reason": "frontier_id_not_rendered",
                            "value": frontier_id,
                        }
                    )
                    continue
            if self.action_progress_mode == "enforce" and name in {
                "investigate",
                "navigate_to_obs",
                "verify_siglip",
                "explore_frontier",
            }:
                decision = self._action_gate_decision(name, args)
                if not decision.allowed:
                    rejected.append(
                        {
                            "tool": name,
                            "reason": decision.disposition,
                            "value": decision.signature.target.stable_id,
                            "detail": decision.reason,
                        }
                    )
                    continue
                if decision.signature.family == "inspect_place":
                    approach = decision.signature.variant_value("approach_index")
                    if approach is not None:
                        args = {**args, "approach_index": int(approach)}
            accepted.append((name, args))
        return accepted, rejected

    def _fallback_tool(self) -> tuple[str, dict[str, Any]]:
        """Deterministic tool when VLM emits nothing parseable (or router is off).

        Thin scaffold only — prefer the VLM router. Interactive loop:
          (1) explore / inspect → hypotheses
          (2) navigate → capture → verify (+ Qwen assess)
          (3) Qwen answerable → submit with its suggested letter; else keep exploring
        """
        if self.mode == "explore":
            no_eligible_frontier = (
                self.action_progress_mode == "enforce"
                and self._last_agent_state_snapshot is not None
                and not self._rendered_action_allowlist()["frontier_ids"]
            )
            if self._explore_done() or no_eligible_frontier:
                return "finish", {}
            return "explore_frontier", self._rendered_frontier_args()
        budget_left = self._n_nav + self._n_explore < self.max_nav_steps
        if budget_left and self._ovmm_prefers_nearby_investigate():
            near = self._nearby_untried_investigate_hyp()
            if near is not None:
                return "investigate", {"obs_id": int(near.obs_id)}
        # (3) Qwen said this view is enough
        if self._evidence_policy.state == AgenticState.ANSWER and self._verified:
            prefer = ""
            if self._last_vlm_assess:
                prefer = str(self._last_vlm_assess.get("suggested_answer") or "").strip()
            return "submit_answer", ({"answer": prefer} if prefer else {})
        # If Qwen asked for more views, honor that before burning the budget on submit.
        budget_left = self._n_nav + self._n_explore < self.max_nav_steps
        need_more = bool(self._last_vlm_assess and self._last_vlm_assess.get("need_more_views"))
        frontiers_gone = (self._n_nav + self._n_explore) > 0 and self._frontier_count() == 0
        if (
            self.action_progress_mode == "enforce"
            and self._last_agent_state_snapshot is not None
            and not self._rendered_action_allowlist()["frontier_ids"]
        ):
            frontiers_gone = True
        if need_more and budget_left and not frontiers_gone:
            return "explore_frontier", self._rendered_frontier_args()
        # After a close ABSENT look, grow coverage before the next investigate —
        # but only once; if we already explored this streak and place cards remain,
        # look closer instead of frontier-only loops.
        if self.decision_policy != "grounded_v2" and budget_left and not frontiers_gone and self._prefer_explore:
            streak = int(getattr(self, "_n_consecutive_explore", 0) or 0)
            near = self._nearby_untried_investigate_hyp() if self._ovmm_prefers_nearby_investigate() else None
            if near is not None:
                return "investigate", {"obs_id": int(near.obs_id)}
            hyp = (
                self._next_rendered_hypothesis()
                if self.decision_policy == "grounded_v2" and self._last_agent_state_snapshot is not None
                else self._next_untried_hypothesis()
            )
            if streak >= 1 and hyp is not None:
                return "investigate", {"obs_id": int(hyp.obs_id)}
            return "explore_frontier", self._rendered_frontier_args()
        # Soft cap: too many explores in a row with unused place cards → investigate.
        if budget_left and self.decision_policy != "grounded_v2":
            streak = int(getattr(self, "_n_consecutive_explore", 0) or 0)
            if streak >= EXPLORE_STREAK_FORCE_INVESTIGATE:
                hyp = self._next_untried_hypothesis()
                if hyp is not None:
                    return "investigate", {"obs_id": int(hyp.obs_id)}
        # (2) move in to an untried hypothesis (verify is chained after nav)
        if budget_left:
            h = (
                self._next_rendered_hypothesis()
                if self.decision_policy == "grounded_v2" and self._last_agent_state_snapshot is not None
                else self._next_untried_hypothesis()
            )
            if h is not None:
                return "investigate", {"obs_id": int(h.obs_id)}
        # (1) keep exploring for new views while budget remains
        if budget_left and not frontiers_gone:
            return "explore_frontier", self._rendered_frontier_args()
        # One first-look verify only if we have a fresh untried current view.
        latest = self._latest_obs_id()
        if self._last_verify is None and latest is not None and not self._obs_already_verified(latest):
            return "verify_siglip", {"obs_id": int(latest)}
        if self._require_verified and not self._verified:
            if budget_left and not frontiers_gone:
                return "explore_frontier", self._rendered_frontier_args()
            return "submit_answer", {}
        prefer = ""
        if self._last_vlm_assess:
            prefer = str(self._last_vlm_assess.get("suggested_answer") or "").strip()
        return "submit_answer", ({"answer": prefer} if prefer else {})

    def _ensure_router_prompt(self) -> None:
        """Build the tool registry + fixed system prompt once (stable string → prefix-cache hits)."""
        if self._tools is None:
            self._tools = build_agentic_eqa_tools(self)
            self._tool_names = {t.name for t in self._tools}
            self._system_prompt = build_graph_eqa_system_prompt(
                self._tools,
                room_policy=self.room_policy,
                decision_policy=self.decision_policy,
            )

    def _live_rgb(self) -> np.ndarray | None:
        robot = getattr(self.agent, "robot", None)
        if robot is not None and hasattr(robot, "get_observation"):
            try:
                live_obs = robot.get_observation()
                if live_obs is not None and getattr(live_obs, "rgb", None) is not None:
                    return np.asarray(live_obs.rgb)
            except Exception:
                pass
        gm = self.graph_memory
        if gm is not None:
            obs_list = list(getattr(gm, "_observations", None) or [])
            for obs in reversed(obs_list):
                rgb = getattr(obs, "rgb", None)
                if isinstance(rgb, np.ndarray) and rgb.ndim == 3:
                    return np.asarray(rgb)
        return None

    def _room_visual_pack(self) -> tuple[list[Any], list[dict[str, Any]], str]:
        """Live RGB + nearby object RGBs for multimodal router room judgment.

        Returns (payload_parts, nearby_meta, caption_prefix). ``EMET_EQA_ROUTER_ROOM_IMAGES=0``
        disables images (text-only / speed baseline).
        """
        from PIL import Image

        k = env_eqa_router_room_images()
        nearby_meta: list[dict[str, Any]] = []
        if k <= 0:
            self._last_router_n_images = 0
            return [], nearby_meta, ""

        parts: list[Any] = []
        captions: list[str] = []
        live = self._live_rgb()
        if live is not None:
            parts.append(Image.fromarray(np.asarray(live, dtype=np.uint8)))
            captions.append("Image 1: current robot view")

        gm = self.graph_memory
        xyt = self._robot_xyt_world()
        if gm is not None and hasattr(gm, "nearby_object_observations") and xyt is not None:
            try:
                nearby = gm.nearby_object_observations(xyt, k=k, max_dist_m=5.0)
            except Exception as e:
                _logger.warning(f"nearby_object_observations failed: {e}")
                nearby = []
            if not isinstance(nearby, list):
                nearby = []
            for item in nearby:
                if not isinstance(item, dict):
                    continue
                rgb = item.get("rgb")
                if not isinstance(rgb, np.ndarray):
                    continue
                parts.append(Image.fromarray(np.asarray(rgb, dtype=np.uint8)))
                idx = len(parts)
                cap = (
                    f"Image {idx}: nearby obs_id={item.get('obs_id')} "
                    f"phrase={item.get('phrase')!r} labels={item.get('labels')} "
                    f"dist={item.get('dist_m')}m"
                )
                captions.append(cap)
                nearby_meta.append(
                    {
                        "obs_id": item.get("obs_id"),
                        "dist_m": item.get("dist_m"),
                        "phrase": item.get("phrase"),
                        "labels": item.get("labels"),
                    }
                )

        self._last_router_n_images = len(parts)
        prefix = ""
        if captions:
            prefix = "Room context images (use for current_room):\n" + "\n".join(captions) + "\n\n"
        return parts, nearby_meta, prefix

    def _route_tool_calls(self) -> tuple[list[tuple[str, dict[str, Any]]], str, dict[str, Any]]:
        """One routing turn: state (+ optional room images) → VLM → tool calls.

        Returns (tool_calls, picked_by, router_meta) where router_meta feeds the offline tuner.
        """
        meta: dict[str, Any] = {"raw_reply_chars": 0, "parse_ok": False, "tool_calls": []}
        gm = self.graph_memory
        client = getattr(gm, "eqa_client", None) if gm is not None else None
        if not self._router_enabled or client is None:
            if self.decision_policy == "grounded_v2":
                build_state_message(self)
                decisions = [
                    dict(item.__dict__)
                    for item in list(getattr(self._last_agent_state_snapshot, "gate_decisions", ()) or ())
                ]
                meta["action_gate_decisions"] = decisions
                if decisions:
                    self._append_trace(
                        {
                            "event": "action_gate_snapshot",
                            "picked_by": "fallback",
                            "action_gate_decisions": decisions,
                        }
                    )
            tool, args = self._fallback_tool()
            return [(tool, args)], "fallback", meta
        self._ensure_router_prompt()
        self._refresh_graph_room_estimate()
        img_parts, nearby_meta, img_prefix = self._room_visual_pack()
        state = build_state_message(self)
        from emet.memory.graph_eqa.agentic_state import state_text_digest

        self._router_call_seq += 1
        router_call_id = f"{self._question_id}:router:{self._router_call_seq:04d}"
        pose = self._robot_xyt_world()
        pose_list = (
            [float(value) for value in np.asarray(pose, dtype=float).reshape(-1)[:3]] if pose is not None else None
        )
        if pose_list is not None:
            if self._router_path_world:
                self._router_path_m += float(
                    np.linalg.norm(
                        np.asarray(pose_list[:2], dtype=float)
                        - np.asarray(self._router_path_world[-1][:2], dtype=float)
                    )
                )
            self._router_path_world.append(pose_list)
        action_allowlist = self._rendered_action_allowlist()
        self._last_rendered_action_allowlist = action_allowlist
        self._router_action_allowlists[router_call_id] = {key: tuple(value) for key, value in action_allowlist.items()}
        visible_event_ids = list(action_allowlist["event_ids"])
        gate_decisions = [
            dict(item.__dict__) for item in list(getattr(self._last_agent_state_snapshot, "gate_decisions", ()) or ())
        ]
        meta["router_call_id"] = router_call_id
        meta["state_text_digest"] = state_text_digest(state)
        meta["visible_event_ids"] = visible_event_ids
        meta["action_allowlist"] = {key: list(value) for key, value in action_allowlist.items()}
        meta["action_gate_decisions"] = gate_decisions
        self._append_trace(
            {
                "event": "router_call",
                "router_call_id": router_call_id,
                "state_text": state,
                "state_text_digest": meta["state_text_digest"],
                "visible_event_ids": visible_event_ids,
                "action_allowlist": meta["action_allowlist"],
                "action_gate_decisions": gate_decisions,
                "robot_world_pose": pose_list,
                "robot_world_path": list(self._router_path_world),
                "robot_world_path_m": round(float(self._router_path_m), 4),
            }
        )
        user_text = f"{img_prefix}{state}" if img_prefix else state
        payload: list[Any] = [user_text, *img_parts] if img_parts else [user_text]
        t_router0 = time.monotonic()
        try:
            reply = client(
                payload,
                system_prompt=self._system_prompt,
                max_new_tokens=ROUTER_MAX_NEW_TOKENS,
            )
        except TypeError:
            try:
                reply = client(
                    [f"{self._system_prompt}\n\n{user_text}", *img_parts]
                    if img_parts
                    else [f"{self._system_prompt}\n\n{user_text}"]
                )
            except Exception as e:
                _logger.warning(f"agentic router VLM call failed: {e}")
                tool, args = self._fallback_tool()
                return [(tool, args)], "fallback", meta
        except Exception as e:
            _logger.warning(f"agentic router VLM call failed: {e}")
            tool, args = self._fallback_tool()
            return [(tool, args)], "fallback", meta
        router_ms = (time.monotonic() - t_router0) * 1000.0
        self._last_router_ms = router_ms
        text = str(reply or "")
        meta["raw_reply_chars"] = len(text)
        meta["router_ms"] = round(router_ms, 1)
        meta["n_room_images"] = int(self._last_router_n_images)
        meta["nearby_obs"] = nearby_meta
        parsed = parse_tool_calls_response(text)
        vlm_room = coerce_room_label(parsed.get("current_room"), room_policy=self.room_policy)
        graph_room = "unknown"
        xyt = self._robot_xyt_world()
        if gm is not None:
            if vlm_room != "unknown" and hasattr(gm, "stamp_vlm_room_at_robot"):
                try:
                    latest_obs = self._latest_obs_id()
                    gm.stamp_vlm_room_at_robot(
                        xyt,
                        vlm_room,
                        source="router_vlm",
                        source_view_id=(
                            gm.view_id_for_obs(int(latest_obs))
                            if latest_obs is not None and hasattr(gm, "view_id_for_obs")
                            else None
                        ),
                        agent_round=int(self._round) + 1,
                        pose_round=int(self._round) + 1,
                    )
                except Exception as e:
                    _logger.warning(f"stamp_vlm_room_at_robot failed: {e}")
            room_fn = getattr(gm, "graph_room_at_robot", None)
            if callable(room_fn):
                try:
                    graph_room = coerce_room_label(room_fn(xyt), room_policy=self.room_policy)
                except Exception as e:
                    _logger.warning(f"graph_room_at_robot failed: {e}")
                    graph_room = "unknown"
        room = merge_room_estimates(vlm_room, graph_room, room_policy=self.room_policy)
        self._graph_room_estimate = graph_room
        self._graph_room_stale = graph_room == "unknown"
        self._last_router_room_estimate = vlm_room
        self._last_room_estimate = room
        self._current_room_source = (
            "graph+router_vlm"
            if graph_room != "unknown" and vlm_room != "unknown"
            else (
                "router_vlm"
                if vlm_room != "unknown"
                else ("graph_current_pose" if graph_room != "unknown" else "unknown")
            )
        )
        self._room_estimate_stale = room == "unknown"
        self._router_room_stale = vlm_room == "unknown"
        if room != "unknown":
            self._room_pose_round = int(self._round) + 1
            self._room_world_step = self._graph_world_step()
        self._room_estimates.append(room)
        if len(self._room_estimates) > 8:
            self._room_estimates = self._room_estimates[-8:]
        meta["current_room"] = room
        meta["current_room_vlm"] = vlm_room
        meta["current_room_graph"] = graph_room
        meta["room_policy"] = self.room_policy
        in_target: bool | None = None
        if "in_target_area" in parsed:
            raw_ita = parsed.get("in_target_area")
            if isinstance(raw_ita, bool):
                in_target = raw_ita
            elif raw_ita is not None:
                s = str(raw_ita).strip().lower()
                if s in _TRUE:
                    in_target = True
                elif s in _FALSE:
                    in_target = False
        if self.room_policy == "llm" and in_target is not None:
            self._in_target_area = in_target
        elif self.room_policy == "canonical":
            targets_now = question_target_rooms(self.question)
            if room != "unknown" and targets_now:
                self._in_target_area = not room_mismatches_question(room, self.question)
            else:
                self._in_target_area = None
        meta["in_target_area"] = self._in_target_area
        rooms_line = ""
        if gm is not None:
            rooms_fn = getattr(gm, "format_rooms_line", None)
            if callable(rooms_fn):
                try:
                    rooms_line = str(rooms_fn() or "").strip()
                except Exception as e:
                    _logger.warning(f"format_rooms_line for router_room trace failed: {e}")
                    rooms_line = ""
        # Diagnostic only — do not hard-redirect investigate on room_mismatch (Hydra
        # exposes rooms to the VLM; it does not force leave-wrong-room).
        target_rooms = sorted(question_target_rooms(self.question))
        meta["rooms_line"] = rooms_line
        meta["question_target_rooms"] = target_rooms
        meta["room_mismatch_diagnostic"] = bool(
            room_leave_needed(
                room_policy=self.room_policy,
                current_room=room,
                question=self.question,
                in_target_area=self._in_target_area,
            )
        )
        calls: list[tuple[str, dict[str, Any]]] = []
        for tc in parsed.get("tool_calls", []):
            name = str(tc.get("name") or "").strip().lower()
            if name in self._tool_names:
                calls.append((name, dict(tc.get("arguments") or {})))
            else:
                _logger.warning(f"agentic router: ignoring unknown tool {name!r}")
        calls, rejected_calls = self._validate_rendered_tool_calls(calls)
        meta["rejected_tool_calls"] = rejected_calls
        if rejected_calls:
            self._append_trace(
                {
                    "event": "router_action_rejected",
                    "router_call_id": router_call_id,
                    "action_allowlist": meta["action_allowlist"],
                    "rejected_tool_calls": rejected_calls,
                }
            )
        if not calls:
            tool, args = self._fallback_tool()
            meta["fallback_after_rejection"] = bool(rejected_calls)
            meta["tool_calls"] = [tool]
            return [(tool, args)], "fallback", meta
        meta["parse_ok"] = True
        meta["tool_calls"] = [n for n, _ in calls]
        self._append_trace(
            {
                "event": "router_room",
                "router_call_id": router_call_id,
                "state_text_digest": meta["state_text_digest"],
                "visible_event_ids": visible_event_ids,
                "action_allowlist": meta["action_allowlist"],
                "rejected_tool_calls": rejected_calls,
                "current_room": room,
                "current_room_vlm": vlm_room,
                "current_room_graph": graph_room,
                "room_policy": self.room_policy,
                "in_target_area": self._in_target_area,
                "question_target_rooms": target_rooms,
                "rooms_line": rooms_line,
                "prefer_explore_reason": str(self._prefer_explore_reason or ""),
                "room_mismatch_diagnostic": bool(meta.get("room_mismatch_diagnostic")),
                "tool_calls": list(meta["tool_calls"]),
                "router_ms": meta["router_ms"],
                "n_room_images": meta["n_room_images"],
                "nearby_obs": nearby_meta,
            }
        )
        return calls, "vlm", meta

    def _auto_submit_allowed(self, *, round_idx: int) -> bool:
        """May the ANSWER state auto-submit at this round?

        Verified answers (corroborated) submit anytime. Unverified answers submit
        only at the last round (or when the no-early-unverified guard is off) —
        budget-exhausted episodes still get a forced best guess from the ladder.
        """
        if self._verified or not self._no_early_unverified:
            return True
        return round_idx >= self.max_rounds - 1

    def _finalize_at_budget(self) -> dict[str, Any]:
        """Produce the scored answer once rounds / nav budget are spent.

        Always runs the four-image EQA through the forced-answer ladder rather than
        returning a bare Unknown, and still honors one EQA ``Action: N`` follow-up.
        """
        if self.mode != "answer":
            return self._do_finish()
        if self._evidence_policy.state != AgenticState.ANSWER:
            final = self._forced_answer_fallback(reason="budget exhausted without VLM answerable")
        elif self._require_verified and not self._verified:
            final = self._forced_answer_fallback()
        elif self._mcq_debias and not self._verified:
            # Unverified ANSWER at exhaustion: still run the ladder so the letter is
            # debiased (the raw EQA letter alone showed a last-option bias).
            final = self._forced_answer_fallback(reason="budget exhausted unverified (ANSWER state)")
        else:
            final = self._do_submit_answer()
        if self._evidence_policy.state == AgenticState.ANSWER and self._maybe_follow_eqa_explore_action(final):
            final = self._do_submit_answer()
        return final

    def _recover_failed_router_motion(self, *, tool: str, out: dict[str, Any]) -> bool:
        """Recover a router turn that selected an unusable navigation target.

        A rejected tool call performs no motion, so letting it consume the round
        unchanged invites the router to repeat the same stale id until exhaustion.
        Redirect invalid evidence ids to the best listed place card and force one
        exploration step when a selected place is exhausted or unusable.
        """
        if (
            tool not in ("navigate_to_obs", "investigate")
            or out.get("ok")
            or self.mode != "answer"
            or self._n_nav + self._n_explore >= self.max_nav_steps
        ):
            return False
        status = str(out.get("status") or "")
        invalid_pick = status in {"OBS_NOT_IN_EVIDENCE", "NOT_INVESTIGATE_CARD"}
        blocked_pick = status in {
            "NAV_LOOP_BLOCKED",
            "STATION_OBS_NOT_PLACE",
        }
        if not invalid_pick and not blocked_pick:
            return False

        redirect_tool = "explore_frontier"
        redirect_args = self._rendered_frontier_args(toward=self.query_text)
        if invalid_pick:
            hyp = (
                self._next_rendered_hypothesis()
                if self.decision_policy == "grounded_v2" and self._last_agent_state_snapshot is not None
                else self._next_untried_hypothesis()
            )
            if hyp is not None:
                redirect_tool = "investigate"
                redirect_args = {"obs_id": int(hyp.obs_id)}
        if (
            self.action_progress_mode == "enforce"
            and redirect_tool == "explore_frontier"
            and not redirect_args.get("frontier_id")
        ):
            self._append_trace(
                {
                    "event": "nav_loop_redirect_skipped",
                    "from_obs_id": out.get("obs_id"),
                    "status": status,
                    "reason": "NO_ELIGIBLE_ACTION",
                }
            )
            return False

        self._append_trace(
            {
                "event": "nav_loop_redirect",
                "from_obs_id": out.get("obs_id"),
                "status": status,
                "listed_obs_ids": out.get("listed_obs_ids"),
                "to": redirect_tool,
                "to_obs_id": redirect_args.get("obs_id"),
            }
        )
        selected_by = self._action_selected_by
        self._action_selected_by = "recovery"
        self.handle_tool(redirect_tool, redirect_args)
        self._action_selected_by = selected_by
        return True

    def _effective_state_contract_knobs(self) -> dict[str, Any]:
        return {
            "decision_policy": self.decision_policy,
            "room_policy": self.room_policy,
            "graph_evidence_mode": self.graph_evidence_mode,
            "room_history_mode": self.room_history_mode,
            "attempt_ledger_mode": self.attempt_ledger_mode,
            "action_progress_mode": self.action_progress_mode,
            "agent_state_max_chars": int(self.agent_state_max_chars),
            "question_id": self._question_id,
            "session_id": self._session_id,
        }

    def run(self) -> AgenticEQAResult:
        t0 = time.monotonic()
        final: dict[str, Any] | None = None
        # Agents are reused across episodes; do not inherit a previous escape floor.
        self._not_present_streak = 0
        self._frontier_pick_waypoints = []
        self._frontier_pick_dir = None
        self.agent._explore_min_travel_m = 0.0
        self._last_agent_state_snapshot = None
        self._router_call_seq = 0
        self._last_rendered_action_allowlist = {
            "place_ids": (),
            "place_obs_ids": (),
            "frontier_ids": (),
            "event_ids": (),
        }
        self._router_action_allowlists = {}
        self._router_path_world = []
        self._router_path_m = 0.0
        self._recent_actions = []
        self._action_history = []
        self._tool_dispatch_depth = 0
        self._action_selected_by = "internal"
        self._last_action_gate_decisions = []
        self._station_obs_ids = set()
        self._assess_history = {}
        self._answer_evidence = []
        self._confirmed_answer_evidence = None
        self._final_answer_decision = None
        self._verified_evidence_event_ids = ()
        self._last_positive_letter = ""
        self._last_positive_obs_id = None
        self._last_vlm_assess = None
        self._pending_answerable = None
        self._verified = False
        self._verified_obs_id = None
        self._last_verify = None
        self._vlm_assessed_obs_ids = set()
        self._evidence_policy = EvidencePolicy()
        self._tried = {}
        self._nav_to_obs_counts = {}
        self._nav_loop_flags = []
        self._place_inspect = {}
        self._consecutive_nav_fail = 0
        self._unreachable_obs_ids = set()
        self._prefer_explore = False
        self._prefer_explore_reason = ""
        self._n_consecutive_explore = 0
        self._last_room_estimate = "unknown"
        self._last_router_room_estimate = "unknown"
        self._graph_room_estimate = "unknown"
        self._current_room_source = "unknown"
        self._room_estimate_stale = True
        self._graph_room_stale = True
        self._router_room_stale = True
        self._room_pose_round = None
        self._room_world_step = None
        self._room_estimates = []
        self._in_target_area = None
        self._last_router_n_images = 0
        self._last_router_ms = None
        gm0 = self.graph_memory
        if gm0 is not None:
            gm0.last_agentic_decision = None
            bind_context = getattr(gm0, "bind_episode_context", None)
            if callable(bind_context):
                bind_context(
                    question_id=self._question_id,
                    session_id=self._session_id,
                )
            else:
                set_qid = getattr(gm0, "set_attempt_ledger_question_id", None)
                if callable(set_qid):
                    set_qid(self._question_id)
        if gm0 is not None and hasattr(gm0, "clear_retracted_nav_claims"):
            gm0.clear_retracted_nav_claims()
        if gm0 is not None and hasattr(gm0, "clear_room_events"):
            gm0.clear_room_events()
        # Resolve panel dir early so HM-EQA bundles get picks even without trace_path.
        try:
            self._frontier_pick_out_dir()
        except Exception:
            pass
        budget_hit = False
        # Deferred clients: build the shared VLM now — keyword extraction in inspect_graph
        # and the tool router both need it (text-only turns coexist with warm SigLIP).
        gm = self.graph_memory
        if gm is not None and hasattr(gm, "_ensure_llm_clients"):
            try:
                gm._ensure_llm_clients()
            except Exception as e:
                # Router / keyword extract need a real VLM. Do not limp along with silent fallback.
                if self._router_enabled:
                    raise RuntimeError(
                        "Agentic EQA VLM router is enabled but LLM clients failed to load. "
                        "Fix the VLM install (CUDA + flash-attn / bitsandbytes) or set "
                        "EMET_EQA_AGENTIC_ROUTER=0 for deterministic tools only."
                    ) from e
                _logger.warning(f"agentic: LLM client init failed (fallback-only mode): {e}")
        if self.mode == "answer":
            self._extract_vlm_target()
        self._apply_ovmm_trace_target()
        # Always start with inspect to seed hypotheses.
        self.handle_tool("inspect_graph", {})
        for r in range(self.max_rounds):
            self._round = r
            # Only VLM-assessed ANSWER may auto-submit.
            if self.mode == "answer" and self._evidence_policy.state == AgenticState.ANSWER and r > 0:
                # Corroborated (verified) ANSWER may submit early. An unverified
                # ANSWER with budget remaining must keep gathering evidence — early
                # unverified submits were the 2026-08 forced-letter wrongs; the
                # ladder commits a best guess at exhaustion instead.
                if self._auto_submit_allowed(round_idx=r):
                    out = self._do_submit_answer()
                    if self._maybe_follow_eqa_explore_action(out):
                        continue
                    final = out
                    break
                self._append_trace(
                    {
                        "event": "hold_submit_unverified",
                        "round": r,
                        "reason": "no_early_unverified",
                        "state": "ANSWER",
                    }
                )
            # Reserve the last round for answering. Otherwise the router could spend it
            # on an explore call and the loop fell through to the exhaustion branch
            # without submit_answer ever being offered.
            if self.mode == "answer" and r >= self.max_rounds - 1:
                budget_hit = True
                final = self._finalize_at_budget()
                break
            calls, picked_by, router_meta = self._route_tool_calls()
            selected_calls = [(name, dict(args)) for name, args in calls]
            action_rewrite: dict[str, Any] | None = None
            # Close+VLM-absent → force explore so coverage grows before the next investigate.
            # OVMM find / close-look: keep investigating nearby place cards instead of frontier drift.
            skip_prefer_explore = (
                self._ovmm_prefers_nearby_investigate() and self._nearby_untried_investigate_hyp() is not None
            )
            if (
                self.decision_policy != "grounded_v2"
                and self._prefer_explore
                and not skip_prefer_explore
                and self.mode == "answer"
                and calls
                and calls[0][0] in ("investigate", "navigate_to_obs")
                and self._n_nav + self._n_explore < self.max_nav_steps
                and self._frontier_count() > 0
                and int(getattr(self, "_n_consecutive_explore", 0) or 0) < 1
            ):
                self._append_trace(
                    {
                        "event": "prefer_explore_redirect",
                        "from": calls[0][0],
                        "from_args": calls[0][1],
                        "to": "explore_frontier",
                    }
                )
                calls = [("explore_frontier", {"toward": self.query_text})]
                picked_by = f"{picked_by}+prefer_explore"
                action_rewrite = {
                    "reason": "prefer_explore",
                    "selected": selected_calls,
                    "executed": calls,
                }
            # Leave/ABSENT explore-only loops: after a streak of frontiers, force a close look.
            if (
                self.decision_policy != "grounded_v2"
                and self.mode == "answer"
                and calls
                and calls[0][0] == "explore_frontier"
                and int(getattr(self, "_n_consecutive_explore", 0) or 0) >= EXPLORE_STREAK_FORCE_INVESTIGATE
                and self._n_nav + self._n_explore < self.max_nav_steps
            ):
                hyp = self._next_untried_hypothesis()
                if hyp is not None:
                    self._append_trace(
                        {
                            "event": "explore_streak_investigate",
                            "streak": int(self._n_consecutive_explore),
                            "from": "explore_frontier",
                            "to_obs_id": int(hyp.obs_id),
                        }
                    )
                    calls = [("investigate", {"obs_id": int(hyp.obs_id)})]
                    picked_by = f"{picked_by}+explore_streak"
                    action_rewrite = {
                        "reason": "explore_streak",
                        "selected": selected_calls,
                        "executed": calls,
                    }
                elif self._close_look_required:
                    # Close-look question (clock/state/count) and no place card left:
                    # do a close look at the current station instead of another frontier.
                    self._append_trace(
                        {
                            "event": "close_look_station_look_around",
                            "streak": int(self._n_consecutive_explore),
                            "from": "explore_frontier",
                            "to": "look_around",
                            "source": self._close_look_source,
                        }
                    )
                    calls = [("look_around", {})]
                    picked_by = f"{picked_by}+close_look"
                    action_rewrite = {
                        "reason": "close_look",
                        "selected": selected_calls,
                        "executed": calls,
                    }
            self._append_trace(
                {
                    "event": "tool_pick",
                    "router_call_id": router_meta.get("router_call_id"),
                    "picked_by": picked_by,
                    "tool": calls[0][0],
                    "args": calls[0][1],
                    "selected_actions": selected_calls,
                    "executed_actions": calls,
                    "action_rewrite": action_rewrite,
                    "router_raw_reply_chars": router_meta.get("raw_reply_chars", 0),
                    "router_parse_ok": router_meta.get("parse_ok", False),
                    "router_tool_calls": router_meta.get("tool_calls", []),
                }
            )
            for tool, args in calls:
                self._action_selected_by = picked_by
                out = self.handle_tool(tool, args)
                self._append_trace(
                    {
                        "event": "action_execution",
                        "router_call_id": router_meta.get("router_call_id"),
                        "selected_actions": selected_calls,
                        "executed_action": (tool, dict(args)),
                        "action_rewrite": action_rewrite,
                        "outcome_ok": bool(out.get("ok")),
                        "outcome_status": out.get("status") or out.get("error"),
                    }
                )
                if tool == "submit_answer" and out.get("ok"):
                    # If EQA says Unknown + Action:N (explore image N) and we still have
                    # nav budget, follow that instead of locking a guessed letter.
                    if self._maybe_follow_eqa_explore_action(out):
                        final = None
                        break
                    final = out
                    break
                if tool == "finish" and out.get("ok"):
                    final = out
                    break
                if not out.get("ok") and "budget" in str(out.get("error", "")):
                    # Budget gate rejected — stop dispatching the rest of this reply.
                    break
                self._recover_failed_router_motion(tool=tool, out=out)
                # Motion tools chain verify themselves (router + fallback). Do not
                # double-verify here — that burned rounds on SKIPPED_SAME_VIEW.
            if final is not None:
                break
            if self.mode == "answer" and self._evidence_policy.state == AgenticState.ANSWER:
                if self._auto_submit_allowed(round_idx=self._round):
                    out = self._do_submit_answer()
                    if self._maybe_follow_eqa_explore_action(out):
                        continue
                    final = out
                    break
                self._append_trace(
                    {
                        "event": "hold_submit_unverified",
                        "round": self._round,
                        "reason": "no_early_unverified",
                        "state": "ANSWER",
                    }
                )
        else:
            budget_hit = True
            final = self._finalize_at_budget()

        wall = time.monotonic() - t0
        assert final is not None
        final = self._finalize_unknown_location_letter(final)
        provenance = str(final.get("answer_provenance") or final.get("answer_source") or "")
        confidence_score = final.get("answer_confidence")
        if not isinstance(confidence_score, (int, float)):
            confidence_score = self._confidence_for_provenance(provenance)
        confidence_score = float(confidence_score)
        self._answer_provenance = provenance
        result = AgenticEQAResult(
            discord_text=str(final.get("discord_text") or f"Answer:{final.get('answer', 'Unknown')}"),
            answer=str(final.get("answer") or "Unknown"),
            confidence=bool(final.get("confidence")),
            relevant_images=list(final.get("relevant_images") or []),
            tool_log=list(self._tool_log),
            verified=self._verified,
            verified_obs_id=self._verified_obs_id,
            n_rounds=self._round + 1,
            n_nav=self._n_nav,
            n_explore=self._n_explore,
            wall_s=wall,
            budget_hit=budget_hit,
            salvage_counterfactual_letter=str(self._salvage_counterfactual_letter or ""),
            answer_provenance=provenance,
            answer_confidence=confidence_score,
            decision_rounds=self._round + 1,
        )
        self._sync_scored_answer_to_graph_memory(result, final)
        # Always expose salvage CF for Habitat jsonl even when collect_trace is off.
        salvage_cf = str(result.salvage_counterfactual_letter or "")
        prev_summary = getattr(self.agent, "_agentic_eqa_summary", None)
        summary = dict(prev_summary) if isinstance(prev_summary, dict) else {}
        summary.update(
            {
                "answer": result.answer,
                "confidence": result.confidence,
                "verified": result.verified,
                "salvage_counterfactual_letter": salvage_cf,
                "scored_policy": "no_salvage",
                "answer_provenance": result.answer_provenance,
                "answer_confidence": result.answer_confidence,
                "decision_rounds": result.decision_rounds,
                "budget_hit": result.budget_hit,
                "decision_policy": self.decision_policy,
                "effective_state_contract": self._effective_state_contract_knobs(),
                "router_action_allowlists": {
                    call_id: {key: list(values) for key, values in allowlist.items()}
                    for call_id, allowlist in self._router_action_allowlists.items()
                },
                "final_decision": (
                    self._final_answer_decision.to_dict() if self._final_answer_decision is not None else None
                ),
            }
        )
        self.agent._agentic_eqa_summary = summary
        if self.graph_memory is not None:
            self.graph_memory.last_salvage_counterfactual_letter = salvage_cf
        self._append_trace(
            {
                "tool": "summary",
                "final_answer": result.answer,
                "confidence": result.confidence,
                "verified": result.verified,
                "n_rounds": result.n_rounds,
                "n_nav": result.n_nav,
                "n_explore": result.n_explore,
                "wall_s": result.wall_s,
                "budget_hit": result.budget_hit,
                "tools": result.tool_log,
                "salvage_counterfactual_letter": salvage_cf,
                "answer_provenance": result.answer_provenance,
                "answer_confidence": result.answer_confidence,
                "decision_rounds": result.decision_rounds,
                "decision_policy": self.decision_policy,
                "effective_state_contract": self._effective_state_contract_knobs(),
                "router_action_allowlists": {
                    call_id: {key: list(values) for key, values in allowlist.items()}
                    for call_id, allowlist in self._router_action_allowlists.items()
                },
                "final_decision": (
                    self._final_answer_decision.to_dict() if self._final_answer_decision is not None else None
                ),
            }
        )
        self._flush_trace_to_agent(result)
        return result

    def _sync_scored_answer_to_graph_memory(
        self,
        result: AgenticEQAResult,
        final: dict[str, Any],
    ) -> None:
        """Write the agentic decision into ``last_eqa_*`` so Habitat scores it.

        Habitat ``runner.py`` reads ``graph_memory.last_eqa_raw`` /
        ``last_eqa_parsed``, not ``AgenticEQAResult.answer``. Preserve semantic
        option text here; the runner alone converts it to benchmark encoding.
        """
        gm = self.graph_memory
        if gm is None:
            return
        from emet.habitat.metrics import (
            extract_mcq_letter,
            parse_mcq_choices_from_question,
        )

        choices = parse_mcq_choices_from_question(self.question)
        letter = self._mcq_letter_from_text(result.answer)
        if not letter and choices:
            letter = extract_mcq_letter(str(result.answer or ""), choices)
        if not letter:
            return
        decision = self._final_answer_decision
        answer_text = ""
        if decision is not None and decision.answer == result.answer:
            answer_text = str(decision.answer_text or "").strip()
        answer_text = self._semantic_answer_text(answer_text or str(result.answer or ""), letter)
        if not answer_text:
            return
        source = str(final.get("answer_source") or "agentic")
        prior = getattr(gm, "last_eqa_raw", "") or ""
        gm.last_eqa_raw = f"{prior.rstrip()}\n[agentic_submit]\nsource:{source}\nanswer:\n{answer_text}\n"
        prev = getattr(gm, "last_eqa_parsed", None)
        if isinstance(prev, tuple) and len(prev) >= 5:
            reasoning, _old, _conf, action, conf_reason = prev[:5]
        else:
            reasoning, action, conf_reason = "", "", ""
        gm.last_eqa_parsed = (
            str(reasoning or ""),
            answer_text,
            bool(result.confidence),
            str(action or ""),
            str(conf_reason or ""),
        )
        if self.decision_policy == "grounded_v2" and self._final_answer_decision is not None:
            gm.last_agentic_decision = self._final_answer_decision.to_dict()
        self._append_trace(
            {
                "event": "sync_scored_answer",
                "answer_text": answer_text,
                "choice_index": ord(letter) - ord("A"),
                "source": source,
                "confidence": bool(result.confidence),
                "obs_ids": list(getattr(gm, "last_eqa_obs_ids", []) or []),
                "decision_policy": self.decision_policy,
                "final_decision": (
                    self._final_answer_decision.to_dict() if self._final_answer_decision is not None else None
                ),
            }
        )

    def _flush_trace_to_agent(self, result: AgenticEQAResult) -> None:
        """Stash trace rows on the agent so Habitat debug bundles can persist them."""
        if not self._collect_trace:
            return
        rows = list(self._trace_rows)
        if not rows:
            return
        self.agent._agentic_trace_rows = rows
        salvage_cf = str(result.salvage_counterfactual_letter or self._salvage_counterfactual_letter or "")
        self.agent._agentic_eqa_summary = {
            "answer": result.answer,
            "confidence": result.confidence,
            "verified": result.verified,
            "n_rounds": result.n_rounds,
            "n_nav": result.n_nav,
            "n_explore": result.n_explore,
            "budget_hit": result.budget_hit,
            "tools": list(result.tool_log),
            "salvage_counterfactual_letter": salvage_cf,
            "scored_policy": "no_salvage",
            "answer_provenance": result.answer_provenance,
            "answer_confidence": result.answer_confidence,
            "decision_rounds": result.decision_rounds,
            "decision_policy": self.decision_policy,
            "effective_state_contract": self._effective_state_contract_knobs(),
            "final_decision": (
                self._final_answer_decision.to_dict() if self._final_answer_decision is not None else None
            ),
        }
        gm = self.graph_memory
        if gm is not None:
            gm.last_salvage_counterfactual_letter = salvage_cf
        if self._trace_path is None:
            default = Path.home() / ".cache" / "habitat_eqa" / "agentic_traces" / "last_agentic_trace.jsonl"
            default.parent.mkdir(parents=True, exist_ok=True)
            default.write_text(
                "".join(json.dumps(r, default=str) + "\n" for r in rows),
                encoding="utf-8",
            )
            self._trace_path = default


def build_agentic_eqa_executor(
    agent: Any,
    question: str | None,
    *,
    goal: str = "",
    max_rounds: int | None = None,
    max_nav_steps: int | None = None,
    verify_min_sim: float | None = None,
    trace_path: Path | str | None = None,
    trace_meta: dict[str, Any] | None = None,
    router: bool | None = None,
    require_verified: bool | None = None,
) -> AgenticEQAExecutor:
    """Construct the shared agentic executor (EQA episode and OVMM find use this)."""
    from emet.eval.dynagraph_vram import warm_siglip_confirmed_memory

    cfg = _eqa_cfg(agent)
    warm_siglip_confirmed_memory(agent)
    agent._habitat_blocked_goals = getattr(agent, "_habitat_blocked_goals", set()) or set()
    agent._habitat_recent_goals = getattr(agent, "_habitat_recent_goals", []) or []
    env_max_rounds = _env_positive_int("EMET_EQA_AGENTIC_MAX_TOOL_ROUNDS")
    env_max_nav_steps = _env_positive_int("EMET_EQA_AGENTIC_MAX_NAV_STEPS")
    return AgenticEQAExecutor(
        agent,
        question,
        goal=goal,
        max_rounds=int(
            max_rounds if max_rounds is not None else env_max_rounds or cfg.get("agentic_max_tool_rounds", 8) or 8
        ),
        max_nav_steps=int(
            max_nav_steps
            if max_nav_steps is not None
            else env_max_nav_steps or cfg.get("agentic_max_nav_steps", 8) or 8
        ),
        verify_min_sim=float(
            verify_min_sim
            if verify_min_sim is not None
            else cfg.get("agentic_verify_min_sim", SIGLIP_IMAGE_PRESENT_THRESHOLD) or SIGLIP_IMAGE_PRESENT_THRESHOLD
        ),
        trace_path=trace_path,
        trace_meta=trace_meta,
        router=router,
        require_verified=require_verified,  # None → env/config inside executor
    )


def run_agentic_eqa_result(
    agent: Any,
    question: str | None,
    *,
    goal: str = "",
    max_rounds: int | None = None,
    max_nav_steps: int | None = None,
    verify_min_sim: float | None = None,
    trace_path: Path | str | None = None,
    trace_meta: dict[str, Any] | None = None,
    router: bool | None = None,
    require_verified: bool | None = None,
) -> AgenticEQAResult:
    """Run the unified agentic loop; return the full :class:`AgenticEQAResult`.

    OVMM find phrases the episode as a question and reads ``verified_obs_id`` / pose
    from this result — same executor as HM-EQA, not a parallel find loop.
    """
    ex = build_agentic_eqa_executor(
        agent,
        question,
        goal=goal,
        max_rounds=max_rounds,
        max_nav_steps=max_nav_steps,
        verify_min_sim=verify_min_sim,
        trace_path=trace_path,
        trace_meta=trace_meta,
        router=router,
        require_verified=require_verified,
    )
    result = ex.run()
    print(
        f"\n--- Agentic GraphEQA ({ex.mode}) ---\n{result.discord_text.strip()}\n"
        f"(rounds={result.n_rounds} nav={result.n_nav} explore={result.n_explore} "
        f"verified={result.verified} wall_s={result.wall_s:.1f})\n---\n",
        flush=True,
    )
    return result


def run_agentic_eqa(
    agent: Any,
    question: str | None,
    *,
    goal: str = "",
    max_rounds: int | None = None,
    max_nav_steps: int | None = None,
    verify_min_sim: float | None = None,
    trace_path: Path | str | None = None,
    trace_meta: dict[str, Any] | None = None,
    router: bool | None = None,
) -> tuple[str, list[Any]]:
    """Run the unified agentic loop; returns (discord_text, images) like ``run_eqa``.

    With ``question=None`` the executor runs in explore mode: the VLM router drives
    ``explore_frontier`` / ``look_around`` until frontiers or the nav budget are
    exhausted, then ``finish`` returns a coverage summary instead of an answer.
    """
    result = run_agentic_eqa_result(
        agent,
        question,
        goal=goal,
        max_rounds=max_rounds,
        max_nav_steps=max_nav_steps,
        verify_min_sim=verify_min_sim,
        trace_path=trace_path,
        trace_meta=trace_meta,
        router=router,
    )
    return result.discord_text, result.relevant_images

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Constants and env/config helpers for the agentic GraphEQA loop."""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from emet.memory.graph_eqa.room_clusters import resolve_room_policy

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
# Prefer investigate when a matching place card is this close (locate / close-look).
NEAR_INVESTIGATE_M = 3.5
OVMM_NEAR_INVESTIGATE_M = NEAR_INVESTIGATE_M  # backward-compat alias
# Investigate / close-look standoff: keep outer ring within approached_close (1.0m).
INVESTIGATE_ANNULUS_OUTER_M = 1.0
DEFAULT_INVESTIGATE_ANNULUS_OUTER_M = 1.60

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


# Locate / find phrasing (OVMM find, open "where is X") — nearby place cards first.
_LOCATE_CUES = (
    "where is",
    "where's",
    "find the",
    "locate the",
)


def question_is_locate(question: str) -> bool:
    """True when the question is a find/localize prompt, not a close-look MCQ.

    Close-look (clock/count/color) stays on ``question_requires_close_look_keywords``.
    Locate questions still prefer a nearby matching place card over frontier drift.
    """
    q = str(question or "").strip().lower()
    if not q:
        return False
    return any(cue in q for cue in _LOCATE_CUES)


# Hyp recall: how many evidence cards to show the router / walk in fallback.
DEFAULT_HYP_RECALL_K = 6

# Investigate vs explore: place-card sources worth a closer look.
INVESTIGATE_SOURCES = frozenset({"graph", "confirmed", "siglip", "voxel"})
PLACE_INSPECT_RECENT_K = 3
# Compact tool outcomes shown to the VLM router (avoid re-picking stuck loops).
RECENT_ACTIONS_K = 6

# Routing turns are text-only JSON; a two-call reply with arguments needs more than 64 tokens.
ROUTER_MAX_NEW_TOKENS = 128


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

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unified agentic GraphEQA loop: thin facade over mixins.

Implementation lives in ``agentic_{init,run,router,answer,verify,assess,capture,
investigate,place,explore,action}.py``. Callers and tests keep importing from this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from emet.memory.graph_eqa.agentic_action import AgenticActionMixin
from emet.memory.graph_eqa.agentic_answer import AgenticAnswerMixin
from emet.memory.graph_eqa.agentic_assess import AgenticAssessMixin
from emet.memory.graph_eqa.agentic_capture import AgenticCaptureMixin
from emet.memory.graph_eqa.agentic_config import (
    DEFAULT_INVESTIGATE_ANNULUS_OUTER_M,
    ESCAPE_MIN_TRAVEL_M,
    EXPLORE_STREAK_FORCE_INVESTIGATE,
    INVESTIGATE_ANNULUS_OUTER_M,
    NAV_CONSECUTIVE_FAIL_LIMIT,
    NAV_SAME_OBS_LOOP_LIMIT,
    NEAR_INVESTIGATE_M,
    NOT_PRESENT_ESCAPE_STREAK,
    PLACE_APPROACH_SAMPLES,
    RECENT_ACTIONS_K,
    SIGLIP_IMAGE_PRESENT_THRESHOLD,
    _env_positive_int,
    _eqa_cfg,
    agentic_verify_enabled,
    question_requires_close_look_keywords,
)
from emet.memory.graph_eqa.agentic_explore import AgenticExploreMixin
from emet.memory.graph_eqa.agentic_init import AgenticInitMixin
from emet.memory.graph_eqa.agentic_investigate import AgenticInvestigateMixin
from emet.memory.graph_eqa.agentic_place import AgenticPlaceMixin
from emet.memory.graph_eqa.agentic_policy import AgenticState
from emet.memory.graph_eqa.agentic_router import AgenticRouterMixin
from emet.memory.graph_eqa.agentic_run import AgenticRunMixin
from emet.memory.graph_eqa.agentic_tools import build_state_message
from emet.memory.graph_eqa.agentic_types import (
    AgenticEQAResult,
    AnswerEvidenceRecord,
    FinalAnswerDecision,
    PlaceInspectRecord,
)
from emet.memory.graph_eqa.agentic_verify import AgenticVerifyMixin

__all__ = [
    "DEFAULT_INVESTIGATE_ANNULUS_OUTER_M",
    "ESCAPE_MIN_TRAVEL_M",
    "EXPLORE_STREAK_FORCE_INVESTIGATE",
    "INVESTIGATE_ANNULUS_OUTER_M",
    "NAV_CONSECUTIVE_FAIL_LIMIT",
    "NAV_SAME_OBS_LOOP_LIMIT",
    "NEAR_INVESTIGATE_M",
    "NOT_PRESENT_ESCAPE_STREAK",
    "PLACE_APPROACH_SAMPLES",
    "RECENT_ACTIONS_K",
    "SIGLIP_IMAGE_PRESENT_THRESHOLD",
    "AgenticEQAExecutor",
    "AgenticEQAResult",
    "AgenticState",
    "AnswerEvidenceRecord",
    "FinalAnswerDecision",
    "PlaceInspectRecord",
    "agentic_verify_enabled",
    "build_agentic_eqa_executor",
    "build_state_message",
    "question_requires_close_look_keywords",
    "run_agentic_eqa",
    "run_agentic_eqa_result",
]


class AgenticEQAExecutor(
    AgenticInitMixin,
    AgenticRunMixin,
    AgenticRouterMixin,
    AgenticAnswerMixin,
    AgenticVerifyMixin,
    AgenticAssessMixin,
    AgenticCaptureMixin,
    AgenticInvestigateMixin,
    AgenticPlaceMixin,
    AgenticExploreMixin,
    AgenticActionMixin,
):
    """Bounded tool loop for post-explore / world-change EQA."""


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
    """Construct the shared agentic executor (HM-EQA and OVMM find both use this)."""
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

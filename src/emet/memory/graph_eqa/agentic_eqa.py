# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unified agentic GraphEQA loop: facade over implementation modules.

Implementation lives in ``agentic/{executor_init,run,router,answer,verify,assess,
capture,investigate,place,explore,action}.py``. Callers keep importing from this module.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from emet.memory.graph_eqa._bind import bind_module_methods
from emet.memory.graph_eqa.agentic import (
    action,
    answer,
    assess,
    capture,
    executor_init,
    explore,
    investigate,
    place,
    router,
    run,
    verify,
)
from emet.memory.graph_eqa.agentic.config import (
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
from emet.memory.graph_eqa.agentic.policy import AgenticState, EvidencePhase
from emet.memory.graph_eqa.agentic.session import AgenticSession
from emet.memory.graph_eqa.agentic.tools import build_state_message
from emet.memory.graph_eqa.agentic.types import (
    AgenticEQAResult,
    AnswerEvidenceRecord,
    FinalAnswerDecision,
    PlaceInspectRecord,
)

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
    "AgenticSession",
    "AgenticState",
    "EvidencePhase",
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


class AgenticEQAExecutor:
    """Bounded tool loop for post-explore / world-change EQA."""

    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "session", AgenticSession())
        executor_init.init_executor(self, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "session":
            raise AttributeError(name)
        try:
            sess = object.__getattribute__(self, "session")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        try:
            return getattr(sess, name)
        except AttributeError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "session":
            object.__setattr__(self, name, value)
            return
        cls_attr = type(self).__dict__.get(name)
        if inspect.isfunction(cls_attr) or isinstance(cls_attr, (staticmethod, classmethod)):
            object.__setattr__(self, name, value)
            return
        try:
            sess = object.__getattribute__(self, "session")
        except AttributeError:
            object.__setattr__(self, name, value)
            return
        setattr(sess, name, value)

    def handle_tool(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        return executor_init.handle_tool(self, name, args)


for _mod in (
    executor_init,
    run,
    router,
    answer,
    verify,
    assess,
    capture,
    investigate,
    place,
    explore,
    action,
):
    bind_module_methods(AgenticEQAExecutor, _mod, skip=frozenset({"handle_tool"}))


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
        require_verified=require_verified,
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
    """Run the unified agentic loop; return the full :class:`AgenticEQAResult`."""
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
    """Run the unified agentic loop; returns (discord_text, images) like ``run_eqa``."""
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

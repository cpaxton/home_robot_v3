# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""OVMM find as questions into the shared AgenticEQAExecutor loop.

Not a parallel find stack: episode language becomes an EQA-style question; navigate /
verify / retract / explore stay in :mod:`emet.memory.graph_eqa.agentic_eqa`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def ovmm_find_object_question(object_name: str, start_recep: str | None = None) -> str:
    """Phrase FindObj as an open question for the agentic loop."""
    obj = str(object_name or "").strip() or "object"
    recep = str(start_recep or "").strip()
    if recep:
        return f"Where is the {obj} on the {recep}?"
    return f"Where is the {obj}?"


def ovmm_find_recep_question(goal_recep: str) -> str:
    """Phrase FindRec as an open question for the agentic loop."""
    recep = str(goal_recep or "").strip() or "receptacle"
    return f"Where is the {recep}?"


def xyz_from_verified_obs(agent: Any, obs_id: int | None) -> np.ndarray | None:
    """World XYZ for a verified observation / matching graph node."""
    if obs_id is None:
        return None
    gm = getattr(agent, "graph_memory", None)
    if gm is None:
        return None
    oid = int(obs_id)
    for o in getattr(gm, "_observations", None) or []:
        if int(getattr(o, "obs_id", -1)) != oid:
            continue
        xyz = getattr(o, "xyz", None)
        if xyz is None:
            continue
        arr = np.asarray(xyz, dtype=np.float64).reshape(-1)
        if arr.size >= 3:
            return arr[:3]
    for n in gm.get_nodes() if hasattr(gm, "get_nodes") else []:
        if int(getattr(n, "obs_id", -1)) != oid:
            continue
        if getattr(n, "is_frontier", False) or getattr(n, "is_viewpoint", False):
            continue
        xyz = getattr(n, "xyz", None)
        if xyz is None:
            continue
        arr = np.asarray(xyz, dtype=np.float64).reshape(-1)
        if arr.size >= 3:
            return arr[:3]
    return None


@dataclass
class OvmmAgenticLocalizeResult:
    """Outcome of one OVMM phrase through the shared agentic loop."""

    question: str
    verified: bool
    verified_obs_id: int | None
    xyz: np.ndarray | None
    n_rounds: int = 0
    n_nav: int = 0
    n_explore: int = 0
    n_retracted_claims: int = 0
    answer: str = ""
    discord_text: str = ""
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def run_ovmm_agentic_localize(
    agent: Any,
    question: str,
    *,
    goal: str = "",
    max_rounds: int | None = None,
    max_nav_steps: int | None = None,
    require_verified: bool = True,
    router: bool | None = None,
    trace_meta: dict[str, Any] | None = None,
) -> OvmmAgenticLocalizeResult:
    """Run :func:`run_agentic_eqa_result` and map verified obs → world XYZ."""
    from emet.memory.graph_eqa.agentic_eqa import run_agentic_eqa_result

    q = str(question or "").strip()
    if not q:
        return OvmmAgenticLocalizeResult(
            question=q,
            verified=False,
            verified_obs_id=None,
            xyz=None,
            error="empty question",
        )
    goal_text = goal or f"Find and verify: {q}"
    try:
        result = run_agentic_eqa_result(
            agent,
            q,
            goal=goal_text,
            max_rounds=max_rounds,
            max_nav_steps=max_nav_steps,
            require_verified=require_verified,
            router=router,
            trace_meta=trace_meta,
        )
    except Exception as exc:
        return OvmmAgenticLocalizeResult(
            question=q,
            verified=False,
            verified_obs_id=None,
            xyz=None,
            error=str(exc),
        )

    gm = getattr(agent, "graph_memory", None)
    n_retracted = 0
    if gm is not None:
        claims = getattr(gm, "_retracted_nav_claims", None)
        if claims is not None:
            n_retracted = len(claims)

    oid = result.verified_obs_id if result.verified else None
    xyz = xyz_from_verified_obs(agent, oid) if oid is not None else None
    return OvmmAgenticLocalizeResult(
        question=q,
        verified=bool(result.verified),
        verified_obs_id=oid,
        xyz=xyz,
        n_rounds=int(result.n_rounds),
        n_nav=int(result.n_nav),
        n_explore=int(result.n_explore),
        n_retracted_claims=n_retracted,
        answer=str(result.answer or ""),
        discord_text=str(result.discord_text or ""),
        extra={
            "budget_hit": bool(result.budget_hit),
            "answer_provenance": str(result.answer_provenance or ""),
        },
    )


def should_use_agentic_find(backend: str, *, agentic_find: bool | None) -> bool:
    """Default: agentic loop for dynagraph/static_graph; off for dynamem/oracle."""
    if agentic_find is not None:
        return bool(agentic_find)
    b = str(backend or "").lower()
    return b in {"dynagraph", "static_graph", "graph_eqa"}

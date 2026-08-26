# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""OVMM find as questions into the shared AgenticEQAExecutor loop.

Not a parallel find stack and not an OVMM policy inside the executor: the harness
phrases FindObj / FindRec as EQA-style questions; navigate / verify / explore stay
in :mod:`emet.memory.graph_eqa.agentic_eqa`. Trace metadata is logging only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _ovmm_agentic_trace_path(trace_meta: dict[str, Any] | None) -> Path | None:
    """JSONL next to dumped query PNGs when the OVMM harness set an episode dir."""
    ep = os.environ.get("EMET_EQA_EPISODE_DIR", "").strip()
    if not ep:
        return None
    phase = str((trace_meta or {}).get("ovmm_phase") or "agentic")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in phase) or "agentic"
    return Path(ep).expanduser() / f"{safe}_agentic_trace.jsonl"


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


def _localize_phrases(question: str, trace_meta: dict[str, Any] | None) -> list[str]:
    """Target object/recep phrases for graph-node XYZ lookup after verify."""
    out: list[str] = []
    meta = trace_meta or {}
    for key in ("object", "goal_recep", "start_recep"):
        text = str(meta.get(key) or "").strip()
        if text and text not in out:
            out.append(text)
    q = str(question or "").strip()
    if q.lower().startswith("where is the ") and q.endswith("?"):
        inner = q[13:-1].strip()
        if inner.lower().startswith("the "):
            inner = inner[4:].strip()
        on_idx = inner.lower().rfind(" on the ")
        if on_idx > 0:
            inner = inner[:on_idx].strip()
        if inner and inner not in out:
            out.append(inner)
    return out


def xyz_from_verified_obs(
    agent: Any,
    obs_id: int | None,
    *,
    phrases: list[str] | None = None,
) -> np.ndarray | None:
    """World XYZ for a verified view — prefer matching object graph nodes over obs pose."""
    if obs_id is None:
        return None
    gm = getattr(agent, "graph_memory", None)
    if gm is None:
        return None
    from emet.memory.graph_eqa.graph_types import finder_label_texts, label_matches_relevant_object

    oid = int(obs_id)
    phrase_list = [str(p or "").strip() for p in (phrases or []) if str(p or "").strip()]
    object_nodes: list[Any] = []
    for n in gm.get_nodes() if hasattr(gm, "get_nodes") else []:
        if int(getattr(n, "obs_id", -1)) != oid:
            continue
        if getattr(n, "is_frontier", False) or getattr(n, "is_viewpoint", False):
            continue
        object_nodes.append(n)
    if phrase_list and object_nodes:
        for phrase in phrase_list:
            for node in object_nodes:
                texts = finder_label_texts(node)
                if not texts:
                    continue
                if any(label_matches_relevant_object(phrase, text) for text in texts):
                    xyz = getattr(node, "xyz", None)
                    if xyz is None:
                        continue
                    arr = np.asarray(xyz, dtype=np.float64).reshape(-1)
                    if arr.size >= 3:
                        return arr[:3]
    for node in object_nodes:
        xyz = getattr(node, "xyz", None)
        if xyz is None:
            continue
        arr = np.asarray(xyz, dtype=np.float64).reshape(-1)
        if arr.size >= 3:
            return arr[:3]
    for o in getattr(gm, "_observations", None) or []:
        if int(getattr(o, "obs_id", -1)) != oid:
            continue
        xyz = getattr(o, "xyz", None)
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


def _count_retracted_claims(gm: Any) -> int:
    """Size of the graph memory's retracted-nav-claim set (0 when absent)."""
    if gm is None:
        return 0
    claims = getattr(gm, "_retracted_nav_claims", None)
    if claims is None:
        return 0
    try:
        return len(claims)
    except TypeError:
        return 0


def run_ovmm_agentic_localize(
    agent: Any,
    question: str,
    *,
    goal: str = "",
    max_rounds: int | None = None,
    max_nav_steps: int | None = None,
    require_verified: bool = True,
    router: bool | None = None,
    trace_path: Path | str | None = None,
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
    gm = getattr(agent, "graph_memory", None)
    retracted_before = _count_retracted_claims(gm)
    goal_text = goal or f"Find and verify: {q}"
    resolved_trace = Path(trace_path).expanduser() if trace_path else _ovmm_agentic_trace_path(trace_meta)
    try:
        result = run_agentic_eqa_result(
            agent,
            q,
            goal=goal_text,
            max_rounds=max_rounds,
            max_nav_steps=max_nav_steps,
            require_verified=require_verified,
            router=router,
            trace_path=resolved_trace,
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

    n_retracted = max(0, _count_retracted_claims(gm) - retracted_before)

    oid = result.verified_obs_id if result.verified else None
    localize_phrases = _localize_phrases(q, trace_meta)
    xyz = xyz_from_verified_obs(agent, oid, phrases=localize_phrases) if oid is not None else None
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

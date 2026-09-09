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
from typing import Any, Literal

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
    """Target object/recep phrases for graph-node / voxel XYZ after the loop.

    Phase metadata selects **one** OVMM target so a mapping-time pin for the
    cube cannot be scored as FindObj (or the reverse). ``start_recep`` is the
    support surface, not a coordinate.
    """
    out: list[str] = []
    meta = trace_meta or {}
    phase = str(meta.get("ovmm_phase") or "").strip().lower()
    if phase == "find_recep":
        keys = ("goal_recep",)
    elif phase == "find_object":
        keys = ("object",)
    else:
        keys = ("object", "goal_recep")
    for key in keys:
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


def _xyz_from_loop_voxel(
    result: Any,
    phrases: list[str],
) -> tuple[np.ndarray | None, str | None, bool | None]:
    """Object-phrase ``localize_text`` XYZ the executor already queried.

    VLM submit releases SigLIP; do not re-run the voxel encoder for the score.
    Furniture wraps (``red cylinder table``) are rejected even if the loop
    stashed them.
    """
    raw = getattr(result, "voxel_xyz", None)
    if raw is None:
        return None, None, None
    try:
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None, None, None
    if arr.size < 3 or not np.isfinite(arr[:3]).all():
        return None, None, None
    phrase = str(getattr(result, "voxel_phrase", None) or "").strip() or None
    if phrase:
        from emet.memory.graph_eqa.agentic_explore import phrase_is_support_fixture_wrap

        if phrase_is_support_fixture_wrap(phrase):
            return None, None, None
        phrase_list = [str(p or "").strip() for p in phrases if str(p or "").strip()]
        if phrase_list:
            from emet.memory.graph_eqa.graph_types import label_matches_relevant_object

            if not any(
                label_matches_relevant_object(p, phrase) or label_matches_relevant_object(phrase, p)
                for p in phrase_list
            ):
                return None, None, None
    from_pin = getattr(result, "voxel_from_pin", None)
    return arr[:3].copy(), phrase, (None if from_pin is None else bool(from_pin))


def xyz_from_verified_obs(
    agent: Any,
    obs_id: int | None,
    *,
    phrases: list[str] | None = None,
) -> np.ndarray | None:
    """World XYZ for a verified view — matching object graph nodes only (never camera pose)."""
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


def record_ovmm_agentic_result(
    res: OvmmAgenticLocalizeResult,
    *,
    meta: dict[str, Any],
    prefix: str,
    default_query: str,
) -> tuple[np.ndarray | None, bool, str, str | None]:
    """Copy one FindObj/FindRec localize result into harness metrics.

    ``prefix`` is ``obj`` or ``recep``. Returns ``(xyz, ok, query_used, source)``.
    """
    if res.error:
        err_key = "agentic_find_error" if prefix == "obj" else "agentic_find_error_recep"
        meta[err_key] = res.error
    xyz = res.xyz
    ok = xyz is not None
    q_used = default_query
    extra = res.extra or {}
    source = extra.get("xyz_source")
    if not source and ok:
        source = "agentic_verify" if res.verified else None
    meta[f"{prefix}_n_retracted_claims"] = res.n_retracted_claims
    meta[f"{prefix}_agentic_rounds"] = res.n_rounds
    meta[f"{prefix}_n_nav"] = res.n_nav
    meta[f"{prefix}_n_explore"] = res.n_explore
    meta[f"{prefix}_verified_obs_id"] = res.verified_obs_id
    if extra:
        meta[f"{prefix}_xyz_source"] = extra.get("xyz_source")
        meta[f"{prefix}_from_pin"] = extra.get("from_pin")
        if extra.get("voxel_query_used"):
            q_used = str(extra["voxel_query_used"])
    return xyz, ok, q_used, source


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
    """Run :func:`run_agentic_eqa_result` and map the loop → world XYZ.

    Prefer the loop's object-phrase voxel XYZ (captured before submit releases
    SigLIP), then a mapping pin, then a phrase-matched graph node. Never live
    ``localize_text`` after the encoder is dropped. Camera pose is never scored.
    """
    from emet.memory.graph_eqa import run_agentic_eqa_result

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
        # Each find phase must be able to localize_text on the finished map.
        # The previous phase's submit_answer released SigLIP for the VLM; the
        # shared warm only snapshots ranks. Re-attach here (OVMM harness only)
        # so this phase's voxel localize is not a silent no-op.
        from emet.eval.dynagraph_vram import re_attach_siglip_encoder

        if getattr(agent, "encoder", None) is None:
            re_attach_siglip_encoder(agent)
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
    xyz = None
    xyz_source = None
    extra_q = None
    extra_stats: dict[str, Any] = {}
    loop_xyz, loop_phrase, loop_from_pin = _xyz_from_loop_voxel(result, localize_phrases)
    if loop_xyz is not None:
        xyz = loop_xyz
        xyz_source = "voxel"
        extra_q = loop_phrase
        extra_stats = {}
        if loop_from_pin is not None:
            extra_stats["from_pin"] = bool(loop_from_pin)
    if xyz is None:
        from emet.mapping.voxel_localize import pinned_xyz_from_phrases, voxel_map_from_agent

        voxel_xyz, voxel_q, voxel_stats = pinned_xyz_from_phrases(
            voxel_map_from_agent(agent),
            localize_phrases,
        )
        if voxel_xyz is not None:
            xyz = voxel_xyz
            xyz_source = "voxel"
            extra_q = voxel_q
            extra_stats = dict(voxel_stats or {})
    if xyz is None:
        xyz = xyz_from_verified_obs(agent, oid, phrases=localize_phrases) if oid is not None else None
        if xyz is not None:
            xyz_source = "graph_node"
    extra = {
        "budget_hit": bool(result.budget_hit),
        "answer_provenance": str(result.answer_provenance or ""),
        "xyz_source": xyz_source,
    }
    if extra_q:
        extra["voxel_query_used"] = extra_q
    if extra_stats:
        if "from_pin" in extra_stats:
            extra["from_pin"] = bool(extra_stats.get("from_pin"))
        if "yoloe_hit" in extra_stats:
            extra["yoloe_hit"] = bool(extra_stats.get("yoloe_hit"))
        cosine = extra_stats.get("max_cosine")
        if cosine is not None:
            extra["max_cosine"] = cosine
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
        extra=extra,
    )


def empty_ovmm_agentic_meta(*, use_agentic: bool) -> dict[str, Any]:
    """Shared FindObj/FindRec metric keys (agentic or one-shot)."""
    return {
        "agentic_find": bool(use_agentic),
        "obj_agentic_question": None,
        "recep_agentic_question": None,
        "obj_n_retracted_claims": 0,
        "recep_n_retracted_claims": 0,
    }


@dataclass
class OvmmFindQueryOutcome:
    """FindObj + FindRec localization for one episode (agentic or one-shot)."""

    obj_xyz: np.ndarray | None
    obj_ok: bool
    obj_query_used: str
    obj_source: str | None
    recep_xyz: np.ndarray | None
    recep_ok: bool
    recep_query_used: str
    recep_source: str | None
    meta: dict[str, Any]
    obj_detect_stats: dict[str, Any] = field(default_factory=dict)
    recep_detect_stats: dict[str, Any] = field(default_factory=dict)

    def localize_fields(self) -> dict[str, Any]:
        """Shared harness keys for localize success / query / source + agentic meta."""
        return {
            "obj_localize_success": bool(self.obj_ok),
            "recep_localize_success": bool(self.recep_ok),
            "obj_query_used": self.obj_query_used,
            "recep_query_used": self.recep_query_used,
            "obj_localize_source": self.obj_source,
            "recep_localize_source": self.recep_source,
            **self.meta,
        }


def attach_ovmm_episode_debug_dir(agent: Any) -> None:
    """Point query-PNG dumps at ``EMET_EQA_EPISODE_DIR`` when the harness set it."""
    ep_dir = os.environ.get("EMET_EQA_EPISODE_DIR", "").strip()
    if ep_dir:
        agent._episode_debug_dir = ep_dir


def _phase_trace_meta(
    *,
    phase: str,
    episode_id: str,
    object_query: str,
    start_recep: str,
    goal_recep: str,
    object_gt_body: str | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    meta = dict(extra or {})
    meta.update(
        {
            "ovmm_phase": phase,
            "episode_id": episode_id,
            "object": object_query,
            "start_recep": start_recep,
            "goal_recep": goal_recep,
        }
    )
    if phase == "find_object" and object_gt_body:
        meta["gt_body_key"] = object_gt_body
    return meta


def run_ovmm_agentic_find_pair(
    agent: Any,
    *,
    object_query: str,
    start_recep: str,
    goal_recep: str,
    episode_id: str,
    object_gt_body: str | None = None,
    max_rounds: int | None = None,
    max_nav_steps: int | None = None,
    extra_trace_meta: dict[str, Any] | None = None,
) -> OvmmFindQueryOutcome:
    """FindObj then FindRec through the shared AgenticEQA loop (Habitat and sim)."""
    meta = empty_ovmm_agentic_meta(use_agentic=True)
    obj_q = ovmm_find_object_question(object_query, start_recep)
    recep_q = ovmm_find_recep_question(goal_recep)
    meta["obj_agentic_question"] = obj_q
    meta["recep_agentic_question"] = recep_q
    extra = extra_trace_meta
    obj_res = run_ovmm_agentic_localize(
        agent,
        obj_q,
        max_rounds=max_rounds,
        max_nav_steps=max_nav_steps,
        require_verified=True,
        trace_meta=_phase_trace_meta(
            phase="find_object",
            episode_id=episode_id,
            object_query=object_query,
            start_recep=start_recep,
            goal_recep=goal_recep,
            object_gt_body=object_gt_body,
            extra=extra,
        ),
    )
    obj_xyz, obj_ok, obj_q_used, obj_source = record_ovmm_agentic_result(
        obj_res, meta=meta, prefix="obj", default_query=object_query
    )
    recep_res = run_ovmm_agentic_localize(
        agent,
        recep_q,
        max_rounds=max_rounds,
        max_nav_steps=max_nav_steps,
        require_verified=True,
        trace_meta=_phase_trace_meta(
            phase="find_recep",
            episode_id=episode_id,
            object_query=object_query,
            start_recep=start_recep,
            goal_recep=goal_recep,
            object_gt_body=None,
            extra=extra,
        ),
    )
    recep_xyz, recep_ok, recep_q_used, recep_source = record_ovmm_agentic_result(
        recep_res, meta=meta, prefix="recep", default_query=goal_recep
    )
    return OvmmFindQueryOutcome(
        obj_xyz=obj_xyz,
        obj_ok=obj_ok,
        obj_query_used=obj_q_used,
        obj_source=obj_source,
        recep_xyz=recep_xyz,
        recep_ok=recep_ok,
        recep_query_used=recep_q_used,
        recep_source=recep_source,
        meta=meta,
    )


def run_ovmm_find_queries(
    *,
    agent: Any,
    memory: Any,
    use_agentic: bool,
    object_query: str,
    start_recep: str,
    goal_recep: str,
    episode_id: str,
    object_gt_body: str | None = None,
    max_rounds: int | None = None,
    max_nav_steps: int | None = None,
    extra_trace_meta: dict[str, Any] | None = None,
    placements: dict[str, Any] | None = None,
    voxel_map: Any | None = None,
    prefer_voxel: bool = True,
    session: dict[str, Any] | None = None,
    convert_nav_to_world: bool = False,
    planar_frame: Literal["mujoco_xy", "habitat_xz"] = "mujoco_xy",
    phrase_only: bool = False,
    capture_voxel_stats: bool | None = None,
) -> OvmmFindQueryOutcome:
    """Dispatch FindObj/FindRec: agentic loop or one-shot memory localize.

    Habitat and sim both call this so question phrasing, trace keys, and scored
    XYZ provenance cannot drift. Sim ground-truth oracle stays in the sim
    runner (``run_ovmm_gt_oracle_find_pair``) — Habitat GT is memory after
    ``refresh_ground_truth``, not the MuJoCo placement lookup.
    """
    if use_agentic:
        return run_ovmm_agentic_find_pair(
            agent,
            object_query=object_query,
            start_recep=start_recep,
            goal_recep=goal_recep,
            episode_id=episode_id,
            object_gt_body=object_gt_body,
            max_rounds=max_rounds,
            max_nav_steps=max_nav_steps,
            extra_trace_meta=extra_trace_meta,
        )
    # Lazy: oneshot lives next to query_find_phase_localization.
    from emet.eval.ovmm_find_phase import run_ovmm_oneshot_find_pair

    if capture_voxel_stats is None:
        capture_voxel_stats = True
    return run_ovmm_oneshot_find_pair(
        memory,
        object_query=object_query,
        start_recep=start_recep,
        goal_recep=goal_recep,
        placements=placements,
        voxel_map=voxel_map,
        prefer_voxel=prefer_voxel,
        session=session,
        convert_nav_to_world=convert_nav_to_world,
        planar_frame=planar_frame,
        phrase_only=phrase_only,
        capture_voxel_stats=capture_voxel_stats,
    )


def should_use_agentic_find(backend: str, *, agentic_find: bool | None) -> bool:
    """Default: agentic loop for dynagraph/static_graph; off for dynamem/oracle."""
    if agentic_find is not None:
        return bool(agentic_find)
    b = str(backend or "").lower()
    return b in {"dynagraph", "static_graph", "graph_eqa"}

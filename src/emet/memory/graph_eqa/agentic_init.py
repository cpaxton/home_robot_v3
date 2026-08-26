# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Executor construction and tool dispatch for the agentic GraphEQA loop."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from emet.memory.graph_eqa.action_history import ActionHistoryEntry, resolve_action_progress_mode
from emet.memory.graph_eqa.agentic_config import (
    _FALSE,
    _TRUE,
    SIGLIP_IMAGE_PRESENT_THRESHOLD,
    _eqa_cfg,
    env_eqa_agentic_close_look,
    env_eqa_agentic_decision_policy,
    env_eqa_agentic_evidence_image,
    env_eqa_agentic_mcq_debias,
    env_eqa_agentic_no_early_unverified,
    env_eqa_agentic_require_verified,
    env_eqa_agentic_router,
    env_eqa_agentic_single_view_confirm,
    env_eqa_answerable_confirm,
    env_eqa_collect_trace,
    env_eqa_force_answer,
    env_eqa_room_policy,
    env_eqa_room_stamp_investigate,
    resolve_agentic_decision_policy,
)
from emet.memory.graph_eqa.agentic_policy import EvidencePolicy
from emet.memory.graph_eqa.agentic_types import AnswerEvidenceRecord, FinalAnswerDecision, PlaceInspectRecord
from emet.memory.graph_eqa.graph_memory import NavHypothesis, VerifyResult
from emet.memory.graph_eqa.room_clusters import resolve_room_policy


class AgenticInitMixin:
    """``__init__`` field setup and ``handle_tool`` dispatch."""

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
        # Close-map approach attempts per hyp obs_id (escape when unreachable / exhausted).
        self._close_map_attempts: dict[int, int] = {}
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

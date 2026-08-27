# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Episode run loop for the agentic GraphEQA executor."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from emet.memory.graph_eqa.agentic.config import EXPLORE_STREAK_FORCE_INVESTIGATE
from emet.memory.graph_eqa.agentic.policy import AgenticState, EvidencePolicy
from emet.memory.graph_eqa.agentic.types import AgenticEQAResult
from emet.utils.logger import Logger

_logger = Logger(__name__)


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
    self._voxel_score_xyz = None
    self._voxel_score_phrase = None
    self._voxel_score_from_pin = None
    self._last_verify = None
    self._vlm_assessed_obs_ids = set()
    self._evidence_policy = EvidencePolicy()
    self._tried = {}
    self._nav_to_obs_counts = {}
    self._nav_loop_flags = []
    self._place_inspect = {}
    self._close_map_attempts = {}
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
        # Locate / close-look: keep investigating nearby place cards instead of frontier drift.
        skip_prefer_explore = self._hold_detections_before_explore() and (
            self._unused_detection_hypothesis() is not None or self._nearby_untried_investigate_hyp() is not None
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
        # Close-map stay: keep approaching this XY until aimed-close or escape.
        stay_hyp = self._close_map_stay_hypothesis()
        if (
            self.decision_policy != "grounded_v2"
            and stay_hyp is not None
            and self.mode == "answer"
            and calls
            and calls[0][0] == "explore_frontier"
            and self._n_nav + self._n_explore < self.max_nav_steps
        ):
            self._append_trace(
                {
                    "event": "close_map_stay_redirect",
                    "from": "explore_frontier",
                    "to_obs_id": int(stay_hyp.obs_id),
                }
            )
            calls = [("investigate", {"obs_id": int(stay_hyp.obs_id)})]
            picked_by = f"{picked_by}+close_map_stay"
            action_rewrite = {
                "reason": "close_map_stay",
                "selected": selected_calls,
                "executed": calls,
            }
        # Mapping-pose graph views are not objects: rewrite to an unused proposal.
        unused_det = self._unused_detection_hypothesis()
        if (
            self.decision_policy != "grounded_v2"
            and unused_det is not None
            and self.mode == "answer"
            and calls
            and calls[0][0] in ("investigate", "navigate_to_obs")
            and self._n_nav + self._n_explore < self.max_nav_steps
        ):
            try:
                pick_oid = int((calls[0][1] or {}).get("obs_id"))
            except (TypeError, ValueError):
                pick_oid = None
            pick_hyp = self._hypothesis_for_obs_id(pick_oid) if pick_oid is not None else None
            if self._hypothesis_is_camera_pose_place(pick_hyp):
                self._append_trace(
                    {
                        "event": "camera_pose_place_redirect",
                        "from_obs_id": pick_oid,
                        "to_obs_id": int(unused_det.obs_id),
                    }
                )
                calls = [("investigate", {"obs_id": int(unused_det.obs_id)})]
                picked_by = f"{picked_by}+camera_pose_place"
                action_rewrite = {
                    "reason": "camera_pose_place",
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
        voxel_xyz=self._voxel_score_xyz,
        voxel_phrase=self._voxel_score_phrase,
        voxel_from_pin=self._voxel_score_from_pin,
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
            "voxel_xyz": list(result.voxel_xyz) if result.voxel_xyz is not None else None,
            "voxel_phrase": result.voxel_phrase,
            "voxel_from_pin": result.voxel_from_pin,
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
        "final_decision": (self._final_answer_decision.to_dict() if self._final_answer_decision is not None else None),
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

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tool routing and fallbacks for the agentic GraphEQA executor."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from emet.agent.prompt import parse_tool_calls_response
from emet.mapping.voxel_localize import is_proposal_handle
from emet.memory.graph_eqa.agentic.config import (
    _FALSE,
    _TRUE,
    EXPLORE_STREAK_FORCE_INVESTIGATE,
    ROUTER_MAX_NEW_TOKENS,
    env_eqa_router_room_images,
)
from emet.memory.graph_eqa.agentic.policy import AgenticState
from emet.memory.graph_eqa.agentic.tools import (
    build_agentic_eqa_tools,
    build_graph_eqa_system_prompt,
    build_state_message,
    coerce_room_label,
)
from emet.memory.graph_eqa.graph_memory import NavHypothesis
from emet.memory.graph_eqa.spatial.room_clusters import (
    merge_room_estimates,
    question_target_rooms,
    room_leave_needed,
    room_mismatches_question,
)
from emet.utils.logger import Logger

_logger = Logger(__name__)


def _rendered_action_allowlist(self) -> dict[str, tuple[Any, ...]]:
    snapshot = self._last_agent_state_snapshot
    return {
        "place_ids": tuple(getattr(snapshot, "visible_place_ids", ()) or ()),
        "place_obs_ids": tuple(getattr(snapshot, "visible_place_obs_ids", ()) or ()),
        "frontier_ids": tuple(getattr(snapshot, "visible_frontier_ids", ()) or ()),
        "event_ids": tuple(getattr(snapshot, "visible_event_ids", ()) or ()),
    }


def _grounded_visible_place_obs(self) -> set[int] | None:
    """Rendered place-card obs ids, or None when grounded_v2 is not filtering."""
    if self.decision_policy != "grounded_v2" or self._last_agent_state_snapshot is None:
        return None
    return {int(item) for item in self._rendered_action_allowlist()["place_obs_ids"]}


def _next_rendered_hypothesis(self) -> NavHypothesis | None:
    visible = self._grounded_visible_place_obs() or set()
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
    if budget_left:
        stay = self._close_map_stay_hypothesis()
        if stay is not None:
            return "investigate", {"obs_id": int(stay.obs_id)}
    if budget_left and self._hold_detections_before_explore():
        det = self._unused_detection_hypothesis()
        if det is not None:
            return "investigate", {"obs_id": int(det.obs_id)}
    if budget_left and self._prefers_nearby_investigate() and not self._defer_nearby_for_prefer_explore():
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
    if need_more and budget_left:
        # A miss on this RGB is not a reason to leave a remaining detection.
        h = (
            self._next_rendered_hypothesis()
            if self.decision_policy == "grounded_v2" and self._last_agent_state_snapshot is not None
            else self._next_untried_hypothesis()
        )
        if h is not None and (str(h.source) == "voxel" or is_proposal_handle(h.obs_id)):
            return "investigate", {"obs_id": int(h.obs_id)}
        if not frontiers_gone:
            return "explore_frontier", self._rendered_frontier_args()
    # After a close ABSENT look, grow coverage before the next investigate —
    # but only once; if we already explored this streak and place cards remain,
    # look closer instead of frontier-only loops.
    if self.decision_policy != "grounded_v2" and budget_left and not frontiers_gone and self._prefer_explore:
        streak = int(getattr(self, "_n_consecutive_explore", 0) or 0)
        if streak < 1:
            return "explore_frontier", self._rendered_frontier_args()
        near = self._nearby_untried_investigate_hyp() if self._prefers_nearby_investigate() else None
        if near is not None:
            return "investigate", {"obs_id": int(near.obs_id)}
        hyp = (
            self._next_rendered_hypothesis()
            if self.decision_policy == "grounded_v2" and self._last_agent_state_snapshot is not None
            else self._next_untried_hypothesis()
        )
        if hyp is not None:
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
        # Location MCQ: a keyword voxel in this room is not a pin — explore instead.
        if h is not None and self._hypothesis_is_detection(h) and not self._hold_detections_before_explore():
            h = None
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
    pose_list = [float(value) for value in np.asarray(pose, dtype=float).reshape(-1)[:3]] if pose is not None else None
    if pose_list is not None:
        if self._router_path_world:
            self._router_path_m += float(
                np.linalg.norm(
                    np.asarray(pose_list[:2], dtype=float) - np.asarray(self._router_path_world[-1][:2], dtype=float)
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
            "router_vlm" if vlm_room != "unknown" else ("graph_current_pose" if graph_room != "unknown" else "unknown")
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
    if out.get("ok") or self.mode != "answer" or self._n_nav + self._n_explore >= self.max_nav_steps:
        return False
    status = str(out.get("status") or "")
    if tool == "explore_frontier" and status == "DETECTIONS_REMAIN":
        hyp = self._unused_detection_hypothesis() or self._next_untried_hypothesis()
        if hyp is None:
            return False
        redirect_tool = "investigate"
        redirect_args: dict[str, Any] = {"obs_id": int(hyp.obs_id)}
    elif tool in ("navigate_to_obs", "investigate") and status == "CAMERA_POSE_PLACE":
        hyp = self._unused_detection_hypothesis() or self._next_untried_hypothesis()
        if hyp is None:
            return False
        redirect_tool = "investigate"
        redirect_args = {"obs_id": int(hyp.obs_id)}
    elif tool in ("navigate_to_obs", "investigate"):
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
    else:
        return False
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

# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Investigate / navigate_to_obs tools and close-map stay for agentic GraphEQA."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from emet.mapping.close_map import (
    CloseLookDecision,
    close_map_from_voxel_map,
    decide_close_look,
)
from emet.memory.graph_eqa.agentic_config import (
    INVESTIGATE_SOURCES,
    NAV_CONSECUTIVE_FAIL_LIMIT,
    PLACE_APPROACH_SAMPLES,
)
from emet.memory.graph_eqa.agentic_types import PlaceInspectRecord
from emet.memory.graph_eqa.graph_memory import NavHypothesis
from emet.utils.logger import Logger

_logger = Logger(__name__)


class AgenticInvestigateMixin:
    """Close-map stay plus ``investigate`` / ``navigate_to_obs`` tools."""

    def _close_look_query(self, obs_id: int, hyp: NavHypothesis | None = None):
        """Neighborhood close-look stats at this place-card XY, or None if unmapped."""
        voxel_map, _ = self._voxel_planner()
        cm = close_map_from_voxel_map(voxel_map)
        if cm is None:
            return None
        if hyp is None:
            hyp = self._hypothesis_for_obs_id(obs_id)
        xy = self._place_anchor_xy(int(obs_id), hyp)
        if xy is None:
            return None
        return cm.query_xy(xy[0], xy[1])

    def _decide_close_look(
        self,
        obs_id: int,
        *,
        nav_blocked: bool,
        increment: bool = False,
        hyp: NavHypothesis | None = None,
    ) -> CloseLookDecision:
        oid = int(obs_id)
        if increment:
            self._close_map_attempts[oid] = int(self._close_map_attempts.get(oid, 0)) + 1
        rec = self._place_inspect.get(oid)
        approaches_left = rec.approaches_left if rec is not None else PLACE_APPROACH_SAMPLES
        return decide_close_look(
            self._close_look_query(oid, hyp),
            approaches_left=int(approaches_left),
            nav_blocked=bool(nav_blocked),
            attempts=int(self._close_map_attempts.get(oid, 0)),
            is_chat=False,
        )

    def _close_map_unresolved_stay(self, obs_id: int) -> bool:
        """True when the close-map says stay on this XY (not close yet, still reachable)."""
        return bool(self._decide_close_look(int(obs_id), nav_blocked=False, increment=False).stay)

    def _close_map_stay_hypothesis(self) -> NavHypothesis | None:
        visible = self._grounded_visible_place_obs()
        for h in self._investigate_hypotheses():
            oid = int(h.obs_id)
            if visible is not None and oid not in visible:
                continue
            if self._obs_already_verified(oid) or oid in self._unreachable_obs_ids:
                continue
            if self._close_map_unresolved_stay(oid):
                return h
        return None

    def _apply_close_map_after_approach(
        self,
        obs_id: int,
        *,
        hyp: NavHypothesis | None,
        nav_blocked: bool,
    ) -> CloseLookDecision:
        """Increment approach count, stay/escape, and write a close_map trace row."""
        oid = int(obs_id)
        decision = self._decide_close_look(oid, nav_blocked=nav_blocked, increment=True, hyp=hyp)
        rec = self._place_inspect.get(oid) or PlaceInspectRecord()
        rec.close_map_reason = str(decision.reason)
        rec.close_map_resolved = bool(decision.query.resolved) if decision.query is not None else None
        self._place_inspect[oid] = rec
        payload = {
            "event": "close_map",
            "obs_id": oid,
            "attempts": int(self._close_map_attempts.get(oid, 0)),
            **decision.as_dict(),
        }
        self._append_trace(payload)
        if decision.escape:
            self._unreachable_obs_ids.add(oid)
            self._tried.setdefault(oid, f"close_map_{decision.reason}")
            self._prefer_explore = True
            self._prefer_explore_reason = str(decision.reason)
        elif decision.stay:
            self._prefer_explore = False
            self._prefer_explore_reason = ""
        return decision

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
            # Waypoint resolution is deterministic per card (anchor independent of the
            # approach bearing): block it now so the router / fallback advances instead
            # of re-picking a failing card on every remaining approach index.
            self._mark_approach_tried(oid, next_ap)
            self._tried.setdefault(oid, "no waypoint")
            self._unreachable_obs_ids.add(oid)
            close_map = self._apply_close_map_after_approach(oid, hyp=hyp, nav_blocked=True)
            self._append_trace(
                {
                    "tool": trace_tool,
                    "ok": False,
                    "obs_id": oid,
                    "status": "NO_WAYPOINT",
                    "approach_index": int(next_ap),
                    "close_map": close_map.as_dict(),
                }
            )
            return {
                "ok": False,
                "error": f"no waypoint for obs_id={obs_id}",
                "status": "NO_WAYPOINT",
                "obs_id": oid,
                "close_map": close_map.as_dict(),
            }
        start = self._robot_xyt_world() if xyt is not None else np.array([0.0, 0.0, 0.0])
        # Face the OBJECT on arrival, not the approach waypoint. navigate_to_target_pose
        # with target_theta=None leaves the final yaw arbitrary (often a wall), so the
        # arrival capture sees a brick wall and the VLM assess reports present=False.
        # theta toward the object anchor from the standing waypoint makes the head look
        # at the target itself.
        try:
            t_arr = np.asarray(target, dtype=float).reshape(-1)
            look_x, look_y = self._investigate_arrival_look_at_xy(oid, t_arr)
            target_theta = float(np.arctan2(look_y - t_arr[1], look_x - t_arr[0]))
            if math.hypot(look_x - t_arr[0], look_y - t_arr[1]) < 1e-6:
                # Waypoint coincides with the anchor (robot already inside the min
                # standoff): face the object from the current robot pose instead.
                rxy = self._robot_xyt_world()
                if rxy is not None:
                    target_theta = float(np.arctan2(look_y - float(rxy[1]), look_x - float(rxy[0])))
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
            close_map = self._apply_close_map_after_approach(
                oid,
                hyp=hyp,
                nav_blocked=oid in self._unreachable_obs_ids,
            )
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
                "close_map": close_map.as_dict(),
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
        close_map = self._apply_close_map_after_approach(oid, hyp=hyp, nav_blocked=False)
        rec = self._place_inspect.get(oid) or rec
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
                "close_map": close_map.as_dict(),
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
            "close_map": close_map.as_dict(),
        }

    def _tool_navigate_to_obs(self, obs_id: int) -> dict[str, Any]:
        """Compat alias — ``navigate_to_obs`` shares the investigate approach path."""
        return self._tool_investigate(int(obs_id), tool_name="navigate_to_obs")

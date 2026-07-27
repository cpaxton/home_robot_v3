# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unified agentic GraphEQA loop: explore / navigate / verify / answer with tools."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from emet.agent.prompt import parse_tool_calls_response
from emet.memory.graph_eqa.agentic_policy import (
    AgenticState,
    EvidencePolicy,
    EvidenceRecord,
)
from emet.memory.graph_eqa.agentic_tools import (
    build_agentic_eqa_tools,
    build_graph_eqa_system_prompt,
    build_state_message,
)
from emet.memory.graph_eqa.graph_memory import (
    _QUESTION_VERB_FILLERS,
    SIGLIP_CONFIRM_THRESHOLD,
    SIGLIP_PRESENT_THRESHOLD,
    NavHypothesis,
    VerifyResult,
    label_matches_relevant_object,
    question_stem_for_keywords,
)
from emet.utils.logger import Logger

_logger = Logger(__name__)

# Image-space SigLIP is a **high-recall / high-FP** proposal (see agentic_scale.md).
# Three bands on Habitat RGB (offline calib: real hits cluster ~0.10–0.14):
#   >= PRESENT (0.12)  → PRESENT
#   >= ABSENT  (0.10)  → CANDIDATE
#   <  ABSENT  (0.10)  → ABSENT   (true-negative for *this* view — move on)
# Do not treat ABSENT as proof the object is gone from the scene.
# Qwen (vlm_assess / router) decides answerability and whether to explore vs submit;
# SigLIP only appears in the state as a cheap proposal.
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

# Region escape: after this many consecutive "target not visible" view assessments,
# require the next frontier to be at least ESCAPE_MIN_TRAVEL_M away so the robot
# leaves the area instead of re-scanning it (holdout q104/q105 circled their spawn).
NOT_PRESENT_ESCAPE_STREAK = 2
ESCAPE_MIN_TRAVEL_M = 3.0

# Same hyp obs_id navigated this many times without a fresh graph obs → stall / break loop.
NAV_SAME_OBS_LOOP_LIMIT = 2

# Hyp recall: how many evidence cards to show the router / walk in fallback.
DEFAULT_HYP_RECALL_K = 6

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


def env_eqa_hyp_recall_k() -> int:
    """Top-K evidence cards for agentic hyp recall (default 6)."""
    raw = os.environ.get("EMET_EQA_HYP_RECALL_K", "").strip()
    if not raw:
        return DEFAULT_HYP_RECALL_K
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_HYP_RECALL_K


def env_eqa_collect_trace() -> bool:
    v = os.environ.get("EMET_EQA_TRACE", "").strip().lower()
    return v in _TRUE


def env_eqa_agentic_require_verified() -> bool | None:
    """When True, refuse unverified ``submit_answer`` (incl. fallback / budget exhaust).

    Env: ``EMET_EQA_AGENTIC_REQUIRE_VERIFIED=1``. Unverified exhaust → abstain
    (``Unknown``) instead of a guessed letter.
    """
    v = os.environ.get("EMET_EQA_AGENTIC_REQUIRE_VERIFIED", "").strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
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


@dataclass
class AgenticEQAResult:
    discord_text: str
    answer: str
    confidence: bool
    relevant_images: list[Any] = field(default_factory=list)
    tool_log: list[str] = field(default_factory=list)
    verified: bool = False
    verified_obs_id: int | None = None
    n_rounds: int = 0
    n_nav: int = 0
    n_explore: int = 0
    wall_s: float = 0.0
    budget_hit: bool = False


class AgenticEQAExecutor:
    """Bounded tool loop for post-explore / world-change EQA."""

    def __init__(
        self,
        agent: Any,
        question: str | None,
        *,
        goal: str = "",
        max_rounds: int = 8,
        max_nav_steps: int = 8,
        verify_min_sim: float = SIGLIP_CONFIRM_THRESHOLD,
        trace_path: Path | str | None = None,
        trace_meta: dict[str, Any] | None = None,
        collect_trace: bool | None = None,
        router: bool | None = None,
        require_verified: bool | None = None,
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
        self._tool_log: list[str] = []
        self._trace_rows: list[dict[str, Any]] = []
        # obs_id → successful navigate_to_obs attempts (loop detection for the router).
        self._nav_to_obs_counts: dict[int, int] = {}
        self._nav_loop_flags: list[dict[str, Any]] = []
        self._last_capture_status: str | None = None
        self._collect_trace = bool(collect_trace) if collect_trace is not None else (
            env_eqa_collect_trace() or bool(_eqa_cfg(agent).get("collect_agentic_trace", False))
        )
        self._trace_path = Path(trace_path) if trace_path else None
        self._trace_meta = dict(trace_meta or {})
        self._gt_placements: dict[str, Any] | None = None
        self._round = 0
        self._tried: dict[int, str] = {}  # obs_id → last verify summary (never re-verify)
        self._followed_eqa_actions: set[int] = set()
        # Soft explores after Unknown when Action:N is missing/OOB or already followed.
        self._n_unknown_explore = 0
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
            require_verified
            if require_verified is not None
            else (env_req if env_req is not None else cfg_req)
        )
        self._tools: list[Any] | None = None
        self._tool_names: set[str] = set()
        self._system_prompt: str = ""

    @property
    def graph_memory(self) -> Any:
        return getattr(self.agent, "graph_memory", None)

    @property
    def query_text(self) -> str:
        """Phrase used to bias graph inspection / frontier picks (question or explore goal)."""
        return self.question or self.goal

    def _robot_xyt(self) -> np.ndarray | None:
        robot = getattr(self.agent, "robot", None)
        if robot is None or not hasattr(robot, "get_base_pose"):
            return None
        try:
            return np.asarray(robot.get_base_pose(), dtype=float).reshape(-1)
        except Exception:
            return None

    def _append_trace(self, row: dict[str, Any]) -> None:
        if not self._collect_trace:
            return
        payload = {
            **self._trace_meta,
            "question": self.question,
            "mode": self.mode,
            "round": self._round,
            **row,
        }
        self._trace_rows.append(payload)
        if self._trace_path is not None:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._trace_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")

    def _attach_gt(self, row: dict[str, Any], xyz: np.ndarray | None) -> None:
        placements = self._gt_placements
        if placements is None:
            try:
                from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

                robot = getattr(self.agent, "robot", None)
                session = robot.get_emet_session() if robot is not None and hasattr(robot, "get_emet_session") else None
                placements = read_sim_object_placements(session) or {}
                self._gt_placements = placements
            except Exception:
                placements = {}
                self._gt_placements = {}
        gt_key = self._trace_meta.get("gt_body_key") or ""
        if not gt_key or gt_key not in placements:
            return
        info = placements[gt_key]
        gt_xyz = np.asarray(info.get("pos"), dtype=float).reshape(-1)[:3]
        row["gt_body_key"] = gt_key
        row["gt_xyz"] = [float(x) for x in gt_xyz.tolist()]
        if xyz is not None:
            d = float(np.linalg.norm(np.asarray(xyz, dtype=float).reshape(-1)[:2] - gt_xyz[:2]))
            row["gt_dist_m"] = d
            row["gt_present"] = bool(d <= 1.5)

    def handle_tool(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(args or {})
        name = (name or "").strip().lower()
        self._tool_log.append(name)
        if name == "inspect_graph":
            return self._tool_inspect_graph()
        if name == "explore_frontier":
            return self._tool_explore_frontier(str(args.get("toward") or ""))
        if name == "navigate_to_obs":
            return self._tool_navigate_to_obs(int(args.get("obs_id", -1)))
        if name == "look_around":
            return self._tool_look_around()
        if name == "capture_and_update":
            return self._tool_capture_and_update()
        if name == "verify_siglip":
            return self._tool_verify_siglip(
                str(args.get("phrase") or ""),
                int(args.get("obs_id", -1)) if args.get("obs_id") is not None else None,
            )
        if name == "submit_answer":
            return self._tool_submit_answer(str(args.get("answer") or ""))
        if name == "finish":
            return self._tool_finish(str(args.get("summary") or ""))
        return {"ok": False, "error": f"unknown tool {name!r}"}

    def _tool_inspect_graph(self) -> dict[str, Any]:
        gm = self.graph_memory
        if gm is None:
            return {"ok": False, "error": "no graph_memory"}
        if hasattr(gm, "extract_relevant_objects") and getattr(gm, "image_description_client", None) is not None:
            gm.extract_relevant_objects(self.query_text)
        if getattr(gm, "memory_summary_enabled", False) and hasattr(gm, "refresh_siglip_confirmed_memory"):
            gm.refresh_siglip_confirmed_memory()
        try:
            hypotheses = gm.hypothesize_nav_targets(
                self.query_text,
                max_k=env_eqa_hyp_recall_k(),
                robot_xyt=self._robot_xyt(),
            )
        except TypeError:
            hypotheses = gm.hypothesize_nav_targets(
                self.query_text, max_k=env_eqa_hyp_recall_k()
            )
        self._set_hypotheses(hypotheses)
        out = {
            "ok": True,
            "n_hypotheses": len(self._hypotheses),
            "hypotheses": [
                {
                    "phrase": h.phrase,
                    "obs_id": int(h.obs_id),
                    "xyz": [float(x) for x in np.asarray(h.xyz).reshape(-1)[:3]],
                    "source": h.source,
                    "siglip_sim": (
                        float(h.siglip_sim)
                        if getattr(h, "siglip_sim", None) is not None
                        else None
                    ),
                }
                for h in self._hypotheses
            ],
        }
        self._append_trace(
            {
                "tool": "inspect_graph",
                "picked_by": "loop",
                "policy_state": self._evidence_policy.state,
                "n_hypotheses": out["n_hypotheses"],
                "hypotheses": out["hypotheses"],
            }
        )
        return out

    def _tool_explore_frontier(self, toward: str = "") -> dict[str, Any]:
        if self._n_nav + self._n_explore >= self.max_nav_steps:
            return {"ok": False, "error": "nav budget exhausted"}
        bias = (toward or "").strip() or self.query_text
        agent = self.agent
        frontier_xyz = None
        pick_source = "pick_uncovered"
        try:
            from emet.controller.habitat_nav import pick_uncovered_explore_target

            escape_m = self._escape_min_travel_m()
            candidates: list[np.ndarray | None] = []
            # GraphEQA-style: VLM ranks a small pool of reachable frontier RGBs.
            # Agentic always tries this; classic coverage path still uses EMET_VLM_FRONTIER_SCORING.
            if hasattr(agent, "_vlm_frontier_choice"):
                try:
                    vlm_pt = agent._vlm_frontier_choice(bias)
                except Exception as e:
                    _logger.warning(f"vlm_frontier_choice failed: {e}")
                    vlm_pt = None
                if vlm_pt is not None:
                    candidates.append(vlm_pt)
                    pick_source = "vlm_frontier"
            # SigLIP guidance aims at the frontier nearest the best-matching *already
            # observed* point, so while escaping it just pulls us back into the area we
            # already rejected. Let region utility choose instead.
            if escape_m <= 0.0 and hasattr(agent, "_siglip_guided_frontier"):
                candidates.append(agent._siglip_guided_frontier(bias))
            if hasattr(agent, "_best_frontier_point_from_graph"):
                candidates.append(agent._best_frontier_point_from_graph(bias))
            frontier_xyz = pick_uncovered_explore_target(
                agent,
                question=bias,
                candidates=candidates,
                blocked=getattr(agent, "_habitat_blocked_goals", None),
                recent_goals=getattr(agent, "_habitat_recent_goals", None),
                min_travel_m=escape_m,
            )
            if frontier_xyz is not None and pick_source == "vlm_frontier":
                # Confirm the accepted goal is still the VLM pick (not a later fallback).
                vlm0 = candidates[0] if candidates else None
                if vlm0 is None or float(
                    np.linalg.norm(
                        np.asarray(frontier_xyz, dtype=float).reshape(-1)[:2]
                        - np.asarray(vlm0, dtype=float).reshape(-1)[:2]
                    )
                ) > 0.35:
                    pick_source = "pick_uncovered"
        except Exception as e:
            _logger.warning(f"explore_frontier pick failed: {e}")
            pick_source = "pick_uncovered"
        frontier_key = -1_000_000 - self._n_explore
        hypothesis_id = self._begin_policy_approach(
            "frontier",
            frontier_key,
            bias,
        )
        ok = False
        start = self._robot_xyt()
        if start is None:
            start = np.array([0.0, 0.0, 0.0])
        if frontier_xyz is not None and hasattr(agent, "navigate_to_target_pose"):
            try:
                ok = bool(agent.navigate_to_target_pose(frontier_xyz, start, None))
            except TypeError:
                ok = bool(agent.navigate_to_target_pose(frontier_xyz, start))
            self._n_explore += 1
            if ok:
                self._retire_visited_frontier(frontier_xyz=frontier_xyz)
        elif hasattr(agent, "run_exploration"):
            ok = bool(agent.run_exploration())
            self._n_explore += 1
            pick_source = "run_exploration_fallback"
            # Recover a goal for viz/trace when the uncovered picker returned None.
            if frontier_xyz is None:
                recent = list(getattr(agent, "_habitat_recent_goals", None) or [])
                if recent:
                    frontier_xyz = np.array([float(recent[-1][0]), float(recent[-1][1]), 1.0])
                else:
                    after = self._robot_xyt()
                    if after is not None:
                        frontier_xyz = np.asarray(after, dtype=float).reshape(-1)[:3]
            if ok and frontier_xyz is not None:
                self._retire_visited_frontier(frontier_xyz=frontier_xyz)
        cap = self._tool_capture_and_update()
        look_retry = False
        # After a successful explore nav, a mid-floor / already-mapped goal often yields
        # NO_NEW_OBS. Spin in place so we still peel new coverage from this pose.
        if (
            ok
            and not cap.get("ok")
            and str(cap.get("status") or "") == "NO_NEW_OBS"
        ):
            look_retry = True
            look = self._tool_look_around(verify=False)
            look_cap = look.get("capture") if isinstance(look, dict) else None
            if isinstance(look_cap, dict) and look_cap.get("ok"):
                cap = look_cap
        verify_out = None
        if cap.get("ok") and cap.get("obs_id") is not None:
            self._policy_approached(hypothesis_id, int(cap["obs_id"]))
            if self.mode == "answer":
                verify_out = self._verify_after_motion(phrase=self.query_text)
        panel_path = self._save_frontier_pick_panel(
            frontier_xyz,
            robot_xyt_before=start,
        )
        row = {
            "tool": "explore_frontier",
            "ok": ok,
            "frontier_xyz": [float(x) for x in np.asarray(frontier_xyz).reshape(-1)[:3]]
            if frontier_xyz is not None
            else None,
            "source": pick_source,
            "pick_panel": str(panel_path) if panel_path else None,
            "look_around_on_no_new_obs": look_retry,
        }
        self._attach_gt(row, frontier_xyz)
        self._append_trace(row)
        return {
            "ok": ok,
            "capture": cap,
            "frontier_xyz": row["frontier_xyz"],
            "verify": verify_out,
        }

    def _tool_navigate_to_obs(self, obs_id: int) -> dict[str, Any]:
        if self._n_nav + self._n_explore >= self.max_nav_steps:
            return {"ok": False, "error": "nav budget exhausted"}
        gm = self.graph_memory
        agent = self.agent
        if gm is None or not hasattr(agent, "navigate_to_target_pose"):
            return {"ok": False, "error": "nav unavailable"}
        oid = int(obs_id)
        prior_visits = int(self._nav_to_obs_counts.get(oid, 0))
        # Already scored / stalled on this hyp — do not burn another nav round.
        if (
            prior_visits >= NAV_SAME_OBS_LOOP_LIMIT
            or self._obs_already_verified(oid)
            or (
                oid in self._tried
                and str(self._tried.get(oid) or "").startswith("STALLED_NAV_LOOP")
            )
        ):
            flag = {
                "obs_id": oid,
                "visits": prior_visits,
                "status": "NAV_LOOP_BLOCKED",
                "prior": self._tried.get(oid),
            }
            self._nav_loop_flags.append(flag)
            self._append_trace({"tool": "navigate_to_obs", "ok": False, **flag})
            return {
                "ok": False,
                "error": (
                    f"nav loop on obs_id={oid} (visits={prior_visits}, "
                    f"prior={self._tried.get(oid)!r}); explore_frontier or look_around"
                ),
                "status": "NAV_LOOP_BLOCKED",
                "obs_id": oid,
            }
        hyp = next((h for h in self._hypotheses if int(h.obs_id) == oid), None)
        # Router must pick among recalled evidence cards when any are listed.
        if hyp is None and self._hypotheses:
            listed = sorted({int(h.obs_id) for h in self._hypotheses})
            self._append_trace(
                {
                    "tool": "navigate_to_obs",
                    "ok": False,
                    "obs_id": oid,
                    "status": "OBS_NOT_IN_EVIDENCE",
                    "listed_obs_ids": listed,
                }
            )
            return {
                "ok": False,
                "error": (
                    f"obs_id={oid} is not in the current evidence list "
                    f"{listed}; pick a listed obs_id or explore_frontier"
                ),
                "status": "OBS_NOT_IN_EVIDENCE",
                "obs_id": oid,
                "listed_obs_ids": listed,
            }
        phrase = hyp.phrase if hyp is not None else self.query_text
        source = hyp.source if hyp is not None else "graph"
        hypothesis_id = self._begin_policy_approach(source, oid, phrase)
        xyt = self._robot_xyt()
        target = gm._navigation_waypoint_for_obs(oid, xyt)
        if target is None:
            return {"ok": False, "error": f"no waypoint for obs_id={obs_id}"}
        start = xyt if xyt is not None else np.array([0.0, 0.0, 0.0])
        try:
            finished = bool(agent.navigate_to_target_pose(target, start, None, target_obs_id=oid))
        except TypeError:
            finished = bool(agent.navigate_to_target_pose(target, start, None))
        self._n_nav += 1
        self._nav_to_obs_counts[oid] = prior_visits + 1
        nav_res = getattr(agent, "_last_nav_attempt", None)
        dist_m = float(getattr(nav_res, "dist_m", 0.0) or 0.0) if nav_res else 0.0
        note = str(getattr(nav_res, "note", "") or "") if nav_res else ""
        if hasattr(gm, "record_nav_attempt"):
            gm.record_nav_attempt(oid, success=finished, note=note or "agentic", dist_m=dist_m)
        if not finished:
            self._tried.setdefault(oid, "nav failed")
        row = {
            "tool": "navigate_to_obs",
            "obs_id": oid,
            "target_xyz": [float(x) for x in np.asarray(target).reshape(-1)[:3]],
            "nav_success": bool(finished),
            "nav_dist_m": dist_m,
            "nav_note": note,
            "nav_visit_n": self._nav_to_obs_counts[oid],
        }
        self._attach_gt(row, target)
        self._append_trace(row)
        # Frontier waypoints are not evidence — retire after a successful visit.
        if finished and (
            str(source) == "frontier"
            or (hasattr(gm, "_obs_is_frontier") and gm._obs_is_frontier(oid))
        ):
            self._retire_visited_frontier(frontier_obs_id=oid, frontier_xyz=target)
        cap = self._tool_capture_and_update()
        look_retry = False
        # Parity with explore_frontier: already-mapped pose often yields NO_NEW_OBS;
        # spin so voxel+graph can register a new observation from this station.
        if (
            finished
            and not cap.get("ok")
            and str(cap.get("status") or "") == "NO_NEW_OBS"
        ):
            look_retry = True
            look = self._tool_look_around(verify=False)
            look_cap = look.get("capture") if isinstance(look, dict) else None
            if isinstance(look_cap, dict) and look_cap.get("ok"):
                cap = look_cap
        verify_out = None
        if cap.get("ok") and cap.get("obs_id") is not None:
            self._policy_approached(hypothesis_id, int(cap["obs_id"]))
            if self.mode == "answer":
                verify_out = self._verify_after_motion(phrase=phrase)
        else:
            # Capture still stalled: score *this* view once so ABSENT/CANDIDATE updates
            # planner state, then mark the hyp so the router cannot re-pick it forever.
            verify_out = self._verify_stalled_nav_view(oid, phrase=phrase)
            flag = {
                "obs_id": oid,
                "visits": self._nav_to_obs_counts[oid],
                "status": "STALLED_NAV_LOOP",
                "look_around_on_no_new_obs": look_retry,
                "verify_status": (verify_out or {}).get("status")
                if isinstance(verify_out, dict)
                else None,
            }
            self._nav_loop_flags.append(flag)
            self._tried[oid] = (
                f"STALLED_NAV_LOOP verify={flag.get('verify_status') or 'none'}"
            )
            self._append_trace({"event": "nav_loop", **flag})
            _logger.warning(
                f"agentic nav loop: obs_id={oid} visits={flag['visits']} "
                f"look_retry={look_retry} verify={flag.get('verify_status')}"
            )
        return {
            "ok": bool(finished),
            "target_xyz": row["target_xyz"],
            "capture": cap,
            "verify": verify_out,
            "look_around_on_no_new_obs": look_retry,
        }

    def _tool_look_around(self, *, verify: bool = True) -> dict[str, Any]:
        agent = self.agent
        hypothesis_id = self._begin_policy_approach(
            "look",
            -2_000_000 - self._n_nav - self._n_explore,
            self.query_text,
        )
        ok = False
        if hasattr(agent, "look_around"):
            try:
                agent.look_around()
                ok = True
            except Exception as e:
                _logger.warning(f"look_around failed: {e}")
        cap = self._tool_capture_and_update()
        verify_out = None
        if cap.get("ok") and cap.get("obs_id") is not None:
            self._policy_approached(hypothesis_id, int(cap["obs_id"]))
            if verify and self.mode == "answer":
                verify_out = self._verify_after_motion(phrase=self.query_text)
        self._append_trace({"tool": "look_around", "ok": ok})
        return {"ok": ok, "capture": cap, "verify": verify_out}

    def _siglip_phrase(self, phrase: str = "") -> str:
        """Short object phrase for SigLIP — never feed the full MCQ question text."""
        text = (phrase or "").strip()
        q = (self.question or "").strip()
        # Callers sometimes pass query_text (== full question); prefer extracted target.
        if (not text) or (q and text == q) or ("?" in text and len(text.split()) > 6):
            text = (self._target_phrase or "").strip()
        if not text:
            # Last resort: stem without choices / trailing "Answer:"
            stem = question_stem_for_keywords(self.question or "")
            text = (stem or self.query_text or "").strip()
        return text

    def _verify_after_motion(self, *, phrase: str = "") -> dict[str, Any]:
        """Run verify on the newest captured view (router and fallback both need this)."""
        return self.handle_tool(
            "verify_siglip",
            {"phrase": self._siglip_phrase(phrase)},
        )

    def _verify_stalled_nav_view(self, obs_id: int, *, phrase: str = "") -> dict[str, Any]:
        """When capture does not advance, still score the current view once for the planner."""
        oid = int(obs_id)
        # Allow verify despite REQUIRES_FRESH_VIEW — we intentionally revisit this station.
        self._fresh_obs_ids.add(oid)
        # Clear prior same-view skip so this stall path can record ABSENT/CANDIDATE.
        self._tried.pop(oid, None)
        scored = getattr(self._evidence_policy, "_globally_scored_obs_ids", None)
        if isinstance(scored, set):
            scored.discard(oid)
        return self.handle_tool(
            "verify_siglip",
            {
                "phrase": self._siglip_phrase(phrase),
                "obs_id": oid,
            },
        )

    def _obs_revision_snapshot(self, gm: Any) -> dict[int, int]:
        """Safe obs_id→revision map (ignores MagicMock / non-int backends)."""
        out: dict[int, int] = {}
        if gm is None:
            return out
        fn = getattr(gm, "obs_revision", None)
        if not callable(fn):
            return out
        tracked: set[int] = set()
        before = self._latest_obs_id()
        if before is not None:
            tracked.add(int(before))
        for h in self._hypotheses[:5]:
            tracked.add(int(h.obs_id))
        for oid in tracked:
            try:
                out[int(oid)] = int(fn(int(oid)))
            except (TypeError, ValueError):
                continue
        return out

    def _obs_revisions_advanced(self, gm: Any, before_revs: dict[int, int]) -> list[int]:
        if gm is None or not before_revs:
            return []
        fn = getattr(gm, "obs_revision", None)
        if not callable(fn):
            return []
        advanced: list[int] = []
        check_ids = set(before_revs)
        last_u = getattr(gm, "_last_obs_content_update_id", None)
        if isinstance(last_u, int):
            check_ids.add(int(last_u))
        for oid in check_ids:
            try:
                cur = int(fn(int(oid)))
            except (TypeError, ValueError):
                continue
            if cur > int(before_revs.get(int(oid), 0)):
                advanced.append(int(oid))
        return advanced

    def _tool_capture_and_update(self) -> dict[str, Any]:
        before = self._latest_obs_id()
        gm = self.graph_memory
        before_revs = self._obs_revision_snapshot(gm)
        agent = self.agent
        if hasattr(agent, "update"):
            try:
                agent.update()
            except Exception as e:
                _logger.warning(f"capture_and_update agent.update failed: {e}")
        # Always refresh graph-side confirmed memory after a voxel update when enabled.
        if gm is not None and getattr(gm, "memory_summary_enabled", False):
            if hasattr(gm, "refresh_siglip_confirmed_memory"):
                gm.refresh_siglip_confirmed_memory()
        fresh = self._latest_obs_id()
        refreshed_ids = self._obs_revisions_advanced(gm, before_revs)

        # New observation id — full advance.
        if fresh is not None and (before is None or int(fresh) != int(before)):
            self._fresh_obs_ids.add(int(fresh))
            if self.mode == "answer" and before is not None:
                try:
                    self._refresh_hypotheses_from_graph()
                except Exception as exc:
                    _logger.warning(f"hypothesis refresh after capture failed: {exc}")
            self._last_capture_status = "OK"
            self._append_trace({"tool": "capture_and_update", "ok": True, "obs_id": fresh})
            return {"ok": True, "obs_id": fresh, "status": "NEW_OBS"}

        # Same obs_id but candidate RGB/evidence refreshed via spatial merge.
        if refreshed_ids:
            use_id = int(refreshed_ids[0])
            if fresh is not None and int(fresh) in refreshed_ids:
                use_id = int(fresh)
            self._fresh_obs_ids.add(use_id)
            # Allow re-verify: old ABSENT on this id is stale once RGB changed.
            self._tried.pop(use_id, None)
            scored = getattr(self._evidence_policy, "_globally_scored_obs_ids", None)
            if isinstance(scored, set):
                scored.discard(use_id)
            if self.mode == "answer":
                try:
                    self._refresh_hypotheses_from_graph()
                except Exception as exc:
                    _logger.warning(f"hypothesis refresh after content refresh failed: {exc}")
            self._last_capture_status = "CONTENT_REFRESHED"
            self._append_trace(
                {
                    "tool": "capture_and_update",
                    "ok": True,
                    "obs_id": use_id,
                    "status": "CONTENT_REFRESHED",
                    "refreshed_obs_ids": refreshed_ids,
                }
            )
            return {
                "ok": True,
                "obs_id": use_id,
                "status": "CONTENT_REFRESHED",
                "refreshed_obs_ids": refreshed_ids,
            }

        # Reject non-advancing captures (same obs_id, no candidate refresh).
        if fresh is not None and before is not None and int(fresh) == int(before):
            self._last_capture_status = "NO_NEW_OBS"
            self._append_trace(
                {
                    "tool": "capture_and_update",
                    "ok": False,
                    "obs_id": fresh,
                    "prior_obs_id": before,
                    "status": "NO_NEW_OBS",
                }
            )
            return {
                "ok": False,
                "error": "capture did not advance observation — move before re-capturing",
                "obs_id": fresh,
                "status": "NO_NEW_OBS",
            }
        self._last_capture_status = "NO_OBS"
        self._append_trace({"tool": "capture_and_update", "ok": True, "obs_id": fresh})
        return {"ok": True, "obs_id": fresh}

    def _refresh_hypotheses_from_graph(self) -> None:
        """Re-retrieve nav evidence cards after voxel/graph grew — no VLM extract."""
        gm = self.graph_memory
        if gm is None or not hasattr(gm, "hypothesize_nav_targets"):
            return
        try:
            hypotheses = gm.hypothesize_nav_targets(
                self.query_text,
                max_k=env_eqa_hyp_recall_k(),
                robot_xyt=self._robot_xyt(),
            )
        except TypeError:
            hypotheses = gm.hypothesize_nav_targets(
                self.query_text, max_k=env_eqa_hyp_recall_k()
            )
        self._set_hypotheses(hypotheses)

    def _set_hypotheses(self, hypotheses: list[NavHypothesis]) -> None:
        """Install recalled hyps: drop visited frontiers; prefer untried in order."""
        filtered: list[NavHypothesis] = []
        for h in hypotheses:
            oid = int(h.obs_id)
            if str(h.source) == "frontier" and (
                int(self._nav_to_obs_counts.get(oid, 0)) >= 1
                or self._hypothesis_nav_blocked(oid)
            ):
                continue
            filtered.append(h)
        # Anti-echo: untried / low visits first, then tried graph/siglip for context.
        untried: list[NavHypothesis] = []
        tried: list[NavHypothesis] = []
        for h in filtered:
            oid = int(h.obs_id)
            if self._hypothesis_nav_blocked(oid) or int(
                self._nav_to_obs_counts.get(oid, 0)
            ) >= 1:
                if str(h.source) != "frontier":
                    tried.append(h)
            else:
                untried.append(h)
        packed = untried + tried
        self._hypotheses = packed
        self._hyp_i = 0
        _SOURCE_PRIOR = {
            "graph": 0.55,
            "confirmed": 0.5,
            "siglip": 0.4,
            "frontier": 0.2,
        }
        for h in self._hypotheses:
            self._evidence_policy.register_hypothesis(
                f"{h.source}:{int(h.obs_id)}",
                h.phrase,
                prior_probability=_SOURCE_PRIOR.get(str(h.source), 0.3),
            )

    def _latest_obs_id(self) -> int | None:
        """Newest non-frontier observation id (the frame just captured), if any."""
        gm = self.graph_memory
        observations = list(getattr(gm, "_observations", None) or [])
        for obs in reversed(observations):
            oid = int(obs.obs_id)
            usable = getattr(gm, "_obs_usable_for_eqa_image", None)
            if usable is not None and not usable(oid):
                continue
            return oid
        return None

    def _obs_already_verified(self, obs_id: int) -> bool:
        """True when this obs was already verified — do not score it again."""
        return int(obs_id) in self._tried or int(obs_id) in self._evidence_policy.scored_obs_ids

    def _begin_policy_approach(self, source: str, obs_id: int, phrase: str) -> str:
        if self._evidence_policy.state == AgenticState.REPLAN:
            self._evidence_policy.replan()
            self._verified = False
            self._verified_obs_id = None
        hypothesis_id = f"{source}:{int(obs_id)}"
        self._evidence_policy.register_hypothesis(hypothesis_id, phrase)
        if self._evidence_policy.state == AgenticState.SEARCH:
            self._evidence_policy.choose(hypothesis_id)
        return hypothesis_id

    def _policy_approached(self, hypothesis_id: str, fresh_obs_id: int) -> None:
        if self._evidence_policy.active_hypothesis_id != hypothesis_id:
            return
        try:
            self._evidence_policy.approached(int(fresh_obs_id))
        except (RuntimeError, ValueError) as exc:
            _logger.warning(f"evidence-policy approach rejected: {exc}")

    def _next_untried_hypothesis(self) -> NavHypothesis | None:
        while self._hyp_i < len(self._hypotheses):
            h = self._hypotheses[self._hyp_i]
            self._hyp_i += 1
            if not self._obs_already_verified(int(h.obs_id)):
                return h
        return None

    def _hypothesis_nav_blocked(self, obs_id: int) -> bool:
        """True if navigate_to_obs would refuse this id (loop / stall / already verified)."""
        oid = int(obs_id)
        if self._obs_already_verified(oid):
            return True
        if int(self._nav_to_obs_counts.get(oid, 0)) >= NAV_SAME_OBS_LOOP_LIMIT:
            return True
        tried = str(self._tried.get(oid) or "")
        return tried.startswith("STALLED_NAV_LOOP")

    def _retire_visited_frontier(
        self,
        *,
        frontier_obs_id: int | None = None,
        frontier_xyz: Any = None,
    ) -> None:
        """Visited frontiers are not frontiers — drop them from the graph."""
        gm = self.graph_memory
        if gm is None:
            return
        if frontier_obs_id is not None and hasattr(gm, "retire_frontier_obs"):
            try:
                gm.retire_frontier_obs(int(frontier_obs_id))
            except Exception as e:
                _logger.warning(f"retire_frontier_obs({frontier_obs_id}) failed: {e}")
        if frontier_xyz is not None and hasattr(gm, "retire_frontier_near_xy"):
            try:
                gm.retire_frontier_near_xy(frontier_xyz, radius_m=1.25)
            except Exception as e:
                _logger.warning(f"retire_frontier_near_xy failed: {e}")
        # Mirror voxel mask → graph so remaining clusters stay accurate.
        agent = self.agent
        vm = getattr(agent, "voxel_map", None)
        planner = getattr(agent, "planner", None) or getattr(agent, "_planner", None)
        xyt = self._robot_xyt()
        if vm is not None and planner is not None and xyt is not None:
            try:
                from emet.memory.graph_eqa.dynamem_graph_hooks import sync_graph_frontier_nodes

                sync_graph_frontier_nodes(
                    graph_memory=gm,
                    voxel_map=vm,
                    planner=planner,
                    base_xyt=xyt,
                    question=self.query_text,
                )
            except Exception as e:
                _logger.warning(f"sync_graph_frontier_nodes after visit failed: {e}")

    def _dense_max_sim_for_rgb(self, phrase: str, rgb: np.ndarray | None) -> float | None:
        """Max patch-token SigLIP cosine for *phrase* vs *rgb* (MaskSigLIP space).

        Full-frame pool cosines rarely exceed ~0.12; dense max is closer to DynaMem's
        per-point features and can reach the 0.21 PRESENT bar when the object is in view.
        """
        text = (phrase or "").strip()
        if not text or rgb is None:
            return None
        enc = None
        gm = self.graph_memory
        if gm is not None:
            enc = getattr(gm, "_confirmed_memory_siglip_encoder", None)
        if enc is None:
            try:
                from emet.perception.encoders.siglip_encoder import get_shared_mask_siglip_encoder

                enc = get_shared_mask_siglip_encoder()
            except Exception:
                return None
        try:
            import torch
            import torch.nn.functional as F

            text_t = enc.encode_text(text).detach().float().reshape(-1)
            text_t = text_t / (text_t.norm() + 1e-12)
            inputs = enc._to_model_inputs(enc.processor(images=np.asarray(rgb, dtype=np.uint8), return_tensors="pt"))
            with torch.no_grad():
                out = enc.model.vision_model(inputs["pixel_values"], output_hidden_states=True)
                feat = F.normalize(out.last_hidden_state.float(), dim=-1)
                sims = feat @ text_t.to(device=feat.device, dtype=feat.dtype).reshape(-1, 1)
                return float(sims.max().item())
        except Exception as e:
            _logger.warning(f"dense_max_sim_for_rgb failed: {e}")
            return None

    def _voxel_max_sim_for_obs(self, phrase: str, obs_id: int) -> tuple[float, str] | None:
        """Max per-point SigLIP cosine for *phrase* on voxel features in *obs_id*.

        Full-frame ``encode_image`` cosines top out ~0.10–0.16 in Habitat, while DynaMem's
        0.21 / 0.28 thresholds were calibrated on dense voxel point features. Prefer those
        when the voxel map is available so PRESENT can actually fire.
        """
        text = (phrase or "").strip()
        if not text:
            return None
        vm = getattr(self.agent, "voxel_map", None)
        if vm is None or not hasattr(vm, "find_alignment_over_model"):
            return None
        try:
            alignments = vm.find_alignment_over_model(text)
        except Exception:
            return None
        if alignments is None:
            return None
        try:
            import torch

            if not isinstance(alignments, torch.Tensor):
                return None
        except ImportError:
            return None
        a = alignments.detach().cpu().float().reshape(-1)
        if a.numel() == 0:
            return None
        sm = getattr(vm, "semantic_memory", None)
        counts = getattr(sm, "_obs_counts", None) if sm is not None else None
        channel = "voxel_global"
        if counts is not None:
            try:
                import torch

                c = counts.detach().cpu().long().reshape(-1)
                if c.numel() == a.numel():
                    mask = c == int(obs_id)
                    if bool(mask.any()):
                        a = a[mask]
                        channel = "voxel_obs"
            except Exception:
                pass
        return float(a.max().item()), channel

    def _detector_for_verify(self) -> Any | None:
        if self._presence_detector_initialized:
            return self._presence_detector
        self._presence_detector_initialized = True
        backend = os.environ.get("EMET_EQA_AGENTIC_VERIFIER", "").strip().lower()
        if backend in ("", "none", "siglip"):
            return None
        try:
            from emet.eval.presence_verifiers import (
                OwlV2PresenceDetector,
                YoloEPresenceDetector,
            )

            if backend == "owlv2":
                self._presence_detector = OwlV2PresenceDetector()
            elif backend == "yoloe":
                self._presence_detector = YoloEPresenceDetector()
            else:
                raise ValueError(f"unsupported EMET_EQA_AGENTIC_VERIFIER={backend!r}")
        except Exception as exc:
            _logger.warning(f"presence detector unavailable: {exc}")
            self._presence_detector = None
        return self._presence_detector

    def _graph_label_matches(self, phrase: str, obs_id: int) -> bool:
        gm = self.graph_memory
        stored = gm._observation_by_id(int(obs_id)) if gm is not None and hasattr(gm, "_observation_by_id") else None
        labels = getattr(stored, "labels", None) or []
        return any(label_matches_relevant_object(phrase, str(label)) for label in labels)

    def _relation_sufficient_for_obs(self, obs_id: int) -> bool:
        from emet.habitat.metrics import (
            choices_are_count_mcq,
            choices_are_location_mcq,
            parse_mcq_choices_from_question,
        )

        choices = parse_mcq_choices_from_question(self.question)
        if not choices:
            choices = list(getattr(self, "_mcq_choices", None) or [])
        if not choices:
            return True
        # Cardinality MCQs: fused target presence is enough; the answer VLM counts.
        if choices_are_count_mcq(choices):
            return True
        if not choices_are_location_mcq(choices):
            return True
        gm = self.graph_memory
        stored = (
            gm._observation_by_id(int(obs_id))
            if gm is not None and hasattr(gm, "_observation_by_id")
            else None
        )
        labels = [str(label) for label in (getattr(stored, "labels", None) or [])]
        if any(
            label_matches_relevant_object(choice, label)
            for choice in choices
            for label in labels
        ):
            return True
        # Landmark overlap with nearby graph nodes (room/fixture context).
        if gm is not None and hasattr(gm, "labels_near_obs"):
            try:
                near_labels = [str(x) for x in (gm.labels_near_obs(int(obs_id)) or [])]
            except Exception:
                near_labels = []
            if any(
                label_matches_relevant_object(choice, label)
                for choice in choices
                for label in near_labels
            ):
                return True
        return False

    def _inventory_labels(self, *, limit: int = 12) -> list[str]:
        gm = self.graph_memory
        labels: list[str] = []
        seen: set[str] = set()
        for obs in list(getattr(gm, "_observations", None) or []):
            for lab in list(getattr(obs, "labels", None) or []):
                s = str(lab).strip()
                if not s or s.lower() in seen:
                    continue
                seen.add(s.lower())
                labels.append(s)
                if len(labels) >= limit:
                    return labels
        return labels

    def _extract_vlm_target(self) -> None:
        """Text-only VLM: pick the seek/verify phrase once per episode."""
        if self.mode != "answer" or not self.question:
            return
        gm = self.graph_memory
        client = getattr(gm, "eqa_client", None) if gm is not None else None
        from emet.memory.graph_eqa.graph_memory import heuristic_relevant_phrases

        phrases = heuristic_relevant_phrases(self.question)
        fallback = phrases[0] if phrases else question_stem_for_keywords(self.question)
        if client is None:
            self._target_phrase = (fallback or self.question).strip()
            self._question_type = "other"
            self._append_trace(
                {
                    "event": "vlm_target_extract",
                    "target_phrase": self._target_phrase,
                    "question_type": self._question_type,
                    "source": "heuristic",
                }
            )
            return
        from emet.eval.agentic_vlm_assess import extract_target_from_question

        te = extract_target_from_question(
            client, self.question, fallback_phrase=str(fallback or "")
        )
        self._target_phrase = te.target_phrase
        self._question_type = te.question_type
        self._append_trace({"event": "vlm_target_extract", "source": "vlm", **te.to_dict()})

    def _escape_min_travel_m(self) -> float:
        """Distance the next frontier must clear once the target keeps not showing up."""
        if self._not_present_streak < NOT_PRESENT_ESCAPE_STREAK:
            return 0.0
        return ESCAPE_MIN_TRAVEL_M

    def _update_escape_streak(self, *, present: bool) -> None:
        """Track consecutive not-visible views and publish the escape floor to the picker."""
        if present:
            self._not_present_streak = 0
        else:
            self._not_present_streak += 1
        self.agent._explore_min_travel_m = self._escape_min_travel_m()

    def _frontier_pick_out_dir(self) -> Path:
        """Directory for numbered pick panels (episode bundle when available)."""
        if getattr(self, "_frontier_pick_dir", None):
            out = Path(self._frontier_pick_dir)
            out.mkdir(parents=True, exist_ok=True)
            return out
        ep = getattr(self.agent, "_episode_debug_dir", None) or os.environ.get("EMET_EQA_EPISODE_DIR")
        if ep:
            out = Path(str(ep)).expanduser() / "frontier_picks"
        elif self._trace_path is not None:
            out = self._trace_path.parent / "frontier_picks"
        else:
            out = Path.home() / ".cache" / "habitat_eqa" / "frontier_picks"
        out.mkdir(parents=True, exist_ok=True)
        self._frontier_pick_dir = out
        self.agent._frontier_pick_dir = str(out)
        return out

    def _save_frontier_pick_panel(
        self,
        frontier_xyz: Any,
        *,
        robot_xyt_before: np.ndarray | None = None,
    ) -> Path | None:
        """Write a numbered frontier-pick panel into the episode bundle (best-effort)."""
        if frontier_xyz is None:
            return None
        try:
            arr = np.asarray(frontier_xyz, dtype=float).reshape(-1)
            if arr.size < 2:
                return None
            pick = (float(arr[0]), float(arr[1]))
            self._frontier_pick_waypoints.append(pick)

            voxel_map = getattr(self.agent, "voxel_map", None)
            if voxel_map is None or not hasattr(voxel_map, "get_2d_map"):
                return None
            obstacles, explored = voxel_map.get_2d_map()
            go = getattr(voxel_map, "grid_origin", np.array([0.0, 0.0]))
            if hasattr(go, "detach"):
                go = go.detach().cpu().numpy()
            go = np.asarray(go, dtype=np.float64).reshape(-1)[:2]
            res = float(getattr(voxel_map, "grid_resolution", 0.1) or 0.1)

            robot_xy = None
            if robot_xyt_before is not None:
                r = np.asarray(robot_xyt_before, dtype=float).reshape(-1)
                if r.size >= 2:
                    robot_xy = (float(r[0]), float(r[1]))

            from emet.visualization.frontier_pick_viz import (
                frontier_mask_from_explored,
                render_frontier_pick_rgb,
                save_frontier_pick_rgb,
            )

            n = len(self._frontier_pick_waypoints)
            dist_m = 0.0
            if robot_xy is not None:
                dist_m = float(np.hypot(pick[0] - robot_xy[0], pick[1] - robot_xy[1]))
            title = f"iteration {n - 1} — pick {dist_m:.1f} m ahead ({n} waypoints)"
            rgb = render_frontier_pick_rgb(
                obstacles,
                explored,
                frontier=frontier_mask_from_explored(explored, obstacles),
                robot_xy=robot_xy,
                chosen_xy=pick,
                waypoints=list(self._frontier_pick_waypoints),
                grid_origin_xy=go,
                grid_resolution=res,
                title=title,
            )
            out_dir = self._frontier_pick_out_dir()
            path = save_frontier_pick_rgb(rgb, out_dir / f"iter_{n - 1:02d}.png")
            paths = list(getattr(self.agent, "_frontier_pick_panels", []) or [])
            paths.append(str(path))
            self.agent._frontier_pick_panels = paths
            return path
        except Exception as e:
            _logger.warning(f"frontier pick panel failed: {e}")
            return None

    def _run_vlm_view_assess(
        self,
        *,
        rgb: np.ndarray | None,
        phrase: str,
        obs_id: int,
        proposal: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Multimodal answerability gate. One assess per obs_id.

        Qwen looks at pixels + inventory and decides whether this view is enough.
        SigLIP/OWL stay proposals in the inventory — they do not hard-block unlock.
        """
        oid = int(obs_id)
        if oid in self._vlm_assessed_obs_ids:
            return {
                "ok": False,
                "status": "SKIPPED_SAME_VIEW",
                "obs_id": oid,
                "error": f"obs_id {oid} already VLM-assessed",
            }
        gm = self.graph_memory
        client = getattr(gm, "eqa_client", None) if gm is not None else None
        if client is None:
            self._append_trace(
                {
                    "tool": "vlm_assess",
                    "obs_id": oid,
                    "ok": False,
                    "error": "no eqa_client",
                    "answerable": False,
                }
            )
            return {"ok": False, "error": "no eqa_client", "answerable": False, "obs_id": oid}

        from emet.eval.agentic_vlm_assess import assess_view_with_vlm, build_inventory_brief

        inventory = build_inventory_brief(
            n_observations=len(list(getattr(gm, "_observations", None) or [])),
            graph_labels=self._inventory_labels(),
            proposal=proposal,
            tried_obs_ids=sorted(self._tried.keys()),
            n_rounds=self._round,
            n_nav=self._n_nav + self._n_explore,
        )
        assessment = assess_view_with_vlm(
            client,
            question=self.question,
            rgb=rgb,
            inventory=inventory,
            target_phrase=self._target_phrase or phrase,
        )
        self._vlm_assessed_obs_ids.add(oid)
        # Trust the VLM assess. SigLIP ABSENT/CANDIDATE is router context only —
        # do not second-guess answerable with cheap-detector gates (fix2 q65/q104).
        proposal_status = str(
            (proposal or {}).get("decision")
            or getattr(self._last_verify, "status", "")
            or ""
        ).upper()
        vlm_assessment = None
        try:
            vlm_assessment = self._evidence_policy.apply_vlm_assessment(
                present=assessment.present,
                answerable=assessment.answerable,
                need_more_views=assessment.need_more_views,
            )
        except (RuntimeError, ValueError) as exc:
            _logger.warning(f"evidence-policy VLM assess rejected: {exc}")
        if vlm_assessment is not None and vlm_assessment.answerable:
            self._verified = True
            self._verified_obs_id = oid
        self._update_escape_streak(present=assessment.present)
        payload = {
            "tool": "vlm_assess",
            "obs_id": oid,
            "phrase": phrase,
            "target": assessment.target,
            "present": assessment.present,
            "answerable": assessment.answerable,
            "need_more_views": assessment.need_more_views,
            "suggested_answer": assessment.suggested_answer,
            "reason": assessment.reason,
            "policy_state": str(self._evidence_policy.state),
            "verified": self._verified,
            "proposal_status": proposal_status or None,
            "not_present_streak": self._not_present_streak,
            "explore_min_travel_m": self._escape_min_travel_m(),
            "inventory": inventory,
        }
        self._last_vlm_assess = payload
        self._append_trace(payload)
        return {
            "ok": True,
            "obs_id": oid,
            "present": assessment.present,
            "answerable": assessment.answerable,
            "need_more_views": assessment.need_more_views,
            "suggested_answer": assessment.suggested_answer,
            "verified": self._verified,
            "policy_state": str(self._evidence_policy.state),
        }

    def _tool_verify_siglip(self, phrase: str, obs_id: int | None) -> dict[str, Any]:
        gm = self.graph_memory
        if gm is None:
            return {"ok": False, "error": "no graph_memory"}
        text = (phrase or "").strip()
        if not text:
            if self._target_phrase:
                text = self._target_phrase
            else:
                phrases = list(getattr(gm, "_relevant_phrases", None) or []) + list(
                    getattr(gm, "_relevant_objects", None) or []
                )
                # Prefer phrases from the question stem over MCQ-option nouns
                # (``fruit bowl`` > ``kitchen island``), then noun compounds over
                # leading verb fillers (``fruit bowl`` > ``looking``).
                stem = question_stem_for_keywords(self.question).lower()
                ranked = sorted(
                    phrases,
                    key=lambda p: (
                        1 if (p or "").strip().lower() in stem else 0,
                        0
                        if (p or "").split()[:1]
                        and (p or "").split()[0].lower() in _QUESTION_VERB_FILLERS
                        else 1,
                        len((p or "").split()),
                        len(p or ""),
                    ),
                    reverse=True,
                )
                text = ranked[0] if ranked else self.question
        oid = obs_id
        if oid is None or int(oid) < 0:
            # No obs_id means "verify what the robot is looking at now". Motion tools call
            # this right after capture_and_update, so the newest observation is the frame
            # just taken; falling back to a hypothesis re-verified the same stale obs every
            # round while the robot explored the far side of the scene (q104/q105).
            oid = self._latest_obs_id()
        if oid is None:
            if self._hypotheses:
                oid = int(self._hypotheses[min(self._hyp_i, len(self._hypotheses) - 1)].obs_id)
            elif getattr(gm, "last_eqa_obs_ids", None):
                oid = int(gm.last_eqa_obs_ids[0])
            else:
                return {"ok": False, "error": "no obs_id"}
        oid = int(oid)
        if (
            self._router_enabled
            and self._evidence_policy.state != AgenticState.VERIFY
            and oid not in self._fresh_obs_ids
        ):
            return {
                "ok": False,
                "error": (
                    f"obs_id {oid} is stale; SEARCH must APPROACH/capture a fresh view "
                    "before VERIFY"
                ),
                "status": "REQUIRES_FRESH_VIEW",
                "obs_id": oid,
                "verified": self._verified,
            }
        # Interactive rule: one verify per view. Re-checking the same obs burns rounds
        # without new evidence — move first (nav / explore), then verify the fresh frame.
        if self._obs_already_verified(oid):
            self._append_trace(
                {
                    "tool": "verify_siglip",
                    "phrase": text,
                    "obs_id": oid,
                    "decision": "SKIPPED_SAME_VIEW",
                    "sim": 0.0,
                    "prior": self._tried.get(oid),
                }
            )
            return {
                "ok": False,
                "error": f"obs_id {oid} already verified ({self._tried.get(oid)}); navigate or explore for a new view",
                "status": "SKIPPED_SAME_VIEW",
                "obs_id": oid,
                "verified": self._verified,
            }
        rgb = None
        live_obs = None
        robot = getattr(self.agent, "robot", None)
        if robot is not None and hasattr(robot, "get_observation"):
            try:
                live_obs = robot.get_observation()
                if live_obs is not None and getattr(live_obs, "rgb", None) is not None:
                    rgb = np.asarray(live_obs.rgb)
            except Exception:
                pass
        if rgb is None:
            stored = gm._observation_by_id(int(oid)) if hasattr(gm, "_observation_by_id") else None
            stored_rgb = getattr(stored, "rgb", None) if stored is not None else None
            if isinstance(stored_rgb, np.ndarray) and stored_rgb.ndim == 3:
                rgb = np.asarray(stored_rgb)
        result = gm.verify_phrase_at_obs(text, int(oid), rgb=rgb, min_sim=self.verify_min_sim)
        full_frame_sim = float(result.sim)
        voxel = self._voxel_max_sim_for_obs(text, int(oid))
        voxel_sim = float(voxel[0]) if voxel is not None else None
        voxel_ch = voxel[1] if voxel is not None else None
        dense_sim = self._dense_max_sim_for_rgb(text, rgb)
        detector_evidence = None
        detector = self._detector_for_verify()
        if detector is not None and rgb is not None:
            try:
                from emet.eval.presence_verifiers import detector_crop_evidence

                enc = getattr(gm, "_confirmed_memory_siglip_encoder", None)
                detector_evidence = (
                    detector_crop_evidence(detector, enc, rgb, text)
                    if enc is not None
                    else detector.score(rgb, text)
                )
            except Exception as exc:
                _logger.warning(f"hybrid presence verify failed: {exc}")
        # Best image-space score (full-frame pool vs dense patch).
        best_img = full_frame_sim
        verify_channel = "full_frame"
        if dense_sim is not None and float(dense_sim) > best_img:
            best_img = float(dense_sim)
            verify_channel = "dense_patch"
        # Voxel wins when it clears the DynaMem bar; else use image three-band.
        if voxel_sim is not None and voxel_sim >= SIGLIP_PRESENT_THRESHOLD:
            status, ok, sim_out, verify_channel = "PRESENT", True, float(voxel_sim), str(voxel_ch)
        elif best_img >= SIGLIP_IMAGE_PRESENT_THRESHOLD:
            status, ok, sim_out = "PRESENT", True, float(best_img)
        elif best_img >= SIGLIP_IMAGE_ABSENT_THRESHOLD:
            status, ok, sim_out = "CANDIDATE", False, float(best_img)
        else:
            status, ok, sim_out = "ABSENT", False, float(best_img)
        result = VerifyResult(
            status=status,
            sim=float(sim_out),
            obs_id=int(oid),
            phrase=text,
            ok=ok,
            text_feat=result.text_feat,
            img_feat=result.img_feat,
        )
        self._last_verify = result
        self._tried[int(result.obs_id)] = f"verify {result.status} sim={float(result.sim):.2f}"
        self._fresh_obs_ids.discard(int(result.obs_id))
        # After motion, capture may not mint a new obs_id (unit mocks / no map growth).
        # Complete APPROACH → VERIFY on the view we are scoring; otherwise open a view hyp.
        if self._evidence_policy.state == AgenticState.APPROACH:
            active = self._evidence_policy.active_hypothesis_id
            if active is not None:
                self._policy_approached(active, int(result.obs_id))
        if self._evidence_policy.state != AgenticState.VERIFY:
            hypothesis_id = self._begin_policy_approach("view", int(result.obs_id), text)
            self._policy_approached(hypothesis_id, int(result.obs_id))
        hypothesis_id = self._evidence_policy.active_hypothesis_id or f"view:{int(result.obs_id)}"
        graph_label_match = self._graph_label_matches(text, int(result.obs_id))
        assessment = None
        try:
            evidence = EvidenceRecord(
                hypothesis_id=hypothesis_id,
                obs_id=int(result.obs_id),
                phrase=text,
                full_frame_sim=full_frame_sim,
                dense_sim=dense_sim,
                voxel_sim=voxel_sim,
                detector_score=(
                    float(detector_evidence.score) if detector_evidence is not None else None
                ),
                crop_siglip_sim=(
                    detector_evidence.crop_siglip_sim if detector_evidence is not None else None
                ),
                graph_label_match=graph_label_match,
                detector_backend=(
                    str(detector_evidence.backend) if detector_evidence is not None else None
                ),
                bbox_xyxy=(
                    detector_evidence.bbox_xyxy if detector_evidence is not None else None
                ),
                provenance=tuple(
                    channel
                    for channel, value in (
                        ("full_frame", full_frame_sim),
                        ("dense_patch", dense_sim),
                        (str(voxel_ch or "voxel"), voxel_sim),
                        (
                            str(detector_evidence.backend)
                            if detector_evidence is not None
                            else "detector",
                            detector_evidence.score if detector_evidence is not None else None,
                        ),
                    )
                    if value is not None
                ),
            )
            self._evidence_policy.add_evidence(evidence)
            assessment = self._evidence_policy.assess(
                relation_sufficient=self._relation_sufficient_for_obs(int(result.obs_id))
            )
        except (RuntimeError, ValueError) as exc:
            _logger.warning(f"evidence-policy verify rejected: {exc}")
        # Cheap channels never unlock submit — multimodal VLM assess is the gate.
        proposal = {
            "phrase": text,
            "detector_score": (
                float(detector_evidence.score) if detector_evidence is not None else None
            ),
            "dense_sim": dense_sim,
            "full_frame_sim": full_frame_sim,
            "decision": result.status,
            "obs_id": int(result.obs_id),
        }
        vlm_out = self._run_vlm_view_assess(
            rgb=rgb,
            phrase=text,
            obs_id=int(result.obs_id),
            proposal=proposal,
        )
        row = {
            "tool": "verify_siglip",
            "phrase": text,
            "obs_id": int(result.obs_id),
            "sim": float(result.sim),
            "decision": result.status,
            "verify_channel": verify_channel,
            "full_frame_sim": full_frame_sim,
            "voxel_sim": voxel_sim,
            "dense_sim": dense_sim,
            "detector_backend": (
                detector_evidence.backend if detector_evidence is not None else None
            ),
            "detector_score": (
                float(detector_evidence.score) if detector_evidence is not None else None
            ),
            "detector_bbox_xyxy": (
                list(detector_evidence.bbox_xyxy)
                if detector_evidence is not None and detector_evidence.bbox_xyxy is not None
                else None
            ),
            "crop_siglip_sim": (
                detector_evidence.crop_siglip_sim if detector_evidence is not None else None
            ),
            "graph_label_match": graph_label_match,
            "policy_state": self._evidence_policy.state,
            "presence_probability": (
                assessment.presence_probability if assessment is not None else None
            ),
            "answerability_probability": (
                self._evidence_policy.beliefs[
                    self._evidence_policy.active_hypothesis_id
                ].answerability_probability
                if self._evidence_policy.active_hypothesis_id
                and self._evidence_policy.active_hypothesis_id in self._evidence_policy.beliefs
                else (assessment.answerability_probability if assessment is not None else None)
            ),
            "positive_channels": (
                list(assessment.positive_channels) if assessment is not None else []
            ),
            "contradiction_channels": (
                list(assessment.contradiction_channels) if assessment is not None else []
            ),
            # Submit unlock (VLM answerable on a non-ABSENT view), not cheap-channel alone.
            "fused_verified": bool(self._verified),
            "answerable": bool(self._evidence_policy.state == AgenticState.ANSWER),
            "vlm_assess": vlm_out,
            "present_bar": (
                SIGLIP_PRESENT_THRESHOLD
                if verify_channel.startswith("voxel")
                else SIGLIP_IMAGE_PRESENT_THRESHOLD
            ),
            "absent_bar": None if verify_channel.startswith("voxel") else SIGLIP_IMAGE_ABSENT_THRESHOLD,
            "text_feat": _feat_list(result.text_feat),
            "img_feat": _feat_list(result.img_feat),
        }
        labeler = getattr(robot, "hm3d_semantic_labeler", None) if robot is not None else None
        semantic = getattr(live_obs, "semantic", None) if live_obs is not None else None
        if labeler is not None and semantic is not None and hasattr(labeler, "visibility_for_phrase"):
            row.update(
                labeler.visibility_for_phrase(
                    semantic,
                    text,
                    getattr(live_obs, "depth", None),
                )
            )
        xyt = self._robot_xyt()
        if xyt is not None:
            row["xyt"] = [float(x) for x in xyt.reshape(-1)[:3]]
        hyp = next((h for h in self._hypotheses if int(h.obs_id) == int(result.obs_id)), None)
        if hyp is not None:
            row["target_xyz"] = [float(x) for x in np.asarray(hyp.xyz).reshape(-1)[:3]]
            row["source"] = hyp.source
            self._attach_gt(row, hyp.xyz)
        else:
            row["source"] = "current_view"
        self._append_trace(row)
        return {
            "ok": True,
            "status": result.status,
            "sim": float(result.sim),
            "verified": self._verified,
            "obs_id": int(result.obs_id),
            "verify_channel": verify_channel,
            "fused_verified": bool(self._verified),
            "answerable": bool(self._evidence_policy.state == AgenticState.ANSWER),
            "presence_probability": (
                assessment.presence_probability if assessment is not None else None
            ),
            "answerability_probability": (
                self._evidence_policy.beliefs[
                    self._evidence_policy.active_hypothesis_id
                ].answerability_probability
                if self._evidence_policy.active_hypothesis_id
                and self._evidence_policy.active_hypothesis_id in self._evidence_policy.beliefs
                else (assessment.answerability_probability if assessment is not None else None)
            ),
            "vlm_assess": vlm_out,
        }

    def _tool_submit_answer(self, answer: str) -> dict[str, Any]:
        if self.mode == "explore":
            return {"ok": False, "error": "submit_answer unavailable in explore mode — use finish"}
        nav_exhausted = self._n_nav + self._n_explore >= self.max_nav_steps
        if self._verified and self._evidence_policy.state != AgenticState.ANSWER:
            if nav_exhausted or self._round >= self.max_rounds - 1:
                return self._abstain_unverified(
                    reason="target evidence did not establish answer sufficiency"
                )
            return {
                "ok": False,
                "error": "target present but answer relation/count is unresolved — replan for a disambiguating view",
            }
        if self._require_verified and not self._verified:
            # Exhausted candidates → honest abstain (do not burn rounds on rejected submits).
            if nav_exhausted or self._round >= self.max_rounds - 1:
                return self._abstain_unverified()
            return {
                "ok": False,
                "error": "not verified — require_verified=1; VLM assess must mark answerable before submit_answer",
            }
        # Allow submit once nav budget is spent so EQA can emit Action:N for follow-up,
        # even if SigLIP never hit PRESENT (holdout q104/q105).
        if not self._verified and self._round < self.max_rounds - 1 and not nav_exhausted:
            return {
                "ok": False,
                "error": "not verified — call verify_siglip (+ VLM assess) or exhaust budget before submit_answer",
            }
        return self._do_submit_answer(prefer_answer=answer)

    def _frontier_count(self) -> int:
        gm = self.graph_memory
        try:
            nodes = gm.get_nodes() if gm is not None else []
            return sum(1 for n in nodes if getattr(n, "is_frontier", False))
        except Exception:
            return 0

    def _explore_done(self) -> bool:
        if self._n_nav + self._n_explore >= self.max_nav_steps:
            return True
        return (self._n_nav + self._n_explore) > 0 and self._frontier_count() == 0

    def _tool_finish(self, summary: str) -> dict[str, Any]:
        if self.mode != "explore":
            return {"ok": False, "error": "finish unavailable in answer mode — use submit_answer"}
        if not self._explore_done() and self._round < self.max_rounds - 1:
            return {
                "ok": False,
                "error": "exploration not finished — frontiers remain and nav budget left",
            }
        return self._do_finish(summary)

    def _do_finish(self, summary: str = "") -> dict[str, Any]:
        text = (summary or "").strip() or (
            f"Explored {self._n_explore + self._n_nav} waypoints; "
            f"{self._frontier_count()} frontiers remain."
        )
        self._append_trace(
            {
                "tool": "finish",
                "summary": text,
                "n_explore": self._n_explore,
                "n_nav": self._n_nav,
                "frontiers_left": self._frontier_count(),
            }
        )
        return {
            "ok": True,
            "answer": text,
            "discord_text": f"Explore finished: {text}",
            "confidence": True,
            "relevant_images": [],
        }

    def _abstain_unverified(
        self,
        reason: str = "require_verified and no fused verification before budget/rounds exhausted",
    ) -> dict[str, Any]:
        """Budget exhausted without PRESENT — do not guess a letter."""
        msg = f"Unknown (unverified — {reason})"
        self._append_trace(
            {
                "tool": "abstain_unverified",
                "reason": reason,
                "n_nav": self._n_nav,
                "n_explore": self._n_explore,
                "last_verify": (
                    None
                    if self._last_verify is None
                    else {
                        "status": self._last_verify.status,
                        "sim": float(self._last_verify.sim),
                        "obs_id": int(self._last_verify.obs_id),
                        "phrase": self._last_verify.phrase,
                    }
                ),
            }
        )
        return {
            "ok": True,
            "answer": "Unknown",
            "discord_text": f"Answer:{msg}",
            "confidence": False,
            "relevant_images": [],
            "verified": False,
        }

    @staticmethod
    def _looks_like_coordinate_dump(text: str) -> bool:
        """True for nearest-furniture XYZ dumps (``The fan is at approximately (x,y,z) m``)."""
        return bool(_COORD_DUMP_RE.search(text or ""))

    def _mcq_letter_from_text(self, text: str) -> str:
        """Extract a canonical A–D letter when the question is MCQ-shaped."""
        from emet.habitat.metrics import extract_mcq_letter, parse_mcq_choices_from_question

        raw = (text or "").strip()
        if not raw:
            return ""
        choices = parse_mcq_choices_from_question(self.question)
        letter = extract_mcq_letter(raw, choices or None)
        if letter:
            return letter
        if len(raw) == 1 and raw.upper() in "ABCDE":
            return raw.upper()
        return ""

    def _resolve_submit_answer_text(
        self,
        *,
        prefer_answer: str,
        query_answer: str,
    ) -> tuple[str, str]:
        """Keep Qwen's MCQ letter when ``query_answer`` collapses to graph XYZ prose.

        Dynagraph may put meter coordinates in the EQA prompt; small VLMs sometimes
        echo those instead of A–D. If the router or view-assess already gave a letter,
        do not let that echo overwrite it. Otherwise trust ``query_answer`` (also Qwen).
        """
        prefer = (prefer_answer or "").strip()
        qa = (query_answer or "").strip()
        suggested = ""
        if self._last_vlm_assess:
            suggested = str(self._last_vlm_assess.get("suggested_answer") or "").strip()

        prefer_letter = self._mcq_letter_from_text(prefer)
        if prefer_letter:
            return prefer_letter, "prefer"

        suggested_letter = self._mcq_letter_from_text(suggested)
        if suggested_letter and self._looks_like_coordinate_dump(qa):
            return suggested_letter, "vlm_suggested"

        if qa:
            return qa, "query"
        if prefer:
            return prefer, "prefer"
        if suggested:
            return suggested, "vlm_suggested"
        return "Unknown", "query"

    def _do_submit_answer(self, prefer_answer: str = "") -> dict[str, Any]:
        from emet.eval.dynagraph_vram import release_siglip_for_vlm

        agent = self.agent
        gm = self.graph_memory
        release_siglip_for_vlm(agent)
        discord_text = ""
        confidence = False
        relevant_images: list[Any] = []
        prefer = (prefer_answer or "").strip()
        query_ans = ""
        answer_source = "prefer"
        force_obs_ids: list[int] | None = None
        if gm is not None and hasattr(gm, "query_answer"):
            # Prefer verified observation as Image 1 (query_answer must honor force_obs_ids;
            # setting last_eqa_obs_ids alone was overwritten by diversified selection).
            if self._verified_obs_id is not None and hasattr(gm, "select_obs_ids_for_verified_answer"):
                force_obs_ids = gm.select_obs_ids_for_verified_answer(
                    self._verified_obs_id, max_images=1
                )
                gm.last_eqa_obs_ids = list(force_obs_ids)
            # Do not clamp EMET_EQA_ANSWER_MAX_NEW_TOKENS here. A prior setdefault("64")
            # truncated Caption/Reasoning mid-stream and forced [salvage] on every bal-32
            # agentic answer; graph_memory.query_answer defaults to 256.
            xyt = self._robot_xyt()
            planner = getattr(agent, "planner", None)
            try:
                (
                    _reasoning,
                    ans,
                    confidence,
                    _cr,
                    _tp,
                    relevant_images,
                ) = gm.query_answer(
                    self.question,
                    xyt,
                    planner,
                    force_obs_ids=force_obs_ids,
                )
                query_ans = (ans or "").strip()
            except Exception as e:
                discord_text = f"Answer:Unknown\nEQA failed: {e}"
                query_ans = "Unknown"
                confidence = False
            answer, answer_source = self._resolve_submit_answer_text(
                prefer_answer=prefer,
                query_answer=query_ans,
            )
            # Letter from VLM assess / tool arg is the decision we want to score; do not
            # inherit False confidence from a coordinate-dump query_answer path.
            if answer_source in ("prefer", "vlm_suggested") and self._mcq_letter_from_text(answer):
                confidence = bool(self._verified) or bool(confidence)
            discord_text = (
                f"Answer:{answer}\nConfidence:{confidence}"
                f"\n[submit_source:{answer_source}]"
            )
            if query_ans and query_ans != answer:
                discord_text += f"\n[query_answer:{query_ans[:160]}]"
        elif prefer:
            answer, answer_source = self._resolve_submit_answer_text(
                prefer_answer=prefer,
                query_answer="",
            )
            discord_text = f"Answer:{answer}\n[submit_source:{answer_source}]"
            confidence = bool(self._verified)
        else:
            discord_text = "Answer:Unknown\nNo graph memory"
            answer = "Unknown"
            answer_source = "query"
        self._append_trace(
            {
                "tool": "submit_answer",
                "final_answer": answer,
                "confidence": bool(confidence),
                "verified": self._verified,
                "verified_obs_id": self._verified_obs_id,
                "answer_source": answer_source,
                "query_answer": query_ans or None,
                "prefer_answer": prefer or None,
                "vlm_suggested": (
                    None
                    if self._last_vlm_assess is None
                    else self._last_vlm_assess.get("suggested_answer")
                ),
                "answerable": self._evidence_policy.state == AgenticState.ANSWER,
                "answerability_probability": (
                    self._evidence_policy.beliefs[
                        self._evidence_policy.active_hypothesis_id
                    ].answerability_probability
                    if self._evidence_policy.active_hypothesis_id
                    in self._evidence_policy.beliefs
                    else None
                ),
                "force_obs_ids": list(force_obs_ids) if force_obs_ids else None,
                "last_eqa_obs_ids": list(getattr(gm, "last_eqa_obs_ids", []) or [])
                if gm is not None
                else None,
                "spatial_rag": getattr(gm, "last_eqa_spatial_rag", None) if gm is not None else None,
            }
        )
        return {
            "ok": True,
            "answer": answer,
            "answer_source": answer_source,
            "discord_text": discord_text,
            "confidence": bool(confidence),
            "relevant_images": relevant_images,
        }

    @staticmethod
    def _answer_unknownish(answer: Any) -> bool:
        ans = str(answer or "").strip().lower()
        return (not ans) or ans in {"unknown", "none", "n/a", "na"} or "frontier" in ans

    def _finalize_unknown_location_letter(self, submit_out: dict[str, Any]) -> dict[str, Any]:
        """Last-chance VLM letter when Action:/explore is done and answer is still Unknown.

        Mid-episode we keep Unknown so the loop can follow ``Action:N`` / frontiers
        (memory invent caused failfix5 B/B). Once that path is exhausted, an empty
        letter is worse than a terse image re-ask — same helper as truncated-stream
        salvage, not nearest-furniture memory.
        """
        if self.mode != "answer" or not self._answer_unknownish(submit_out.get("answer")):
            return submit_out
        gm = self.graph_memory
        if gm is None or not hasattr(gm, "_salvage_location_mcq_letter"):
            return submit_out
        from emet.habitat.metrics import (
            choices_are_attribute_state,
            choices_are_location_mcq,
            parse_mcq_choices_from_question,
            question_is_attribute_state,
        )

        choices = parse_mcq_choices_from_question(self.question)
        if (
            not choices
            or question_is_attribute_state(self.question)
            or choices_are_attribute_state(choices)
            or not choices_are_location_mcq(choices)
        ):
            return submit_out
        images = list(submit_out.get("relevant_images") or [])
        if not images:
            # Fall back to whatever the last query_answer attached, if still around.
            for attr in ("last_eqa_images", "last_relevant_images"):
                cand = getattr(gm, attr, None)
                if cand:
                    images = list(cand)
                    break
        letter = str(gm._salvage_location_mcq_letter(self.question, choices, images) or "").strip()
        if not letter:
            return submit_out
        prior = submit_out.get("answer")
        self._append_trace(
            {
                "event": "final_location_salvage",
                "letter": letter,
                "prior_answer": prior,
                "n_unknown_explore": self._n_unknown_explore,
                "n_images": len(images),
            }
        )
        return {
            **submit_out,
            "answer": letter,
            "discord_text": (
                f"Answer:{letter}\n[final-location-salvage]\nprior:{prior}\n"
                f"{submit_out.get('discord_text') or ''}"
            ).strip(),
            "confidence": False,
        }

    def _maybe_follow_eqa_explore_action(self, submit_out: dict[str, Any]) -> bool:
        """Navigate to EQA ``Action: N`` when submit returned unconfident Unknown.

        Location MCQs often answer Unknown with an image index to explore. Inventing
        a salvage letter (holdout q104/q105) is worse than following that action.
        Allows one soft-over-budget nav so Action:N still runs after explore used
        the nominal ``max_nav_steps``.

        When Action:N is missing or out of range for the prompt image list (q105:
        ``Action:2`` with only one image), or the action target was already followed
        and the model is still Unknown, fall back to ``explore_frontier`` a few times
        instead of locking an empty letter.
        """
        if self.mode != "answer":
            return False
        gm = self.graph_memory
        if gm is None:
            return False
        ans = str(submit_out.get("answer") or "").strip().lower()
        conf = bool(submit_out.get("confidence"))
        unknownish = (not ans) or ans in {"unknown", "none", "n/a", "na"} or "frontier" in ans
        if conf and not unknownish:
            return False
        if not unknownish:
            return False
        obs_id = getattr(gm, "last_eqa_action_obs_id", None)
        if obs_id is not None:
            oid = int(obs_id)
            if oid not in self._followed_eqa_actions:
                # Soft +1 budget so Action:N is not starved by prior explore_frontier calls.
                if self._n_nav + self._n_explore >= self.max_nav_steps + 1:
                    return False
                self._followed_eqa_actions.add(oid)
                gm.last_eqa_action_obs_id = None
                # Force re-verify at the action target before the next submit.
                self._verified = False
                self._verified_obs_id = None
                self._last_verify = None
                # Temporarily raise budget so navigate_to_obs accepts the Action follow.
                old_budget = self.max_nav_steps
                self.max_nav_steps = max(old_budget, self._n_nav + self._n_explore + 1)
                try:
                    nav = self.handle_tool("navigate_to_obs", {"obs_id": oid})
                    # Verify the post-nav capture, not the historical hyp obs_id.
                    self.handle_tool("verify_siglip", {})
                finally:
                    self.max_nav_steps = old_budget
                self._append_trace(
                    {
                        "event": "follow_eqa_action",
                        "obs_id": oid,
                        "nav_ok": bool(nav.get("ok")),
                        "prior_answer": submit_out.get("answer"),
                    }
                )
                return True
        # No resolvable Action:N, or already followed that obs and still Unknown.
        # Cap soft explores so we do not loop forever on location MCQs.
        # Soft +2 beyond max_nav_steps: Action follow may already have used +1.
        if self._n_unknown_explore >= 2:
            return False
        if self._n_nav + self._n_explore >= self.max_nav_steps + 2:
            return False
        self._n_unknown_explore += 1
        gm.last_eqa_action_obs_id = None
        self._verified = False
        self._verified_obs_id = None
        self._last_verify = None
        old_budget = self.max_nav_steps
        self.max_nav_steps = max(old_budget, self._n_nav + self._n_explore + 1)
        try:
            nav = self.handle_tool("explore_frontier", {})
            self.handle_tool("verify_siglip", {})
        finally:
            self.max_nav_steps = old_budget
        self._append_trace(
            {
                "event": "follow_unknown_explore",
                "nav_ok": bool(nav.get("ok")),
                "prior_answer": submit_out.get("answer"),
                "n_unknown_explore": self._n_unknown_explore,
            }
        )
        return True

    def _fallback_tool(self) -> tuple[str, dict[str, Any]]:
        """Deterministic tool when VLM emits nothing parseable (or router is off).

        Thin scaffold only — prefer the VLM router. Interactive loop:
          (1) explore / inspect → hypotheses
          (2) navigate → capture → verify (+ Qwen assess)
          (3) Qwen answerable → submit with its suggested letter; else keep exploring
        """
        if self.mode == "explore":
            if self._explore_done():
                return "finish", {}
            return "explore_frontier", {}
        # (3) Qwen said this view is enough
        if self._evidence_policy.state == AgenticState.ANSWER and self._verified:
            prefer = ""
            if self._last_vlm_assess:
                prefer = str(self._last_vlm_assess.get("suggested_answer") or "").strip()
            return "submit_answer", ({"answer": prefer} if prefer else {})
        # If Qwen asked for more views, honor that before burning the budget on submit.
        budget_left = self._n_nav + self._n_explore < self.max_nav_steps
        need_more = bool(
            self._last_vlm_assess and self._last_vlm_assess.get("need_more_views")
        )
        frontiers_gone = (self._n_nav + self._n_explore) > 0 and self._frontier_count() == 0
        if need_more and budget_left and not frontiers_gone:
            return "explore_frontier", {}
        # (2) move in to an untried hypothesis (verify is chained after nav)
        if budget_left:
            h = self._next_untried_hypothesis()
            if h is not None:
                return "navigate_to_obs", {"obs_id": int(h.obs_id)}
        # (1) keep exploring for new views while budget remains
        if budget_left and not frontiers_gone:
            return "explore_frontier", {}
        # One first-look verify only if we have a fresh untried current view.
        latest = self._latest_obs_id()
        if self._last_verify is None and latest is not None and not self._obs_already_verified(latest):
            return "verify_siglip", {"obs_id": int(latest)}
        if self._require_verified and not self._verified:
            if budget_left and not frontiers_gone:
                return "explore_frontier", {}
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
            self._system_prompt = build_graph_eqa_system_prompt(self._tools)

    def _route_tool_calls(self) -> tuple[list[tuple[str, dict[str, Any]]], str, dict[str, Any]]:
        """One routing turn: state message → VLM → parsed tool calls (or deterministic fallback).

        Returns (tool_calls, picked_by, router_meta) where router_meta feeds the offline tuner.
        """
        meta: dict[str, Any] = {"raw_reply_chars": 0, "parse_ok": False, "tool_calls": []}
        gm = self.graph_memory
        client = getattr(gm, "eqa_client", None) if gm is not None else None
        if not self._router_enabled or client is None:
            tool, args = self._fallback_tool()
            return [(tool, args)], "fallback", meta
        self._ensure_router_prompt()
        state = build_state_message(self)
        try:
            reply = client(
                [state], system_prompt=self._system_prompt, max_new_tokens=ROUTER_MAX_NEW_TOKENS
            )
        except TypeError:
            try:
                reply = client([f"{self._system_prompt}\n\n{state}"])
            except Exception as e:
                _logger.warning(f"agentic router VLM call failed: {e}")
                tool, args = self._fallback_tool()
                return [(tool, args)], "fallback", meta
        except Exception as e:
            _logger.warning(f"agentic router VLM call failed: {e}")
            tool, args = self._fallback_tool()
            return [(tool, args)], "fallback", meta
        text = str(reply or "")
        meta["raw_reply_chars"] = len(text)
        parsed = parse_tool_calls_response(text)
        calls: list[tuple[str, dict[str, Any]]] = []
        for tc in parsed.get("tool_calls", []):
            name = str(tc.get("name") or "").strip().lower()
            if name in self._tool_names:
                calls.append((name, dict(tc.get("arguments") or {})))
            else:
                _logger.warning(f"agentic router: ignoring unknown tool {name!r}")
        if not calls:
            tool, args = self._fallback_tool()
            return [(tool, args)], "fallback", meta
        meta["parse_ok"] = True
        meta["tool_calls"] = [n for n, _ in calls]
        return calls, "vlm", meta

    def run(self) -> AgenticEQAResult:
        t0 = time.monotonic()
        final: dict[str, Any] | None = None
        # Agents are reused across episodes; do not inherit a previous escape floor.
        self._not_present_streak = 0
        self._frontier_pick_waypoints = []
        self._frontier_pick_dir = None
        self.agent._explore_min_travel_m = 0.0
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
            if (
                self.mode == "answer"
                and self._evidence_policy.state == AgenticState.ANSWER
                and r > 0
            ):
                out = self._do_submit_answer()
                if self._maybe_follow_eqa_explore_action(out):
                    continue
                final = out
                break
            calls, picked_by, router_meta = self._route_tool_calls()
            self._append_trace(
                {
                    "event": "tool_pick",
                    "picked_by": picked_by,
                    "tool": calls[0][0],
                    "args": calls[0][1],
                    "router_raw_reply_chars": router_meta.get("raw_reply_chars", 0),
                    "router_parse_ok": router_meta.get("parse_ok", False),
                    "router_tool_calls": router_meta.get("tool_calls", []),
                }
            )
            for tool, args in calls:
                out = self.handle_tool(tool, args)
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
                # Router re-picked a stalled hyp — force explore so we do not burn the round.
                if (
                    tool == "navigate_to_obs"
                    and not out.get("ok")
                    and str(out.get("status") or "") == "NAV_LOOP_BLOCKED"
                    and self.mode == "answer"
                    and self._n_nav + self._n_explore < self.max_nav_steps
                ):
                    self._append_trace(
                        {
                            "event": "nav_loop_redirect",
                            "from_obs_id": out.get("obs_id"),
                            "to": "explore_frontier",
                        }
                    )
                    self.handle_tool("explore_frontier", {"toward": self.query_text})
                # Motion tools chain verify themselves (router + fallback). Do not
                # double-verify here — that burned rounds on SKIPPED_SAME_VIEW.
            if final is not None:
                break
            if self.mode == "answer" and self._evidence_policy.state == AgenticState.ANSWER:
                out = self._do_submit_answer()
                if self._maybe_follow_eqa_explore_action(out):
                    continue
                final = out
                break
        else:
            budget_hit = True
            if self.mode == "answer" and self._require_verified:
                if not self._verified:
                    final = self._abstain_unverified()
                elif self._evidence_policy.state != AgenticState.ANSWER:
                    final = self._abstain_unverified(
                        reason="target evidence did not establish answer sufficiency"
                    )
                else:
                    final = self._do_submit_answer()
            else:
                final = self._do_submit_answer() if self.mode == "answer" else self._do_finish()
            # Rounds ran out before a submit round; still honor one Action:N /
            # unknown-explore follow-up so the EQA hint is not silently dropped.
            if (
                self.mode == "answer"
                and self._evidence_policy.state == AgenticState.ANSWER
                and self._maybe_follow_eqa_explore_action(final)
            ):
                final = self._do_submit_answer()

        wall = time.monotonic() - t0
        assert final is not None
        final = self._finalize_unknown_location_letter(final)
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
        )
        self._sync_scored_answer_to_graph_memory(result, final)
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
        ``last_eqa_parsed``, not ``AgenticEQAResult.answer``. Without this sync,
        a correct ``vlm_suggested`` letter is overwritten by truncated
        ``query_answer`` ``[salvage]`` (bal-32r2 q28/q39).
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
        source = str(final.get("answer_source") or "agentic")
        prior = getattr(gm, "last_eqa_raw", "") or ""
        gm.last_eqa_raw = (
            f"{prior.rstrip()}\n[agentic_submit]\nsource:{source}\nanswer:\n{letter}\n"
        )
        prev = getattr(gm, "last_eqa_parsed", None)
        if isinstance(prev, tuple) and len(prev) >= 5:
            reasoning, _old, _conf, action, conf_reason = prev[:5]
        else:
            reasoning, action, conf_reason = "", "", ""
        gm.last_eqa_parsed = (
            str(reasoning or ""),
            letter,
            bool(result.confidence),
            str(action or ""),
            str(conf_reason or ""),
        )
        self._append_trace(
            {
                "event": "sync_scored_answer",
                "letter": letter,
                "source": source,
                "confidence": bool(result.confidence),
                "obs_ids": list(getattr(gm, "last_eqa_obs_ids", []) or []),
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
        self.agent._agentic_eqa_summary = {
            "answer": result.answer,
            "confidence": result.confidence,
            "verified": result.verified,
            "n_rounds": result.n_rounds,
            "n_nav": result.n_nav,
            "n_explore": result.n_explore,
            "budget_hit": result.budget_hit,
            "tools": list(result.tool_log),
        }
        if self._trace_path is None:
            default = Path.home() / ".cache" / "habitat_eqa" / "agentic_traces" / "last_agentic_trace.jsonl"
            default.parent.mkdir(parents=True, exist_ok=True)
            default.write_text(
                "".join(json.dumps(r, default=str) + "\n" for r in rows),
                encoding="utf-8",
            )
            self._trace_path = default


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
    from emet.eval.dynagraph_vram import warm_siglip_confirmed_memory

    cfg = _eqa_cfg(agent)
    warm_siglip_confirmed_memory(agent)
    agent._habitat_blocked_goals = getattr(agent, "_habitat_blocked_goals", set()) or set()
    agent._habitat_recent_goals = getattr(agent, "_habitat_recent_goals", []) or []
    ex = AgenticEQAExecutor(
        agent,
        question,
        goal=goal,
        max_rounds=int(max_rounds if max_rounds is not None else cfg.get("agentic_max_tool_rounds", 8) or 8),
        max_nav_steps=int(max_nav_steps if max_nav_steps is not None else cfg.get("agentic_max_nav_steps", 8) or 8),
        verify_min_sim=float(
            verify_min_sim
            if verify_min_sim is not None
            else cfg.get("agentic_verify_min_sim", SIGLIP_CONFIRM_THRESHOLD)
            or SIGLIP_CONFIRM_THRESHOLD
        ),
        trace_path=trace_path,
        trace_meta=trace_meta,
        router=router,
        require_verified=None,  # resolved from env/config inside executor
    )
    result = ex.run()
    print(
        f"\n--- Agentic GraphEQA ({ex.mode}) ---\n{result.discord_text.strip()}\n"
        f"(rounds={result.n_rounds} nav={result.n_nav} explore={result.n_explore} "
        f"verified={result.verified} wall_s={result.wall_s:.1f})\n---\n",
        flush=True,
    )
    return result.discord_text, result.relevant_images

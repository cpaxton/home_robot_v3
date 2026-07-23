# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unified agentic GraphEQA loop: explore / navigate / verify / answer with tools."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from emet.agent.prompt import parse_tool_calls_response
from emet.memory.graph_eqa.agentic_tools import (
    build_agentic_eqa_tools,
    build_graph_eqa_system_prompt,
    build_state_message,
)
from emet.memory.graph_eqa.graph_memory import (
    SIGLIP_CONFIRM_THRESHOLD,
    NavHypothesis,
    VerifyResult,
)
from emet.utils.logger import Logger

_logger = Logger(__name__)

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})

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


def env_eqa_collect_trace() -> bool:
    v = os.environ.get("EMET_EQA_TRACE", "").strip().lower()
    return v in _TRUE


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
        max_rounds: int = 6,
        max_nav_steps: int = 3,
        verify_min_sim: float = SIGLIP_CONFIRM_THRESHOLD,
        trace_path: Path | str | None = None,
        trace_meta: dict[str, Any] | None = None,
        collect_trace: bool | None = None,
        router: bool | None = None,
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
        self._collect_trace = bool(collect_trace) if collect_trace is not None else (
            env_eqa_collect_trace() or bool(_eqa_cfg(agent).get("collect_agentic_trace", False))
        )
        self._trace_path = Path(trace_path) if trace_path else None
        self._trace_meta = dict(trace_meta or {})
        self._gt_placements: dict[str, Any] | None = None
        self._round = 0
        self._tried: dict[int, str] = {}
        env_router = env_eqa_agentic_router()
        cfg_router = _eqa_cfg(agent).get("agentic_vlm_router", True)
        self._router_enabled = bool(
            router if router is not None else (env_router if env_router is not None else cfg_router)
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
        self._hypotheses = list(gm.hypothesize_nav_targets(self.query_text, max_k=3))
        self._hyp_i = 0
        out = {
            "ok": True,
            "n_hypotheses": len(self._hypotheses),
            "hypotheses": [
                {
                    "phrase": h.phrase,
                    "obs_id": int(h.obs_id),
                    "xyz": [float(x) for x in np.asarray(h.xyz).reshape(-1)[:3]],
                    "score": float(h.score),
                    "source": h.source,
                }
                for h in self._hypotheses
            ],
        }
        self._append_trace({"tool": "inspect_graph", "picked_by": "loop", **{k: out[k] for k in ("n_hypotheses",)}})
        return out

    def _tool_explore_frontier(self, toward: str = "") -> dict[str, Any]:
        if self._n_nav + self._n_explore >= self.max_nav_steps:
            return {"ok": False, "error": "nav budget exhausted"}
        bias = (toward or "").strip() or self.query_text
        agent = self.agent
        frontier_xyz = None
        try:
            from emet.controller.habitat_nav import pick_uncovered_explore_target

            candidates: list[np.ndarray | None] = []
            if hasattr(agent, "_siglip_guided_frontier"):
                candidates.append(agent._siglip_guided_frontier(bias))
            if hasattr(agent, "_best_frontier_point_from_graph"):
                candidates.append(agent._best_frontier_point_from_graph(bias))
            frontier_xyz = pick_uncovered_explore_target(
                agent,
                question=bias,
                candidates=candidates,
                blocked=getattr(agent, "_habitat_blocked_goals", None),
                recent_goals=getattr(agent, "_habitat_recent_goals", None),
            )
        except Exception as e:
            _logger.warning(f"explore_frontier pick failed: {e}")
        ok = False
        if frontier_xyz is not None and hasattr(agent, "navigate_to_target_pose"):
            start = self._robot_xyt()
            if start is None:
                start = np.array([0.0, 0.0, 0.0])
            try:
                ok = bool(agent.navigate_to_target_pose(frontier_xyz, start, None))
            except TypeError:
                ok = bool(agent.navigate_to_target_pose(frontier_xyz, start))
            self._n_explore += 1
        elif hasattr(agent, "run_exploration"):
            ok = bool(agent.run_exploration())
            self._n_explore += 1
        cap = self._tool_capture_and_update()
        row = {
            "tool": "explore_frontier",
            "ok": ok,
            "frontier_xyz": [float(x) for x in np.asarray(frontier_xyz).reshape(-1)[:3]]
            if frontier_xyz is not None
            else None,
            "source": "pick_uncovered",
        }
        self._attach_gt(row, frontier_xyz)
        self._append_trace(row)
        return {"ok": ok, "capture": cap, "frontier_xyz": row["frontier_xyz"]}

    def _tool_navigate_to_obs(self, obs_id: int) -> dict[str, Any]:
        if self._n_nav + self._n_explore >= self.max_nav_steps:
            return {"ok": False, "error": "nav budget exhausted"}
        gm = self.graph_memory
        agent = self.agent
        if gm is None or not hasattr(agent, "navigate_to_target_pose"):
            return {"ok": False, "error": "nav unavailable"}
        xyt = self._robot_xyt()
        target = gm._navigation_waypoint_for_obs(int(obs_id), xyt)
        if target is None:
            return {"ok": False, "error": f"no waypoint for obs_id={obs_id}"}
        start = xyt if xyt is not None else np.array([0.0, 0.0, 0.0])
        try:
            finished = bool(agent.navigate_to_target_pose(target, start, None, target_obs_id=int(obs_id)))
        except TypeError:
            finished = bool(agent.navigate_to_target_pose(target, start, None))
        self._n_nav += 1
        nav_res = getattr(agent, "_last_nav_attempt", None)
        dist_m = float(getattr(nav_res, "dist_m", 0.0) or 0.0) if nav_res else 0.0
        note = str(getattr(nav_res, "note", "") or "") if nav_res else ""
        if hasattr(gm, "record_nav_attempt"):
            gm.record_nav_attempt(int(obs_id), success=finished, note=note or "agentic", dist_m=dist_m)
        if not finished:
            self._tried.setdefault(int(obs_id), "nav failed")
        row = {
            "tool": "navigate_to_obs",
            "obs_id": int(obs_id),
            "target_xyz": [float(x) for x in np.asarray(target).reshape(-1)[:3]],
            "nav_success": bool(finished),
            "nav_dist_m": dist_m,
            "nav_note": note,
        }
        self._attach_gt(row, target)
        self._append_trace(row)
        cap = self._tool_capture_and_update()
        return {"ok": bool(finished), "target_xyz": row["target_xyz"], "capture": cap}

    def _tool_look_around(self) -> dict[str, Any]:
        agent = self.agent
        ok = False
        if hasattr(agent, "look_around"):
            try:
                agent.look_around()
                ok = True
            except Exception as e:
                _logger.warning(f"look_around failed: {e}")
        cap = self._tool_capture_and_update()
        self._append_trace({"tool": "look_around", "ok": ok})
        return {"ok": ok, "capture": cap}

    def _tool_capture_and_update(self) -> dict[str, Any]:
        agent = self.agent
        if hasattr(agent, "update"):
            try:
                agent.update()
            except Exception as e:
                _logger.warning(f"capture_and_update agent.update failed: {e}")
        gm = self.graph_memory
        if gm is not None and getattr(gm, "memory_summary_enabled", False):
            if hasattr(gm, "refresh_siglip_confirmed_memory"):
                gm.refresh_siglip_confirmed_memory()
        self._append_trace({"tool": "capture_and_update", "ok": True})
        return {"ok": True}

    def _tool_verify_siglip(self, phrase: str, obs_id: int | None) -> dict[str, Any]:
        gm = self.graph_memory
        if gm is None:
            return {"ok": False, "error": "no graph_memory"}
        text = (phrase or "").strip()
        if not text:
            phrases = list(getattr(gm, "_relevant_phrases", None) or []) + list(
                getattr(gm, "_relevant_objects", None) or []
            )
            text = phrases[0] if phrases else self.question
        oid = obs_id
        if oid is None or int(oid) < 0:
            if self._hypotheses:
                oid = int(self._hypotheses[min(self._hyp_i, len(self._hypotheses) - 1)].obs_id)
            elif getattr(gm, "last_eqa_obs_ids", None):
                oid = int(gm.last_eqa_obs_ids[0])
            else:
                return {"ok": False, "error": "no obs_id"}
        rgb = None
        robot = getattr(self.agent, "robot", None)
        if robot is not None and hasattr(robot, "get_observation"):
            try:
                obs = robot.get_observation()
                if obs is not None and getattr(obs, "rgb", None) is not None:
                    rgb = np.asarray(obs.rgb)
            except Exception:
                pass
        result = gm.verify_phrase_at_obs(text, int(oid), rgb=rgb, min_sim=self.verify_min_sim)
        self._last_verify = result
        self._tried[int(result.obs_id)] = f"verify {result.status} sim={float(result.sim):.2f}"
        if result.ok and result.status == "PRESENT":
            self._verified = True
            self._verified_obs_id = int(result.obs_id)
        row = {
            "tool": "verify_siglip",
            "phrase": text,
            "obs_id": int(result.obs_id),
            "sim": float(result.sim),
            "decision": result.status,
            "text_feat": _feat_list(result.text_feat),
            "img_feat": _feat_list(result.img_feat),
        }
        xyt = self._robot_xyt()
        if xyt is not None:
            row["xyt"] = [float(x) for x in xyt.reshape(-1)[:3]]
        hyp = next((h for h in self._hypotheses if int(h.obs_id) == int(result.obs_id)), None)
        if hyp is not None:
            row["target_xyz"] = [float(x) for x in np.asarray(hyp.xyz).reshape(-1)[:3]]
            row["source"] = hyp.source
            self._attach_gt(row, hyp.xyz)
        self._append_trace(row)
        return {
            "ok": True,
            "status": result.status,
            "sim": float(result.sim),
            "verified": self._verified,
            "obs_id": int(result.obs_id),
        }

    def _tool_submit_answer(self, answer: str) -> dict[str, Any]:
        if self.mode == "explore":
            return {"ok": False, "error": "submit_answer unavailable in explore mode — use finish"}
        if not self._verified and self._round < self.max_rounds - 1:
            return {
                "ok": False,
                "error": "not verified — call verify_siglip (or exhaust budget) before submit_answer",
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

    def _do_submit_answer(self, prefer_answer: str = "") -> dict[str, Any]:
        from emet.eval.dynagraph_vram import release_siglip_for_vlm

        agent = self.agent
        gm = self.graph_memory
        release_siglip_for_vlm(agent)
        discord_text = ""
        confidence = False
        relevant_images: list[Any] = []
        answer = (prefer_answer or "").strip()
        if gm is not None and hasattr(gm, "query_answer"):
            # Prefer verified single-image selection for the prompt.
            if self._verified_obs_id is not None and hasattr(gm, "select_obs_ids_for_verified_answer"):
                ids = gm.select_obs_ids_for_verified_answer(self._verified_obs_id, max_images=1)
                gm.last_eqa_obs_ids = list(ids)
            os.environ.setdefault("EMET_EQA_ANSWER_MAX_NEW_TOKENS", "64")
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
                ) = gm.query_answer(self.question, xyt, planner)
                answer = (ans or answer or "").strip()
                discord_text = f"Answer:{answer}\nConfidence:{confidence}"
            except Exception as e:
                discord_text = f"Answer:Unknown\nEQA failed: {e}"
                answer = "Unknown"
        elif answer:
            discord_text = f"Answer:{answer}"
            confidence = bool(self._verified)
        else:
            discord_text = "Answer:Unknown\nNo graph memory"
            answer = "Unknown"
        self._append_trace(
            {
                "tool": "submit_answer",
                "final_answer": answer,
                "confidence": bool(confidence),
                "verified": self._verified,
                "verified_obs_id": self._verified_obs_id,
            }
        )
        return {
            "ok": True,
            "answer": answer,
            "discord_text": discord_text,
            "confidence": bool(confidence),
            "relevant_images": relevant_images,
        }

    def _fallback_tool(self) -> tuple[str, dict[str, Any]]:
        """Deterministic tool when VLM emits nothing parseable (or router is off)."""
        if self.mode == "explore":
            if self._explore_done():
                return "finish", {}
            return "explore_frontier", {}
        if self._verified:
            return "submit_answer", {}
        if not self._hypotheses:
            if self._n_nav + self._n_explore < self.max_nav_steps:
                return "explore_frontier", {}
            return "submit_answer", {}
        if self._n_nav + self._n_explore >= self.max_nav_steps:
            if self._last_verify is None:
                return "verify_siglip", {}
            return "submit_answer", {}
        if self._hyp_i < len(self._hypotheses):
            h = self._hypotheses[self._hyp_i]
            self._hyp_i += 1
            return "navigate_to_obs", {"obs_id": int(h.obs_id)}
        if self._last_verify is None or self._last_verify.status != "PRESENT":
            if self._n_explore < 1 and self._n_nav + self._n_explore < self.max_nav_steps:
                return "explore_frontier", {}
            return "verify_siglip", {}
        return "submit_answer", {}

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
        # Always start with inspect to seed hypotheses.
        self.handle_tool("inspect_graph", {})
        for r in range(self.max_rounds):
            self._round = r
            if self.mode == "answer" and self._verified and r > 0:
                final = self._do_submit_answer()
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
                if tool in ("submit_answer", "finish") and out.get("ok"):
                    final = out
                    break
                if not out.get("ok") and "budget" in str(out.get("error", "")):
                    # Budget gate rejected — stop dispatching the rest of this reply.
                    break
                if picked_by == "fallback" and self.mode == "answer":
                    # Deterministic policy chains verify after motion; the VLM router
                    # is expected to call verify_siglip itself next round.
                    if tool == "navigate_to_obs" and out.get("ok"):
                        self.handle_tool("verify_siglip", {"obs_id": args.get("obs_id")})
                    elif tool == "explore_frontier":
                        self.handle_tool("verify_siglip", {})
            if final is not None:
                break
            if self.mode == "answer" and self._verified:
                final = self._do_submit_answer()
                break
        else:
            budget_hit = True
            final = self._do_submit_answer() if self.mode == "answer" else self._do_finish()

        wall = time.monotonic() - t0
        assert final is not None
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
    from emet.eval.dynagraph_vram import warm_siglip_confirmed_memory

    cfg = _eqa_cfg(agent)
    warm_siglip_confirmed_memory(agent)
    agent._habitat_blocked_goals = getattr(agent, "_habitat_blocked_goals", set()) or set()
    agent._habitat_recent_goals = getattr(agent, "_habitat_recent_goals", []) or []
    ex = AgenticEQAExecutor(
        agent,
        question,
        goal=goal,
        max_rounds=int(max_rounds if max_rounds is not None else cfg.get("agentic_max_tool_rounds", 6) or 6),
        max_nav_steps=int(max_nav_steps if max_nav_steps is not None else cfg.get("agentic_max_nav_steps", 3) or 3),
        verify_min_sim=float(
            verify_min_sim
            if verify_min_sim is not None
            else cfg.get("agentic_verify_min_sim", SIGLIP_CONFIRM_THRESHOLD)
            or SIGLIP_CONFIRM_THRESHOLD
        ),
        trace_path=trace_path,
        trace_meta=trace_meta,
        router=router,
    )
    result = ex.run()
    print(
        f"\n--- Agentic GraphEQA ({ex.mode}) ---\n{result.discord_text.strip()}\n"
        f"(rounds={result.n_rounds} nav={result.n_nav} explore={result.n_explore} "
        f"verified={result.verified} wall_s={result.wall_s:.1f})\n---\n",
        flush=True,
    )
    return result.discord_text, result.relevant_images

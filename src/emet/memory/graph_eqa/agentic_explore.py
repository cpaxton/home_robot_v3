# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Explore, inspect_graph, and frontier-pick for the agentic GraphEQA executor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from emet.memory.graph_eqa.agentic_config import (
    INVESTIGATE_SOURCES,
    NEAR_INVESTIGATE_M,
    env_eqa_hyp_recall_k,
    question_is_locate,
)
from emet.memory.graph_eqa.agentic_tools import coerce_room_label, normalize_current_room
from emet.memory.graph_eqa.graph_memory import NavHypothesis, label_matches_relevant_object
from emet.memory.graph_eqa.room_clusters import question_target_rooms, room_leave_needed
from emet.utils.logger import Logger

_logger = Logger(__name__)


class AgenticExploreMixin:
    """Explore_frontier, inspect_graph, and nearby-investigate preference."""

    _FIXTURE_LABEL_TOKENS = frozenset(
        {
            "cabinet",
            "counter",
            "shelf",
            "table",
            "desk",
            "dresser",
            "chest",
            "drawer",
            "stove",
            "oven",
            "refrigerator",
            "fridge",
            "microwave",
            "countertop",
        }
    )

    def _target_boost_phrases(self) -> list[str]:
        """Phrases to bias graph recall — from the question, not episode metadata."""
        ordered: list[str] = []
        phrase = str(getattr(self, "_target_phrase", "") or "").strip()
        if phrase:
            ordered.append(phrase)
        from emet.memory.graph_eqa.graph_memory import heuristic_relevant_phrases

        for raw in heuristic_relevant_phrases(self.query_text):
            val = str(raw or "").strip()
            if val and val not in ordered:
                ordered.append(val)
        return ordered

    def _recall_nav_hypotheses(self) -> list[NavHypothesis]:
        gm = self.graph_memory
        if gm is None:
            return []
        boost = self._target_boost_phrases()
        try:
            hypotheses = gm.hypothesize_nav_targets(
                self.query_text,
                max_k=env_eqa_hyp_recall_k(),
                robot_xyt=self._robot_xyt_world(),
                boost_phrases=boost or None,
            )
        except TypeError:
            hypotheses = gm.hypothesize_nav_targets(self.query_text, max_k=env_eqa_hyp_recall_k())
        if not any(str(h.source) in INVESTIGATE_SOURCES for h in hypotheses):
            adjacent = self._receptacle_adjacent_hypotheses(gm)
            if adjacent:
                hypotheses = adjacent + list(hypotheses)
        return hypotheses

    def _prefers_nearby_investigate(self) -> bool:
        """Look at a nearby matching place card before frontier travel."""
        if bool(self._close_look_required):
            return True
        return question_is_locate(self.question)

    def _investigate_matches_target(self, hyp: NavHypothesis | None, obs_id: int) -> bool:
        """Absent-at-close only nudges explore when this card matches the seek phrase."""
        target = str(getattr(self, "_target_phrase", "") or "").strip().lower()
        if not target:
            return True
        if hyp is not None:
            phrase = str(hyp.phrase or "").lower()
            if target in phrase or label_matches_relevant_object(target, phrase):
                return True
            for tok in self._FIXTURE_LABEL_TOKENS:
                if tok in target and tok in phrase:
                    return True
        gm = self.graph_memory
        if gm is not None and hasattr(gm, "_observation_by_id"):
            obs = gm._observation_by_id(int(obs_id))
            if obs is not None:
                for lab in getattr(obs, "labels", None) or []:
                    if label_matches_relevant_object(target, str(lab)):
                        return True
        return False

    def _nearby_untried_investigate_hyp(self, max_dist_m: float = NEAR_INVESTIGATE_M) -> NavHypothesis | None:
        best: NavHypothesis | None = None
        best_d = float("inf")
        visible = self._grounded_visible_place_obs()
        for h in self._investigate_hypotheses():
            oid = int(h.obs_id)
            if visible is not None and oid not in visible:
                continue
            if self._obs_already_verified(oid) or self._hypothesis_nav_blocked(oid):
                continue
            if self._place_approaches_exhausted(oid):
                continue
            d = self._dist_to_anchor_m(oid, h)
            if d is None or d > float(max_dist_m):
                continue
            if d < best_d:
                best_d = d
                best = h
        return best

    def _tool_explore_frontier(self, toward: str = "", *, frontier_id: str = "") -> dict[str, Any]:
        if self._n_nav + self._n_explore >= self.max_nav_steps:
            return {"ok": False, "error": "nav budget exhausted"}
        toward = (toward or "").strip()
        requested_frontier_id = str(frontier_id or "").strip()
        grounded = self.decision_policy == "grounded_v2"
        if grounded and self.action_progress_mode == "enforce" and not requested_frontier_id:
            eligible = tuple(self._rendered_action_allowlist().get("frontier_ids", ()))
            if not eligible:
                return {
                    "ok": False,
                    "status": "NO_ELIGIBLE_ACTION",
                    "error": "no eligible rendered frontier remains while static progress gating is enforced",
                }
            requested_frontier_id = str(eligible[0])
        leave = False
        if not grounded:
            leave = room_leave_needed(
                room_policy=self.room_policy,
                current_room=self._last_room_estimate,
                question=self.question,
                in_target_area=self._in_target_area,
            )
        # Do not invent toward= from MCQ options / room enums — pass the Question
        # to the frontier VLM and let it pick among graph-contexted views.
        bias = toward or self.query_text
        agent = self.agent
        gm = self.graph_memory
        frontier_xyz = None
        frontier_id = requested_frontier_id
        pick_source = "pick_uncovered"
        frontier_room = "unknown"
        history_signature = None
        history_progress_before = None
        try:
            from emet.controller.habitat_nav import pick_uncovered_explore_target

            escape_m = 0.0 if grounded else self._escape_min_travel_m()
            candidates: list[np.ndarray | None] = []
            if requested_frontier_id:
                world = getattr(gm, "world_evidence", None) if gm is not None else None
                record = world.frontiers.get(requested_frontier_id) if world is not None else None
                if record is None or record.status != "active":
                    return {
                        "ok": False,
                        "error": f"stale or unknown frontier_id: {requested_frontier_id}",
                        "status": "STALE_FRONTIER_ID",
                        "frontier_id": requested_frontier_id,
                    }
                frontier_xyz = np.asarray(record.centroid_xyz, dtype=float)
                pick_source = "router_frontier_id"
            # GraphEQA-style: VLM ranks a small pool of reachable frontier RGBs.
            # Agentic always tries this; classic coverage path still uses EMET_VLM_FRONTIER_SCORING.
            if frontier_xyz is None and hasattr(agent, "_vlm_frontier_choice"):
                try:
                    vlm_pt = agent._vlm_frontier_choice(
                        bias,
                        current_room=self._last_room_estimate,
                        room_policy=self.room_policy,
                        leave_hint=leave,
                    )
                except TypeError:
                    # Older controllers without room kwargs.
                    try:
                        vlm_pt = agent._vlm_frontier_choice(bias)
                    except Exception as e:
                        _logger.warning(f"vlm_frontier_choice failed: {e}")
                        vlm_pt = None
                except Exception as e:
                    _logger.warning(f"vlm_frontier_choice failed: {e}")
                    vlm_pt = None
                if vlm_pt is not None:
                    candidates.append(vlm_pt)
                    pick_source = "vlm_frontier"
            # SigLIP guidance aims at the frontier nearest the best-matching *already
            # observed* point, so while escaping it just pulls us back into the area we
            # already rejected. Let region utility choose instead.
            if frontier_xyz is None and escape_m <= 0.0 and hasattr(agent, "_siglip_guided_frontier"):
                candidates.append(agent._siglip_guided_frontier(bias))
            if frontier_xyz is None and hasattr(agent, "_best_frontier_point_from_graph"):
                candidates.append(agent._best_frontier_point_from_graph(bias))
            if frontier_xyz is None:
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
                if (
                    vlm0 is None
                    or float(
                        np.linalg.norm(
                            np.asarray(frontier_xyz, dtype=float).reshape(-1)[:2]
                            - np.asarray(vlm0, dtype=float).reshape(-1)[:2]
                        )
                    )
                    > 0.35
                ):
                    pick_source = "pick_uncovered"
        except Exception as e:
            _logger.warning(f"explore_frontier pick failed: {e}")
            pick_source = "pick_uncovered"
        if frontier_xyz is not None:
            room_fn = getattr(gm, "graph_room_at_robot", None) if gm is not None else None
            if callable(room_fn):
                try:
                    frontier_room = coerce_room_label(
                        room_fn(
                            (
                                float(np.asarray(frontier_xyz).reshape(-1)[0]),
                                float(np.asarray(frontier_xyz).reshape(-1)[1]),
                            )
                        ),
                        room_policy=self.room_policy,
                    )
                except Exception:
                    frontier_room = "unknown"
            frontier_id_fn = getattr(gm, "frontier_id_near_xy", None) if gm is not None else None
            if not frontier_id and callable(frontier_id_fn):
                try:
                    frontier_id = str(frontier_id_fn(frontier_xyz) or "")
                except (TypeError, ValueError):
                    frontier_id = ""
        if frontier_id:
            history_signature = self._action_signature(
                "explore_frontier",
                {"frontier_id": frontier_id, "toward": toward},
            )
            history_progress_before = self._action_progress_token(history_signature)
        frontier_key = -1_000_000 - self._n_explore
        hypothesis_id = self._begin_policy_approach(
            "frontier",
            frontier_key,
            bias,
        )
        ok = False
        motion_progress = False
        reached = False
        nav_outcome_str = ""
        used_nav_target = False
        start = self._robot_xyt_world()
        if start is None:
            start = np.array([0.0, 0.0, 0.0])
        if frontier_xyz is not None and hasattr(agent, "navigate_to_target_pose"):
            used_nav_target = True
            try:
                nav_outcome = agent.navigate_to_target_pose(frontier_xyz, start, None)
            except TypeError:
                nav_outcome = agent.navigate_to_target_pose(frontier_xyz, start)
            nav_outcome_str = str(nav_outcome)
            ok = bool(nav_outcome)
            nav_outcome_str = str(nav_outcome)
            motion_progress = ok
            self._n_explore += 1
            if ok:
                self._consecutive_nav_fail = 0
                reached = bool(getattr(nav_outcome, "finished", ok))
                if self.decision_policy != "grounded_v2" or reached:
                    self._retire_visited_frontier(frontier_xyz=frontier_xyz)
            elif self._explore_nav_progressed():
                # Chunked path: robot moved toward the frontier even if not finished.
                self._consecutive_nav_fail = 0
                motion_progress = True
        elif hasattr(agent, "run_exploration"):
            ok = bool(agent.run_exploration())
            motion_progress = ok
            self._n_explore += 1
            if ok:
                self._consecutive_nav_fail = 0
            pick_source = "run_exploration_fallback"
            # Recover a goal for viz/trace when the uncovered picker returned None.
            if frontier_xyz is None:
                recent = list(getattr(agent, "_habitat_recent_goals", None) or [])
                if recent:
                    frontier_xyz = np.array([float(recent[-1][0]), float(recent[-1][1]), 1.0])
                else:
                    after = self._robot_xyt_world()
                    if after is not None:
                        frontier_xyz = np.asarray(after, dtype=float).reshape(-1)[:3]
            if ok and frontier_xyz is not None:
                self._retire_visited_frontier(frontier_xyz=frontier_xyz)
                reached = True
        nav_result = getattr(agent, "_last_nav_attempt", None) if used_nav_target else None
        if nav_result is not None:
            from emet.controller.nav_attempt import nav_status_code

            nav_status = nav_status_code(nav_result)
        else:
            nav_status = nav_outcome_str or ("ok" if ok else "failed")
        if motion_progress:
            self._refresh_room_after_motion()
        cap = self._tool_capture_and_update()
        look_retry = False
        # After a successful explore nav, a mid-floor / already-mapped goal often yields
        # NO_NEW_OBS. Spin in place so we still peel new coverage from this pose.
        if ok and not cap.get("ok") and str(cap.get("status") or "") == "NO_NEW_OBS":
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
        room_aligned: bool | None = None
        if self.room_policy == "canonical":
            targets = question_target_rooms(self.question)
            fr = normalize_current_room(frontier_room)
            if fr != "unknown" and targets:
                room_aligned = fr in targets
        elif self._in_target_area is False:
            # Soft leave explore: alignment unknown until next router turn.
            room_aligned = None
        row = {
            "tool": "explore_frontier",
            "ok": ok,
            "frontier_xyz": [float(x) for x in np.asarray(frontier_xyz).reshape(-1)[:3]]
            if frontier_xyz is not None
            else None,
            "frontier_id": frontier_id or None,
            "source": pick_source,
            "pick_panel": str(panel_path) if panel_path else None,
            "look_around_on_no_new_obs": look_retry,
            "room_policy": self.room_policy,
            "current_room": self._last_room_estimate,
            "frontier_room": frontier_room,
            "room_leave_hint": leave,
            "toward": toward or None,
            "room_aligned": room_aligned,
            "in_target_area": self._in_target_area,
            "nav_outcome": nav_outcome_str or None,
            "nav_status_code": nav_status,
        }
        self._attach_gt(row, frontier_xyz)
        self._append_trace(row)
        if ok:
            self._prefer_explore = False
            self._prefer_explore_reason = ""
            self._n_consecutive_explore = int(getattr(self, "_n_consecutive_explore", 0) or 0) + 1
        return {
            "ok": ok,
            "capture": cap,
            "frontier_xyz": row["frontier_xyz"],
            "xyz": row["frontier_xyz"],
            "frontier_id": frontier_id or None,
            "target_kind": "frontier" if frontier_id else "",
            "target_id": frontier_id or "",
            "phrase": toward or self.query_text,
            "room": frontier_room,
            "motion_progress": bool(motion_progress),
            "nav_outcome": nav_outcome_str or None,
            "nav_progress": bool(motion_progress),
            "nav_finished": bool(reached),
            "nav_status_code": nav_status,
            "status": nav_status,
            "verify": verify_out,
            "_action_history_signature": history_signature,
            "_action_history_progress_before": history_progress_before,
        }

    def _explore_hypotheses(self) -> list[NavHypothesis]:
        return [h for h in self._hypotheses if str(h.source) not in INVESTIGATE_SOURCES]

    def _explore_nav_progressed(self) -> bool:
        """True if the last nav attempt made real progress (chunked path is not a miss)."""
        agent = self.agent
        if agent is None:
            return False
        nav_res = getattr(agent, "_last_nav_attempt", None)
        if nav_res is None:
            return False
        return bool(getattr(nav_res, "success", False))

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
        xyt = self._robot_xyt_world()
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

    def _tool_inspect_graph(self) -> dict[str, Any]:
        gm = self.graph_memory
        if gm is None:
            return {"ok": False, "error": "no graph_memory"}
        if hasattr(gm, "extract_relevant_objects") and getattr(gm, "image_description_client", None) is not None:
            gm.extract_relevant_objects(self.query_text)
        if getattr(gm, "memory_summary_enabled", False) and hasattr(gm, "refresh_siglip_confirmed_memory"):
            gm.refresh_siglip_confirmed_memory()
        hypotheses = self._recall_nav_hypotheses()
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
                    "siglip_sim": (float(h.siglip_sim) if getattr(h, "siglip_sim", None) is not None else None),
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

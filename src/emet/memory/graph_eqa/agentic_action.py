# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Pose, trace, and action-gate helpers for the agentic GraphEQA executor."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from emet.memory.graph_eqa.action_history import (
    ActionHistoryEntry,
    ActionSignature,
    ActionTarget,
    GateDecision,
    ProgressToken,
    decide_candidate,
    quantized_xy,
    render_history_entry,
    stable_digest,
    status_outcome_class,
)
from emet.memory.graph_eqa.agentic_config import (
    PLACE_APPROACH_SAMPLES,
    RECENT_ACTIONS_K,
)
from emet.memory.graph_eqa.agentic_tools import coerce_room_label, normalize_current_room
from emet.memory.graph_eqa.room_clusters import question_target_rooms, room_mismatches_question
from emet.utils.logger import Logger

_logger = Logger(__name__)


class AgenticActionMixin:
    """Pose, GT trace attach, and action-progress gating."""

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
            pose = np.asarray(robot.get_base_pose(), dtype=float).reshape(-1)
        except Exception:
            return None
        if pose.size < 2 or not np.isfinite(pose[:2]).all():
            return None
        return pose

    def _robot_xyt_world(self) -> np.ndarray | None:
        """Robot base ``(x, y, θ)`` in the voxel-map / world frame for A* planning.

        ``get_base_pose`` is episode-relative (ZMQ gps/compass), but the voxel map
        and ``navigate_to_target_pose`` plan in the world frame anchored at
        ``navigation_origin_xyt``. For sims whose spawn is not at world (0,0)
        (robocasa origin ≈ (2.9,-1.7)) planning from the raw episode pose puts the
        A* start at grid center / an unexplored cell → "non navigable point".
        """
        local = self._robot_xyt()
        if local is None:
            return None
        agent = self.agent
        convert = getattr(agent, "_planning_base_xyt", None)
        if callable(convert):
            try:
                world = np.asarray(convert(local), dtype=float).reshape(-1)
                if world.size >= 2 and np.isfinite(world[:2]).all():
                    if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
                        print(
                            f"[navstart] local={local.round(3).tolist()} world={world.round(3).tolist()}",
                            flush=True,
                        )
                    return world
                if os.environ.get("EMET_DYNAMEM_MAP_DEBUG"):
                    print(
                        f"[navstart] invalid world pose={world.tolist()}; using local={local.round(3).tolist()}",
                        flush=True,
                    )
            except Exception:
                pass
        return local

    def _graph_world_step(self) -> int:
        gm = self.graph_memory
        getter = getattr(gm, "_effective_timestep", None) if gm is not None else None
        if callable(getter):
            try:
                return int(getter())
            except (TypeError, ValueError):
                pass
        return int(getattr(gm, "_graph_timestep", 0) or 0) if gm is not None else 0

    def _refresh_graph_room_estimate(self, *, after_motion: bool = False) -> str:
        """Recompute current-pose canonical room and room-target status."""
        gm = self.graph_memory
        room_fn = getattr(gm, "graph_room_at_robot", None) if gm is not None else None
        if after_motion:
            self._graph_room_stale = True
            self._router_room_stale = True
        if not callable(room_fn):
            if after_motion:
                self._room_estimate_stale = True
                self._current_room_source = "stale_router"
                self._in_target_area = None
            return self._graph_room_estimate
        try:
            graph_room = coerce_room_label(
                room_fn(self._robot_xyt_world()),
                room_policy=self.room_policy,
            )
        except Exception as e:
            _logger.warning(f"graph room refresh before router state failed: {e}")
            if after_motion:
                self._room_estimate_stale = True
                self._current_room_source = "stale_router"
                self._in_target_area = None
            return self._graph_room_estimate
        self._graph_room_estimate = graph_room
        if graph_room != "unknown":
            self._last_room_estimate = graph_room
            self._current_room_source = "graph_current_pose"
            self._room_estimate_stale = False
            self._graph_room_stale = False
            self._room_pose_round = int(self._round) + 1
            self._room_world_step = self._graph_world_step()
        elif after_motion:
            self._room_estimate_stale = True
            self._current_room_source = "stale_router"
        targets = question_target_rooms(self.question)
        if self._last_room_estimate != "unknown" and targets and not self._room_estimate_stale:
            self._in_target_area = not room_mismatches_question(
                self._last_room_estimate,
                self.question,
            )
        elif after_motion:
            self._in_target_area = None
        return graph_room

    def _refresh_room_after_motion(self) -> str:
        """Invalidate prior-pose router room and establish a current-pose room."""
        return self._refresh_graph_room_estimate(after_motion=True)

    def _observation_room(self, obs_id: int | None) -> str:
        """Resolve evidence room from the immutable observation view/place."""
        if obs_id is None:
            return ""
        gm = self.graph_memory
        room_fn = getattr(gm, "observation_room", None) if gm is not None else None
        if callable(room_fn):
            try:
                _room_id, room_name = room_fn(int(obs_id))
                normalized = normalize_current_room(room_name)
                if normalized != "unknown":
                    return normalized
            except (TypeError, ValueError):
                pass
        return ""

    def _append_trace(self, row: dict[str, Any]) -> None:
        if not self._collect_trace:
            return
        payload = {
            **self._trace_meta,
            "trace_schema_version": 2,
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

    def _graph_node_for_obs(self, obs_id: int) -> Any | None:
        gm = self.graph_memory
        for node in list(getattr(gm, "_nodes", None) or ()):
            if int(getattr(node, "obs_id", -1)) != int(obs_id):
                continue
            if bool(getattr(node, "is_frontier", False)) or bool(getattr(node, "is_viewpoint", False)):
                continue
            return node
        return None

    def _action_target_for_obs(self, obs_id: int) -> ActionTarget:
        """Resolve a mutable adapter ID to stable place/view semantics."""
        oid = int(obs_id)
        gm = self.graph_memory
        world = getattr(gm, "world_evidence", None) if gm is not None else None
        node = self._graph_node_for_obs(oid)
        view = world.view_for_obs(oid) if world is not None else None
        entity = None
        if world is not None and node is not None:
            entity = world.entity_for_node(int(node.node_id))
        place_id = str(getattr(view, "place_id", "") or getattr(entity, "place_id", "") or f"obs:{oid}")
        labels = tuple(str(item) for item in (getattr(node, "labels", None) or ()))
        if not labels:
            labels = tuple(str(item) for item in (getattr(view, "labels", None) or ()))
        if not labels:
            hypothesis = next(
                (item for item in self._hypotheses if int(item.obs_id) == oid),
                None,
            )
            phrase = str(getattr(hypothesis, "phrase", "") or "").strip()
            labels = (phrase,) if phrase else ()
        room = self._observation_room(oid) or "unknown"
        if room == "unknown" and world is not None:
            place = world.places.get(place_id)
            room_record = world.rooms.get(place.room_id) if place is not None and place.room_id else None
            room = str(getattr(room_record, "room_name", "") or "unknown")
        xyz_value = getattr(node, "xyz", None) if node is not None else getattr(view, "object_xyz", None)
        xyz = None
        if xyz_value is not None:
            values = np.asarray(xyz_value, dtype=float).reshape(-1)
            if values.size >= 2:
                xyz = (
                    float(values[0]),
                    float(values[1]),
                    float(values[2]) if values.size >= 3 else 0.0,
                )
        return ActionTarget(
            kind="place",
            stable_id=place_id,
            labels=labels,
            room=room,
            adapter_id=oid,
            view_id=str(getattr(view, "view_id", "") or "") or None,
            revision=(int(view.revision) if view is not None else None),
            xyz=xyz,
        )

    def _action_target_for_frontier(self, frontier_id: str) -> ActionTarget:
        fid = str(frontier_id or "").strip()
        gm = self.graph_memory
        world = getattr(gm, "world_evidence", None) if gm is not None else None
        record = world.frontiers.get(fid) if world is not None and fid else None
        labels: list[str] = []
        room = "unknown"
        if record is not None and world is not None:
            for place_id in tuple(record.attachment_ids)[:3]:
                place = world.places.get(str(place_id))
                entity = world.entities.get(place.entity_id) if place is not None else None
                labels.extend(str(item) for item in (getattr(entity, "labels", None) or ()))
                room_record = world.rooms.get(place.room_id) if place is not None and place.room_id else None
                if room == "unknown" and room_record is not None:
                    room = str(room_record.room_name or "unknown")
        xyz = tuple(record.centroid_xyz) if record is not None else None
        return ActionTarget(
            kind="frontier",
            stable_id=fid or "unresolved",
            labels=tuple(dict.fromkeys(labels))[:3],
            room=room,
            adapter_id=(int(record.obs_id) if record is not None and record.obs_id is not None else None),
            revision=(int(record.revision) if record is not None else None),
            xyz=xyz,
        )

    def _frontier_geometry_id(self, frontier_id: str) -> str:
        gm = self.graph_memory
        world = getattr(gm, "world_evidence", None) if gm is not None else None
        record = world.frontiers.get(str(frontier_id)) if world is not None else None
        if record is None:
            return "unknown"
        centroid = quantized_xy(record.centroid_xyz, cell_m=0.5)
        return stable_digest(
            "frontier-material",
            {
                "status": str(record.status),
                "centroid_cell_0p5m": centroid,
                "cells": tuple(sorted(record.cells)),
                "parents": tuple(sorted(record.parent_ids)),
            },
        )

    def _relevant_evidence_digest(self, target: ActionTarget) -> str:
        """Hash target-local non-attempt evidence only."""
        gm = self.graph_memory
        world = getattr(gm, "world_evidence", None) if gm is not None else None
        if world is None:
            return "none"
        rows: list[tuple[Any, ...]] = []
        for event in list(getattr(world, "events", None) or ()):
            payload = dict(getattr(event, "payload", None) or {})
            if "outcome" in payload and "status_code" in payload:
                continue
            relevant = str(getattr(event, "subject_id", "")) == target.stable_id
            relevant = relevant or str(getattr(event, "place_id", "") or "") == target.stable_id
            relevant = relevant or str(getattr(event, "frontier_id", "") or "") == target.stable_id
            relevant = relevant or bool(target.view_id and getattr(event, "view_id", None) == target.view_id)
            if not relevant:
                continue
            rows.append(
                (
                    str(getattr(event, "event_id", "")),
                    str(getattr(event, "predicate", "")),
                    str(getattr(event, "polarity", "")),
                    str(getattr(event, "view_id", "") or ""),
                )
            )
        return stable_digest("target-evidence", rows)

    def _action_signature(
        self,
        name: str,
        args: dict[str, Any],
        *,
        out: dict[str, Any] | None = None,
    ) -> ActionSignature:
        tool = str(name or "").strip().lower()
        result = dict(out or {})
        intent = str(
            args.get("phrase")
            or args.get("toward")
            or getattr(self, "_target_phrase", "")
            or self.query_text
            or self.question
        )
        if tool in {"investigate", "navigate_to_obs"}:
            raw_obs = args.get("obs_id", result.get("obs_id", -1))
            try:
                obs_id = int(raw_obs)
            except (TypeError, ValueError):
                obs_id = -1
            target = self._action_target_for_obs(obs_id)
            raw_approach = result.get("approach_index")
            if raw_approach is None and obs_id >= 0:
                requested = args.get("approach_index", args.get("approach"))
                try:
                    preferred = int(requested) if requested is not None else None
                except (TypeError, ValueError):
                    preferred = None
                if preferred is not None and self.action_progress_mode == "enforce":
                    raw_approach = preferred % PLACE_APPROACH_SAMPLES
                else:
                    raw_approach = self._next_approach_index(obs_id, prefer=preferred)
            return ActionSignature.build(
                tool_name=tool,
                family="inspect_place",
                intent=intent,
                target=target,
                variant={"approach_index": raw_approach},
            )
        if tool == "verify_siglip":
            raw_obs = args.get("obs_id", result.get("obs_id"))
            if raw_obs is None:
                raw_obs = self._latest_obs_id()
            target = self._action_target_for_obs(int(raw_obs) if raw_obs is not None else -1)
            phrase = str(args.get("phrase") or getattr(self, "_target_phrase", "") or self.query_text)
            return ActionSignature.build(
                tool_name=tool,
                family="verify_view",
                intent=phrase,
                target=target,
                variant={
                    "view_id": target.view_id or f"obs:{target.adapter_id}",
                    "verifier_profile": "siglip+vlm",
                },
            )
        if tool == "explore_frontier":
            frontier_id = str(args.get("frontier_id") or result.get("frontier_id") or "")
            target = self._action_target_for_frontier(frontier_id)
            goal_xyz = result.get("frontier_xyz") or target.xyz
            frontier_intent = str(getattr(self, "_target_phrase", "") or self.query_text or self.question)
            return ActionSignature.build(
                tool_name=tool,
                family="explore_frontier",
                intent=intent,
                target=target,
                # ``toward`` is useful display context but only a weak navigation
                # hint: rewording it must not bypass the same static frontier work.
                work_intent=frontier_intent,
                variant={
                    "frontier_geometry_id": self._frontier_geometry_id(frontier_id),
                    "goal_cell": quantized_xy(goal_xyz),
                },
            )
        pose = self._robot_xyt_world()
        pose_cell = quantized_xy(pose)
        if tool == "look_around":
            target = ActionTarget(kind="pose", stable_id=f"pose:{pose_cell}", xyz=None)
            return ActionSignature.build(
                tool_name=tool,
                family="scan_view",
                intent=intent,
                target=target,
                variant={"pose_cell": pose_cell, "sensor_profile": "head-rgb"},
            )
        if tool == "submit_answer":
            target = ActionTarget(kind="question", stable_id=self._question_id)
            return ActionSignature.build(
                tool_name=tool,
                family="submit_answer",
                intent=self.question,
                target=target,
                variant={"answer": str(args.get("answer") or result.get("final_answer") or "")},
            )
        target = ActionTarget(kind="episode", stable_id=self._question_id or self._session_id)
        return ActionSignature.build(
            tool_name=tool,
            family="finish" if tool == "finish" else tool,
            intent=intent,
            target=target,
            variant={},
        )

    def _action_progress_token(self, signature: ActionSignature) -> ProgressToken:
        target = signature.target
        components: dict[str, Any] = {
            "family": signature.family,
            "robot_pose_cell": quantized_xy(self._robot_xyt_world()),
        }
        if signature.family in {"inspect_place", "verify_view"}:
            current = self._action_target_for_obs(target.adapter_id) if target.adapter_id is not None else target
            components.update(
                {
                    "target_id": current.stable_id,
                    "view_id": current.view_id,
                    "view_revision": current.revision,
                    "relevant_evidence_digest": self._relevant_evidence_digest(current),
                }
            )
            rec = self._place_inspect.get(int(target.adapter_id)) if target.adapter_id is not None else None
            if rec is not None:
                components["coverage"] = str(rec.coverage)
                components["local_frontier_cells"] = int(rec.local_frontier_cells)
        elif signature.family == "explore_frontier":
            components.update(
                {
                    "target_id": target.stable_id,
                    "frontier_geometry_id": self._frontier_geometry_id(target.stable_id),
                    "relevant_evidence_digest": self._relevant_evidence_digest(target),
                }
            )
        return ProgressToken.build(components)

    @staticmethod
    def _action_progress_reasons(
        signature: ActionSignature,
        before: ProgressToken,
        after: ProgressToken,
        out: dict[str, Any],
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        for key, reason in (
            ("view_id", "new_view"),
            ("view_revision", "view_revision"),
            ("relevant_evidence_digest", "target_evidence"),
            ("coverage", "coverage"),
            ("local_frontier_cells", "local_geometry"),
            ("frontier_geometry_id", "frontier_geometry"),
        ):
            if before.value(key) != after.value(key):
                reasons.append(reason)
        pose_changed = before.value("robot_pose_cell") != after.value("robot_pose_cell")
        partial_nav = bool(out.get("nav_progress")) and not bool(out.get("nav_finished"))
        if pose_changed and (signature.family == "explore_frontier" or partial_nav):
            reasons.append("motion")
        capture = out.get("capture")
        capture_status = str(capture.get("status") or "") if isinstance(capture, dict) else ""
        if capture_status in {"NEW_OBS", "CONTENT_REFRESHED"} and "new_view" not in reasons:
            reasons.append("new_view")
        return tuple(dict.fromkeys(reasons))

    def _record_action_history(
        self,
        name: str,
        args: dict[str, Any],
        out: dict[str, Any],
        *,
        signature: ActionSignature,
        progress_before: ProgressToken,
    ) -> None:
        if name in {"inspect_graph", "capture_and_update"}:
            return
        if signature.family == "inspect_place" and out.get("ok"):
            self._n_consecutive_explore = 0
        progress_after = self._action_progress_token(signature)
        progress_reasons = self._action_progress_reasons(
            signature,
            progress_before,
            progress_after,
            out,
        )
        verify = out.get("verify")
        verify_status = ""
        if isinstance(verify, dict):
            verify_status = str(verify.get("status") or verify.get("decision") or "")
        elif name == "verify_siglip":
            verify_status = str(out.get("status") or out.get("decision") or "")
        capture = out.get("capture")
        capture_status = str(capture.get("status") or "") if isinstance(capture, dict) else ""
        status = str(
            out.get("status")
            or out.get("nav_status_code")
            or out.get("error")
            or verify_status
            or ("ok" if out.get("ok") else "failed")
        )
        closest = None
        adapter = signature.target.adapter_id
        rec = self._place_inspect.get(int(adapter)) if adapter is not None else None
        if rec is not None and rec.closest_m is not None:
            closest = float(rec.closest_m)
        entry = ActionHistoryEntry(
            schema_version=1,
            round_index=int(self._round) + 1,
            selected_by=str(self._action_selected_by or "internal"),
            signature=signature,
            progress_before=progress_before,
            progress_after=progress_after,
            outcome_class=status_outcome_class(
                family=signature.family,
                ok=bool(out.get("ok")),
                status=verify_status or status,
                progress_reasons=progress_reasons,
            ),
            status=status[:120],
            ok=bool(out.get("ok")),
            progress_reasons=progress_reasons,
            closest_m=closest,
            capture_status=capture_status,
            verify_status=verify_status,
            nav_outcome=str(out.get("nav_outcome") or "")[:80],
        )
        self._action_history.append(entry)
        self._action_history = self._action_history[-32:]
        line = render_history_entry(entry)
        self._recent_actions.append(line)
        self._recent_actions = self._recent_actions[-RECENT_ACTIONS_K:]
        self._append_trace({"event": "action_history", "entry": entry.to_dict()})

    def _inspect_action_gate_decision(
        self,
        name: str,
        args: dict[str, Any],
    ) -> GateDecision:
        try:
            obs_id = int(args.get("obs_id"))
        except (TypeError, ValueError):
            signature = self._action_signature(name, args)
            progress = self._action_progress_token(signature)
            return decide_candidate(self._action_history, signature, progress)

        requested = args.get("approach_index", args.get("approach"))
        try:
            preferred = int(requested) % PLACE_APPROACH_SAMPLES if requested is not None else None
        except (TypeError, ValueError):
            preferred = None

        rec = self._place_inspect.get(obs_id)
        tried = [int(item) % PLACE_APPROACH_SAMPLES for item in (rec.tried_approaches if rec is not None else ())]
        probe = self._action_signature(name, args, out={"approach_index": 0})
        current_progress = self._action_progress_token(probe)
        continuation: list[int] = []
        if preferred is None:
            for entry in reversed(self._action_history):
                if (
                    entry.signature.work_key == probe.work_key
                    and entry.outcome_class == "progress"
                    and "motion" in entry.progress_reasons
                    and entry.progress_after.digest == current_progress.digest
                ):
                    approach = entry.signature.variant_value("approach_index")
                    if approach is not None:
                        continuation.append(int(approach) % PLACE_APPROACH_SAMPLES)
                    break

        if preferred is not None:
            order = [preferred]
        else:
            order = [
                *continuation,
                *(index for index in range(PLACE_APPROACH_SAMPLES) if index not in tried),
                *tried,
            ]
        order = list(dict.fromkeys(order))
        decisions: list[GateDecision] = []
        for approach in order:
            signature = self._action_signature(name, args, out={"approach_index": approach})
            progress = self._action_progress_token(signature)
            decision = decide_candidate(self._action_history, signature, progress)
            decisions.append(decision)
            if decision.allowed:
                return decision

        if preferred is not None and decisions:
            return decisions[0]
        prior_rounds = tuple(
            dict.fromkeys(round_index for decision in decisions for round_index in decision.prior_rounds)
        )
        base = (
            decisions[0]
            if decisions
            else decide_candidate(
                self._action_history,
                probe,
                current_progress,
            )
        )
        return GateDecision(
            allowed=False,
            disposition="would_suppress_saturated",
            reason="all finite approach variants are temporarily ineligible for the unchanged place state",
            signature=base.signature,
            progress=base.progress,
            prior_rounds=prior_rounds[-4:],
        )

    def _action_gate_decision(self, name: str, args: dict[str, Any]) -> GateDecision:
        tool = str(name or "").strip().lower()
        if tool in {"investigate", "navigate_to_obs"}:
            return self._inspect_action_gate_decision(tool, args)
        signature = self._action_signature(tool, args)
        progress = self._action_progress_token(signature)
        return decide_candidate(self._action_history, signature, progress)

    def _prepare_action_progress_dispatch(
        self,
        name: str,
        args: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if self.action_progress_mode not in {"shadow", "enforce"} or name not in {
            "investigate",
            "navigate_to_obs",
            "verify_siglip",
            "explore_frontier",
        }:
            return args, None

        prepared = dict(args)
        if (
            self.action_progress_mode == "enforce"
            and name == "explore_frontier"
            and not str(prepared.get("frontier_id") or "").strip()
        ):
            frontier_ids = tuple(self._rendered_action_allowlist().get("frontier_ids", ()))
            if not frontier_ids:
                self._append_trace(
                    {
                        "event": "action_gate_dispatch",
                        "mode": self.action_progress_mode,
                        "tool": name,
                        "allowed": False,
                        "disposition": "no_eligible_frontier",
                    }
                )
                return prepared, {
                    "ok": False,
                    "status": "NO_ELIGIBLE_ACTION",
                    "error": "no eligible rendered frontier remains while static progress gating is enforced",
                }
            prepared["frontier_id"] = str(frontier_ids[0])

        decision = self._action_gate_decision(name, prepared)
        self._append_trace(
            {
                "event": "action_gate_dispatch",
                "mode": self.action_progress_mode,
                "tool": name,
                "decision": decision.to_dict(),
            }
        )
        if self.action_progress_mode == "shadow":
            return args, None
        if not decision.allowed:
            return prepared, {
                "ok": False,
                "status": "ACTION_PROGRESS_SUPPRESSED",
                "disposition": decision.disposition,
                "target_id": decision.signature.target.stable_id,
                "error": decision.reason,
            }
        if decision.signature.family == "inspect_place":
            approach = decision.signature.variant_value("approach_index")
            if approach is not None:
                prepared["approach_index"] = int(approach)
        return prepared, None

    def _record_recent_action(
        self,
        name: str,
        args: dict[str, Any],
        out: dict[str, Any],
    ) -> None:
        """Compatibility helper for tests and callers with a completed result."""
        signature = self._action_signature(name, dict(args or {}), out=out)
        self._record_action_history(
            name,
            dict(args or {}),
            dict(out or {}),
            signature=signature,
            progress_before=self._action_progress_token(signature),
        )

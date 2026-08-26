# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Room clusters, VLM room stamps, attempt ledger, and EQA history."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from emet.memory.graph_eqa.attempt_ledger import (
    AttemptRecord,
    AttemptSource,
    infer_nav_outcome,
    infer_nav_status_code,
    records_from_dicts,
    records_to_dicts,
)
from emet.memory.graph_eqa.graph_types import GraphNode


class GraphRoomsMixin:
    """Room clusters, VLM room stamps, attempt ledger, and EQA history."""

    def _room_link_radius_m(self) -> float:
        env = os.environ.get("EMET_EQA_ROOM_LINK_RADIUS_M", "").strip()
        if env:
            try:
                return float(env)
            except ValueError:
                pass
        try:
            return float(self._eqa_cfg_value("room_link_radius_m", 2.0))
        except Exception:
            return 2.0

    def _room_assign_max_m(self) -> float:
        env = os.environ.get("EMET_EQA_ROOM_ASSIGN_MAX_M", "").strip()
        if env:
            try:
                return float(env)
            except ValueError:
                pass
        try:
            return float(self._eqa_cfg_value("room_assign_max_m", 3.0))
        except Exception:
            return 3.0

    def set_room_connectivity_checker(
        self,
        checker: Callable[[tuple[float, float], tuple[float, float]], bool] | None,
    ) -> None:
        """Set a free-space line/connectivity gate used by room clustering."""
        self._room_connectivity_fn = checker

    def refresh_room_clusters(self) -> list[Any]:
        """Recompute near+planar connected components over object nodes."""
        from emet.memory.graph_eqa.room_clusters import cluster_object_nodes

        clusters = cluster_object_nodes(
            self._nodes,
            self._edges,
            link_radius_m=self._room_link_radius_m(),
            connectivity_fn=self._room_connectivity_fn,
        )
        if self.world_evidence.enabled:
            self._reindex_world_entities()
            node_to_place: dict[int, str] = {}
            for node in self._nodes:
                entity = self.world_evidence.entity_for_node(int(node.node_id))
                if entity is not None:
                    node_to_place[int(node.node_id)] = entity.place_id
            hypotheses = self.world_evidence.update_room_hypotheses(
                list(clusters),
                node_to_place=node_to_place,
                step=self._effective_timestep(),
            )
            clusters = [
                replace(
                    cluster,
                    room_id=hypothesis.room_id,
                    room_name=(hypothesis.room_name if hypothesis.room_name != "unknown" else cluster.room_name),
                )
                for cluster, hypothesis in zip(clusters, hypotheses, strict=True)
            ]
        self._room_clusters = list(clusters)
        self.last_room_clusters = list(clusters)
        return self._room_clusters

    def graph_room_at_robot(self, robot_xy: Any) -> str:
        """Nearest graph room-cluster label at ``robot_xy``, or ``unknown``."""
        from emet.memory.graph_eqa.room_clusters import estimate_room_at_xy

        if not self._room_clusters:
            self.refresh_room_clusters()
        if robot_xy is None:
            return "unknown"
        try:
            xy = (float(robot_xy[0]), float(robot_xy[1]))
        except Exception:
            return "unknown"
        return estimate_room_at_xy(
            self._room_clusters,
            xy,
            max_dist_m=self._room_assign_max_m(),
        )

    def format_rooms_line(self, *, max_chars: int = 200) -> str:
        """Compact ``Rooms: …`` summary for router / memory prompts."""
        from emet.memory.graph_eqa.room_clusters import format_rooms_compact

        if not self._room_clusters:
            self.refresh_room_clusters()
        return format_rooms_compact(self._room_clusters, max_chars=max_chars)

    def stamp_vlm_room_at_robot(
        self,
        robot_xy: Any,
        room: str | None,
        *,
        protect_indoor_from_outdoor: bool = True,
        corroborating_labels: list[str] | tuple[str, ...] | None = None,
        source: str = "router_vlm",
        source_view_id: str | None = None,
        agent_round: int | None = None,
        pose_round: int | None = None,
    ) -> str:
        """Stamp VLM ``current_room`` onto the nearest cluster; return stamped name or unknown.

        ``room`` should already be policy-coerced (canonical bucket or free-text phrase).
        By default refuses outdoor overwriting a named indoor cluster without corroboration.
        Returns ``unknown`` when the stamp is skipped/blocked.
        """
        from emet.memory.graph_eqa.agentic_tools import sanitize_room_phrase
        from emet.memory.graph_eqa.room_clusters import estimate_room_at_xy, stamp_room_at_xy

        name = sanitize_room_phrase(room)
        if name == "unknown" or robot_xy is None:
            return "unknown"
        if not self._room_clusters:
            self.refresh_room_clusters()
        try:
            xy = (float(robot_xy[0]), float(robot_xy[1]))
        except Exception:
            return "unknown"
        prev = estimate_room_at_xy(
            self._room_clusters,
            xy,
            max_dist_m=self._room_assign_max_m(),
        )
        labs = [str(x) for x in (corroborating_labels or ()) if str(x).strip()] or None
        self._room_clusters = stamp_room_at_xy(
            self._room_clusters,
            xy,
            name,
            max_dist_m=self._room_assign_max_m(),
            protect_indoor_from_outdoor=bool(protect_indoor_from_outdoor),
            corroborating_labels=labs,
        )
        self.last_room_clusters = list(self._room_clusters)
        after = estimate_room_at_xy(
            self._room_clusters,
            xy,
            max_dist_m=self._room_assign_max_m(),
        )
        after_s = sanitize_room_phrase(after)
        if after_s != name:
            # Protection blocked the write (or nearest cluster out of range).
            return "unknown" if sanitize_room_phrase(prev) != name else after_s
        if self.world_evidence.enabled:
            nearest = min(
                self._room_clusters,
                key=lambda cluster: float(
                    (cluster.centroid_xy[0] - xy[0]) ** 2 + (cluster.centroid_xy[1] - xy[1]) ** 2
                ),
                default=None,
            )
            if nearest is not None and nearest.room_id:
                confidence = 0.75 if source == "investigate_vlm" else 0.65
                hypothesis = self.world_evidence.stamp_room(
                    nearest.room_id,
                    name,
                    source=source,
                    confidence=confidence,
                    step=self._effective_timestep(),
                    view_id=source_view_id,
                    agent_round=agent_round,
                    pose_round=pose_round,
                )
                if hypothesis is not None:
                    self._room_clusters = [
                        replace(cluster, room_name=hypothesis.room_name)
                        if cluster.room_id == hypothesis.room_id
                        else cluster
                        for cluster in self._room_clusters
                    ]
                    self.last_room_clusters = list(self._room_clusters)
                    if hypothesis.room_name != name:
                        return "unknown"
        return name

    def nearby_object_observations(
        self,
        robot_xy: Any,
        *,
        k: int = 3,
        max_dist_m: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Nearest object observations with RGB for multimodal router room context."""
        if robot_xy is None or k <= 0:
            return []
        try:
            rxy = np.asarray(robot_xy, dtype=float).reshape(-1)[:2]
        except Exception:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for obs in list(getattr(self, "_observations", None) or []):
            rgb = getattr(obs, "rgb", None)
            if not isinstance(rgb, np.ndarray) or rgb.ndim != 3:
                continue
            labels = [str(x).strip() for x in list(getattr(obs, "labels", None) or []) if str(x).strip()]
            if any(lab.lower() == "frontier" for lab in labels) and len(labels) <= 1:
                continue
            xyz = getattr(obs, "xyz", None)
            if xyz is None:
                continue
            try:
                oxy = np.asarray(xyz, dtype=float).reshape(-1)[:2]
                dist = float(np.linalg.norm(oxy - rxy))
            except Exception:
                continue
            if dist > float(max_dist_m):
                continue
            phrase = labels[0] if labels else f"obs_{int(obs.obs_id)}"
            scored.append(
                (
                    dist,
                    {
                        "obs_id": int(obs.obs_id),
                        "dist_m": round(dist, 2),
                        "labels": labels[:6],
                        "phrase": phrase,
                        "rgb": np.asarray(rgb),
                    },
                )
            )
        scored.sort(key=lambda t: t[0])
        return [item for _, item in scored[: int(k)]]

    def _node_nav_status_suffix(self, node: GraphNode) -> str:
        failures = int(getattr(node, "nav_failures", 0) or 0)
        if failures <= 0:
            return ""
        note = (getattr(node, "last_nav_note", None) or "").strip()
        tail = f", last: {note}" if note else ""
        return f"; unreachable ({failures} nav failure(s){tail})"

    def _attempt_ledger_enabled(self) -> bool:
        """True when ``eqa.attempt_ledger`` / ``EMET_EQA_ATTEMPT_LEDGER`` is on (default off)."""
        env = os.environ.get("EMET_EQA_ATTEMPT_LEDGER", "").strip().lower()
        if env in ("1", "true", "yes", "on"):
            return True
        if env in ("0", "false", "no", "off"):
            return False
        raw = self._eqa_cfg_value("attempt_ledger", False)
        if isinstance(raw, dict):
            raw = raw.get("enabled", False)
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)

    def set_attempt_ledger_question_id(self, question_id: str | None) -> None:
        """Tag subsequent ledger rows with a question/episode id (does not clear the store)."""
        self._attempt_ledger_question_id = str(question_id) if question_id else None
        self.world_evidence.set_question_id(self._attempt_ledger_question_id)

    def record_attempt(
        self,
        *,
        action_kind: str,
        outcome: str,
        status_code: str,
        note: str = "",
        step: int | None = None,
        target_node_id: int | None = None,
        obs_id: int | None = None,
        xyz: tuple[float, float, float] | None = None,
        source: AttemptSource | str = "unknown",
        question_id: str | None = None,
        phrase: str = "",
        room: str = "",
        target_kind: str = "",
        target_id: str = "",
        view_id: str = "",
        force: bool = False,
    ) -> AttemptRecord | None:
        """Append one :class:`AttemptRecord` when the ledger is enabled (or ``force``).

        Returns the stored record, or ``None`` when the ledger is off.
        """
        st = int(step if step is not None else self._effective_timestep())
        qid = question_id if question_id is not None else self._attempt_ledger_question_id
        stable_kind = str(target_kind or "")
        stable_id = str(target_id or "")
        stable_view = str(view_id or "")
        if target_node_id is not None and not stable_id:
            entity = self.world_evidence.entity_for_node(int(target_node_id))
            if entity is not None:
                stable_kind = "place"
                stable_id = entity.place_id
        if obs_id is not None and not stable_view:
            stable_view = self.view_id_for_obs(int(obs_id))
        if not stable_id and stable_view:
            view = self.world_evidence.views.get(stable_view)
            if view is not None and view.place_id:
                stable_kind = "place"
                stable_id = view.place_id
        ledger_enabled = bool(force or self._attempt_ledger_enabled())
        if self.world_evidence.enabled:
            subject_kind = stable_kind or ("observation" if obs_id is not None else "action")
            subject_id = stable_id or (str(obs_id) if obs_id is not None else str(action_kind))
            self.world_evidence.record_event(
                subject_kind=subject_kind,
                subject_id=subject_id,
                predicate=str(action_kind),
                polarity="positive" if str(outcome) in {"ok", "present"} else "negative",
                source=str(source or "unknown"),
                confidence=1.0,
                step=st,
                view_id=stable_view or None,
                place_id=stable_id if stable_kind == "place" else None,
                payload={
                    "outcome": str(outcome),
                    "status_code": str(status_code),
                    "note": str(note)[:240],
                    "legacy_obs_id": obs_id,
                    "legacy_node_id": target_node_id,
                },
            )
        if not ledger_enabled:
            return None
        src = str(source or "unknown")
        if src not in ("chat", "eqa", "unknown"):
            src = "unknown"
        rec = AttemptRecord.from_dict(
            {
                "action_kind": action_kind,
                "outcome": outcome,
                "status_code": status_code,
                "note": note,
                "step": st,
                "target_node_id": target_node_id,
                "obs_id": obs_id,
                "xyz": list(xyz) if xyz is not None else None,
                "source": src,
                "question_id": qid,
                "phrase": phrase,
                "room": room,
                "target_kind": stable_kind,
                "target_id": stable_id,
                "view_id": stable_view,
            }
        )
        self._attempt_records.append(rec)
        max_n = max(1, int(self._attempt_ledger_max))
        if len(self._attempt_records) > max_n:
            self._attempt_records = self._attempt_records[-max_n:]
        return rec

    def clear_room_events(self) -> None:
        """Drop the room timeline (new episode / world-change)."""
        self._room_events = []

    def record_room_event(
        self,
        *,
        room: str | None,
        kind: str,
        step: int | None = None,
        phrase: str = "",
        obs_id: int | None = None,
        note: str = "",
        agent_round: int | None = None,
    ) -> dict[str, Any] | None:
        """Append a room-scoped timeline event when ``room`` is a known label.

        Does **not** invent ``unknown``. Survives room-cluster rebuilds. Independent
        of the attempt-ledger opt-in so agentic state can still show history.
        """
        from emet.memory.graph_eqa.room_clusters import normalize_current_room

        room_n = normalize_current_room(room)
        if room_n == "unknown":
            return None
        kind_s = str(kind or "").strip().lower()
        if not kind_s:
            return None
        st = int(step if step is not None else self._effective_timestep())
        event = {
            "step": st,
            "room": room_n,
            "kind": kind_s[:40],
            "phrase": str(phrase or "").strip().lower()[:80],
            "obs_id": int(obs_id) if obs_id is not None else None,
            "note": str(note or "").strip()[:120],
            "world_step": st,
            "agent_round": int(agent_round) if agent_round is not None else None,
        }
        world_event = self.world_evidence.record_event(
            subject_kind="room",
            subject_id=room_n,
            predicate=kind_s,
            polarity="negative" if "absent" in kind_s else "positive",
            source="room_timeline",
            confidence=0.7,
            step=st,
            view_id=self.view_id_for_obs(int(obs_id)) if obs_id is not None else None,
            payload={
                "phrase": str(phrase or "").strip()[:120],
                "legacy_obs_id": int(obs_id) if obs_id is not None else None,
                "note": str(note or "").strip()[:120],
                "agent_round": int(agent_round) if agent_round is not None else None,
            },
        )
        if world_event is not None:
            event["event_id"] = world_event.event_id
        self._room_events.append(event)
        max_n = max(8, int(self._room_events_max))
        if len(self._room_events) > max_n:
            self._room_events = self._room_events[-max_n:]
        return dict(event)

    def get_room_events(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return room timeline rows (oldest first)."""
        rows = [dict(e) for e in self._room_events]
        if limit is not None and int(limit) >= 0:
            rows = rows[-int(limit) :]
        return rows

    def format_room_history(
        self,
        *,
        max_chars: int = 220,
        target_rooms: list[str] | set[str] | None = None,
    ) -> str:
        """Newest-first compact room timeline for the agentic state card."""
        if not self._room_events:
            targets = sorted({str(t).strip().lower() for t in (target_rooms or []) if str(t).strip()})
            if not targets:
                return ""
            return "Room history: (none) | targets=" + ",".join(targets)
        bits: list[str] = []
        for ev in reversed(self._room_events):
            step = int(ev.get("step") or 0)
            room = str(ev.get("room") or "").strip() or "?"
            kind = str(ev.get("kind") or "").strip() or "?"
            phrase = str(ev.get("phrase") or "").strip()
            bit = f"r{step} {room} {kind}"
            if phrase and kind.startswith("verify"):
                bit += f"({phrase})"
            bits.append(bit)
        targets = sorted({str(t).strip().lower() for t in (target_rooms or []) if str(t).strip()})
        prefix = "Room history: "
        tail = (" | targets=" + ",".join(targets)) if targets else ""
        # Drop oldest (end of newest-first list) until under budget.
        while bits:
            body = "; ".join(bits)
            text = prefix + body + tail
            if len(text) <= int(max_chars):
                return text
            bits = bits[:-1]
        return (prefix + "(truncated)" + tail)[: int(max_chars)]

    def get_attempt_records(
        self,
        *,
        obs_id: int | None = None,
        action_kind: str | None = None,
        target_node_id: int | None = None,
        question_id: str | None = None,
        limit: int | None = None,
    ) -> list[AttemptRecord]:
        """Return ledger rows matching optional filters (oldest first)."""
        rows = list(self._attempt_records)
        if obs_id is not None:
            oid = int(obs_id)
            rows = [r for r in rows if r.obs_id is not None and int(r.obs_id) == oid]
        if action_kind is not None:
            kind = str(action_kind)
            rows = [r for r in rows if r.action_kind == kind]
        if target_node_id is not None:
            nid = int(target_node_id)
            rows = [r for r in rows if r.target_node_id is not None and int(r.target_node_id) == nid]
        if question_id is not None:
            qid = str(question_id)
            rows = [r for r in rows if r.question_id == qid]
        if limit is not None and int(limit) >= 0:
            rows = rows[-int(limit) :]
        return rows

    def export_attempt_ledger(self) -> list[dict[str, Any]]:
        """JSON-serializable snapshot of the attempt ledger."""
        return records_to_dicts(self._attempt_records)

    def import_attempt_ledger(self, items: list[Any], *, replace: bool = True) -> int:
        """Load ledger rows from dicts (or :class:`AttemptRecord`). Returns count loaded."""
        loaded = records_from_dicts(list(items or []))
        if replace:
            self._attempt_records = loaded
        else:
            self._attempt_records.extend(loaded)
        max_n = max(1, int(self._attempt_ledger_max))
        if len(self._attempt_records) > max_n:
            self._attempt_records = self._attempt_records[-max_n:]
        return len(loaded)

    def clear_attempt_ledger(self) -> None:
        self._attempt_records.clear()

    def derive_nav_counters_from_ledger(self, obs_id: int) -> tuple[int, int, str | None, int]:
        """Compute ``(attempts, failures, last_note, last_step)`` for ``obs_id`` from the ledger."""
        rows = self.get_attempt_records(obs_id=obs_id, action_kind="navigate")
        if not rows:
            return 0, 0, None, 0
        failures = sum(1 for r in rows if r.outcome != "ok")
        last = rows[-1]
        note = (last.note or last.status_code or "").strip() or None
        return len(rows), failures, note, int(last.step)

    def record_nav_attempt(
        self,
        obs_id: int | None,
        *,
        success: bool,
        note: str,
        dist_m: float = 0.0,
        step: int | None = None,
        status_code: str | None = None,
        source: AttemptSource | str = "eqa",
        question_id: str | None = None,
        target_node_id: int | None = None,
    ) -> None:
        """Update graph node(s) tied to ``obs_id`` after an EQA navigation attempt.

        When the attempt ledger is enabled, also append a ``navigate``
        :class:`AttemptRecord`. Node ``nav_attempts`` / ``nav_failures`` counters
        remain dual-written for compatibility.
        """
        if obs_id is None:
            self.last_nav_result_note = note
            return
        oid = int(obs_id)
        self._obs_nav_dists.setdefault(oid, []).append(float(dist_m))
        st = int(step if step is not None else self._effective_timestep())
        moved = float(dist_m) >= 0.12
        ok = bool(success) and moved
        matched_node_id = target_node_id
        xyz_t: tuple[float, float, float] | None = None
        for idx, node in enumerate(self._nodes):
            if int(node.obs_id) != oid:
                continue
            if matched_node_id is None:
                matched_node_id = int(node.node_id)
            if xyz_t is None:
                try:
                    arr = np.asarray(node.xyz, dtype=float).reshape(-1)
                    if arr.size >= 3:
                        xyz_t = (float(arr[0]), float(arr[1]), float(arr[2]))
                except Exception:
                    xyz_t = None
            failures = int(getattr(node, "nav_failures", 0)) + (0 if ok else 1)
            self._nodes[idx] = replace(
                node,
                nav_attempts=int(getattr(node, "nav_attempts", 0)) + 1,
                nav_failures=failures,
                last_nav_note=str(note or "")[:120] or None,
                last_nav_at_step=st,
            )
        code = status_code or infer_nav_status_code(success=ok, note=str(note or ""))
        outcome = infer_nav_outcome(success=ok, status_code=code)
        self.record_attempt(
            action_kind="navigate",
            outcome=outcome,
            status_code=code,
            note=str(note or "")[:240],
            step=st,
            target_node_id=matched_node_id,
            obs_id=oid,
            xyz=xyz_t,
            source=source,
            question_id=question_id,
        )
        self.last_nav_result_note = note

    @staticmethod
    def strip_caption_block_from_history(text: str) -> str:
        """Drop a leading ``Caption:`` block so HISTORY cannot reinforce caption loops."""
        if not text:
            return text
        return re.sub(
            r"(?is)^\s*Caption:\s*.*?(?=\n\s*(?:Reasoning|Answer)\s*:|\Z)",
            "",
            text,
            count=1,
        ).lstrip()

    def _append_eqa_history(self, text: str) -> None:
        self._history_outputs.append(self.strip_caption_block_from_history(text))

    def append_nav_outcome_to_last_history(self, *, dist_m: float, success: bool, note: str) -> None:
        if not self._history_outputs:
            return
        status = "ok" if success else "failed"
        self._history_outputs[-1] += f"\nNav_result: moved {float(dist_m):.2f}m ({status}; {note})"

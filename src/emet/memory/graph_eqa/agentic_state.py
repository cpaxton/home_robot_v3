# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Pure, bounded state compilation for the grounded GraphEQA router."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PlaceState:
    place_id: str
    obs_adapter_id: int
    labels: tuple[str, ...]
    room_id: str | None
    room_name: str
    xyz: tuple[float, float, float]
    source: str
    path_cost: float | None
    attempts: tuple[str, ...]
    approach_bearings: tuple[float, ...]
    coverage: str
    coverage_gain: float
    information_gain: float
    revisit_change: str
    failure_risk: float


@dataclass(frozen=True)
class FrontierState:
    frontier_id: str
    obs_adapter_id: int | None
    xyz: tuple[float, float, float]
    status: str
    cell_count: int
    attachment_ids: tuple[str, ...]
    parent_ids: tuple[str, ...]
    information_gain: float
    path_cost: float | None


@dataclass(frozen=True)
class RoomState:
    room_id: str
    room_name: str
    confidence: float
    sources: tuple[str, ...]
    member_place_ids: tuple[str, ...]
    centroid_xy: tuple[float, float]


@dataclass(frozen=True)
class EvidenceState:
    event_id: str
    step: int
    subject_kind: str
    subject_id: str
    predicate: str
    polarity: str
    source: str
    confidence: float
    view_id: str | None
    labels: tuple[str, ...] = ()
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentStateSnapshot:
    schema_version: int
    question: str
    mode: str
    round_index: int
    max_rounds: int
    nav_used: int
    max_nav_steps: int
    robot_pose_world: tuple[float, float, float] | None
    current_room_graph: str
    current_room_router: str
    in_target_area: bool | None
    target_rooms: tuple[str, ...]
    verified: bool
    last_capture_status: str | None
    pending_answer: str | None = None
    pending_answer_obs_id: int | None = None
    pending_answer_present: bool | None = None
    places: tuple[PlaceState, ...] = ()
    frontiers: tuple[FrontierState, ...] = ()
    rooms: tuple[RoomState, ...] = ()
    evidence: tuple[EvidenceState, ...] = ()
    recent_actions: tuple[str, ...] = ()
    loop_flags: tuple[str, ...] = ()
    visible_event_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _xyz(value: Any) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    return (
        float(arr[0]) if arr.size > 0 else 0.0,
        float(arr[1]) if arr.size > 1 else 0.0,
        float(arr[2]) if arr.size > 2 else 0.0,
    )


def _attempt_bits(graph_memory: Any, obs_id: int) -> tuple[tuple[str, ...], float]:
    getter = getattr(graph_memory, "get_attempt_records", None)
    if not callable(getter):
        return (), 0.0
    rows = list(
        getter(
            obs_id=int(obs_id),
            question_id=getattr(graph_memory, "_attempt_ledger_question_id", None),
            limit=8,
        )
        or ()
    )
    bits = tuple(f"{row.action_kind}:{row.outcome}:{row.status_code}" for row in rows[-6:])
    nav = [row for row in rows if row.action_kind == "navigate"]
    failed = [row for row in nav if row.outcome != "ok"]
    risk = float(len(failed)) / float(len(nav)) if nav else 0.0
    return bits, risk


def _approach_bearings(record: Any, anchor_xyz: Any) -> tuple[float, ...]:
    anchor = np.asarray(anchor_xyz, dtype=float).reshape(-1)[:2]
    out: list[float] = []
    for xy in list(getattr(record, "tried_xy", None) or ()):
        point = np.asarray(xy, dtype=float).reshape(-1)[:2]
        if point.size < 2:
            continue
        bearing = round(float(np.degrees(np.arctan2(point[1] - anchor[1], point[0] - anchor[0]))), 1)
        if bearing not in out:
            out.append(bearing)
    return tuple(out)


def compile_agent_state(executor: Any, *, max_events: int = 16) -> AgentStateSnapshot:
    """Read executor/graph state without refreshing coverage, rooms, or evidence."""
    graph_memory = getattr(executor, "graph_memory", None)
    world = getattr(graph_memory, "world_evidence", None)
    graph_mode = str(getattr(executor, "graph_evidence_mode", "off") or "off")
    history_mode = str(getattr(executor, "room_history_mode", "off") or "off")
    attempt_mode = str(getattr(executor, "attempt_ledger_mode", "off") or "off")
    agent_graph = graph_mode == "agent" and world is not None
    agent_history = history_mode == "agent"
    agent_attempts = attempt_mode == "agent"

    pose = None
    pose_fn = getattr(executor, "_robot_xyt_world", None)
    if callable(pose_fn):
        raw_pose = pose_fn()
        if raw_pose is not None:
            pose = _xyz(raw_pose)

    places: list[PlaceState] = []
    hypotheses = list(getattr(executor, "_hypotheses", None) or ())
    inspect_ledger = dict(getattr(executor, "_place_inspect", None) or {})
    nodes = list(getattr(graph_memory, "_nodes", None) or ())
    node_by_obs = {
        int(node.obs_id): node
        for node in nodes
        if not bool(getattr(node, "is_frontier", False)) and not bool(getattr(node, "is_viewpoint", False))
    }
    for hypothesis in hypotheses:
        source = str(getattr(hypothesis, "source", ""))
        if source not in {"graph", "confirmed", "siglip"}:
            continue
        obs_id = int(hypothesis.obs_id)
        node = node_by_obs.get(obs_id)
        entity = world.entity_for_node(int(node.node_id)) if agent_graph and node is not None else None
        place_id = entity.place_id if entity is not None else f"obs:{obs_id}"
        place = world.places.get(place_id) if entity is not None else None
        room = world.rooms.get(place.room_id) if place is not None and place.room_id else None
        attempts, risk = _attempt_bits(graph_memory, obs_id) if agent_attempts else ((), 0.0)
        inspect = inspect_ledger.get(obs_id)
        approaches_left = int(getattr(inspect, "approaches_left", 4) if inspect is not None else 4)
        coverage = str(getattr(inspect, "coverage", "unknown") if inspect is not None else "unknown")
        local_cells = int(getattr(inspect, "local_frontier_cells", 0) if inspect is not None else 0)
        current_view = world.view_for_obs(obs_id) if agent_graph else None
        revision = int(current_view.revision) if current_view is not None else 0
        labels = tuple(str(x) for x in getattr(node, "labels", ()) or ())
        if not labels:
            labels = (str(getattr(hypothesis, "phrase", "") or "unknown"),)
        places.append(
            PlaceState(
                place_id=place_id,
                obs_adapter_id=obs_id,
                labels=labels,
                room_id=room.room_id if room is not None else None,
                room_name=room.room_name if room is not None else "unknown",
                xyz=_xyz(hypothesis.xyz),
                source=source,
                path_cost=(
                    float(hypothesis.path_cost)
                    if isinstance(getattr(hypothesis, "path_cost", None), (int, float))
                    else None
                ),
                attempts=attempts,
                approach_bearings=_approach_bearings(inspect, hypothesis.xyz) if inspect is not None else (),
                coverage=coverage,
                coverage_gain=float(local_cells),
                information_gain=float(max(0, approaches_left) + max(0, local_cells)),
                revisit_change=f"view_revision={revision}",
                failure_risk=risk,
            )
        )

    frontiers: list[FrontierState] = []
    if agent_graph:
        hyp_cost_by_obs = {
            int(hyp.obs_id): (
                float(hyp.path_cost) if isinstance(getattr(hyp, "path_cost", None), (int, float)) else None
            )
            for hyp in hypotheses
        }
        for record in world.frontiers.values():
            if record.status != "active":
                continue
            frontiers.append(
                FrontierState(
                    frontier_id=record.frontier_id,
                    obs_adapter_id=record.obs_id,
                    xyz=record.centroid_xyz,
                    status=record.status,
                    cell_count=len(record.cells),
                    attachment_ids=record.attachment_ids,
                    parent_ids=record.parent_ids,
                    information_gain=float(len(record.cells)),
                    path_cost=hyp_cost_by_obs.get(int(record.obs_id)) if record.obs_id is not None else None,
                )
            )
    else:
        for hypothesis in hypotheses:
            if str(getattr(hypothesis, "source", "")) in {"graph", "confirmed", "siglip"}:
                continue
            obs_id = int(hypothesis.obs_id)
            frontiers.append(
                FrontierState(
                    frontier_id=f"obs:{obs_id}",
                    obs_adapter_id=obs_id,
                    xyz=_xyz(hypothesis.xyz),
                    status="active",
                    cell_count=0,
                    attachment_ids=(),
                    parent_ids=(),
                    information_gain=1.0,
                    path_cost=(
                        float(hypothesis.path_cost)
                        if isinstance(getattr(hypothesis, "path_cost", None), (int, float))
                        else None
                    ),
                )
            )

    rooms: list[RoomState] = []
    if agent_graph:
        events_by_id = {event.event_id: event for event in world.events}
        for record in world.rooms.values():
            if not record.active:
                continue
            sources = tuple(
                dict.fromkeys(
                    events_by_id[event_id].source for event_id in record.evidence_event_ids if event_id in events_by_id
                )
            )
            rooms.append(
                RoomState(
                    room_id=record.room_id,
                    room_name=record.room_name,
                    confidence=float(record.confidence),
                    sources=sources,
                    member_place_ids=record.member_place_ids,
                    centroid_xy=record.centroid_xy,
                )
            )

    evidence: list[EvidenceState] = []
    if agent_history and world is not None:
        question_id = str(getattr(world, "question_id", "") or "")
        candidates = [event for event in world.events if not question_id or event.question_id in {None, question_id}]
        for event in candidates[-int(max_events) :]:
            payload = dict(event.payload or {})
            if event.polarity == "negative":
                qualified_negative = bool(payload.get("visibility_qualified", False))
                qualified_negative = qualified_negative or bool(payload.get("outcome"))
                qualified_negative = qualified_negative or event.source == "room_timeline"
                if not qualified_negative:
                    continue
            labels: tuple[str, ...] = ()
            if event.subject_kind == "entity":
                entity = world.entities.get(event.subject_id)
                labels = tuple(entity.labels) if entity is not None else ()
            elif event.subject_kind == "place":
                place = world.places.get(event.subject_id)
                entity = world.entities.get(place.entity_id) if place is not None else None
                labels = tuple(entity.labels) if entity is not None else ()
            details = tuple(
                f"{key}={str(payload[key]).strip()[:80]}"
                for key in ("phrase", "outcome", "status_code", "note")
                if payload.get(key) not in (None, "")
            )
            evidence.append(
                EvidenceState(
                    event_id=event.event_id,
                    step=int(event.step),
                    subject_kind=event.subject_kind,
                    subject_id=event.subject_id,
                    predicate=event.predicate,
                    polarity=event.polarity,
                    source=event.source,
                    confidence=float(event.confidence),
                    view_id=event.view_id,
                    labels=labels,
                    details=details,
                )
            )

    target_rooms: tuple[str, ...] = ()
    if bool(getattr(executor, "room_target_hints", False)):
        try:
            from emet.memory.graph_eqa.room_clusters import question_target_rooms

            target_rooms = tuple(sorted(question_target_rooms(str(getattr(executor, "question", "") or ""))))
        except ImportError:
            target_rooms = ()

    loop_flags = tuple(
        f"obs={item.get('obs_id')} visits={item.get('visits')} status={item.get('status')}"
        for item in list(getattr(executor, "_nav_loop_flags", None) or ())[-4:]
    )
    pending = dict(getattr(executor, "_pending_answerable", None) or {})
    pending_answer = str(pending.get("answer_text") or "").strip()
    if len(pending_answer) == 1 and pending_answer.upper() in "ABCDE":
        pending_answer = ""
    if not pending_answer and pending.get("letter"):
        choice_text = getattr(executor, "_choice_text_for_letter", None)
        if callable(choice_text):
            pending_answer = str(choice_text(pending.get("letter")) or "").strip()
    return AgentStateSnapshot(
        schema_version=1,
        question=str(getattr(executor, "question", "") or ""),
        mode=str(getattr(executor, "mode", "answer") or "answer"),
        round_index=int(getattr(executor, "_round", 0)) + 1,
        max_rounds=int(getattr(executor, "max_rounds", 0)),
        nav_used=int(getattr(executor, "_n_nav", 0)) + int(getattr(executor, "_n_explore", 0)),
        max_nav_steps=int(getattr(executor, "max_nav_steps", 0)),
        robot_pose_world=pose,
        current_room_graph=str(getattr(executor, "_graph_room_estimate", "") or "unknown"),
        current_room_router=str(getattr(executor, "_last_room_estimate", "") or "unknown"),
        in_target_area=getattr(executor, "_in_target_area", None),
        target_rooms=target_rooms,
        verified=bool(getattr(executor, "_verified", False)),
        last_capture_status=getattr(executor, "_last_capture_status", None),
        pending_answer=pending_answer or None,
        pending_answer_obs_id=(int(pending["obs_id"]) if pending.get("obs_id") is not None else None),
        pending_answer_present=(bool(pending["present"]) if pending.get("present") is not None else None),
        places=tuple(places),
        frontiers=tuple(frontiers),
        rooms=tuple(rooms),
        evidence=tuple(evidence),
        recent_actions=tuple(str(x) for x in list(getattr(executor, "_recent_actions", None) or ())[-8:]),
        loop_flags=loop_flags,
        visible_event_ids=tuple(item.event_id for item in evidence),
        metadata={
            "decision_policy": str(getattr(executor, "decision_policy", "legacy")),
            "graph_evidence_mode": graph_mode,
            "room_history_mode": history_mode,
            "attempt_ledger_mode": attempt_mode,
        },
    )


def render_agent_state(snapshot: AgentStateSnapshot, *, max_chars: int = 6000) -> str:
    """Render facts only; action semantics remain in the fixed system prompt."""
    lines = [
        f"Question: {snapshot.question}",
        f"State schema: grounded_agent_state/v{snapshot.schema_version}",
        (
            f"Budget: round={snapshot.round_index}/{snapshot.max_rounds} "
            f"nav={snapshot.nav_used}/{snapshot.max_nav_steps} verified={snapshot.verified}"
        ),
        f"Robot world pose: {snapshot.robot_pose_world}",
        (
            f"Room estimates: graph={snapshot.current_room_graph} "
            f"router={snapshot.current_room_router} in_target_area={snapshot.in_target_area}"
        ),
    ]
    if snapshot.target_rooms:
        lines.append("Question room hints: " + ",".join(snapshot.target_rooms))
    if snapshot.last_capture_status:
        lines.append(f"Last capture: {snapshot.last_capture_status}")
    if snapshot.pending_answer and not snapshot.verified:
        lines.append(
            f"Pending answer evidence: answer={snapshot.pending_answer} "
            f"obs_id={snapshot.pending_answer_obs_id} present={snapshot.pending_answer_present} "
            "(needs corroboration; do not submit yet)"
        )

    lines.append("Evidence history:")
    for event in snapshot.evidence:
        labels = ",".join(event.labels) or "unknown"
        details = ";".join(event.details) or "none"
        lines.append(
            f"- event_id={event.event_id} step={event.step} {event.polarity} "
            f"{event.subject_kind}:{event.subject_id} labels={labels} predicate={event.predicate} "
            f"source={event.source} confidence={event.confidence:.2f} "
            f"view_id={event.view_id or 'none'} details={details}"
        )
    if not snapshot.evidence:
        lines.append("- none")

    lines.append("Room hypotheses:")
    for room in snapshot.rooms:
        lines.append(
            f"- room_id={room.room_id} name={room.room_name} confidence={room.confidence:.2f} "
            f"sources={','.join(room.sources) or 'none'} places={','.join(room.member_place_ids)}"
        )
    if not snapshot.rooms:
        lines.append("- none")

    lines.append("Places:")
    for place in sorted(snapshot.places, key=lambda item: (-item.information_gain, item.failure_risk)):
        lines.append(
            f"- place_id={place.place_id} obs_adapter={place.obs_adapter_id} "
            f"labels={','.join(place.labels) or 'unknown'} room_id={place.room_id or 'unknown'} "
            f"room={place.room_name} xyz={place.xyz} source={place.source} "
            f"path_cost={place.path_cost} info_gain={place.information_gain:.1f} "
            f"coverage={place.coverage} coverage_gain={place.coverage_gain:.1f} "
            f"bearings={place.approach_bearings or 'none'} failure_risk={place.failure_risk:.2f} "
            f"attempts={place.attempts or 'none'} {place.revisit_change}"
        )
    if not snapshot.places:
        lines.append("- none")

    lines.append("Frontiers:")
    for frontier in sorted(snapshot.frontiers, key=lambda item: -item.information_gain):
        lines.append(
            f"- frontier_id={frontier.frontier_id} obs_adapter={frontier.obs_adapter_id} "
            f"xyz={frontier.xyz} status={frontier.status} cells={frontier.cell_count} "
            f"attachments={','.join(frontier.attachment_ids) or 'none'} "
            f"parents={','.join(frontier.parent_ids) or 'none'} "
            f"info_gain={frontier.information_gain:.1f} path_cost={frontier.path_cost}"
        )
    if not snapshot.frontiers:
        lines.append("- none")

    if snapshot.recent_actions:
        lines.append("Recent action outcomes: " + " | ".join(snapshot.recent_actions))
    if snapshot.loop_flags:
        lines.append("Loop observations: " + " | ".join(snapshot.loop_flags))

    text = "\n".join(lines)
    if len(text) <= int(max_chars):
        return text
    marker = "\n[state truncated to fixed prompt budget]"
    return text[: max(0, int(max_chars) - len(marker))].rstrip() + marker


def state_text_digest(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

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
    assertions: tuple[RoomAssertionState, ...] = ()
    conflict: bool = False


@dataclass(frozen=True)
class RoomAssertionState:
    source: str
    room_name: str
    confidence: float
    world_step: int
    agent_round: int | None
    pose_round: int | None
    stale: bool
    event_id: str | None = None
    view_id: str | None = None


@dataclass(frozen=True)
class RoomEventState:
    event_id: str | None
    world_step: int
    agent_round: int | None
    room_name: str
    kind: str
    phrase: str
    obs_adapter_id: int | None
    view_id: str | None
    note: str


@dataclass(frozen=True)
class AttemptState:
    action_kind: str
    outcome: str
    status_code: str
    world_step: int
    question_id: str | None
    target_kind: str
    target_id: str
    obs_adapter_id: int | None
    view_id: str | None
    room_name: str
    note: str


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
    question_id: str
    session_id: str
    mode: str
    world_step: int
    agent_round: int
    round_index: int
    max_rounds: int
    nav_used: int
    max_nav_steps: int
    robot_pose_world: tuple[float, float, float] | None
    current_room: str
    current_room_source: str
    current_room_pose_round: int | None
    current_room_world_step: int | None
    current_room_stale: bool
    current_room_conflict: bool
    current_room_graph: str
    current_room_router: str
    in_target_area: bool | None
    target_rooms: tuple[str, ...]
    verified: bool
    last_capture_status: str | None
    verified_evidence_event_ids: tuple[str, ...] = ()
    pending_answer: str | None = None
    pending_answer_obs_id: int | None = None
    pending_answer_present: bool | None = None
    places: tuple[PlaceState, ...] = ()
    frontiers: tuple[FrontierState, ...] = ()
    rooms: tuple[RoomState, ...] = ()
    room_assertions: tuple[RoomAssertionState, ...] = ()
    room_events: tuple[RoomEventState, ...] = ()
    attempts: tuple[AttemptState, ...] = ()
    evidence: tuple[EvidenceState, ...] = ()
    recent_actions: tuple[str, ...] = ()
    loop_flags: tuple[str, ...] = ()
    visible_place_ids: tuple[str, ...] = ()
    visible_place_obs_ids: tuple[int, ...] = ()
    visible_frontier_ids: tuple[str, ...] = ()
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


def _is_attempt_event(event: Any) -> bool:
    """Return whether a world-evidence event mirrors an action-attempt row."""
    payload = dict(getattr(event, "payload", None) or {})
    return "outcome" in payload and "status_code" in payload


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


def _world_step(graph_memory: Any) -> int:
    getter = getattr(graph_memory, "_effective_timestep", None)
    if callable(getter):
        try:
            return int(getter())
        except (TypeError, ValueError):
            pass
    return int(getattr(graph_memory, "_graph_timestep", 0) or 0)


def _room_assertion(
    *,
    source: str,
    room_name: Any,
    confidence: float,
    world_step: int,
    agent_round: int | None,
    pose_round: int | None,
    stale: bool,
    event_id: str | None = None,
    view_id: str | None = None,
) -> RoomAssertionState | None:
    name = str(room_name or "").strip() or "unknown"
    if name == "unknown":
        return None
    return RoomAssertionState(
        source=str(source or "unknown"),
        room_name=name,
        confidence=float(confidence),
        world_step=int(world_step),
        agent_round=int(agent_round) if agent_round is not None else None,
        pose_round=int(pose_round) if pose_round is not None else None,
        stale=bool(stale),
        event_id=str(event_id) if event_id else None,
        view_id=str(view_id) if view_id else None,
    )


def _assertions_conflict(assertions: list[RoomAssertionState] | tuple[RoomAssertionState, ...]) -> bool:
    names = {item.room_name for item in assertions if not item.stale and item.room_name != "unknown"}
    return len(names) > 1


def compile_agent_state(
    executor: Any,
    *,
    max_events: int = 16,
    max_rooms: int = 8,
    max_room_events: int = 8,
    max_attempts: int = 8,
) -> AgentStateSnapshot:
    """Read executor/graph state without refreshing coverage, rooms, or evidence."""
    graph_memory = getattr(executor, "graph_memory", None)
    world = getattr(graph_memory, "world_evidence", None)
    graph_mode = str(getattr(executor, "graph_evidence_mode", "off") or "off")
    history_mode = str(getattr(executor, "room_history_mode", "off") or "off")
    attempt_mode = str(getattr(executor, "attempt_ledger_mode", "off") or "off")
    agent_graph = graph_mode == "agent" and world is not None
    agent_history = history_mode == "agent"
    agent_attempts = attempt_mode == "agent"
    world_step = _world_step(graph_memory)
    agent_round = int(getattr(executor, "_round", 0)) + 1
    trace_meta = dict(getattr(executor, "_trace_meta", None) or {})
    question_id = str(
        getattr(executor, "_question_id", "")
        or trace_meta.get("question_id")
        or trace_meta.get("qid")
        or getattr(world, "question_id", "")
        or hashlib.sha1(str(getattr(executor, "question", "") or "").encode("utf-8")).hexdigest()[:12]
    )
    session_id = str(
        getattr(executor, "_session_id", "")
        or trace_meta.get("session_id")
        or getattr(world, "session_id", "")
        or "session"
    )

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
        for record in sorted(
            world.rooms.values(),
            key=lambda item: (-float(item.confidence), item.room_id),
        )[: max(0, int(max_rooms))]:
            if not record.active:
                continue
            room_assertions_by_source: dict[str, RoomAssertionState] = {}
            for event_id in record.evidence_event_ids:
                event = events_by_id.get(event_id)
                if event is None or event.predicate != "room_label":
                    continue
                assertion = _room_assertion(
                    source=event.source,
                    room_name=event.payload.get("room_name"),
                    confidence=event.confidence,
                    world_step=event.step,
                    agent_round=event.payload.get("agent_round"),
                    pose_round=event.payload.get("pose_round"),
                    stale=False,
                    event_id=event.event_id,
                    view_id=event.view_id,
                )
                if assertion is not None:
                    room_assertions_by_source[assertion.source] = assertion
            room_assertions = tuple(
                sorted(room_assertions_by_source.values(), key=lambda item: (item.source, item.room_name))
            )
            rooms.append(
                RoomState(
                    room_id=record.room_id,
                    room_name=record.room_name,
                    confidence=float(record.confidence),
                    sources=tuple(item.source for item in room_assertions),
                    member_place_ids=record.member_place_ids,
                    centroid_xy=record.centroid_xy,
                    assertions=room_assertions,
                    conflict=_assertions_conflict(room_assertions),
                )
            )

    current_room_stale = bool(getattr(executor, "_room_estimate_stale", False))
    current_pose_round = getattr(executor, "_room_pose_round", None)
    current_room_world_step = getattr(executor, "_room_world_step", None)
    graph_room = str(getattr(executor, "_graph_room_estimate", "") or "unknown")
    router_room = str(
        getattr(
            executor,
            "_last_router_room_estimate",
            getattr(executor, "_last_room_estimate", ""),
        )
        or "unknown"
    )
    current_room = str(getattr(executor, "_last_room_estimate", "") or graph_room or "unknown")
    current_source = str(getattr(executor, "_current_room_source", "") or "")
    current_assertions: list[RoomAssertionState] = []
    graph_assertion = _room_assertion(
        source="graph",
        room_name=graph_room,
        confidence=0.8,
        world_step=world_step,
        agent_round=agent_round,
        pose_round=current_pose_round,
        stale=bool(getattr(executor, "_graph_room_stale", current_room_stale)),
    )
    if graph_assertion is not None:
        current_assertions.append(graph_assertion)
    router_assertion = _room_assertion(
        source="router_vlm",
        room_name=router_room,
        confidence=0.65,
        world_step=int(current_room_world_step if current_room_world_step is not None else world_step),
        agent_round=agent_round,
        pose_round=current_pose_round,
        stale=bool(getattr(executor, "_router_room_stale", current_room_stale)),
    )
    if router_assertion is not None:
        current_assertions.append(router_assertion)
    current_conflict = _assertions_conflict(current_assertions)
    if not current_source:
        if graph_assertion is not None and router_assertion is not None and not current_conflict:
            current_source = "graph+router_vlm"
        elif graph_assertion is not None:
            current_source = "graph"
        elif router_assertion is not None:
            current_source = "router_vlm"
        else:
            current_source = "unknown"

    room_events: list[RoomEventState] = []
    room_events_fn = getattr(graph_memory, "get_room_events", None)
    if agent_history and callable(room_events_fn):
        for row in reversed(list(room_events_fn(limit=max_room_events) or ())):
            obs_id = row.get("obs_id")
            event_id = str(row.get("event_id") or "") or None
            view_id = None
            if event_id and world is not None:
                world_event = next((event for event in world.events if event.event_id == event_id), None)
                view_id = world_event.view_id if world_event is not None else None
            room_events.append(
                RoomEventState(
                    event_id=event_id,
                    world_step=int(row.get("world_step", row.get("step", 0)) or 0),
                    agent_round=(int(row["agent_round"]) if row.get("agent_round") is not None else None),
                    room_name=str(row.get("room") or "unknown"),
                    kind=str(row.get("kind") or "unknown"),
                    phrase=str(row.get("phrase") or ""),
                    obs_adapter_id=int(obs_id) if obs_id is not None else None,
                    view_id=view_id,
                    note=str(row.get("note") or ""),
                )
            )

    attempts: list[AttemptState] = []
    attempts_fn = getattr(graph_memory, "get_attempt_records", None)
    if agent_attempts and callable(attempts_fn):
        rows = list(
            attempts_fn(
                question_id=getattr(graph_memory, "_attempt_ledger_question_id", None),
                limit=max_attempts,
            )
            or ()
        )
        for row in reversed(rows):
            attempts.append(
                AttemptState(
                    action_kind=str(row.action_kind),
                    outcome=str(row.outcome),
                    status_code=str(row.status_code),
                    world_step=int(row.step),
                    question_id=str(row.question_id) if row.question_id is not None else None,
                    target_kind=str(row.target_kind or ""),
                    target_id=str(row.target_id or ""),
                    obs_adapter_id=int(row.obs_id) if row.obs_id is not None else None,
                    view_id=str(row.view_id) if row.view_id else None,
                    room_name=str(row.room or "unknown"),
                    note=str(row.note or ""),
                )
            )

    evidence: list[EvidenceState] = []
    if agent_history and world is not None:
        evidence_question_id = str(getattr(world, "question_id", "") or "")
        candidates = [
            event
            for event in world.events
            if (not evidence_question_id or event.question_id in {None, evidence_question_id})
            and (agent_attempts or not _is_attempt_event(event))
        ]
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

    loop_flags = (
        tuple(
            f"obs={item.get('obs_id')} visits={item.get('visits')} status={item.get('status')}"
            for item in list(getattr(executor, "_nav_loop_flags", None) or ())[-4:]
        )
        if agent_attempts
        else ()
    )
    pending = dict(getattr(executor, "_pending_answerable", None) or {})
    pending_answer = str(pending.get("answer_text") or "").strip()
    if len(pending_answer) == 1 and pending_answer.upper() in "ABCDE":
        pending_answer = ""
    if not pending_answer and pending.get("letter"):
        choice_text = getattr(executor, "_choice_text_for_letter", None)
        if callable(choice_text):
            pending_answer = str(choice_text(pending.get("letter")) or "").strip()
    confirmed = getattr(executor, "_confirmed_answer_evidence", None)
    verified_event_ids = tuple(
        str(item)
        for item in (
            getattr(executor, "_verified_evidence_event_ids", ()) or getattr(confirmed, "evidence_event_ids", ()) or ()
        )
    )
    return AgentStateSnapshot(
        schema_version=2,
        question=str(getattr(executor, "question", "") or ""),
        question_id=question_id,
        session_id=session_id,
        mode=str(getattr(executor, "mode", "answer") or "answer"),
        world_step=world_step,
        agent_round=agent_round,
        round_index=agent_round,
        max_rounds=int(getattr(executor, "max_rounds", 0)),
        nav_used=int(getattr(executor, "_n_nav", 0)) + int(getattr(executor, "_n_explore", 0)),
        max_nav_steps=int(getattr(executor, "max_nav_steps", 0)),
        robot_pose_world=pose,
        current_room=current_room,
        current_room_source=current_source,
        current_room_pose_round=(int(current_pose_round) if current_pose_round is not None else None),
        current_room_world_step=(int(current_room_world_step) if current_room_world_step is not None else None),
        current_room_stale=current_room_stale,
        current_room_conflict=current_conflict,
        current_room_graph=graph_room,
        current_room_router=router_room,
        in_target_area=getattr(executor, "_in_target_area", None),
        target_rooms=target_rooms,
        verified=bool(getattr(executor, "_verified", False)),
        last_capture_status=getattr(executor, "_last_capture_status", None),
        verified_evidence_event_ids=verified_event_ids,
        pending_answer=pending_answer or None,
        pending_answer_obs_id=(int(pending["obs_id"]) if pending.get("obs_id") is not None else None),
        pending_answer_present=(bool(pending["present"]) if pending.get("present") is not None else None),
        places=tuple(places),
        frontiers=tuple(frontiers),
        rooms=tuple(rooms),
        room_assertions=tuple(current_assertions),
        room_events=tuple(room_events),
        attempts=tuple(attempts),
        evidence=tuple(evidence),
        recent_actions=(
            tuple(str(x) for x in list(getattr(executor, "_recent_actions", None) or ())[-8:])
            if agent_attempts
            else ()
        ),
        loop_flags=loop_flags,
        metadata={
            "decision_policy": str(getattr(executor, "decision_policy", "legacy")),
            "graph_evidence_mode": graph_mode,
            "room_history_mode": history_mode,
            "attempt_ledger_mode": attempt_mode,
            "agent_state_max_chars": int(getattr(executor, "agent_state_max_chars", 6000)),
            "room_policy": str(getattr(executor, "room_policy", "canonical")),
        },
    )


def _limited(values: tuple[str, ...], *, limit: int = 8) -> str:
    shown = tuple(str(item) for item in values[: max(0, int(limit))])
    return ",".join(shown) or "none"


def _render_sections(
    fixed_lines: list[str],
    sections: list[tuple[str, list[str], int]],
    *,
    max_chars: int,
) -> str:
    """Fit complete rows, retaining required rows and explicit omission counts."""
    selected = [min(max(0, minimum), len(rows)) for _title, rows, minimum in sections]

    def assemble(counts: list[int]) -> str:
        lines = list(fixed_lines)
        for (title, rows, _minimum), count in zip(sections, counts, strict=True):
            lines.append(title)
            if rows:
                lines.extend(rows[:count])
                omitted = len(rows) - count
                if omitted:
                    lines.append(f"- omitted={omitted} additional rows")
            else:
                lines.append("- none")
        return "\n".join(lines)

    text = assemble(selected)
    if len(text) > max_chars:
        return _render_compact_sections(
            fixed_lines,
            sections,
            max_chars=max_chars,
        )

    # Fill optional rows in deterministic section priority.
    for index, (_title, rows, _minimum) in enumerate(sections):
        while selected[index] < len(rows):
            candidate = list(selected)
            candidate[index] += 1
            candidate_text = assemble(candidate)
            if len(candidate_text) > max_chars:
                break
            selected = candidate
            text = candidate_text
    return text


def _row_fields(row: str, prefixes: tuple[str, ...]) -> str:
    tokens = row.split()
    selected = [token for token in tokens if any(token.startswith(prefix) for prefix in prefixes)]
    return "- " + " ".join(selected)


def _stable_compact(value: Any, *, limit: int = 24) -> str:
    text = str(value)
    if len(text) <= int(limit):
        return text
    return f"sha1:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


def _whole_word_limit(value: Any, *, limit: int) -> str:
    text = str(value).strip()
    if len(text) <= int(limit):
        return text
    words = text.split()
    kept: list[str] = []
    for word in words:
        omitted = len(words) - len(kept) - 1
        suffix = f" … omitted={omitted}w"
        candidate = " ".join((*kept, word))
        if len(candidate) + len(suffix) > int(limit):
            break
        kept.append(word)
    omitted = len(words) - len(kept)
    return f"{' '.join(kept)} … omitted={omitted}w".strip()


def _compact_row(title: str, row: str) -> str:
    if title == "Places:":
        return _row_fields(row, ("place_id=", "obs_adapter=", "room=", "source=", "info_gain="))
    if title == "Frontiers:":
        return _row_fields(row, ("frontier_id=", "obs_adapter=", "status=", "cells=", "info_gain="))
    if title == "Current-room assertions:":
        return _row_fields(row, ("source=", "room=", "stale=", "event_id="))
    if title.startswith("Room events"):
        return _row_fields(
            row,
            ("event_id=", "world_step=", "room=", "kind=", "obs_adapter="),
        ).replace("obs_adapter=", "obs=")
    if title.startswith("Global attempts"):
        return _row_fields(
            row,
            ("world_step=", "action=", "outcome=", "status=", "obs_adapter="),
        ).replace("obs_adapter=", "obs=")
    if title == "Room hypotheses:":
        return _row_fields(row, ("room_id=", "name=", "confidence=", "conflict="))
    if title.startswith("Bulk evidence"):
        return _row_fields(
            row,
            ("event_id=", "step=", "predicate=", "source=", "view_id="),
        )
    return row


def _compact_fixed_lines(fixed_lines: list[str]) -> list[str]:
    lines: list[str] = []
    for line in fixed_lines:
        if line.startswith("Current room:"):
            fields = " ".join(
                token
                for token in line.split()
                if token.startswith(
                    (
                        "canonical=",
                        "source=",
                        "pose_round=",
                        "stale=",
                        "conflict=",
                        "in_target_area=",
                    )
                )
            )
            lines.append(f"Room: {fields}")
        elif line.startswith("Pending answer evidence:"):
            if "answer=" in line and " obs_id=" in line:
                answer = line.split("answer=", 1)[1].split(" obs_id=", 1)[0]
                obs_id = line.split(" obs_id=", 1)[1].split()[0]
                present = line.split(" present=", 1)[1].split()[0]
                lines.append(f"Pending: answer={_whole_word_limit(answer, limit=40)!r} obs={obs_id} present={present}")
            else:
                lines.append("Pending: none")
        elif line.startswith("IDs:"):
            values = {token.split("=", 1)[0]: token.split("=", 1)[1] for token in line.split() if "=" in token}
            lines.append(
                f"IDs: q={_stable_compact(values.get('question_id'))} "
                f"s={_stable_compact(values.get('session_id'))} "
                f"w={values.get('world_step')} r={values.get('agent_round')}"
            )
        elif line.startswith("Budget:"):
            fields = " ".join(token for token in line.split() if token.startswith(("round=", "nav=")))
            lines.append(f"Budget: {fields}")
        elif line.startswith("Verification:"):
            fields = " ".join(token for token in line.split() if token.startswith(("verified=", "evidence_event_ids=")))
            lines.append(f"Verified: {fields.replace('evidence_event_ids=', 'events=')}")
        elif line.startswith("Question:"):
            lines.append(f"Question: {_whole_word_limit(line.removeprefix('Question:'), limit=150)}")
    return lines


def _render_compact_sections(
    fixed_lines: list[str],
    sections: list[tuple[str, list[str], int]],
    *,
    max_chars: int,
) -> str:
    required = {
        "Recent action outcomes:",
        "Loop flags:",
        "Places:",
        "Frontiers:",
        "Room events (latest first):",
        "Global attempts (latest first):",
    }
    counts = [min(1, len(rows)) if title in required else 0 for title, rows, _minimum in sections]

    def assemble(selected: list[int]) -> str:
        lines = _compact_fixed_lines(fixed_lines)
        room_detail_omitted = 0
        short_titles = {
            "Recent action outcomes:": "Recent",
            "Loop flags:": "Loops",
            "Places:": "Places",
            "Frontiers:": "Frontiers",
            "Current-room assertions:": "Room assertions",
            "Room events (latest first):": "Room events",
            "Global attempts (latest first):": "Attempts",
            "Room hypotheses:": "Room hypotheses",
            "Bulk evidence (latest first):": "Evidence",
        }
        for (title, rows, _minimum), count in zip(sections, selected, strict=True):
            if title in {"Current-room assertions:", "Room hypotheses:"} and count == 0:
                room_detail_omitted += len(rows)
                continue
            omitted = len(rows) - count
            suffix = f" omitted={omitted}" if rows else " none"
            lines.append(f"{short_titles.get(title, title.rstrip(':'))}:{suffix}")
            lines.extend(_compact_row(title, row) for row in rows[:count])
        if room_detail_omitted:
            lines.append(f"Rooms: omitted={room_detail_omitted}")
        return "\n".join(lines)

    text = assemble(counts)
    if len(text) > max_chars:
        raise ValueError(
            f"max_chars={max_chars} is too small for the grounded state contract "
            f"(minimum {len(text)} complete characters)"
        )
    for index, (_title, rows, _minimum) in enumerate(sections):
        while counts[index] < len(rows):
            candidate = list(counts)
            candidate[index] += 1
            candidate_text = assemble(candidate)
            if len(candidate_text) > max_chars:
                break
            counts = candidate
            text = candidate_text
    return text


def render_agent_state(snapshot: AgentStateSnapshot, *, max_chars: int = 6000) -> str:
    """Render facts with deterministic section-aware, whole-row budgeting."""
    fixed_lines = [
        f"Question: {snapshot.question}",
        f"State schema: grounded_agent_state/v{snapshot.schema_version}",
        (
            f"IDs: question_id={snapshot.question_id} session_id={snapshot.session_id} "
            f"world_step={snapshot.world_step} agent_round={snapshot.agent_round}"
        ),
        (
            f"Budget: round={snapshot.round_index}/{snapshot.max_rounds} "
            f"nav={snapshot.nav_used}/{snapshot.max_nav_steps} verified={snapshot.verified}"
        ),
        f"Robot world pose: {snapshot.robot_pose_world}",
        (
            f"Current room: canonical={snapshot.current_room} source={snapshot.current_room_source} "
            f"pose_round={snapshot.current_room_pose_round} world_step={snapshot.current_room_world_step} "
            f"stale={snapshot.current_room_stale} conflict={snapshot.current_room_conflict} "
            f"in_target_area={snapshot.in_target_area}"
        ),
        f"Room sources: graph={snapshot.current_room_graph} router={snapshot.current_room_router}",
        (
            f"Verification: verified={snapshot.verified} "
            f"evidence_event_ids={snapshot.verified_evidence_event_ids or 'none'}"
        ),
    ]
    if snapshot.target_rooms:
        fixed_lines.append("Question room hints: " + ",".join(snapshot.target_rooms))
    if snapshot.last_capture_status:
        fixed_lines.append(f"Last capture: {snapshot.last_capture_status}")
    if snapshot.pending_answer and not snapshot.verified:
        fixed_lines.append(
            f"Pending answer evidence: answer={snapshot.pending_answer} "
            f"obs_id={snapshot.pending_answer_obs_id} present={snapshot.pending_answer_present} "
            "(needs corroboration; do not submit yet)"
        )
    else:
        fixed_lines.append("Pending answer evidence: none")

    recent_rows = [f"- {item}" for item in reversed(snapshot.recent_actions)]
    loop_rows = [f"- {item}" for item in reversed(snapshot.loop_flags)]
    place_rows: list[str] = []
    for place in sorted(
        snapshot.places,
        key=lambda item: (-item.information_gain, item.failure_risk, item.place_id),
    ):
        place_rows.append(
            f"- place_id={place.place_id} obs_adapter={place.obs_adapter_id} "
            f"labels={_limited(place.labels, limit=6)} room_id={place.room_id or 'unknown'} "
            f"room={place.room_name} xyz={place.xyz} source={place.source} "
            f"path_cost={place.path_cost} info_gain={place.information_gain:.1f} "
            f"coverage={place.coverage} coverage_gain={place.coverage_gain:.1f} "
            f"bearings={place.approach_bearings or 'none'} failure_risk={place.failure_risk:.2f} "
            f"attempts={_limited(place.attempts, limit=4)} {place.revisit_change}"
        )
    frontier_rows = [
        (
            f"- frontier_id={frontier.frontier_id} obs_adapter={frontier.obs_adapter_id} "
            f"xyz={frontier.xyz} status={frontier.status} cells={frontier.cell_count} "
            f"attachments={_limited(frontier.attachment_ids, limit=4)} "
            f"parents={_limited(frontier.parent_ids, limit=4)} "
            f"info_gain={frontier.information_gain:.1f} path_cost={frontier.path_cost}"
        )
        for frontier in sorted(
            snapshot.frontiers,
            key=lambda item: (-item.information_gain, item.frontier_id),
        )
    ]
    assertion_rows = [
        (
            f"- source={item.source} room={item.room_name} confidence={item.confidence:.2f} "
            f"world_step={item.world_step} agent_round={item.agent_round} "
            f"pose_round={item.pose_round} stale={item.stale} "
            f"event_id={item.event_id or 'none'} view_id={item.view_id or 'none'}"
        )
        for item in snapshot.room_assertions
    ]
    room_event_rows = [
        (
            f"- event_id={item.event_id or 'none'} world_step={item.world_step} "
            f"agent_round={item.agent_round} room={item.room_name} kind={item.kind} "
            f"phrase={item.phrase or 'none'} obs_adapter={item.obs_adapter_id} "
            f"view_id={item.view_id or 'none'} note={item.note or 'none'}"
        )
        for item in snapshot.room_events
    ]
    attempt_rows = [
        (
            f"- world_step={item.world_step} action={item.action_kind} outcome={item.outcome} "
            f"status={item.status_code} question_id={item.question_id or 'none'} "
            f"target={item.target_kind or 'none'}:{item.target_id or 'none'} "
            f"obs_adapter={item.obs_adapter_id} view_id={item.view_id or 'none'} "
            f"room={item.room_name} note={item.note or 'none'}"
        )
        for item in snapshot.attempts
    ]
    room_rows = [
        (
            f"- room_id={room.room_id} name={room.room_name} confidence={room.confidence:.2f} "
            f"sources={_limited(room.sources)} conflict={room.conflict} "
            f"places={_limited(room.member_place_ids, limit=8)}"
        )
        for room in snapshot.rooms
    ]
    evidence_rows: list[str] = []
    for event in reversed(snapshot.evidence):
        labels = ",".join(event.labels) or "unknown"
        details = ";".join(event.details) or "none"
        evidence_rows.append(
            f"- event_id={event.event_id} step={event.step} {event.polarity} "
            f"{event.subject_kind}:{event.subject_id} labels={labels} predicate={event.predicate} "
            f"source={event.source} confidence={event.confidence:.2f} "
            f"view_id={event.view_id or 'none'} details={details}"
        )
    sections = [
        ("Recent action outcomes:", recent_rows, min(1, len(recent_rows))),
        ("Loop flags:", loop_rows, min(1, len(loop_rows))),
        ("Places:", place_rows, min(1, len(place_rows))),
        ("Frontiers:", frontier_rows, min(1, len(frontier_rows))),
        ("Current-room assertions:", assertion_rows, min(1, len(assertion_rows))),
        ("Room events (latest first):", room_event_rows, min(1, len(room_event_rows))),
        ("Global attempts (latest first):", attempt_rows, min(1, len(attempt_rows))),
        ("Room hypotheses:", room_rows, min(1, len(room_rows))),
        ("Bulk evidence (latest first):", evidence_rows, 0),
    ]
    return _render_sections(fixed_lines, sections, max_chars=max(1, int(max_chars)))


def rendered_state_allowlists(snapshot: AgentStateSnapshot, state_text: str) -> dict[str, tuple[Any, ...]]:
    """Derive the exact stable/action IDs present in rendered whole rows."""
    text = str(state_text)
    rendered_event_ids = {
        token.removeprefix("event_id=")
        for line in text.splitlines()
        for token in line.split()
        if token.startswith("event_id=")
    }
    visible_places = tuple(
        item.place_id
        for item in snapshot.places
        if f"- place_id={item.place_id} obs_adapter={item.obs_adapter_id} " in text
    )
    visible_obs = tuple(
        item.obs_adapter_id
        for item in snapshot.places
        if f"- place_id={item.place_id} obs_adapter={item.obs_adapter_id} " in text
    )
    visible_frontiers = tuple(
        item.frontier_id
        for item in snapshot.frontiers
        if f"- frontier_id={item.frontier_id} obs_adapter={item.obs_adapter_id} " in text
    )
    candidate_events = [
        *(item.event_id for item in snapshot.evidence),
        *(item.event_id for item in snapshot.room_events if item.event_id),
        *(item.event_id for item in snapshot.room_assertions if item.event_id),
        *(assertion.event_id for room in snapshot.rooms for assertion in room.assertions if assertion.event_id),
    ]
    visible_events = tuple(dict.fromkeys(event_id for event_id in candidate_events if event_id in rendered_event_ids))
    return {
        "place_ids": visible_places,
        "place_obs_ids": visible_obs,
        "frontier_ids": visible_frontiers,
        "event_ids": visible_events,
    }


def state_text_digest(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

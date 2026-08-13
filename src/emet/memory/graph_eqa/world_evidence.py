# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Stable identities, immutable views, and append-only evidence for GraphEQA."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

WORLD_EVIDENCE_SCHEMA_VERSION = 1
WORLD_EVIDENCE_FILENAME = "world_evidence.json"
WORLD_EVIDENCE_VIEWS_DIR = "world_evidence_views"
WORLD_EVIDENCE_MODES = frozenset({"off", "shadow", "agent"})


def resolve_world_evidence_mode(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    return value if value in WORLD_EVIDENCE_MODES else "off"


def _xyz(value: Any) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    return (
        float(arr[0]) if arr.size > 0 else 0.0,
        float(arr[1]) if arr.size > 1 else 0.0,
        float(arr[2]) if arr.size > 2 else 0.0,
    )


def _pose(value: Any) -> tuple[tuple[float, float, float, float], ...] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.shape != (4, 4):
        return None
    return tuple(tuple(float(x) for x in row) for row in arr)


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    identity_key: str
    place_id: str
    current_node_id: int | None
    labels: tuple[str, ...]
    xyz: tuple[float, float, float]
    first_seen_step: int
    last_seen_step: int
    active: bool = True
    aliases: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntityRecord:
        return cls(
            entity_id=str(data["entity_id"]),
            identity_key=str(data.get("identity_key") or data["entity_id"]),
            place_id=str(data.get("place_id") or f"place_{data['entity_id']}"),
            current_node_id=(
                int(data["current_node_id"]) if data.get("current_node_id") is not None else None
            ),
            labels=tuple(str(x) for x in data.get("labels") or ()),
            xyz=_xyz(data.get("xyz")),
            first_seen_step=int(data.get("first_seen_step") or 0),
            last_seen_step=int(data.get("last_seen_step") or 0),
            active=bool(data.get("active", True)),
            aliases=tuple(str(x) for x in data.get("aliases") or ()),
        )


@dataclass(frozen=True)
class PlaceRecord:
    place_id: str
    entity_id: str
    anchor_xyz: tuple[float, float, float]
    room_id: str | None = None
    active: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaceRecord:
        return cls(
            place_id=str(data["place_id"]),
            entity_id=str(data["entity_id"]),
            anchor_xyz=_xyz(data.get("anchor_xyz")),
            room_id=str(data["room_id"]) if data.get("room_id") else None,
            active=bool(data.get("active", True)),
        )


@dataclass(frozen=True)
class ViewRecord:
    view_id: str
    obs_id: int
    revision: int
    entity_id: str | None
    place_id: str | None
    captured_step: int
    session_id: str
    question_id: str | None
    camera_pose_world: tuple[tuple[float, float, float, float], ...] | None
    base_pose_world: tuple[float, float, float] | None
    object_xyz: tuple[float, float, float]
    labels: tuple[str, ...]
    description: str = ""
    rgb: np.ndarray | None = field(default=None, repr=False, compare=False)

    def to_dict(self, *, rgb_file: str | None = None) -> dict[str, Any]:
        data = asdict(self)
        data.pop("rgb", None)
        data["rgb_file"] = rgb_file
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, rgb: np.ndarray | None = None) -> ViewRecord:
        camera_pose = data.get("camera_pose_world")
        return cls(
            view_id=str(data["view_id"]),
            obs_id=int(data["obs_id"]),
            revision=int(data.get("revision") or 0),
            entity_id=str(data["entity_id"]) if data.get("entity_id") else None,
            place_id=str(data["place_id"]) if data.get("place_id") else None,
            captured_step=int(data.get("captured_step") or 0),
            session_id=str(data.get("session_id") or ""),
            question_id=str(data["question_id"]) if data.get("question_id") is not None else None,
            camera_pose_world=_pose(camera_pose),
            base_pose_world=_xyz(data["base_pose_world"]) if data.get("base_pose_world") is not None else None,
            object_xyz=_xyz(data.get("object_xyz")),
            labels=tuple(str(x) for x in data.get("labels") or ()),
            description=str(data.get("description") or ""),
            rgb=rgb.copy() if isinstance(rgb, np.ndarray) else None,
        )


@dataclass(frozen=True)
class EvidenceEvent:
    event_id: str
    step: int
    session_id: str
    question_id: str | None
    subject_kind: str
    subject_id: str
    predicate: str
    polarity: str
    source: str
    confidence: float
    view_id: str | None = None
    room_id: str | None = None
    place_id: str | None = None
    frontier_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    supersedes: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceEvent:
        return cls(
            event_id=str(data["event_id"]),
            step=int(data.get("step") or 0),
            session_id=str(data.get("session_id") or ""),
            question_id=str(data["question_id"]) if data.get("question_id") is not None else None,
            subject_kind=str(data.get("subject_kind") or "unknown"),
            subject_id=str(data.get("subject_id") or ""),
            predicate=str(data.get("predicate") or ""),
            polarity=str(data.get("polarity") or "unknown"),
            source=str(data.get("source") or "unknown"),
            confidence=float(data.get("confidence") or 0.0),
            view_id=str(data["view_id"]) if data.get("view_id") else None,
            room_id=str(data["room_id"]) if data.get("room_id") else None,
            place_id=str(data["place_id"]) if data.get("place_id") else None,
            frontier_id=str(data["frontier_id"]) if data.get("frontier_id") else None,
            payload=dict(data.get("payload") or {}),
            supersedes=tuple(str(x) for x in data.get("supersedes") or ()),
            contradicts=tuple(str(x) for x in data.get("contradicts") or ()),
        )


@dataclass(frozen=True)
class RoomHypothesis:
    room_id: str
    member_place_ids: tuple[str, ...]
    centroid_xy: tuple[float, float]
    room_name: str = "unknown"
    confidence: float = 0.0
    active: bool = True
    last_seen_step: int = 0
    evidence_event_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoomHypothesis:
        xy = np.asarray(data.get("centroid_xy") or (0.0, 0.0), dtype=float).reshape(-1)
        return cls(
            room_id=str(data["room_id"]),
            member_place_ids=tuple(str(x) for x in data.get("member_place_ids") or ()),
            centroid_xy=(
                float(xy[0]) if xy.size > 0 else 0.0,
                float(xy[1]) if xy.size > 1 else 0.0,
            ),
            room_name=str(data.get("room_name") or "unknown"),
            confidence=float(data.get("confidence") or 0.0),
            active=bool(data.get("active", True)),
            last_seen_step=int(data.get("last_seen_step") or 0),
            evidence_event_ids=tuple(str(x) for x in data.get("evidence_event_ids") or ()),
        )


@dataclass(frozen=True)
class FrontierTrack:
    frontier_id: str
    revision: int
    centroid_xyz: tuple[float, float, float]
    cells: tuple[tuple[int, int], ...]
    status: str = "active"
    obs_id: int | None = None
    support_view_ids: tuple[str, ...] = ()
    attachment_ids: tuple[str, ...] = ()
    parent_ids: tuple[str, ...] = ()
    last_seen_step: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrontierTrack:
        return cls(
            frontier_id=str(data["frontier_id"]),
            revision=int(data.get("revision") or 0),
            centroid_xyz=_xyz(data.get("centroid_xyz")),
            cells=tuple((int(cell[0]), int(cell[1])) for cell in data.get("cells") or ()),
            status=str(data.get("status") or "active"),
            obs_id=int(data["obs_id"]) if data.get("obs_id") is not None else None,
            support_view_ids=tuple(str(x) for x in data.get("support_view_ids") or ()),
            attachment_ids=tuple(str(x) for x in data.get("attachment_ids") or ()),
            parent_ids=tuple(str(x) for x in data.get("parent_ids") or ()),
            last_seen_step=int(data.get("last_seen_step") or 0),
        )


class WorldEvidenceStore:
    """Identity-complete sidecar that can dual-write without changing policy."""

    def __init__(self, *, mode: str = "off", session_id: str = "") -> None:
        self.mode = resolve_world_evidence_mode(mode)
        self.session_id = str(session_id or "")
        self.question_id: str | None = None
        self.entities: dict[str, EntityRecord] = {}
        self.places: dict[str, PlaceRecord] = {}
        self.views: dict[str, ViewRecord] = {}
        self.events: list[EvidenceEvent] = []
        self.rooms: dict[str, RoomHypothesis] = {}
        self.frontiers: dict[str, FrontierTrack] = {}
        self.entity_aliases: dict[str, str] = {}
        self._identity_to_entity: dict[str, str] = {}
        self._node_to_entity: dict[int, str] = {}
        self._obs_revision_to_view: dict[tuple[int, int], str] = {}
        self._obs_current_view: dict[int, str] = {}
        self._next_view = 1
        self._next_event = 1
        self._next_room = 1
        self._next_frontier = 1

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def agent_visible(self) -> bool:
        return self.mode == "agent"

    def set_question_id(self, question_id: str | None) -> None:
        self.question_id = str(question_id) if question_id is not None else None

    def ensure_entity(
        self,
        *,
        identity_key: str,
        node_id: int,
        labels: list[str] | tuple[str, ...],
        xyz: Any,
        step: int,
    ) -> EntityRecord | None:
        if not self.enabled:
            return None
        key = str(identity_key or f"node:{int(node_id)}")
        entity_id = self._identity_to_entity.get(key)
        if entity_id is None:
            suffix = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
            entity_id = f"entity_{suffix}"
            if entity_id in self.entities and self.entities[entity_id].identity_key != key:
                suffix = hashlib.sha1(f"{key}:{len(self.entities)}".encode("utf-8")).hexdigest()[:16]
                entity_id = f"entity_{suffix}"
            place_id = f"place_{suffix}"
            record = EntityRecord(
                entity_id=entity_id,
                identity_key=key,
                place_id=place_id,
                current_node_id=int(node_id),
                labels=tuple(str(x) for x in labels),
                xyz=_xyz(xyz),
                first_seen_step=int(step),
                last_seen_step=int(step),
            )
            self.entities[entity_id] = record
            self.places[place_id] = PlaceRecord(
                place_id=place_id,
                entity_id=entity_id,
                anchor_xyz=record.xyz,
            )
            self._identity_to_entity[key] = entity_id
        else:
            prior = self.entities[entity_id]
            record = replace(
                prior,
                current_node_id=int(node_id),
                labels=tuple(sorted({*prior.labels, *(str(x) for x in labels)})),
                xyz=_xyz(xyz),
                last_seen_step=max(prior.last_seen_step, int(step)),
                active=True,
            )
            self.entities[entity_id] = record
            place = self.places[record.place_id]
            self.places[record.place_id] = replace(place, anchor_xyz=record.xyz, active=True)
        self._node_to_entity[int(node_id)] = entity_id
        return self.entities[entity_id]

    def append_view(
        self,
        *,
        obs_id: int,
        revision: int,
        rgb: Any,
        object_xyz: Any,
        labels: list[str] | tuple[str, ...],
        description: str | None,
        entity_id: str | None,
        place_id: str | None,
        captured_step: int,
        camera_pose_world: Any = None,
        base_pose_world: Any = None,
    ) -> ViewRecord | None:
        if not self.enabled:
            return None
        key = (int(obs_id), int(revision))
        existing = self._obs_revision_to_view.get(key)
        if existing is not None:
            return self.views[existing]
        view_id = f"view_{self._next_view:08d}"
        self._next_view += 1
        rgb_array = np.asarray(rgb).copy() if rgb is not None else None
        base_pose = _xyz(base_pose_world) if base_pose_world is not None else None
        record = ViewRecord(
            view_id=view_id,
            obs_id=int(obs_id),
            revision=int(revision),
            entity_id=entity_id,
            place_id=place_id,
            captured_step=int(captured_step),
            session_id=self.session_id,
            question_id=self.question_id,
            camera_pose_world=_pose(camera_pose_world),
            base_pose_world=base_pose,
            object_xyz=_xyz(object_xyz),
            labels=tuple(str(x) for x in labels),
            description=str(description or ""),
            rgb=rgb_array,
        )
        self.views[view_id] = record
        self._obs_revision_to_view[key] = view_id
        self._obs_current_view[int(obs_id)] = view_id
        return record

    def view_id_for_obs(self, obs_id: int) -> str:
        return self._obs_current_view.get(int(obs_id), "")

    def view_for_obs(self, obs_id: int) -> ViewRecord | None:
        view_id = self.view_id_for_obs(obs_id)
        return self.views.get(view_id) if view_id else None

    def entity_for_node(self, node_id: int) -> EntityRecord | None:
        entity_id = self._node_to_entity.get(int(node_id))
        if entity_id is None:
            return None
        entity_id = self.entity_aliases.get(entity_id, entity_id)
        return self.entities.get(entity_id)

    def reindex_entities(self, nodes: list[Any], *, step: int) -> None:
        if not self.enabled:
            return
        self._node_to_entity = {}
        active_ids: set[str] = set()
        for node in nodes:
            if bool(getattr(node, "is_viewpoint", False)) or bool(getattr(node, "is_frontier", False)):
                continue
            record = self.ensure_entity(
                identity_key=str(getattr(node, "identity_key", None) or f"obs:{int(node.obs_id)}"),
                node_id=int(node.node_id),
                labels=list(getattr(node, "labels", None) or ()),
                xyz=node.xyz,
                step=step,
            )
            if record is not None:
                active_ids.add(record.entity_id)
        for entity_id, record in list(self.entities.items()):
            if entity_id not in active_ids and entity_id not in self.entity_aliases:
                self.entities[entity_id] = replace(record, active=False, current_node_id=None)

    def absorb_entity(self, *, src_node_id: int, dst_node_id: int) -> None:
        src = self.entity_for_node(src_node_id)
        dst = self.entity_for_node(dst_node_id)
        if src is None or dst is None or src.entity_id == dst.entity_id:
            return
        self.entity_aliases[src.entity_id] = dst.entity_id
        self.entities[src.entity_id] = replace(src, active=False, current_node_id=None)
        self.entities[dst.entity_id] = replace(
            dst,
            aliases=tuple(sorted({*dst.aliases, src.entity_id, *src.aliases})),
            labels=tuple(sorted({*dst.labels, *src.labels})),
        )
        self.places[src.place_id] = replace(self.places[src.place_id], active=False)
        self._node_to_entity[int(src_node_id)] = dst.entity_id

    def record_event(
        self,
        *,
        subject_kind: str,
        subject_id: str,
        predicate: str,
        polarity: str,
        source: str,
        confidence: float,
        step: int,
        view_id: str | None = None,
        room_id: str | None = None,
        place_id: str | None = None,
        frontier_id: str | None = None,
        payload: dict[str, Any] | None = None,
        supersedes: tuple[str, ...] = (),
        contradicts: tuple[str, ...] = (),
    ) -> EvidenceEvent | None:
        if not self.enabled:
            return None
        event = EvidenceEvent(
            event_id=f"event_{self._next_event:08d}",
            step=int(step),
            session_id=self.session_id,
            question_id=self.question_id,
            subject_kind=str(subject_kind),
            subject_id=str(subject_id),
            predicate=str(predicate),
            polarity=str(polarity),
            source=str(source),
            confidence=float(confidence),
            view_id=view_id,
            room_id=room_id,
            place_id=place_id,
            frontier_id=frontier_id,
            payload=dict(payload or {}),
            supersedes=tuple(supersedes),
            contradicts=tuple(contradicts),
        )
        self._next_event += 1
        self.events.append(event)
        return event

    def _room_event_exists(self, room_id: str, source: str, room_name: str) -> bool:
        for event in reversed(self.events):
            if event.subject_kind != "room" or event.subject_id != room_id:
                continue
            if event.predicate != "room_label":
                continue
            return (
                event.source == source
                and str(event.payload.get("room_name") or "") == room_name
            )
        return False

    def update_room_hypotheses(
        self,
        clusters: list[Any],
        *,
        node_to_place: dict[int, str],
        step: int,
    ) -> list[RoomHypothesis]:
        """Match disposable clusters onto persistent rooms by members then centroid."""
        if not self.enabled:
            return []
        active = [record for record in self.rooms.values() if record.active]
        unmatched = {record.room_id for record in active}
        output: list[RoomHypothesis] = []
        for cluster in clusters:
            members = tuple(
                sorted(
                    {
                        node_to_place[int(node_id)]
                        for node_id in tuple(getattr(cluster, "node_ids", ()) or ())
                        if int(node_id) in node_to_place
                    }
                )
            )
            centroid = tuple(float(x) for x in getattr(cluster, "centroid_xy", (0.0, 0.0))[:2])
            member_set = set(members)
            best: RoomHypothesis | None = None
            best_score = -1.0
            for candidate in active:
                if candidate.room_id not in unmatched:
                    continue
                prior_set = set(candidate.member_place_ids)
                union = member_set | prior_set
                overlap = len(member_set & prior_set) / len(union) if union else 0.0
                distance = float(
                    np.hypot(
                        centroid[0] - candidate.centroid_xy[0],
                        centroid[1] - candidate.centroid_xy[1],
                    )
                )
                if overlap <= 0.0 and distance > 1.5:
                    continue
                score = 3.0 * overlap + max(0.0, 1.5 - distance) / 1.5
                if score > best_score:
                    best, best_score = candidate, score
            if best is None:
                room_id = f"room_{self._next_room:06d}"
                self._next_room += 1
                best = RoomHypothesis(
                    room_id=room_id,
                    member_place_ids=members,
                    centroid_xy=centroid,
                    last_seen_step=int(step),
                )
            else:
                unmatched.discard(best.room_id)
                best = replace(
                    best,
                    member_place_ids=members,
                    centroid_xy=centroid,
                    active=True,
                    last_seen_step=int(step),
                )
            graph_name = str(getattr(cluster, "room_name", "") or "unknown")
            if graph_name != "unknown" and not self._room_event_exists(
                best.room_id,
                "graph_labels",
                graph_name,
            ):
                event = self.record_event(
                    subject_kind="room",
                    subject_id=best.room_id,
                    predicate="room_label",
                    polarity="positive",
                    source="graph_labels",
                    confidence=0.8,
                    step=step,
                    room_id=best.room_id,
                    payload={"room_name": graph_name},
                )
                event_ids = (
                    (*best.evidence_event_ids, event.event_id)
                    if event is not None
                    else best.evidence_event_ids
                )
                best = replace(
                    best,
                    room_name=graph_name,
                    confidence=max(best.confidence, 0.8),
                    evidence_event_ids=event_ids,
                )
            self.rooms[best.room_id] = best
            for place_id in members:
                place = self.places.get(place_id)
                if place is not None:
                    self.places[place_id] = replace(place, room_id=best.room_id)
            output.append(best)
        for room_id in unmatched:
            prior = self.rooms[room_id]
            self.rooms[room_id] = replace(prior, active=False)
        return output

    def stamp_room(
        self,
        room_id: str,
        room_name: str,
        *,
        source: str,
        confidence: float,
        step: int,
        view_id: str | None = None,
    ) -> RoomHypothesis | None:
        record = self.rooms.get(str(room_id))
        if record is None:
            return None
        name = str(room_name or "unknown")
        if name == "unknown":
            return record
        event = self.record_event(
            subject_kind="room",
            subject_id=record.room_id,
            predicate="room_label",
            polarity="positive",
            source=str(source),
            confidence=float(confidence),
            step=step,
            view_id=view_id,
            room_id=record.room_id,
            payload={"room_name": name},
        )
        event_ids = (
            (*record.evidence_event_ids, event.event_id)
            if event is not None
            else record.evidence_event_ids
        )
        # Keep higher-confidence graph evidence unless this source is at least as strong.
        if record.room_name == "unknown" or float(confidence) >= record.confidence:
            record = replace(
                record,
                room_name=name,
                confidence=float(confidence),
                evidence_event_ids=event_ids,
            )
        else:
            record = replace(record, evidence_event_ids=event_ids)
        self.rooms[record.room_id] = record
        return record

    def update_frontier_tracks(
        self,
        components: list[dict[str, Any]],
        *,
        step: int,
    ) -> list[FrontierTrack]:
        """Match frontier masks across refreshes, retaining split/merge lineage."""
        if not self.enabled:
            return []
        prior_active = [
            record for record in self.frontiers.values() if record.status == "active"
        ]
        used_ids: set[str] = set()
        seen_ids: set[str] = set()
        output: list[FrontierTrack] = []
        for component in components:
            cells = tuple(
                sorted(
                    {
                        (int(cell[0]), int(cell[1]))
                        for cell in component.get("cells") or ()
                    }
                )
            )
            cell_set = set(cells)
            centroid = _xyz(component.get("centroid_xyz"))
            overlaps: list[tuple[float, float, FrontierTrack]] = []
            for prior in prior_active:
                prior_cells = set(prior.cells)
                union = cell_set | prior_cells
                overlap = len(cell_set & prior_cells) / len(union) if union else 0.0
                distance = float(
                    np.linalg.norm(
                        np.asarray(centroid[:2], dtype=float)
                        - np.asarray(prior.centroid_xyz[:2], dtype=float)
                    )
                )
                if overlap > 0.0 or distance <= 0.75:
                    overlaps.append((overlap, -distance, prior))
            overlaps.sort(key=lambda item: (item[0], item[1]), reverse=True)
            parents = tuple(item[2].frontier_id for item in overlaps)
            best = overlaps[0][2] if overlaps else None
            if best is not None and best.frontier_id not in used_ids:
                frontier_id = best.frontier_id
                revision = best.revision + 1
                used_ids.add(frontier_id)
            else:
                frontier_id = f"frontier_{self._next_frontier:06d}"
                self._next_frontier += 1
                revision = 1
            support_view_ids = tuple(
                str(x) for x in component.get("support_view_ids") or () if str(x)
            )
            attachment_ids = tuple(
                str(x) for x in component.get("attachment_ids") or () if str(x)
            )
            record = FrontierTrack(
                frontier_id=frontier_id,
                revision=revision,
                centroid_xyz=centroid,
                cells=cells,
                status="active",
                obs_id=(
                    int(component["obs_id"])
                    if component.get("obs_id") is not None
                    else (best.obs_id if best is not None else None)
                ),
                support_view_ids=support_view_ids,
                attachment_ids=attachment_ids,
                parent_ids=parents if frontier_id not in parents or len(parents) > 1 else best.parent_ids,
                last_seen_step=int(step),
            )
            self.frontiers[frontier_id] = record
            output.append(record)
            seen_ids.add(frontier_id)
            # A merge keeps the best identity and closes the absorbed tracks.
            for _overlap, _distance, prior in overlaps[1:]:
                if prior.frontier_id != frontier_id:
                    self.frontiers[prior.frontier_id] = replace(prior, status="merged")
                    seen_ids.add(prior.frontier_id)
        for prior in prior_active:
            if prior.frontier_id not in seen_ids and prior.frontier_id not in used_ids:
                self.frontiers[prior.frontier_id] = replace(prior, status="stale")
        return output

    def set_frontier_status(self, frontier_id: str, status: str) -> FrontierTrack | None:
        record = self.frontiers.get(str(frontier_id))
        if record is None:
            return None
        record = replace(record, status=str(status))
        self.frontiers[record.frontier_id] = record
        return record

    def set_frontier_obs(self, frontier_id: str, obs_id: int) -> FrontierTrack | None:
        record = self.frontiers.get(str(frontier_id))
        if record is None:
            return None
        record = replace(record, obs_id=int(obs_id))
        self.frontiers[record.frontier_id] = record
        return record

    def frontier_near_xyz(
        self,
        xyz: Any,
        *,
        max_dist_m: float = 1.5,
        active_only: bool = True,
    ) -> FrontierTrack | None:
        target = np.asarray(xyz, dtype=float).reshape(-1)[:2]
        best: FrontierTrack | None = None
        best_distance = float("inf")
        for record in self.frontiers.values():
            if active_only and record.status != "active":
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(record.centroid_xyz[:2], dtype=float) - target
                )
            )
            if distance <= float(max_dist_m) and distance < best_distance:
                best, best_distance = record, distance
        return best

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORLD_EVIDENCE_SCHEMA_VERSION,
            "mode": self.mode,
            "session_id": self.session_id,
            "question_id": self.question_id,
            "next_ids": {
                "view": self._next_view,
                "event": self._next_event,
                "room": self._next_room,
                "frontier": self._next_frontier,
            },
            "entities": [asdict(x) for x in self.entities.values()],
            "places": [asdict(x) for x in self.places.values()],
            "views": [x.to_dict() for x in self.views.values()],
            "events": [asdict(x) for x in self.events],
            "rooms": [asdict(x) for x in self.rooms.values()],
            "frontiers": [asdict(x) for x in self.frontiers.values()],
            "entity_aliases": dict(self.entity_aliases),
        }

    def save(self, directory: str | Path) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        views_dir = root / WORLD_EVIDENCE_VIEWS_DIR
        views_dir.mkdir(exist_ok=True)
        data = self.to_dict()
        view_rows: list[dict[str, Any]] = []
        for record in self.views.values():
            rgb_file = None
            if record.rgb is not None:
                rgb_file = f"{record.view_id}.png"
                Image.fromarray(np.asarray(record.rgb, dtype=np.uint8), mode="RGB").save(
                    views_dir / rgb_file
                )
            view_rows.append(
                record.to_dict(
                    rgb_file=(
                        f"{WORLD_EVIDENCE_VIEWS_DIR}/{rgb_file}" if rgb_file is not None else None
                    )
                )
            )
        data["views"] = view_rows
        path = root / WORLD_EVIDENCE_FILENAME
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path, *, mode: str | None = None) -> WorldEvidenceStore:
        root = Path(directory)
        path = root / WORLD_EVIDENCE_FILENAME
        data = json.loads(path.read_text(encoding="utf-8"))
        store = cls(mode=mode or data.get("mode") or "shadow", session_id=str(data.get("session_id") or ""))
        store.question_id = str(data["question_id"]) if data.get("question_id") is not None else None
        for row in data.get("entities") or ():
            record = EntityRecord.from_dict(row)
            store.entities[record.entity_id] = record
            store._identity_to_entity[record.identity_key] = record.entity_id
            if record.current_node_id is not None:
                store._node_to_entity[record.current_node_id] = record.entity_id
        for row in data.get("places") or ():
            record = PlaceRecord.from_dict(row)
            store.places[record.place_id] = record
        for row in data.get("views") or ():
            rgb = None
            rgb_file = row.get("rgb_file")
            if rgb_file and (root / rgb_file).is_file():
                rgb = np.asarray(Image.open(root / rgb_file).convert("RGB"))
            record = ViewRecord.from_dict(row, rgb=rgb)
            store.views[record.view_id] = record
            store._obs_revision_to_view[(record.obs_id, record.revision)] = record.view_id
            prior = store.view_for_obs(record.obs_id)
            if prior is None or record.revision >= prior.revision:
                store._obs_current_view[record.obs_id] = record.view_id
        store.events = [EvidenceEvent.from_dict(row) for row in data.get("events") or ()]
        store.rooms = {
            record.room_id: record
            for record in (RoomHypothesis.from_dict(row) for row in data.get("rooms") or ())
        }
        store.frontiers = {
            record.frontier_id: record
            for record in (FrontierTrack.from_dict(row) for row in data.get("frontiers") or ())
        }
        store.entity_aliases = {
            str(key): str(value) for key, value in dict(data.get("entity_aliases") or {}).items()
        }
        next_ids = dict(data.get("next_ids") or {})
        store._next_view = max(int(next_ids.get("view") or 1), len(store.views) + 1)
        store._next_event = max(int(next_ids.get("event") or 1), len(store.events) + 1)
        store._next_room = max(int(next_ids.get("room") or 1), len(store.rooms) + 1)
        store._next_frontier = max(int(next_ids.get("frontier") or 1), len(store.frontiers) + 1)
        return store

    def transform_se2(self, transform: Any) -> None:
        matrix = np.asarray(transform, dtype=float).reshape(4, 4)

        def point(value: tuple[float, float, float]) -> tuple[float, float, float]:
            src = np.array([value[0], value[1], value[2], 1.0], dtype=float)
            return _xyz((matrix @ src)[:3])

        for entity_id, record in list(self.entities.items()):
            self.entities[entity_id] = replace(record, xyz=point(record.xyz))
        for place_id, record in list(self.places.items()):
            self.places[place_id] = replace(record, anchor_xyz=point(record.anchor_xyz))
        for view_id, record in list(self.views.items()):
            camera = None
            if record.camera_pose_world is not None:
                camera = _pose(matrix @ np.asarray(record.camera_pose_world, dtype=float))
            base = point(record.base_pose_world) if record.base_pose_world is not None else None
            self.views[view_id] = replace(
                record,
                object_xyz=point(record.object_xyz),
                camera_pose_world=camera,
                base_pose_world=base,
            )
        for room_id, record in list(self.rooms.items()):
            transformed = point((record.centroid_xy[0], record.centroid_xy[1], 0.0))
            self.rooms[room_id] = replace(record, centroid_xy=transformed[:2])
        for frontier_id, record in list(self.frontiers.items()):
            self.frontiers[frontier_id] = replace(record, centroid_xyz=point(record.centroid_xyz))

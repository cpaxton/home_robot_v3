# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Task-local retrieval hypotheses; geometry is trusted only after fresh grounding."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from emet.mapping.voxel_localize import voxel_proposal_id


@dataclass
class QueryCandidate:
    handle: int
    query: str
    source_obs_id: int
    source_revision: int
    xyz: list[float]
    retrieval_score: float | None = None
    instance_id: int | None = None
    grounded_revision: int | None = None
    invalidation_reason: str = ""

    def require_grounding(self, observation_revision: int) -> int:
        """Return the instance identity only for the observation just verified."""
        if self.instance_id is None or self.grounded_revision != observation_revision or self.invalidation_reason:
            raise ValueError("Target requires fresh object-specific grounding")
        return self.instance_id


class QueryCandidates:
    """Bounded candidates referencing existing observations and instance IDs.

    Scores remain retrieval scores. Promotion is an explicit operation performed
    after the detector/depth admission path succeeds, never by score accumulation.
    """

    def __init__(self, max_candidates: int = 64):
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self.max_candidates = max_candidates
        self.records: dict[int, QueryCandidate] = {}
        self._next_index = 0

    def propose(
        self,
        query: str,
        source_obs_id: int,
        source_revision: int,
        xyz,
        *,
        retrieval_score: float | None = None,
    ) -> QueryCandidate:
        query = " ".join(query.lower().split())
        point = np.asarray(xyz, dtype=float)
        if not query or source_obs_id < 1 or source_revision < 0:
            raise ValueError("Candidate requires a query and source observation")
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Candidate requires finite world XYZ")
        for record in self.records.values():
            if (
                record.query == query
                and record.source_obs_id == source_obs_id
                and record.source_revision == source_revision
                and np.allclose(record.xyz, point, atol=1e-6, rtol=0)
            ):
                return record
        # Active references are never silently evicted. The caller releases
        # finished candidates or handles this budget failure explicitly.
        if len(self.records) >= self.max_candidates:
            raise ValueError("Query candidate budget exhausted")
        handle = voxel_proposal_id(self._next_index)
        self._next_index += 1
        record = QueryCandidate(handle, query, source_obs_id, source_revision, point.tolist(), retrieval_score)
        self.records[handle] = record
        return record

    def ground(self, handle: int, *, instance_id: int, observation_revision: int) -> QueryCandidate:
        record = self.records[handle]
        if instance_id < 0 or observation_revision < record.source_revision:
            raise ValueError("Grounding must reference a valid instance and current observation")
        record.instance_id = instance_id
        record.grounded_revision = observation_revision
        record.invalidation_reason = ""
        return record

    def invalidate_instance(self, instance_id: int, reason: str) -> None:
        if not reason:
            raise ValueError("Invalidation requires a reason")
        for record in self.records.values():
            if record.instance_id == instance_id:
                record.grounded_revision = None
                record.invalidation_reason = reason

    def release(self, handle: int) -> None:
        del self.records[handle]

    def to_dict(self) -> dict:
        return {
            "max_candidates": self.max_candidates,
            "next_index": self._next_index,
            "records": [asdict(record) for record in self.records.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> QueryCandidates:
        store = cls(max_candidates=int(data["max_candidates"]))
        store._next_index = int(data["next_index"])
        store.records = {row["handle"]: QueryCandidate(**row) for row in data["records"]}
        # Restoring a checkpoint never authorizes a physical action without a
        # new observation, even if its numerical revision happens to match.
        for record in store.records.values():
            record.grounded_revision = None
            record.invalidation_reason = "checkpoint restored; reacquire geometry"
        return store

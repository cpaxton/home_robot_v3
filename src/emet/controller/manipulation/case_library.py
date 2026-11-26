# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Scenario record + ever-growing case library for whole-body RL training data.

A scenario is a *perception-level* reconstruction: the local occupancy crop, the goal,
and the initial robot state. It is reconstructible either by re-instantiating the sim
scene that produced the voxel, or by synthesizing matching occupancy. The case library
stores these, deduplicated by fingerprint and tagged by failure mode, and supports
stratified sampling so training favors the hardest / least-seen cells.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _canonical(value: Any) -> Any:
    """Recursively convert to JSON-stable primitives for hashing / serialization."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    return value


@dataclass
class ScenarioRecord:
    """One reconstructible training scenario.

    ``initial_voxel`` is the **initial** map (the scene geometry before the attempt):
    either a local 3D binary occupancy crop (numpy array, from a generator) or a 2D
    occupancy dict (``{"kind": "voxel_map_2d", "obstacles", "explored", ...}``, from
    :func:`~emet.controller.manipulation.failure_record.export_initial_voxel_map`).
    It is never a mid-attempt or final snapshot.
    """

    initial_voxel: Any  # initial map (3D occupancy array or 2D occupancy dict)
    goal: dict[str, Any]  # {"kind": "ee_target"|"ee_targets"|"pick_place", ...}
    robot_state: dict[str, Any]  # legacy {"q": [...]} or v2 {"base": ..., "arms": {...}}
    meta: dict[str, Any] = field(default_factory=dict)  # episode_id, seed, robot_id, scene, failure_mode, ...

    def fingerprint(self) -> str:
        """Stable content hash of the scenario identity for dedup.

        Embodiment, profile fingerprint, and schema version are part of the
        identity: the same map, goal, and joint vector on two robots is not the
        same training case, and changing a profile's limits/actuator model makes
        previously collected records a different reset contract. Episode/job
        identifiers are intentionally excluded so re-collection can still
        deduplicate.
        """
        payload = json.dumps(
            {
                "initial_voxel": _canonical(self.initial_voxel),
                "goal": _canonical(self.goal),
                "robot_state": _canonical(self.robot_state),
                "embodiment": str(self.meta.get("embodiment") or ""),
                "embodiment_fingerprint": str(self.meta.get("embodiment_fingerprint") or ""),
                "schema_version": self.meta.get("schema_version"),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def failure_mode(self) -> str:
        return str(self.meta.get("failure_mode") or "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_voxel": _canonical(self.initial_voxel),
            "goal": _canonical(self.goal),
            "robot_state": _canonical(self.robot_state),
            "meta": _canonical(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScenarioRecord:
        initial = d.get("initial_voxel")
        if isinstance(initial, list) and all(isinstance(row, list) for row in initial):
            initial = np.asarray(initial, dtype=np.uint8)
        return cls(
            initial_voxel=initial,
            goal=dict(d.get("goal") or {}),
            robot_state=dict(d.get("robot_state") or {}),
            meta=dict(d.get("meta") or {}),
        )


class CaseLibrary:
    """Persistent, deduplicated store of scenario records."""

    def __init__(self, path: str | Path | None = None):
        self._records: list[ScenarioRecord] = []
        self._fingerprints: set[str] = set()
        self.path = Path(path).expanduser() if path else None

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def add(self, record: ScenarioRecord, *, dedupe: bool = True) -> bool:
        """Append a record; returns True if it was actually added (not a duplicate)."""
        fp = record.fingerprint()
        if dedupe and fp in self._fingerprints:
            return False
        self._records.append(record)
        self._fingerprints.add(fp)
        return True

    def extend(self, records) -> int:
        """Add many records; returns the number actually added."""
        return sum(1 for r in records if self.add(r))

    def by_failure_mode(self) -> dict[str, list[ScenarioRecord]]:
        buckets: dict[str, list[ScenarioRecord]] = {}
        for r in self._records:
            buckets.setdefault(r.failure_mode, []).append(r)
        return buckets

    def sample(
        self,
        n: int,
        *,
        rng: np.random.Generator | None = None,
        failure_mode: str | None = None,
    ) -> list[ScenarioRecord]:
        """Stratified sample *with replacement*: round-robin across failure-mode buckets so
        rare modes aren't starved, and a batch can be larger than the library."""
        rng = rng or np.random.default_rng()
        pool = self._records if failure_mode is None else [r for r in self._records if r.failure_mode == failure_mode]
        if not pool:
            return []
        buckets: dict[str, list[ScenarioRecord]] = {}
        for r in pool:
            buckets.setdefault(r.failure_mode, []).append(r)
        keys = list(buckets)
        out: list[ScenarioRecord] = []
        while len(out) < n:
            for k in keys:
                bucket = buckets[k]
                out.append(bucket[int(rng.integers(0, len(bucket)))])
                if len(out) >= n:
                    break
        return out[:n]

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path or self.path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for r in self._records:
                fh.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")
        return target

    @classmethod
    def load(cls, path: str | Path) -> CaseLibrary:
        lib = cls(path)
        target = Path(path).expanduser()
        if target.is_file():
            for line in target.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    lib.add(ScenarioRecord.from_dict(json.loads(line)))
        return lib

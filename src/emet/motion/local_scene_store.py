# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bound local collision evidence and invalidate opaque leases on retirement.

The navigation map is owned separately. Plans retain leases, not snapshots.
Borrowing serializes geometry queries with eviction; the command boundary must
validate its lease again. Acquisition times and unknown cells are never renewed.
"""

import sys
import threading
import time
import uuid
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass

import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.motion.observed_admission import ObservedEnvironmentAdmission
from emet.motion.observed_support_contact import ObservedSupportContact


class LocalMapUnavailable(ObservationUnavailable):
    def __init__(self, reason):
        self.reason = reason
        super().__init__("stale_observation" if reason == "expired" else "scene_changed_replan")


@dataclass(frozen=True)
class LocalScene:
    admission: ObservedEnvironmentAdmission
    support: ObservedSupportContact

    def __post_init__(self):
        if not isinstance(self.admission, ObservedEnvironmentAdmission) or not isinstance(
            self.support, ObservedSupportContact
        ):
            raise ValueError("invalid_local_scene")
        # Material ownership is already immutable; force its derived allocation
        # before accounting so later dense checks cannot grow an uncounted cache.
        self.admission.background.exclusively_owned_cells(self.support.support_ref)

    def require(self, now):
        self.support.require(self.admission.background, now=now, static_assumption=self.admission.static_assumption)


def resident_bytes(value):
    """Count owned Python/NumPy storage once, including immutable array bases."""
    seen = set()

    def size(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        total = sys.getsizeof(obj)
        if isinstance(obj, np.ndarray):
            return total + (size(obj.base) if obj.base is not None else 0)
        if isinstance(obj, Mapping):
            return total + sum(size(k) + size(v) for k, v in obj.items())
        if isinstance(obj, (tuple, list, set, frozenset)):
            return total + sum(map(size, obj))
        if is_dataclass(obj):
            total += size(vars(obj))
        return total

    return size(value)


@dataclass(frozen=True)
class LocalSceneLease:
    _store: object
    key: str
    generation: str

    @contextmanager
    def borrow(self):
        store = self._store()
        if store is None:
            raise LocalMapUnavailable("retired")
        with store._borrow(self.key, self.generation) as scene:
            yield scene

    def require_current(self):
        with self.borrow():
            pass

    def retire(self):
        store = self._store()
        if store is not None:
            store.retire(self.key, self.generation)


class LocalSceneStore:
    def __init__(self, *, max_bytes=64 * 1024**2, max_entries=1, clock=time.monotonic):
        if type(max_bytes) is not int or max_bytes <= 0 or type(max_entries) is not int or max_entries <= 0:
            raise ValueError("invalid_local_scene_budget")
        self.max_bytes, self.max_entries, self.clock = max_bytes, max_entries, clock
        self._entries, self._lock = OrderedDict(), threading.RLock()
        self._pins = 0
        self.stats = {
            "loads": 0,
            "unloads": 0,
            "stale_rejections": 0,
            "resident_bytes": 0,
            "peak_resident_bytes": 0,
            "load_seconds": 0.0,
        }

    def _retire(self, key):
        _, _, size = self._entries.pop(key)
        self.stats["resident_bytes"] -= size
        self.stats["unloads"] += 1

    def load(self, key, build, *, reserve_bytes):
        """Evict before construction; the builder must also bound its scratch.

        Failure leaves old leases invalid. Nothing falls back to an evicted
        scene. The reservation bounds admitted retained evidence; its measured
        owned size is checked before publication.
        """
        if (
            not isinstance(key, str)
            or not key
            or type(reserve_bytes) is not int
            or not 0 < reserve_bytes <= self.max_bytes
        ):
            raise ValueError("local_scene_budget_exceeded")
        with self._lock:
            if self._pins:
                raise LocalMapUnavailable("busy")
            if key in self._entries:
                self._retire(key)
            while self._entries and (
                len(self._entries) >= self.max_entries or self.stats["resident_bytes"] + reserve_bytes > self.max_bytes
            ):
                self._retire(next(iter(self._entries)))
            start = time.perf_counter()
            scene = build()
            if not isinstance(scene, LocalScene):
                raise ValueError("invalid_local_scene")
            scene.require(self.clock())
            count = resident_bytes(scene)
            if count > reserve_bytes:
                raise ValueError("local_scene_budget_exceeded")
            generation = uuid.uuid4().hex
            self._entries[key] = (generation, scene, count)
            self.stats["loads"] += 1
            self.stats["load_seconds"] += time.perf_counter() - start
            self.stats["resident_bytes"] += count
            self.stats["peak_resident_bytes"] = max(self.stats["peak_resident_bytes"], self.stats["resident_bytes"])
            return LocalSceneLease(weakref.ref(self), key, generation)

    @contextmanager
    def _borrow(self, key, generation):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry[0] != generation:
                self.stats["stale_rejections"] += 1
                raise LocalMapUnavailable("retired")
            try:
                entry[1].require(self.clock())
            except ObservationUnavailable as error:
                self._retire(key)
                self.stats["stale_rejections"] += 1
                raise LocalMapUnavailable("expired") from error
            self._entries.move_to_end(key)
            self._pins += 1
            try:
                yield entry[1]
            finally:
                self._pins -= 1

    def retire(self, key, generation):
        """Idempotently retire this generation without removing a newer map."""
        with self._lock:
            if self._pins:
                raise LocalMapUnavailable("busy")
            entry = self._entries.get(key)
            if entry is not None and entry[0] == generation:
                self._retire(key)

    def invalidate(self):
        with self._lock:
            if self._pins:
                raise LocalMapUnavailable("busy")
            for key in tuple(self._entries):
                self._retire(key)

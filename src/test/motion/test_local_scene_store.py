# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Retiring local evidence invalidates plans without retaining map arrays."""

import gc
import weakref

import numpy as np
import pytest

from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.motion.local_scene_store import LocalMapUnavailable, LocalScene, LocalSceneStore
from emet.motion.observed_admission import ObservedEnvironmentAdmission
from emet.motion.observed_background import ObservedBackground, StaticBackgroundAssumption
from emet.motion.observed_support import ObservedSupportPatch
from emet.motion.observed_support_contact import ObservedSupportContact
from emet.motion.observed_workspace import ObservedWorkspace


def scene():
    cap = CaptureInterval("capture", 10.0, 10.001, 10.01, "clock")
    grid = ObservedWorkspace([-0.2, -0.2, -0.05], 0.05, np.ones((8, 8, 6), np.uint8))
    bg = ObservedBackground(grid, np.full(grid.cells.shape, 10.0), "epoch", frozenset(("capture",)), cap)
    patch = ObservedSupportPatch([[-0.1, -0.1, -0.001], [0.1, 0.1, 0.001]], [0, 0, 0], 0.0, "capture", 10.01, 0.001)
    camera = np.diag([1.0, -1.0, -1.0, 1.0])
    camera[2, 3] = 1.0
    contact = ObservedSupportContact(
        "target",
        "table",
        "epoch",
        "ideal-test",
        patch,
        cap,
        np.ones((21, 21)),
        [[20, 0, 10], [0, 20, 10], [0, 0, 1]],
        camera,
        np.ones((21, 21), bool),
        payload_motion_error_m=0.0,
    )
    admission = ObservedEnvironmentAdmission(
        bg, static_assumption=StaticBackgroundAssumption("epoch", 9.0, 20.0, "isolated-test")
    )
    return LocalScene(admission, contact)


def test_eviction_releases_arrays_and_cannot_revive_an_old_plan_lease():
    store = LocalSceneStore(clock=lambda: 10.02)
    lease = store.load("source", scene, reserve_bytes=1_000_000)
    with lease.borrow() as borrowed:
        ref = weakref.ref(borrowed)
    del borrowed
    newer = store.load("source", scene, reserve_bytes=1_000_000)
    gc.collect()
    assert ref() is None
    newer.require_current()
    with pytest.raises(LocalMapUnavailable, match="scene_changed_replan"):
        lease.require_current()
    store.invalidate()
    assert store.stats["resident_bytes"] == 0
    with pytest.raises(LocalMapUnavailable):
        newer.require_current()


def test_failed_reacquisition_does_not_fall_back_to_old_geometry():
    store = LocalSceneStore(clock=lambda: 10.02)
    lease = store.load("source", scene, reserve_bytes=1_000_000)

    def failed():
        raise RuntimeError("camera unavailable")

    with pytest.raises(RuntimeError):
        store.load("destination", failed, reserve_bytes=1_000_000)
    assert store.stats["resident_bytes"] == 0
    with pytest.raises(LocalMapUnavailable):
        lease.require_current()


def test_acquisition_age_is_preserved_and_expiry_unloads_the_map():
    clock = [10.02]
    store = LocalSceneStore(clock=lambda: clock[0])
    lease = store.load("source", scene, reserve_bytes=1_000_000)
    clock[0] = 21.0
    with pytest.raises(LocalMapUnavailable, match="stale_observation"):
        lease.require_current()
    assert store.stats["resident_bytes"] == 0


def test_retained_byte_budget_and_active_query_prevent_unsafe_replacement():
    store = LocalSceneStore(max_bytes=1_000_000, clock=lambda: 10.02)
    with pytest.raises(ValueError, match="budget_exceeded"):
        store.load("too-big", scene, reserve_bytes=1)
    lease = store.load("source", scene, reserve_bytes=1_000_000)
    with lease.borrow():
        with pytest.raises(LocalMapUnavailable):
            store.invalidate()
    assert 0 < store.stats["resident_bytes"] <= 1_000_000

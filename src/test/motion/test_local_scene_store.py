# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Retiring local evidence invalidates plans without retaining map arrays."""

import gc
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.controller.task.tamp.native_approach import sweep_clear
from emet.motion.local_scene_store import LocalMapUnavailable, LocalScene, LocalSceneStore
from emet.motion.mujoco_collision import MujocoSceneCollisionChecker
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


def test_retiring_old_generation_is_idempotent_and_cannot_retire_replacement():
    store = LocalSceneStore(clock=lambda: 10.02)
    old = store.load('source',scene,reserve_bytes=1_000_000)
    old.retire()
    newer = store.load('source',scene,reserve_bytes=1_000_000)
    old.retire()
    newer.require_current()
    assert store.stats['unloads']==1
    newer.retire()
    assert store.stats['resident_bytes']==0


@pytest.mark.parametrize("blocked", [None, "unknown", "occupied", "native"])
def test_batched_sweep_keeps_map_and_native_rejections_and_lease(monkeypatch, blocked):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    import eval_mapped_native_contact as mapped
    import eval_mobile_mapped_contact as mobile

    clock = [10.02]
    for module in (mapped, mobile):
        monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    obstacle = '<geom type="box" pos=".02 .14 .1" size=".004 .004 .004"/>' if blocked == "native" else ""
    model = mujoco.MjModel.from_xml_string(f"""<mujoco><worldbody>
      <body name="base_link" pos="0 0 .1"><freejoint/><geom size=".006"/>
        <body name="link_lift" pos="0 .14 0"><geom size=".006"/></body>
      </body>
      <body name="target" pos="0 -.1 .1"><freejoint/><geom size=".005"/></body>
      {obstacle}</worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = scene()
    if blocked in ("unknown", "occupied"):
        grid = base.admission.background.workspace
        cells = grid.cells.copy()
        index = np.floor((np.array([.02, .14, .1]) - grid.origin) / grid.resolution).astype(int)
        cells[tuple(index)] = 0 if blocked == "unknown" else 2
        background = replace(base.admission.background, workspace=replace(grid, cells=cells))
        base = LocalScene(ObservedEnvironmentAdmission(background, static_assumption=base.admission.static_assumption),
                          base.support)
    store = LocalSceneStore(clock=lambda: clock[0])
    lease = store.load("source", lambda: base, reserve_bytes=1_000_000)
    start, goal = np.array([0., 0., 0.]), np.array([.04, 0., 0.])
    guard = mobile.LeaseGuard(model, lease)
    exact = MujocoSceneCollisionChecker(
        model, robot_body="base_link",
        additional_scene_checks=(lambda *args: guard.configuration_collides(*args),),
    )
    batched = MujocoSceneCollisionChecker(
        model, robot_body="base_link", additional_scene_checks=(guard.configuration_collides,),
    )
    assert sweep_clear(model, data, start, goal, scene_checker=exact) is (blocked is None)
    exact_queries = guard.map_queries
    guard.map_queries = 0
    assert sweep_clear(model, data, start, goal, scene_checker=batched) is (blocked is None)
    assert (guard.batched_sweeps > 0) is (blocked in (None, "native"))
    if blocked is None:
        assert guard.map_queries < exact_queries
        scoped = guard.for_straight_base_sweep(model, data, start, goal, xy_error=.01, yaw_error=.05)
        clock[0] = 21.
        with pytest.raises(LocalMapUnavailable, match="stale_observation"):
            scoped(batched, model, data)


@pytest.mark.parametrize("case", ["outside", "native_collision", "reenters_map"])
def test_reused_bounds_preserve_native_only_coverage_and_freshness(monkeypatch, case):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    import eval_mapped_native_contact as mapped
    import eval_mobile_mapped_contact as mobile

    clock = [10.02]
    for module in (mapped, mobile):
        monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    obstacle = '<geom type="box" pos=".044 .206 .1" size=".004 .004 .004"/>' if case == "native_collision" else ""
    model = mujoco.MjModel.from_xml_string(f"""<mujoco><worldbody>
      <body name="base_link" pos="0 0 .1"><freejoint/><geom size=".006"/>
        <body name="link_lift" pos="0 .206 0"><geom size=".025"/></body>
      </body>
      <body name="target" pos="0 -.1 .1"><freejoint/><geom size=".005"/></body>
      {obstacle}</worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    store = LocalSceneStore(clock=lambda: clock[0])
    lease = store.load("source", scene, reserve_bytes=1_000_000)
    guard = mobile.LeaseGuard(model, lease)
    start = np.zeros(3)
    goal = np.array([.04, -.055 if case == "reenters_map" else 0., 0.])
    robot_solid_queries = 0
    query = mapped.MappedGuard._query

    def record_query(self, solid, *, payload=False):
        nonlocal robot_solid_queries
        if not payload:
            robot_solid_queries += 1
        return query(self, solid, payload=payload)

    monkeypatch.setattr(mapped.MappedGuard, "_query", record_query)

    def verdict(check):
        rows = []
        checker = MujocoSceneCollisionChecker(model, robot_body="base_link", additional_scene_checks=(check,))
        clear = sweep_clear(model, data, start, goal, scene_checker=checker, event=lambda **row: rows.append(row))
        return clear, rows

    exact = verdict(lambda *args: guard.configuration_collides(*args))
    detailed_queries, out_queries = guard.map_queries, guard.native_only_queries
    guard.map_queries = guard.native_only_queries = 0
    robot_solid_queries = 0

    assert verdict(guard.configuration_collides) == exact
    assert exact[0] is (case != "native_collision")
    assert (guard.native_only_swept_geoms == 1) is (case != "reenters_map")
    assert guard.map_queries == detailed_queries
    assert guard.native_only_queries == out_queries > 0
    assert (robot_solid_queries > 0) is (case == "reenters_map")
    clock[0] = 21.
    with pytest.raises(LocalMapUnavailable, match="stale_observation"):
        verdict(guard.configuration_collides)

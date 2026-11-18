# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Stable physical placement must get the same simulation interval in a slow sim."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_native_adapter import native, table_model  # noqa: F401


def test_release_verification_allows_two_sim_seconds_at_slow_rate(native, monkeypatch):  # noqa: F811
    import emet.controller.task.tamp.native_adapter as module

    robot, _, context, _ = native
    adapter = context["_native_tamp_adapter"]
    adapter.data.qvel[:] = 0
    robot._state["sim_to_real_ratio"] = 0.05
    goal = adapter.data.body("object1").xpos.copy() + [0, 0, 0.02]
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    # Coarse polling keeps this CPU test small; the observed rate remains 0.05x.
    monkeypatch.setattr(module.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + 20 * dt))
    monkeypatch.setattr(adapter, "synchronize", lambda data: setattr(data, "time", data.time + 0.05))
    plan = SimpleNamespace(
        object_body="object1",
        support_body="table",
        goal=goal,
        executor=adapter._executor("object1"),
        placement_region=None,
    )
    assert adapter._verify(plan, released=True, seconds=2.0)
    assert clock[0] >= 40.0
    assert np.isfinite(clock[0])


@pytest.mark.parametrize("rate, wall_limit", [(0.05, 300.0), (None, 30.0), (0.0, 30.0)])
def test_frozen_sim_clock_still_expires(monkeypatch, rate, wall_limit):
    import emet.controller.manipulation.physical_pick_place as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    executor = SimpleNamespace(
        use_sim_time=True,
        settling_wall_timeout_s=30.0,
        robot=SimpleNamespace(_state={"sim_to_real_ratio": rate}),
        _motion_time=lambda: 0.0,
    )
    active = module.PhysicalPickPlaceExecutor._settling_window(executor, 12.0)
    clock[0] = wall_limit - 0.01
    assert active()
    clock[0] = wall_limit
    assert not active()

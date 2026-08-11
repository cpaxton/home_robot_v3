# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""execute_trajectory aborts remaining waypoints after wait_for_waypoint timeout."""

from __future__ import annotations

import numpy as np

from emet.controller.zmq_client import StretchZmqClient


def test_execute_trajectory_aborts_after_wait_timeout(monkeypatch):
    client = StretchZmqClient.__new__(StretchZmqClient)
    moves: list[tuple] = []
    wait_kwargs: list[dict] = []

    def fake_move(pt, **kwargs):
        moves.append((tuple(np.asarray(pt).reshape(-1)[:3]), dict(kwargs)))

    waits = [True, False]  # second intermediate wait fails

    def fake_wait(*_a, **kwargs):
        wait_kwargs.append(kwargs)
        return waits.pop(0) if waits else True

    client.move_base_to = fake_move  # type: ignore[method-assign]
    client.wait_for_waypoint = fake_wait  # type: ignore[method-assign]

    traj = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([3.0, 0.0, 0.0]),
    ]
    ok = StretchZmqClient.execute_trajectory(
        client,
        traj,
        per_waypoint_timeout=1.0,
        final_timeout=1.0,
        world_frame=True,
    )
    assert ok is False
    # First waypoint: non-blocking + wait path; second: starts then wait fails — never reaches index 2/3.
    # Each waypoint calls move_base_to twice (non-blocking then blocking/reliable).
    reached_x = {m[0][0] for m in moves}
    assert 0.0 in reached_x
    assert 1.0 in reached_x
    assert 2.0 not in reached_x
    assert 3.0 not in reached_x
    assert len(wait_kwargs) == 2
    assert all(np.isinf(kwargs["rot_err_threshold"]) for kwargs in wait_kwargs)

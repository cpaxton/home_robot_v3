# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

import numpy as np

from emet.controller.zmq_client import StretchZmqClient
from emet.utils.geometry import nav_xyt_to_world_xyt


def test_get_base_pose_world_composes_navigation_origin(monkeypatch):
    client = object.__new__(StretchZmqClient)
    local = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    origin = [10.0, 20.0, 0.0]
    monkeypatch.setattr(client, "get_base_pose", lambda timeout=5.0: local)
    monkeypatch.setattr(client, "get_emet_session", lambda: {"navigation_origin_xyt": origin})
    world = client.get_base_pose_world()
    expected = nav_xyt_to_world_xyt(local, {"navigation_origin_xyt": origin})
    assert world is not None
    np.testing.assert_allclose(world, expected)


def test_wait_for_waypoint_world_frame_uses_world_pose(monkeypatch):
    client = object.__new__(StretchZmqClient)
    client._finish = False
    # Episode gps far from world goal; world pose matches goal → must succeed with world_frame.
    monkeypatch.setattr(client, "get_base_pose", lambda timeout=5.0: np.array([0.1, 0.2, 0.0]))
    monkeypatch.setattr(client, "get_base_pose_world", lambda timeout=5.0: np.array([1.5, -1.6, 1.57]))
    monkeypatch.setenv("EMET_SIM_NAV_TELEPORT", "0")

    ok = StretchZmqClient.wait_for_waypoint(
        client,
        np.array([1.5, -1.6, 1.57]),
        pos_err_threshold=0.2,
        rot_err_threshold=0.2,
        timeout=1.0,
        world_frame=True,
    )
    assert ok is True


def test_wait_for_waypoint_episode_frame_ignores_world_helper(monkeypatch):
    client = object.__new__(StretchZmqClient)
    client._finish = False
    monkeypatch.setattr(client, "get_base_pose", lambda timeout=5.0: np.array([0.5, -0.5, 0.0]))
    monkeypatch.setattr(
        client,
        "get_base_pose_world",
        lambda timeout=5.0: (_ for _ in ()).throw(AssertionError("should not use world pose")),
    )
    monkeypatch.setenv("EMET_SIM_NAV_TELEPORT", "0")

    ok = StretchZmqClient.wait_for_waypoint(
        client,
        np.array([0.5, -0.5, 0.0]),
        pos_err_threshold=0.2,
        rot_err_threshold=0.2,
        timeout=1.0,
        world_frame=False,
    )
    assert ok is True
